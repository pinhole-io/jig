//! One turn for any agent spec.
//!
//! The runner builds the prompt, asks the chooser when more than one driver is
//! registered, narrows the tools, streams the model call, and runs the verifier.

use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::agent::AgentSpec;
use crate::completer::SuperGlueCompleter;
use crate::control::{
    Budget, ChooserPick, LaneSpec, TaskType, VerifierRegistry, WorkflowRegistry, WorkflowSpec,
    intersect_tools, triage, with_workflow_prompt,
};
use crate::prompt::{PromptBlocks, PromptBuilder, PromptContext, PromptSlot, PromptSlots};
use crate::stream::{PieceLane, StreamPiece};
use crate::turn::{
    Completer, CompleterError, CompletionRequest, TokenCounts, TurnMessage, TurnRole,
};

/// One control path the runner can select.
pub trait EngineDriver: Send + Sync {
    fn id(&self) -> &'static str;
    fn chooser_line(&self) -> &'static str;
    fn budget(&self) -> Budget;
    fn task_override(&self) -> Option<TaskType> {
        None
    }
    /// Tool names to keep. Empty keeps the workflow tool set.
    fn allowlist(&self) -> &'static [&'static str] {
        &[]
    }
    /// Extra volatile instructions for this lane.
    fn prompt(&self) -> &'static str {
        ""
    }
    /// When true, the turn gets an empty tool registry.
    fn deny_tools(&self) -> bool {
        false
    }
    fn max_rounds(&self, workflow_rounds: u32) -> u32 {
        workflow_rounds
    }
}

/// A driver made of static fields. Use it for a fixed lane such as answer or tool loop.
pub struct StaticDriver {
    pub id: &'static str,
    pub chooser_line: &'static str,
    pub budget: Budget,
    pub task: Option<TaskType>,
    pub allowlist: &'static [&'static str],
    pub prompt: &'static str,
    pub rounds: Option<u32>,
    pub deny_tools: bool,
}

impl EngineDriver for StaticDriver {
    fn id(&self) -> &'static str {
        self.id
    }

    fn chooser_line(&self) -> &'static str {
        self.chooser_line
    }

    fn budget(&self) -> Budget {
        self.budget
    }

    fn task_override(&self) -> Option<TaskType> {
        self.task
    }

    fn allowlist(&self) -> &'static [&'static str] {
        self.allowlist
    }

    fn prompt(&self) -> &'static str {
        self.prompt
    }

    fn deny_tools(&self) -> bool {
        self.deny_tools
    }

    fn max_rounds(&self, workflow_rounds: u32) -> u32 {
        self.rounds.unwrap_or(workflow_rounds)
    }
}

/// Inputs for one agent turn.
pub struct TurnInput<'a> {
    pub spec: &'static AgentSpec,
    pub drivers: &'a [Arc<dyn EngineDriver>],
    pub workflows: &'a WorkflowRegistry,
    pub verifiers: &'a VerifierRegistry,
    pub tools: &'a superglue::tools::ToolRegistry,
    pub history: &'a [TurnMessage],
    /// Prior tool transcript. Empty means `history` is the transcript.
    pub transcript: &'a [superglue::openai::ChatMessage],
    pub user_text: &'a str,
    /// Volatile blocks such as a research brief. They do not change the cached identity.
    pub volatile: &'a [String],
    pub cancel: CancellationToken,
    pub single: bool,
    pub unattended: bool,
    pub server_rounds: u32,
    pub swarm_steps: u32,
    pub model: &'a str,
    pub thinking: &'a str,
    pub temperature: Option<f32>,
    pub steering: Option<superglue::chat::RoundBoundary>,
}

/// The finished turn.
pub struct TurnOutcome {
    pub reply: String,
    pub reasoning: String,
    pub usage: Option<TokenCounts>,
    pub lane: String,
    pub workflow_id: String,
    pub history: Vec<TurnMessage>,
    pub transcript: Vec<superglue::openai::ChatMessage>,
}

/// Why a turn stopped before a reply.
#[derive(Debug)]
pub enum TurnError {
    Cancelled,
    Prompt(&'static str),
    Failed(String),
}

impl std::fmt::Display for TurnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled => write!(f, "turn cancelled"),
            Self::Prompt(reason) => write!(f, "{reason}"),
            Self::Failed(reason) => write!(f, "{reason}"),
        }
    }
}

impl std::error::Error for TurnError {}

pub struct AgentRunner;

impl AgentRunner {
    pub async fn run(
        completer: &SuperGlueCompleter,
        input: TurnInput<'_>,
        mut on_piece: impl FnMut(StreamPiece) + Send,
    ) -> Result<TurnOutcome, TurnError> {
        let drivers = selected_drivers(input.spec, input.drivers);
        if drivers.is_empty() {
            return Err(TurnError::Failed("no driver".to_string()));
        }
        let prompt = build_prompt(input.spec, input.user_text, input.volatile)?;
        let decision = if drivers.len() == 1 {
            sole_decision(input.workflows, drivers[0].as_ref())
        } else {
            let lanes: Vec<(&str, &str)> = drivers
                .iter()
                .map(|driver| (driver.id(), driver.chooser_line()))
                .collect();
            let pick = completer
                .choose(crate::turn::ChooseRequest {
                    request: input.user_text,
                    workflows: input.workflows,
                    lanes: &lanes,
                    cancel: input.cancel.clone(),
                })
                .await
                .map_err(error_from_completer)?;
            let allowed: Vec<&str> = drivers.iter().map(|driver| driver.id()).collect();
            let specs: Vec<LaneSpec> = drivers
                .iter()
                .map(|driver| lane_spec(driver.as_ref()))
                .collect();
            triage(input.workflows, Some(&pick), &allowed, &specs)
        };
        let driver = drivers
            .iter()
            .find(|driver| driver.id() == decision.state.lane)
            .copied()
            .unwrap_or(drivers[0]);
        let mut prompt = with_workflow_prompt(&prompt, &decision.workflow);
        if !driver.prompt().is_empty() {
            prompt.volatile.push(driver.prompt().to_string());
        }
        let tools = narrow_tools(
            input.tools,
            input.spec,
            input.single,
            input.unattended,
            &decision.workflow,
            driver.allowlist(),
            driver.deny_tools(),
        )
        .await;
        let rounds = driver
            .max_rounds(decision.workflow.max_rounds)
            .min(input.spec.rounds(input.server_rounds, input.swarm_steps));
        let mut messages = input.history.to_vec();
        messages.push(TurnMessage {
            role: TurnRole::User,
            text: input.user_text.to_string(),
            images: Vec::new(),
        });
        let mut transcript = input.transcript.to_vec();
        if !transcript.is_empty() {
            transcript.push(superglue::openai::ChatMessage::text(
                "user",
                input.user_text.to_string(),
            ));
        }
        let lane = Arc::new(PieceLane::new());
        let mut reply = String::new();
        let mut reasoning = String::new();
        let result = {
            let run = completer.complete(CompletionRequest {
                prompt: &prompt,
                messages: &messages,
                transcript: &transcript,
                tools: &tools,
                model: input.model,
                thinking: input.thinking,
                max_tool_rounds: rounds,
                cancel: input.cancel.clone(),
                lane: Arc::clone(&lane),
                steering: input.steering,
                answer_after_tool_cap: true,
                temperature: input.temperature,
                watch_tools: true,
            });
            tokio::pin!(run);
            let mut result = None;
            loop {
                let batch = lane.take();
                if batch.is_empty() {
                    if result.is_some() {
                        break;
                    }
                    let notified = lane.notified();
                    tokio::pin!(notified);
                    tokio::select! {
                        biased;
                        finished = &mut run => {
                            result = Some(finished);
                        }
                        _ = &mut notified => {}
                    }
                } else {
                    for piece in batch {
                        match &piece {
                            StreamPiece::Text(text) => reply.push_str(text),
                            StreamPiece::Reasoning(text) => reasoning.push_str(text),
                            StreamPiece::Tool(_) => {}
                        }
                        on_piece(piece);
                    }
                }
            }
            result
        };
        let finished = result.unwrap_or(Err(CompleterError::Failed(
            "turn produced no result".to_string(),
        )));
        let (model_reply, usage, next_transcript) = match finished {
            Ok(reply) => (reply.text, reply.usage, reply.transcript),
            Err(CompleterError::Cancelled)
                if completer.tool_stuck() || input.cancel.is_cancelled() =>
            {
                if input.cancel.is_cancelled() && !completer.tool_stuck() {
                    return Err(TurnError::Cancelled);
                }
                (reply.clone(), None, Vec::new())
            }
            Err(err) => return Err(error_from_completer(err)),
        };
        let reply = if model_reply.is_empty() {
            reply
        } else {
            model_reply
        };
        let _verified = input
            .verifiers
            .get(&decision.state.verifier_id)
            .verify(&decision.state, &tools)
            .await;
        let transcript = if next_transcript.is_empty() {
            if transcript.is_empty() {
                fallback_transcript(&messages, &reply)
            } else {
                if !reply.is_empty() {
                    transcript.push(superglue::openai::ChatMessage::text(
                        "assistant",
                        reply.clone(),
                    ));
                }
                transcript
            }
        } else {
            next_transcript
        };
        let history = history_from_transcript(&transcript);
        Ok(TurnOutcome {
            reply,
            reasoning,
            usage,
            lane: decision.state.lane,
            workflow_id: decision.state.workflow_id,
            history,
            transcript,
        })
    }
}

fn selected_drivers<'a>(
    spec: &AgentSpec,
    drivers: &'a [Arc<dyn EngineDriver>],
) -> Vec<&'a Arc<dyn EngineDriver>> {
    if spec.drivers.is_empty() {
        return drivers.iter().collect();
    }
    spec.drivers
        .iter()
        .filter_map(|id| drivers.iter().find(|driver| driver.id() == *id))
        .collect()
}

fn lane_spec(driver: &dyn EngineDriver) -> LaneSpec {
    LaneSpec {
        id: driver.id(),
        budget: driver.budget(),
        task: driver.task_override(),
    }
}

fn sole_decision(
    registry: &WorkflowRegistry,
    driver: &dyn EngineDriver,
) -> crate::control::TriageDecision {
    let pick = ChooserPick {
        workflow_id: None,
        lane: Some(driver.id().to_string()),
    };
    let allowed = [driver.id()];
    let specs = [lane_spec(driver)];
    triage(registry, Some(&pick), &allowed, &specs)
}

fn build_prompt(
    spec: &AgentSpec,
    user_text: &str,
    volatile: &[String],
) -> Result<PromptBlocks, TurnError> {
    let mut slots = PromptSlots::with_identity(spec.prompt);
    for block in volatile {
        slots
            .add(PromptSlot::DynamicPlugin, block)
            .map_err(TurnError::Prompt)?;
    }
    let context = PromptContext {
        user_text,
        system_note: "",
        current_time: jiff::Timestamp::now(),
    };
    PromptBuilder::new(context, slots)
        .build()
        .map_err(TurnError::Prompt)
}

async fn narrow_tools(
    source: &superglue::tools::ToolRegistry,
    spec: &AgentSpec,
    single: bool,
    unattended: bool,
    workflow: &WorkflowSpec,
    allowlist: &[&str],
    deny_tools: bool,
) -> superglue::tools::ToolRegistry {
    if deny_tools {
        return superglue::tools::ToolRegistry::new();
    }
    let set = spec.tools.select(single, unattended);
    let mut names: Vec<String> = source
        .list_specs()
        .await
        .into_iter()
        .map(|spec| spec.name)
        .filter(|name| set.allows(name))
        .collect();
    if !workflow.tools.is_empty() {
        names.retain(|name| workflow.tools.iter().any(|kept| kept == name));
    }
    if !allowlist.is_empty() {
        names.retain(|name| allowlist.contains(&name.as_str()));
    }
    intersect_tools(source, &names).await
}

fn fallback_transcript(
    messages: &[TurnMessage],
    reply: &str,
) -> Vec<superglue::openai::ChatMessage> {
    let mut transcript = Vec::new();
    for message in messages {
        transcript.push(superglue::openai::ChatMessage::text(
            message.role.as_str(),
            message.text.clone(),
        ));
    }
    if !reply.is_empty() {
        transcript.push(superglue::openai::ChatMessage::text(
            "assistant",
            reply.to_string(),
        ));
    }
    transcript
}

fn history_from_transcript(transcript: &[superglue::openai::ChatMessage]) -> Vec<TurnMessage> {
    transcript
        .iter()
        .filter_map(|message| {
            let role = match message.role.as_str() {
                "user" => TurnRole::User,
                "assistant" => TurnRole::Assistant,
                _ => return None,
            };
            let text = message
                .content
                .as_ref()
                .map(superglue::openai::MessageContent::text_for_summary)
                .unwrap_or_default();
            if role == TurnRole::Assistant && text.is_empty() {
                return None;
            }
            Some(TurnMessage {
                role,
                text,
                images: Vec::new(),
            })
        })
        .collect()
}

fn error_from_completer(err: CompleterError) -> TurnError {
    match err {
        CompleterError::Cancelled => TurnError::Cancelled,
        CompleterError::Credentials(reason)
        | CompleterError::Failed(reason)
        | CompleterError::Plugin(reason) => TurnError::Failed(reason),
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, mpsc};
    use std::thread;

    use async_trait::async_trait;
    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use super::{AgentRunner, EngineDriver, StaticDriver, TurnInput};
    use crate::agent::{
        AgentSpec, CoordinationSource, DoneCheck, RoundBudget, ToolPolicy, ToolSet,
    };
    use crate::completer::{ApiMode, SuperGlueCompleter};
    use crate::control::{Budget, TaskType, VerifierRegistry, WorkflowRegistry};
    use crate::turn::{TurnMessage, TurnRole};

    const ANSWER_SPEC: AgentSpec = AgentSpec {
        id: "test-answer",
        prompt: "Answer briefly.",
        instructions: "",
        tools: ToolPolicy::fixed(ToolSet::All),
        max_tool_rounds: RoundBudget::Fixed(1),
        coordination: CoordinationSource::Single,
        drivers: &["answer"],
        worktree: false,
        done_check: DoneCheck::None,
        children: &[],
        swarm: false,
        primary: true,
        awaits_verify: false,
        tools_error: "",
    };

    const TOOL_SPEC: AgentSpec = AgentSpec {
        id: "test-tools",
        prompt: "Use tools when needed.",
        instructions: "",
        tools: ToolPolicy::fixed(ToolSet::All),
        max_tool_rounds: RoundBudget::Fixed(4),
        coordination: CoordinationSource::Single,
        drivers: &["tool_loop"],
        worktree: false,
        done_check: DoneCheck::None,
        children: &[],
        swarm: false,
        primary: true,
        awaits_verify: false,
        tools_error: "",
    };

    struct EchoTool;

    #[async_trait]
    impl superglue::tools::Tool for EchoTool {
        fn spec(&self) -> superglue::tools::ToolSpec {
            superglue::tools::ToolSpec {
                name: "echo".to_string(),
                description: None,
                parameters_schema: json!({"type": "object"}),
                static_tool: false,
            }
        }

        async fn call(
            &self,
            arguments: serde_json::Value,
        ) -> Result<serde_json::Value, superglue::tools::ToolInvokeError> {
            Ok(json!({"echo": arguments}))
        }
    }

    fn spawn_multi_sse_server(
        responses: Vec<String>,
    ) -> (String, mpsc::Receiver<serde_json::Value>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback server");
        let addr = listener.local_addr().expect("local addr");
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            for sse_body in responses {
                let (mut stream, _) = listener.accept().expect("accept request");
                let mut raw = Vec::new();
                let mut chunk = [0u8; 4096];
                let header_end = loop {
                    let n = stream.read(&mut chunk).expect("read headers");
                    assert!(n > 0, "unexpected EOF before headers");
                    raw.extend_from_slice(&chunk[..n]);
                    if let Some(pos) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                        break pos + 4;
                    }
                };
                let headers = String::from_utf8_lossy(&raw[..header_end]);
                let content_len: usize = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse().ok())
                            .flatten()
                    })
                    .unwrap_or(0);
                while raw.len() < header_end + content_len {
                    let n = stream.read(&mut chunk).expect("read body");
                    if n == 0 {
                        break;
                    }
                    raw.extend_from_slice(&chunk[..n]);
                }
                let body: serde_json::Value =
                    serde_json::from_slice(&raw[header_end..header_end + content_len])
                        .expect("parse JSON body");
                let _ = tx.send(body);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    sse_body.len(),
                    sse_body
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("write SSE response");
            }
        });
        (format!("http://{addr}"), rx)
    }

    fn text_sse(text: &str) -> String {
        let first = json!({
            "id": "chatcmpl_text",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-6-luna",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": text},
                "finish_reason": null,
            }]
        });
        let stop = json!({
            "id": "chatcmpl_text",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-6-luna",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }]
        });
        format!("data: {first}\n\ndata: {stop}\n\ndata: [DONE]\n\n")
    }

    fn tool_call_sse(name: &str, args: &str) -> String {
        let start = json!({
            "id": "chatcmpl_tool",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-6-luna",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }]
                },
                "finish_reason": null,
            }]
        });
        let finish = json!({
            "id": "chatcmpl_tool",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-6-luna",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": args},
                    }]
                },
                "finish_reason": "tool_calls",
            }]
        });
        format!("data: {start}\n\ndata: {finish}\n\ndata: [DONE]\n\n")
    }

    fn build_test_completer(base_url: String) -> SuperGlueCompleter {
        let http = superglue::http::HttpClient::new(superglue::http::ClientConfig::default())
            .expect("http client");
        let completer = SuperGlueCompleter::from_client(http)
            .with_mode(ApiMode::ChatCompletions)
            .with_base_url(base_url.clone());
        let mut creds = superglue::providers::ProviderCredentials::default();
        creds.insert_key(superglue::providers::ProviderId::OpenAi, "test-key");
        creds.insert_base_url(superglue::providers::ProviderId::OpenAi, base_url);
        completer.set_credentials(creds);
        completer
    }

    #[tokio::test]
    async fn runner_invokes_round_boundary_steering_between_tool_rounds() {
        let (base_url, rx) =
            spawn_multi_sse_server(vec![tool_call_sse("echo", "{\"x\":1}"), text_sse("done")]);
        let completer = build_test_completer(base_url);
        let tools = superglue::tools::ToolRegistry::new();
        tools
            .register(Arc::new(EchoTool))
            .await
            .expect("register echo");
        let drivers: Vec<Arc<dyn EngineDriver>> = vec![Arc::new(StaticDriver {
            id: "tool_loop",
            chooser_line: "Use tools.",
            budget: Budget::tool_loop(),
            task: None,
            allowlist: &[],
            prompt: "",
            rounds: Some(4),
            deny_tools: false,
        })];
        let workflows = WorkflowRegistry::with_chat(4);
        let verifiers = VerifierRegistry::with_builtins();
        let boundary_calls = Arc::new(AtomicUsize::new(0));
        let boundary_calls_hook = Arc::clone(&boundary_calls);
        let steering: superglue::chat::RoundBoundary = Arc::new(move || {
            let calls = Arc::clone(&boundary_calls_hook);
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(vec![superglue::openai::ChatMessage::text(
                    "user",
                    "steered after tool round",
                )])
            })
        });

        let outcome = AgentRunner::run(
            &completer,
            TurnInput {
                spec: &TOOL_SPEC,
                drivers: &drivers,
                workflows: &workflows,
                verifiers: &verifiers,
                tools: &tools,
                history: &[],
                transcript: &[],
                user_text: "run echo",
                volatile: &[],
                cancel: CancellationToken::new(),
                single: true,
                unattended: false,
                server_rounds: 4,
                swarm_steps: 4,
                model: "gpt-6-luna",
                thinking: "none",
                temperature: None,
                steering: Some(steering),
            },
            |_| {},
        )
        .await
        .expect("turn succeeds");

        assert_eq!(outcome.reply, "done");
        assert_eq!(boundary_calls.load(Ordering::SeqCst), 1);
        let _round_one_body = rx.recv().expect("round 1 request");
        let round_two_body = rx.recv().expect("round 2 request");
        let round_two_msgs = round_two_body["messages"]
            .as_array()
            .expect("round 2 messages");
        assert!(
            round_two_msgs.iter().any(|msg| {
                msg["role"] == "user" && msg["content"] == "steered after tool round"
            }),
            "round 2 request must include steered boundary message: {round_two_msgs:?}"
        );
    }

    #[tokio::test]
    async fn tool_free_lane_preserves_prior_tool_and_image_transcript() {
        let (base_url, _rx) = spawn_multi_sse_server(vec![text_sse("summary reply")]);
        let completer = build_test_completer(base_url);
        let tools = superglue::tools::ToolRegistry::new();
        let drivers: Vec<Arc<dyn EngineDriver>> = vec![Arc::new(StaticDriver {
            id: "answer",
            chooser_line: "Answer without tools.",
            budget: Budget::answer(),
            task: Some(TaskType::Chat),
            allowlist: &[],
            prompt: "",
            rounds: Some(1),
            deny_tools: true,
        })];
        let workflows = WorkflowRegistry::with_chat(1);
        let verifiers = VerifierRegistry::with_builtins();

        let prior_transcript = vec![
            superglue::openai::ChatMessage {
                role: "user".to_string(),
                content: Some(superglue::openai::MessageContent::Parts(vec![
                    superglue::openai::ContentPart::Text {
                        text: "inspect this screenshot".to_string(),
                    },
                    superglue::openai::ContentPart::ImageUrl {
                        image_url: superglue::openai::ImageUrl {
                            url: "data:image/png;base64,aW1hZ2U=".to_string(),
                            detail: None,
                        },
                    },
                ])),
                ..Default::default()
            },
            superglue::openai::ChatMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![superglue::openai::ToolCall {
                    id: "call_prev".to_string(),
                    kind: "function".to_string(),
                    function: superglue::openai::FunctionCall {
                        name: "echo".to_string(),
                        arguments: "{\"ok\":true}".to_string(),
                    },
                }]),
                ..Default::default()
            },
            superglue::openai::ChatMessage {
                role: "tool".to_string(),
                content: Some(superglue::openai::MessageContent::Text(
                    "{\"echo\":{\"ok\":true}}".to_string(),
                )),
                tool_call_id: Some("call_prev".to_string()),
                name: Some("echo".to_string()),
                ..Default::default()
            },
            superglue::openai::ChatMessage::text("assistant", "I checked it."),
        ];
        let prior_history = vec![
            TurnMessage {
                role: TurnRole::User,
                text: "inspect this screenshot".to_string(),
                images: Vec::new(),
            },
            TurnMessage {
                role: TurnRole::Assistant,
                text: "I checked it.".to_string(),
                images: Vec::new(),
            },
        ];

        let outcome = AgentRunner::run(
            &completer,
            TurnInput {
                spec: &ANSWER_SPEC,
                drivers: &drivers,
                workflows: &workflows,
                verifiers: &verifiers,
                tools: &tools,
                history: &prior_history,
                transcript: &prior_transcript,
                user_text: "summarize what happened",
                volatile: &[],
                cancel: CancellationToken::new(),
                single: true,
                unattended: false,
                server_rounds: 1,
                swarm_steps: 1,
                model: "gpt-6-luna",
                thinking: "none",
                temperature: None,
                steering: None,
            },
            |_| {},
        )
        .await
        .expect("turn succeeds");

        assert_eq!(outcome.reply, "summary reply");
        assert_eq!(outcome.transcript.len(), 6);
        assert!(matches!(
            outcome.transcript[0].content.as_ref(),
            Some(superglue::openai::MessageContent::Parts(parts)) if parts.len() == 2
        ));
        assert!(outcome.transcript[1].tool_calls.is_some());
        assert_eq!(outcome.transcript[2].role, "tool");
        assert_eq!(
            outcome.transcript[4]
                .content
                .as_ref()
                .and_then(|c| c.as_text()),
            Some("summarize what happened")
        );
        assert_eq!(
            outcome.transcript[5]
                .content
                .as_ref()
                .and_then(|c| c.as_text()),
            Some("summary reply")
        );
    }
}
