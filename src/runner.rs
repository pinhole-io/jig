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
            steering: None,
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
            fallback_transcript(&messages, &reply)
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
