//! SuperGlue streaming completion.
//!
//! Chat Completions is the default path. Responses keeps tool calls when a model
//! such as `gpt-6-luna` runs with `reasoning_effort = "high"`.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use base64::Engine;
use tokio_util::sync::CancellationToken;

use crate::clock::refresh_clock;
use crate::control::{
    CHOOSER_MIN_CONFIDENCE, ChooserPick, FailureTracker, TRIAGE_MODEL, chooser_questions,
    chooser_state, pick_from_answers,
};
use crate::prompt::PromptBlocks;
use crate::stream::{PieceLane, StreamPiece, ToolPiece, ToolPoint};
use crate::turn::{
    ChooseRequest, Completer, CompleterError, CompletionRequest, ModelReply, TokenCounts,
    TurnMessage,
};

const PREVIEW_CHARS: usize = 4_000;

/// Which provider API sends the turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApiMode {
    /// `POST /v1/chat/completions` with the tool-loop stream.
    ChatCompletions,
    /// `POST /v1/responses` with the tool-loop stream.
    Responses,
}

struct ToolWatch {
    lane: Mutex<Option<Arc<PieceLane>>>,
    cancel: Mutex<Option<CancellationToken>>,
    failures: Mutex<FailureTracker>,
}

impl ToolWatch {
    fn new() -> Self {
        Self {
            lane: Mutex::new(None),
            cancel: Mutex::new(None),
            failures: Mutex::new(FailureTracker::default()),
        }
    }

    fn arm(&self, lane: Arc<PieceLane>, cancel: CancellationToken) {
        let mut failures = lock(&self.failures);
        *failures = FailureTracker::default();
        drop(failures);
        *lock(&self.lane) = Some(lane);
        *lock(&self.cancel) = Some(cancel);
    }

    fn disarm(&self) {
        *lock(&self.lane) = None;
        *lock(&self.cancel) = None;
    }

    fn stuck(&self) -> bool {
        lock(&self.failures).stuck
    }
}

struct WatchHook {
    watch: Arc<ToolWatch>,
}

#[async_trait]
impl superglue::hooks::HookHandler for WatchHook {
    async fn execute(
        &self,
        ctx: superglue::hooks::HookContext,
    ) -> Result<superglue::hooks::HookContext, superglue::hooks::HookError> {
        let lane = lock(&self.watch.lane).clone();
        let Some(lane) = lane else {
            return Ok(ctx);
        };
        let name = ctx
            .metadata
            .get("tool_name")
            .and_then(|value| value.as_str())
            .unwrap_or("tool")
            .to_string();
        if name == "agents" {
            return Ok(ctx);
        }
        match ctx.stage {
            superglue::hooks::HookStage::PreTool => {
                lane.push(StreamPiece::Tool(ToolPiece {
                    name,
                    point: ToolPoint::Call,
                    ok: true,
                    error: String::new(),
                    arguments: clip(&ctx.content, PREVIEW_CHARS),
                    call_id: String::new(),
                    result: String::new(),
                }));
            }
            superglue::hooks::HookStage::PostTool => {
                let arguments = ctx
                    .metadata
                    .get("arguments")
                    .and_then(|value| value.as_str())
                    .unwrap_or("")
                    .to_string();
                let error = tool_error(&ctx.content);
                let ok = error.is_none();
                if let Some(error) = error.as_deref() {
                    let stuck = lock(&self.watch.failures).note(&name, error, &arguments);
                    if stuck && let Some(cancel) = lock(&self.watch.cancel).clone() {
                        cancel.cancel();
                    }
                }
                lane.push(StreamPiece::Tool(ToolPiece {
                    name,
                    point: ToolPoint::Result,
                    ok,
                    error: error.unwrap_or_default(),
                    arguments: clip(&arguments, PREVIEW_CHARS),
                    call_id: String::new(),
                    result: clip(&ctx.content, PREVIEW_CHARS),
                }));
            }
            _ => {}
        }
        Ok(ctx)
    }
}

struct Armed {
    watch: Arc<ToolWatch>,
}

impl Drop for Armed {
    fn drop(&mut self) {
        self.watch.disarm();
    }
}

/// Production completer. Credentials stay in [`secrecy`] types inside SuperGlue.
pub struct SuperGlueCompleter {
    http: superglue::http::HttpClient,
    credentials: Mutex<Arc<superglue::providers::ProviderCredentials>>,
    base_url: String,
    hooks: Arc<superglue::hooks::HookRegistry>,
    guardrails: Arc<superglue::guardrails::GuardrailRegistry>,
    mode: ApiMode,
    watch: Arc<ToolWatch>,
    watch_installed: AtomicBool,
}

impl SuperGlueCompleter {
    /// Build a client and load credentials from the process environment.
    pub fn try_from_env() -> Result<Self, String> {
        let http = superglue::http::HttpClient::new(superglue::http::ClientConfig::default())
            .map_err(|err| safe_line(&err))?;
        Ok(Self::from_client(http))
    }

    /// Use a caller-built HTTP client. Credentials start as [`ProviderCredentials::from_env`].
    pub fn from_client(http: superglue::http::HttpClient) -> Self {
        Self {
            http,
            credentials: Mutex::new(Arc::new(
                superglue::providers::ProviderCredentials::from_env(),
            )),
            base_url: "https://api.openai.com".to_string(),
            hooks: Arc::new(superglue::hooks::HookRegistry::new()),
            guardrails: Arc::new(superglue::guardrails::GuardrailRegistry::new()),
            mode: ApiMode::ChatCompletions,
            watch: Arc::new(ToolWatch::new()),
            watch_installed: AtomicBool::new(false),
        }
    }

    /// Use caller-owned hook and guardrail registries.
    pub fn from_shared(
        http: superglue::http::HttpClient,
        hooks: Arc<superglue::hooks::HookRegistry>,
        guardrails: Arc<superglue::guardrails::GuardrailRegistry>,
    ) -> Self {
        let mut completer = Self::from_client(http);
        completer.hooks = hooks;
        completer.guardrails = guardrails;
        completer
    }

    /// Register the tool watch. Call this once before the first completion.
    pub async fn install_tool_watch(&self) {
        self.hooks
            .add(
                superglue::hooks::HookStage::PreTool,
                superglue::hooks::HookConfig {
                    name: "jig-tool-watch".to_string(),
                    error_strategy: superglue::hooks::HookErrorStrategy::Skip,
                    handler: Arc::new(WatchHook {
                        watch: Arc::clone(&self.watch),
                    }),
                },
            )
            .await;
        self.hooks
            .add(
                superglue::hooks::HookStage::PostTool,
                superglue::hooks::HookConfig {
                    name: "jig-tool-watch".to_string(),
                    error_strategy: superglue::hooks::HookErrorStrategy::Skip,
                    handler: Arc::new(WatchHook {
                        watch: Arc::clone(&self.watch),
                    }),
                },
            )
            .await;
    }

    pub fn with_mode(mut self, mode: ApiMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Replace the credentials used by the next call.
    pub fn set_credentials(&self, credentials: superglue::providers::ProviderCredentials) {
        *lock(&self.credentials) = Arc::new(credentials);
    }

    pub fn hooks(&self) -> &superglue::hooks::HookRegistry {
        &self.hooks
    }

    pub fn hooks_handle(&self) -> Arc<superglue::hooks::HookRegistry> {
        Arc::clone(&self.hooks)
    }

    pub fn guardrails(&self) -> &superglue::guardrails::GuardrailRegistry {
        &self.guardrails
    }

    pub fn guardrails_handle(&self) -> Arc<superglue::guardrails::GuardrailRegistry> {
        Arc::clone(&self.guardrails)
    }

    /// True after the tool watch cancels a repeated failure.
    pub fn tool_stuck(&self) -> bool {
        self.watch.stuck()
    }
}

impl Completer for SuperGlueCompleter {
    fn complete<'a>(
        &'a self,
        request: CompletionRequest<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<ModelReply, CompleterError>> + Send + 'a>> {
        Box::pin(async move {
            if request.watch_tools && !self.watch_installed.swap(true, Ordering::AcqRel) {
                self.install_tool_watch().await;
            }
            let _armed = if request.watch_tools {
                self.watch
                    .arm(Arc::clone(&request.lane), request.cancel.clone());
                Some(Armed {
                    watch: Arc::clone(&self.watch),
                })
            } else {
                None
            };
            let CompletionRequest {
                prompt,
                messages,
                transcript,
                tools,
                model,
                thinking,
                max_tool_rounds,
                cancel,
                lane,
                steering,
                answer_after_tool_cap,
                temperature,
                watch_tools: _,
            } = request;
            let caller = if transcript.is_empty() {
                messages.iter().map(chat_message).collect()
            } else {
                transcript.to_vec()
            };
            let mut options = superglue::chat::ChatOptions::new(&self.base_url, "", model);
            options.provider_credentials = Some(Arc::clone(&lock(&self.credentials)));
            let prompt = refresh_clock(prompt);
            apply_prompt_blocks(&mut options, &prompt);
            options.cancel = Some(cancel);
            options.max_tool_rounds = max_tool_rounds;
            options.answer_after_tool_cap = answer_after_tool_cap;
            options.reasoning_effort = Some(thinking.to_string());
            options.extra_json = Some(serde_json::json!({
                "reasoning_effort": thinking,
            }));
            if let Some(temperature) = temperature {
                options.temperature = Some(temperature);
            }
            let last_usage = Arc::new(Mutex::new(None));
            let use_tools = !tools.list_specs().await.is_empty();
            watch_model(&mut options, Arc::clone(&last_usage), &lane, !use_tools).await;
            let text_lane = Arc::clone(&lane);
            let reason_lane = Arc::clone(&lane);
            let on_delta = move |chunk: String| {
                text_lane.push(StreamPiece::Text(chunk));
            };
            let on_reasoning = move |chunk: String| {
                reason_lane.push(StreamPiece::Reasoning(chunk));
            };
            let outcome = match self.mode {
                ApiMode::ChatCompletions if use_tools => {
                    superglue::chat::stream_complete_with_tools_at_boundary(
                        &self.http,
                        tools,
                        &self.hooks,
                        &self.guardrails,
                        caller,
                        &options,
                        on_delta,
                        on_reasoning,
                        steering,
                    )
                    .await
                    .map(|item| (item.content, item.usage, item.messages))
                    .map_err(|err| map_chat_error(&err))
                }
                ApiMode::ChatCompletions => superglue::chat::stream_complete(
                    &self.http,
                    &self.hooks,
                    &self.guardrails,
                    caller,
                    &options,
                    on_delta,
                )
                .await
                .map(|item| (item.content, item.usage, Vec::new()))
                .map_err(|err| map_chat_error(&err)),
                ApiMode::Responses => superglue::responses::stream_complete_with_tools(
                    &self.http,
                    tools,
                    &self.hooks,
                    &self.guardrails,
                    caller,
                    &options,
                    on_delta,
                    on_reasoning,
                )
                .await
                .map(|item| (item.content, item.usage, item.messages))
                .map_err(|err| map_response_error(&err)),
            };
            match outcome {
                Ok((text, reported, messages)) => {
                    let usage =
                        take_usage(&last_usage).or_else(|| reported.as_ref().and_then(counts_from));
                    Ok(ModelReply {
                        text,
                        usage,
                        transcript: caller_transcript(messages),
                    })
                }
                Err(err) => Err(err),
            }
        })
    }

    fn choose<'a>(
        &'a self,
        request: ChooseRequest<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<ChooserPick, CompleterError>> + Send + 'a>> {
        Box::pin(async move {
            let state = chooser_state(request.request);
            let questions = chooser_questions(request.workflows, request.lanes);
            let body = superglue::systemone::SystemOneRequest::new(state, questions)
                .with_model(TRIAGE_MODEL);
            let credentials = Arc::clone(&lock(&self.credentials));
            let call = superglue::systemone::system_one(&self.http, credentials.as_ref(), body);
            tokio::pin!(call);
            let response = tokio::select! {
                biased;
                () = request.cancel.cancelled() => {
                    return Err(CompleterError::Cancelled);
                }
                result = &mut call => result,
            };
            match response {
                Ok(response) => {
                    let allowed: Vec<&str> = request.lanes.iter().map(|(id, _)| *id).collect();
                    Ok(pick_from_answers(
                        &response,
                        request.workflows,
                        CHOOSER_MIN_CONFIDENCE,
                        &allowed,
                    ))
                }
                Err(err) => Err(map_system_one(&err)),
            }
        })
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn caller_transcript(
    messages: Vec<superglue::openai::ChatMessage>,
) -> Vec<superglue::openai::ChatMessage> {
    messages
        .into_iter()
        .filter(|message| message.role != "system")
        .collect()
}

pub fn chat_message(message: &TurnMessage) -> superglue::openai::ChatMessage {
    if message.images.is_empty() {
        return superglue::openai::ChatMessage::text(message.role.as_str(), message.text.clone());
    }
    let mut parts = Vec::with_capacity(message.images.len() + 1);
    if !message.text.is_empty() {
        parts.push(superglue::openai::ContentPart::Text {
            text: message.text.clone(),
        });
    }
    for image in &message.images {
        let data = base64::engine::general_purpose::STANDARD.encode(&image.data);
        parts.push(superglue::openai::ContentPart::ImageUrl {
            image_url: superglue::openai::ImageUrl {
                url: format!("data:{};base64,{data}", image.media_type),
                detail: None,
            },
        });
    }
    superglue::openai::ChatMessage {
        role: message.role.as_str().to_string(),
        content: Some(superglue::openai::MessageContent::Parts(parts)),
        ..Default::default()
    }
}

fn apply_prompt_blocks(options: &mut superglue::chat::ChatOptions, prompt: &PromptBlocks) {
    options.system_prompt_blocks = Some(
        prompt
            .cached
            .iter()
            .map(|text| superglue::chat::SystemPromptBlock::cached(text.clone()))
            .collect(),
    );
    if !prompt.volatile.is_empty() {
        options.volatile_suffix_blocks = Some(
            prompt
                .volatile
                .iter()
                .map(|text| superglue::chat::SystemPromptBlock::uncached(text.clone()))
                .collect(),
        );
    }
}

async fn watch_model(
    options: &mut superglue::chat::ChatOptions,
    usage_slot: Arc<Mutex<Option<TokenCounts>>>,
    lane: &Arc<PieceLane>,
    forward_reasoning: bool,
) {
    let emitter = Arc::new(superglue::events::StatusEmitter::new());
    options.status_emitter = Some(Arc::clone(&emitter));
    let lane = Arc::clone(lane);
    emitter
        .subscribe(Arc::new(superglue::events::FnSubscriber(
            move |event: superglue::events::ProcessEvent| {
                if forward_reasoning
                    && event.kind == superglue::events::ProcessEventKind::ReasoningDelta
                {
                    let Some(text) = event.metadata.get("delta") else {
                        return;
                    };
                    if text.is_empty() {
                        return;
                    }
                    lane.push(StreamPiece::Reasoning(text.clone()));
                    return;
                }
                if event.kind != superglue::events::ProcessEventKind::LlmCallEnd {
                    return;
                }
                let Some(usage) = event.usage else {
                    return;
                };
                let Some(counts) = counts_from(&usage) else {
                    return;
                };
                *lock(&usage_slot) = Some(counts);
            },
        )))
        .await;
}

fn take_usage(slot: &Mutex<Option<TokenCounts>>) -> Option<TokenCounts> {
    lock(slot).take()
}

fn counts_from(usage: &superglue::proto::Usage) -> Option<TokenCounts> {
    let cached_tokens = usage.cached_tokens.unwrap_or(0);
    if usage.prompt_tokens == 0 && cached_tokens == 0 {
        return None;
    }
    Some(TokenCounts {
        prompt_tokens: usage.prompt_tokens,
        cached_tokens,
    })
}

fn map_system_one(err: &superglue::systemone::SystemOneError) -> CompleterError {
    match err {
        superglue::systemone::SystemOneError::Credentials(source) => {
            CompleterError::Credentials(safe_line(source))
        }
        other => CompleterError::Failed(safe_line(other)),
    }
}

fn map_chat_error(err: &superglue::chat::ChatError) -> CompleterError {
    match err {
        superglue::chat::ChatError::Cancelled => CompleterError::Cancelled,
        superglue::chat::ChatError::Credentials(source) => {
            CompleterError::Credentials(safe_line(source))
        }
        superglue::chat::ChatError::PartialTurn { cause, .. } => map_chat_error(cause),
        superglue::chat::ChatError::Tool(superglue::tools::ToolInvokeError::HandlerFailed {
            message,
            code,
        }) if code.as_deref() == Some("plugin") => CompleterError::Plugin(safe_text(message)),
        other => CompleterError::Failed(safe_line(other)),
    }
}

fn map_response_error(err: &superglue::responses::ResponseError) -> CompleterError {
    match err {
        superglue::responses::ResponseError::Cancelled => CompleterError::Cancelled,
        superglue::responses::ResponseError::Chat(source) => map_chat_error(source),
        other => CompleterError::Failed(safe_line(other)),
    }
}

fn tool_error(content: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(content).ok()?;
    if value.get("ok").and_then(|item| item.as_bool()) == Some(false) {
        let text = value
            .get("error")
            .and_then(|item| item.as_str())
            .unwrap_or("failed");
        return Some(clip(text, 180));
    }
    None
}

fn clip(text: &str, max: usize) -> String {
    let end = text
        .char_indices()
        .nth(max)
        .map(|(index, _)| index)
        .unwrap_or(text.len());
    if end == text.len() {
        text.to_string()
    } else {
        format!("{}…", &text[..end])
    }
}

/// One safe line. Credential-shaped tokens are removed.
pub(crate) fn safe_line(err: &impl std::fmt::Display) -> String {
    safe_text(&err.to_string())
}

fn safe_text(raw: &str) -> String {
    let mut out = String::new();
    let lower = raw.to_ascii_lowercase();
    let mut index = 0;
    let bytes = raw.as_bytes();
    while index < bytes.len() {
        if let Some(skip) = secret_span(&lower, index) {
            out.push_str("[REDACTED]");
            index = skip;
            continue;
        }
        let ch = raw[index..].chars().next().unwrap_or(' ');
        out.push(if ch.is_control() { ' ' } else { ch });
        index += ch.len_utf8();
        if out.chars().count() >= 512 {
            break;
        }
    }
    let trimmed = out.trim();
    if trimmed.is_empty() {
        "unknown error".to_string()
    } else {
        trimmed.to_string()
    }
}

fn secret_span(lower: &str, index: usize) -> Option<usize> {
    let rest = lower.get(index..)?;
    for marker in ["bearer ", "sk-", "api_key", "api-key"] {
        if rest.starts_with(marker) {
            let start = index + marker.len();
            let tail = lower.get(start..).unwrap_or("");
            let end = tail
                .find(|ch: char| ch.is_whitespace() || ch == '"' || ch == '\'')
                .map(|offset| start + offset)
                .unwrap_or(lower.len());
            return Some(end);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, mpsc};
    use std::thread;

    use tokio_util::sync::CancellationToken;

    use super::{ApiMode, SuperGlueCompleter, apply_prompt_blocks, chat_message};
    use crate::prompt::PromptBlocks;
    use crate::stream::PieceLane;
    use crate::turn::{Completer, CompletionRequest, TurnImage, TurnMessage, TurnRole};

    fn spawn_capture_server(sse_body: &'static str) -> (String, mpsc::Receiver<serde_json::Value>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback test server");
        let addr = listener.local_addr().expect("read local addr");
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept request");
            let mut raw = Vec::new();
            let mut chunk = [0u8; 4096];
            let header_end = loop {
                let n = stream.read(&mut chunk).expect("read request headers");
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
                let n = stream.read(&mut chunk).expect("read request body");
                if n == 0 {
                    break;
                }
                raw.extend_from_slice(&chunk[..n]);
            }
            let body: serde_json::Value =
                serde_json::from_slice(&raw[header_end..header_end + content_len])
                    .expect("parse request JSON");
            let _ = tx.send(body);
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                sse_body.len(),
                sse_body
            );
            stream
                .write_all(response.as_bytes())
                .expect("write SSE response");
        });
        (format!("http://{addr}"), rx)
    }

    async fn run_one_completion(
        mode: ApiMode,
        model: &str,
        thinking: &str,
        sse_body: &'static str,
    ) -> serde_json::Value {
        let (base_url, rx) = spawn_capture_server(sse_body);
        let http = superglue::http::HttpClient::new(superglue::http::ClientConfig::default())
            .expect("build http client");
        let completer = SuperGlueCompleter::from_client(http)
            .with_mode(mode)
            .with_base_url(base_url.clone());
        let mut creds = superglue::providers::ProviderCredentials::default();
        creds.insert_key(superglue::providers::ProviderId::OpenAi, "test-key");
        creds.insert_base_url(superglue::providers::ProviderId::OpenAi, base_url);
        completer.set_credentials(creds);

        let tools = superglue::tools::ToolRegistry::new();
        let messages = vec![TurnMessage::text(TurnRole::User, "hello")];
        let prompt = PromptBlocks::default();
        let lane = Arc::new(PieceLane::new());
        let reply = completer
            .complete(CompletionRequest {
                prompt: &prompt,
                messages: &messages,
                transcript: &[],
                tools: &tools,
                model,
                thinking,
                max_tool_rounds: 1,
                cancel: CancellationToken::new(),
                lane,
                steering: None,
                answer_after_tool_cap: false,
                temperature: None,
                watch_tools: false,
            })
            .await
            .expect("completion should succeed");
        assert_eq!(reply.text, "ok");
        rx.recv().expect("receive captured request body")
    }

    #[test]
    fn prompt_blocks_map_to_cached_prefix_and_volatile_suffix() {
        let prompt = PromptBlocks {
            cached: vec!["core".to_string(), "stable plugin".to_string()],
            volatile: vec!["current time".to_string(), "session".to_string()],
        };
        let mut options = superglue::chat::ChatOptions::new("https://api.openai.com", "", "model");
        apply_prompt_blocks(&mut options, &prompt);
        assert!(options.system_prompt.is_none());
        let Some(cached) = options.system_prompt_blocks else {
            panic!("cached prompt blocks missing");
        };
        assert_eq!(cached.len(), 2);
        assert_eq!(cached[0].text, "core");
        assert!(cached[0].cache);
        let Some(volatile) = options.volatile_suffix_blocks else {
            panic!("volatile prompt blocks missing");
        };
        assert_eq!(volatile.len(), 2);
        assert_eq!(volatile[0].text, "current time");
        assert!(!volatile[0].cache);
    }

    #[test]
    fn image_messages_use_ordered_multipart_content() {
        let message = TurnMessage {
            role: TurnRole::User,
            text: "describe".to_string(),
            images: vec![TurnImage {
                data: Arc::<[u8]>::from(&b"image"[..]),
                media_type: "image/png".to_string(),
                filename: "one.png".to_string(),
            }],
        };
        let actual = chat_message(&message);
        let Some(superglue::openai::MessageContent::Parts(parts)) = actual.content else {
            panic!("image message was not multipart");
        };
        assert!(matches!(
            parts.first(),
            Some(superglue::openai::ContentPart::Text { text }) if text == "describe"
        ));
        assert!(matches!(
            parts.get(1),
            Some(superglue::openai::ContentPart::ImageUrl { image_url })
                if image_url.url.starts_with("data:image/png;base64,aW1hZ2U")
        ));
    }

    #[test]
    fn safe_line_redacts_a_bearer_token() {
        let text = super::safe_text("authorization failed: bearer sk-live-secret tail");
        assert!(!text.contains("sk-live"));
        assert!(text.contains("[REDACTED]"));
    }

    #[tokio::test]
    async fn responses_mode_sends_nested_reasoning_effort_without_top_level_key() {
        let sse = "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"output\":[],\"usage\":{\"input_tokens\":3,\"output_tokens\":2,\"total_tokens\":5}}}\n\n";
        let body = run_one_completion(ApiMode::Responses, "gpt-6-luna", "high", sse).await;
        assert_eq!(body["reasoning"]["effort"], "high");
        assert!(
            body.get("reasoning_effort").is_none(),
            "Responses API rejects top-level reasoning_effort: {body}"
        );
    }

    #[tokio::test]
    async fn chat_completions_mode_keeps_top_level_reasoning_effort() {
        let sse = "data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"custom-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"custom-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
        let body =
            run_one_completion(ApiMode::ChatCompletions, "custom-model", "high", sse).await;
        assert_eq!(body["reasoning_effort"], "high");
    }
}
