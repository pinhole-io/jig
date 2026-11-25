//! One model call and the messages that go into it.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::control::{ChooserPick, WorkflowRegistry, chooser_prompt, parse_chooser_reply};
use crate::prompt::PromptBlocks;
use crate::stream::PieceLane;

pub enum CompleterError {
    /// The cancel token fired.
    Cancelled,
    /// The provider key is missing. The string names the provider, not the key.
    Credentials(String),
    /// The provider or tool round failed. The string is the cause.
    Failed(String),
    /// A hook rejected the call.
    Plugin(String),
}

pub struct TokenCounts {
    pub prompt_tokens: u32,
    pub cached_tokens: u32,
}

/// Image bytes attached to one user message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TurnImage {
    pub data: Arc<[u8]>,
    pub media_type: String,
    pub filename: String,
}

/// Role stored in the transcript.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnRole {
    /// Text from the caller.
    User,
    /// Text from the model.
    Assistant,
}

impl TurnRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// One chat message the model will see.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TurnMessage {
    pub role: TurnRole,
    pub text: String,
    pub images: Vec<TurnImage>,
}

/// Mutable view of the next model call.
#[derive(Clone)]
pub struct TurnPrep {
    pub messages: Vec<TurnMessage>,
}

/// One finished model call.
pub struct ModelReply {
    pub text: String,
    pub usage: Option<TokenCounts>,
    /// Caller messages after the call, without the injected system prefix.
    pub transcript: Vec<superglue::openai::ChatMessage>,
}

/// Inputs for one provider completion.
pub struct CompletionRequest<'a> {
    pub prompt: &'a PromptBlocks,
    pub messages: &'a [TurnMessage],
    /// When this is non-empty, it replaces `messages` as the caller transcript.
    pub transcript: &'a [superglue::openai::ChatMessage],
    pub tools: &'a superglue::tools::ToolRegistry,
    pub model: &'a str,
    pub thinking: &'a str,
    pub max_tool_rounds: u32,
    pub cancel: CancellationToken,
    pub lane: Arc<PieceLane>,
    pub steering: Option<superglue::chat::RoundBoundary>,
    /// One tool-free model call after the tool-round cap.
    pub answer_after_tool_cap: bool,
    pub temperature: Option<f32>,
    /// Record tool calls on `lane` and cancel after a repeated failure.
    pub watch_tools: bool,
}

/// Inputs for one control-path choice.
pub struct ChooseRequest<'a> {
    /// User text for this turn.
    pub request: &'a str,
    /// Workflows the chooser may name.
    pub workflows: &'a WorkflowRegistry,
    /// Driver id and chooser line pairs the spec allows.
    pub lanes: &'a [(&'static str, &'static str)],
    /// Stops the call when the turn is cancelled.
    pub cancel: CancellationToken,
}

/// Streams one reply. Tests supply a fake. Production uses [`crate::SuperGlueCompleter`].
pub trait Completer: Send + Sync {
    /// Push pieces on `lane`, then return the full reply.
    fn complete<'a>(
        &'a self,
        request: CompletionRequest<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<ModelReply, CompleterError>> + Send + 'a>>;

    /// Pick a workflow and a lane.
    ///
    /// The default sends the JSON chooser prompt through [`Self::complete`].
    /// Production overrides this and calls System One.
    fn choose<'a>(
        &'a self,
        request: ChooseRequest<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<ChooserPick, CompleterError>> + Send + 'a>> {
        Box::pin(async move {
            let prompt_text = chooser_prompt(request.workflows, request.request, request.lanes);
            let messages = [TurnMessage {
                role: TurnRole::User,
                text: prompt_text,
                images: Vec::new(),
            }];
            let empty = superglue::tools::ToolRegistry::new();
            let lane = Arc::new(PieceLane::new());
            let prompt = PromptBlocks::cached_only(
                "You route turns. Reply with the JSON object only. No prose.",
            );
            let thinking = "none";
            let reply = self
                .complete(CompletionRequest {
                    prompt: &prompt,
                    messages: &messages,
                    transcript: &[],
                    tools: &empty,
                    model: crate::control::TRIAGE_MODEL,
                    thinking,
                    max_tool_rounds: 0,
                    cancel: request.cancel,
                    lane,
                    steering: None,
                    answer_after_tool_cap: false,
                    temperature: None,
                    watch_tools: false,
                })
                .await?;
            Ok(parse_chooser_reply(&reply.text))
        })
    }
}
