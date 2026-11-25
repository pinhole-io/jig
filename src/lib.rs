//! Protocol-free agent runtime.
//!
//! This crate has no server, sandbox, or tool set. Callers supply the
//! tool registry, the completer, and the drivers.
//!
//! Guides live in `docs/`. Runnable programs live in `examples/`.

pub mod agent;
pub mod clock;
pub mod completer;
pub mod control;
pub mod limits;
pub mod prompt;
pub mod runner;
pub mod stream;
pub mod turn;

pub use agent::{
    AgentSpec, ChildAgent, CoordinationMode, CoordinationSource, DoneCheck, RoundBudget,
    ToolPolicy, ToolSet,
};
pub use clock::{CLOCK_PREFIX, clock_line, refresh_clock};
pub use completer::{ApiMode, SuperGlueCompleter, chat_message};
pub use control::{Budget, CHAT_ID, FailureTracker, TaskType, VerifierRegistry, WorkflowRegistry};
pub use limits::{ContextWindows, ModelBudget, guess_context, model_budget};
pub use prompt::{
    MAX_PROMPT_BYTES, PROMPT_TOO_LARGE, PromptBlocks, PromptBuilder, PromptContext, PromptSlot,
    PromptSlots,
};
pub use runner::{AgentRunner, EngineDriver, StaticDriver, TurnError, TurnInput, TurnOutcome};
pub use stream::{DELTA_CHARS, DeltaBuf, PieceLane, StreamPiece, ToolPiece, ToolPoint};
pub use turn::{
    ChooseRequest, Completer, CompleterError, CompletionRequest, ModelReply, TokenCounts,
    TurnImage, TurnMessage, TurnPrep, TurnRole,
};
