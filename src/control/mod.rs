//! Workflow selection, budgets, and verification.
//!
//! The host owns the hub that runs a turn. This module holds the shared
//! data and the shared functions.

mod lane;
mod state;
mod triage;
mod verify;
mod workflow;

pub use lane::{FailureTracker, LoopLane, TOOL_LOOP, intersect_tools, with_workflow_prompt};
pub use state::{
    Budget, MAX_STEP_RETRIES, PLAN_STEP_CAP, Pause, Risk, RunState, Step, StepStatus, TaskType,
};
pub use triage::{
    CHOOSER_MIN_CONFIDENCE, ChooserPick, LaneSpec, TRIAGE_MODEL, TriageDecision, chooser_prompt,
    chooser_questions, chooser_state, is_chooser_prompt, parse_chooser_reply, pick_from_answers,
    triage,
};
pub use verify::{
    CODE_ID, CONFIG_ID, UNCHECKED_ID, Unchecked, Verifier, VerifierRegistry, VerifyResult,
};
pub use workflow::{CHAT_ID, WorkflowRegistry, WorkflowSpec};
