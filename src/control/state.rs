//! Sidecar control state for one conversation run.

/// Risk level for one plan step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Risk {
    Read,
    Write,
    Destructive,
}

impl Risk {
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "read" => Self::Read,
            "destructive" => Self::Destructive,
            _ => Self::Write,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Destructive => "destructive",
        }
    }
}

/// Task family used when picking a verifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskType {
    Chat,
    Lookup,
    Code,
    Config,
    Research,
    Data,
}

/// Status of one plan step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepStatus {
    Pending,
    Running,
    Done,
    Failed,
}

/// One step in a plan-execute run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    pub id: String,
    pub intent: String,
    pub tools: Vec<String>,
    pub expected: String,
    pub risk: Risk,
    pub status: StepStatus,
    pub result: String,
}

/// Token and step budgets for one run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    pub steps_left: u32,
    pub tokens_left: u32,
}

impl Budget {
    pub const fn answer() -> Self {
        Self {
            steps_left: 1,
            tokens_left: 32_000,
        }
    }

    pub const fn tool_loop() -> Self {
        Self {
            steps_left: 4,
            tokens_left: 64_000,
        }
    }

    pub const fn read() -> Self {
        Self {
            steps_left: 8,
            tokens_left: 64_000,
        }
    }

    pub const fn code() -> Self {
        Self {
            steps_left: 8,
            tokens_left: 128_000,
        }
    }

    pub const fn research() -> Self {
        Self {
            steps_left: 4,
            tokens_left: 64_000,
        }
    }
}

/// Pause waiting for a person to approve a plan step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pause {
    pub command_id: String,
    pub step_id: String,
    pub risk: Risk,
}

/// Control sidecar for one conversation run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunState {
    pub workflow_id: String,
    /// Driver id selected for this run.
    pub lane: String,
    pub task_type: TaskType,
    pub success_criteria: String,
    pub verifier_id: String,
    pub plan: Vec<Step>,
    pub step_idx: u32,
    pub retries: u32,
    pub failure_signatures: Vec<String>,
    pub budget: Budget,
    pub pending_approval: Option<Pause>,
    pub notes: String,
    pub verified: bool,
    pub verify_evidence: String,
    /// Public id of the registry plan for this run.
    pub saved_plan_id: Option<String>,
    /// Task refs parallel to [`Self::plan`], when the registry save succeeded.
    pub saved_task_refs: Vec<String>,
    /// Why the drafted steps were not stored. Empty when they were stored or not drafted.
    pub saved_plan_error: String,
}

impl RunState {
    pub fn clear_finished(&mut self) {
        self.pending_approval = None;
        self.plan.clear();
        self.step_idx = 0;
        self.retries = 0;
        self.failure_signatures.clear();
        self.notes.clear();
        self.saved_plan_id = None;
        self.saved_task_refs.clear();
        self.saved_plan_error.clear();
    }
}

/// Caps for plan-execute retries.
pub const MAX_STEP_RETRIES: u32 = 2;

/// Safety limit for one drafted plan.
///
/// The draft prompt does not ask for this count. A plan may be shorter.
pub const PLAN_STEP_CAP: usize = 32;
