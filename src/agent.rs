//! One agent the host can run.
//!
//! The spec owns the prompt, the tool policy, the limits, and the child ids.
//! The host does not match on a coding role.

/// How many model rounds one agent may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundBudget {
    /// Use the server `max_tool_rounds` value.
    Server,
    /// Use this count.
    Fixed(u32),
    /// Use the swarm task step cap.
    SwarmSteps,
}

/// Where the coordination mode comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinationSource {
    /// Use the server coordination mode.
    Server,
    /// Always single. The agent cannot orchestrate.
    Single,
}

/// Coordination mode for one agent. A host can map its own config enum onto this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinationMode {
    /// The primary agent keeps its tools and cannot spawn children.
    Single,
    /// The primary agent spawns a child when the lane is work.
    Orchestrate,
}

/// How a swarm child result counts as done.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoneCheck {
    /// This agent is not a swarm child.
    None,
    /// The reply text is not empty.
    NonEmpty,
    /// A read tool ran and the reply contains `done: yes`.
    EvidenceLine,
    /// Shell ran and no shell call failed.
    ShellOk,
    /// The reply contains `verdict: accept`.
    VerdictAccept,
}

/// Names one turn may call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolSet {
    /// Every registered tool.
    All,
    /// Only these names.
    Only(&'static [&'static str]),
    /// Every tool except these names.
    Except(&'static [&'static str]),
}

impl ToolSet {
    pub fn only_names(self) -> &'static [&'static str] {
        match self {
            Self::Only(names) => names,
            Self::All | Self::Except(_) => &[],
        }
    }

    pub fn allows(self, name: &str) -> bool {
        match self {
            Self::All => true,
            Self::Only(names) => names.contains(&name),
            Self::Except(names) => !names.contains(&name),
        }
    }
}

/// Tool sets for single and orchestrate, attended and unattended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToolPolicy {
    pub single_attended: ToolSet,
    pub single_unattended: ToolSet,
    pub orchestrate_attended: ToolSet,
    pub orchestrate_unattended: ToolSet,
}

impl ToolPolicy {
    pub const fn same(set: ToolSet) -> Self {
        Self {
            single_attended: set,
            single_unattended: set,
            orchestrate_attended: set,
            orchestrate_unattended: set,
        }
    }

    /// One set for every coordination mode and turn kind.
    pub const fn fixed(set: ToolSet) -> Self {
        Self::same(set)
    }

    /// Pick the set for this coordination mode and turn.
    pub fn select(self, single: bool, unattended: bool) -> ToolSet {
        match (single, unattended) {
            (true, false) => self.single_attended,
            (true, true) => self.single_unattended,
            (false, false) => self.orchestrate_attended,
            (false, true) => self.orchestrate_unattended,
        }
    }
}

/// Fields that differ for each coding child. Shared flags live in [`AgentSpec::child`].
pub struct ChildAgent {
    pub id: &'static str,
    /// Role paragraph. The swarm task starts with this text.
    pub role: &'static str,
    /// System text. The catalog builds this from `role`.
    pub prompt: &'static str,
    pub tools: ToolPolicy,
    pub max_tool_rounds: RoundBudget,
    /// Driver ids the chooser may pick.
    pub drivers: &'static [&'static str],
    pub worktree: bool,
    pub done_check: DoneCheck,
    /// An implementer waits for its judges. Other roles verify themselves.
    pub awaits_verify: bool,
    /// Error when a brief names tools this agent cannot use.
    pub tools_error: &'static str,
}

/// Definition the runtime loads for one agent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AgentSpec {
    pub id: &'static str,
    /// System text. Empty adds no identity block.
    pub prompt: &'static str,
    /// Text the swarm puts at the start of the child task.
    pub instructions: &'static str,
    pub tools: ToolPolicy,
    pub max_tool_rounds: RoundBudget,
    pub coordination: CoordinationSource,
    /// Driver ids the chooser may pick. One id skips the chooser.
    pub drivers: &'static [&'static str],
    pub worktree: bool,
    pub done_check: DoneCheck,
    /// Agent ids this agent may start.
    pub children: &'static [&'static str],
    /// The server may run the swarm path when coordination is orchestrate.
    pub swarm: bool,
    /// This is a user-facing conversation, not a child runtime.
    pub primary: bool,
    /// An implementer waits for its judges. Other roles verify themselves.
    pub awaits_verify: bool,
    /// Error when a brief names tools this agent cannot use.
    pub tools_error: &'static str,
}

impl AgentSpec {
    /// A child agent. Coordination is single, it has no children, and it is not primary.
    pub const fn child(child: ChildAgent) -> Self {
        Self {
            id: child.id,
            prompt: child.prompt,
            instructions: child.role,
            tools: child.tools,
            max_tool_rounds: child.max_tool_rounds,
            coordination: CoordinationSource::Single,
            drivers: child.drivers,
            worktree: child.worktree,
            done_check: child.done_check,
            children: &[],
            swarm: false,
            primary: false,
            awaits_verify: child.awaits_verify,
            tools_error: child.tools_error,
        }
    }

    pub fn allows_child(self, id: &str) -> bool {
        self.children.contains(&id)
    }

    pub fn rounds(self, server: u32, swarm_steps: u32) -> u32 {
        match self.max_tool_rounds {
            RoundBudget::Server => server,
            RoundBudget::Fixed(count) => count,
            RoundBudget::SwarmSteps => swarm_steps,
        }
    }

    pub fn coordination_mode(self, server: CoordinationMode) -> CoordinationMode {
        match self.coordination {
            CoordinationSource::Server => server,
            CoordinationSource::Single => CoordinationMode::Single,
        }
    }

    pub fn single(self, server: CoordinationMode) -> bool {
        self.coordination_mode(server) == CoordinationMode::Single
    }
}
