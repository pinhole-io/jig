//! Tool-loop helpers shared by the hub and the generic runner.

use crate::prompt::PromptBlocks;

use super::workflow::WorkflowSpec;

/// Tracks repeated tool failures inside one Act call.
#[derive(Default)]
pub struct FailureTracker {
    pub signatures: Vec<String>,
    pub stuck: bool,
}

impl FailureTracker {
    /// Record a failed tool. Returns true when the same failure repeats.
    pub fn note(&mut self, name: &str, error: &str, arguments: &str) -> bool {
        if error.is_empty() {
            return false;
        }
        let signature = format!("{name}:{error}:{:016x}", args_fingerprint(arguments));
        if self.signatures.iter().any(|seen| seen == &signature) {
            self.stuck = true;
            return true;
        }
        self.signatures.push(signature);
        false
    }
}

fn args_fingerprint(arguments: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    arguments.hash(&mut hasher);
    hasher.finish()
}

/// Tool-loop settings for one lane.
pub struct LoopLane {
    pub prompt: &'static str,
    /// Tool names to keep. Empty keeps the workflow tool set.
    pub allowlist: &'static [&'static str],
}

pub const TOOL_LOOP: LoopLane = LoopLane {
    prompt: "",
    allowlist: &[],
};

pub fn with_workflow_prompt(prompt: &PromptBlocks, workflow: &WorkflowSpec) -> PromptBlocks {
    let mut volatile = prompt.volatile.clone();
    if !workflow.prompt.is_empty() {
        volatile.push(workflow.prompt.clone());
    }
    if !workflow.criteria.is_empty() {
        volatile.push(format!("Success criteria: {}", workflow.criteria));
    }
    PromptBlocks {
        cached: prompt.cached.clone(),
        volatile,
    }
}

pub async fn intersect_tools<N>(
    source: &superglue::tools::ToolRegistry,
    names: &[N],
) -> superglue::tools::ToolRegistry
where
    N: AsRef<str> + Sync,
{
    let dest = superglue::tools::ToolRegistry::new();
    let specs = source.list_specs().await;
    for spec in specs {
        if !names.is_empty() && !names.iter().any(|name| name.as_ref() == spec.name) {
            continue;
        }
        if let Ok((tool, policy)) = source.resolve_invocation(&spec.name).await {
            let _ = dest.register_with_policy(tool, policy).await;
        }
    }
    dest
}

#[cfg(test)]
mod tests {
    use super::FailureTracker;

    #[test]
    fn the_same_failure_marks_the_tracker_stuck() {
        let mut tracker = FailureTracker::default();
        assert!(!tracker.note("fetch", "timeout", "{}"));
        assert!(tracker.note("fetch", "timeout", "{}"));
        assert!(tracker.stuck);
    }
}
