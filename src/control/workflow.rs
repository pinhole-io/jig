//! Workflow bundles registered by core and plugins.

use super::state::TaskType;
use super::verify::UNCHECKED_ID;

/// One registered workflow bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkflowSpec {
    pub id: String,
    pub description: String,
    /// Driver id. `chat` uses `tool_loop`.
    pub lane: String,
    /// Tool names to keep. Empty means the full conversation registry.
    pub tools: Vec<String>,
    pub task_type: TaskType,
    pub verifier_id: String,
    pub criteria: String,
    pub prompt: String,
    pub max_rounds: u32,
    /// Optional outline that seeds Plan. Empty means the model drafts the plan.
    pub outline: Vec<String>,
}

impl WorkflowSpec {
    pub fn chat(tool_loop_rounds: u32) -> Self {
        Self {
            id: CHAT_ID.to_string(),
            description: "General chat, lookups, and small tool tasks.".to_string(),
            lane: "tool_loop".to_string(),
            tools: Vec::new(),
            task_type: TaskType::Chat,
            verifier_id: UNCHECKED_ID.to_string(),
            criteria: "the reply addresses the user request".to_string(),
            prompt: String::new(),
            max_rounds: tool_loop_rounds.max(1),
            outline: Vec::new(),
        }
    }

    /// Driver id used when the chooser omits one.
    pub fn resolve_lane(&self) -> &str {
        &self.lane
    }
}

/// Reserved built-in workflow id.
pub const CHAT_ID: &str = "chat";

/// Workflows available to one conversation runtime.
///
/// An empty registry cannot answer `chat()`. Construct one with [`WorkflowRegistry::with_chat`].
#[derive(Clone, Debug)]
pub struct WorkflowRegistry {
    workflows: Vec<WorkflowSpec>,
}

impl WorkflowRegistry {
    pub fn with_chat(tool_loop_rounds: u32) -> Self {
        Self {
            workflows: vec![WorkflowSpec::chat(tool_loop_rounds)],
        }
    }

    pub fn get(&self, id: &str) -> Option<&WorkflowSpec> {
        self.workflows.iter().find(|workflow| workflow.id == id)
    }

    pub fn chat(&self) -> &WorkflowSpec {
        match self.workflows.first() {
            Some(workflow) => workflow,
            None => unreachable!("built-in chat workflow is always registered"),
        }
    }

    pub fn list(&self) -> &[WorkflowSpec] {
        &self.workflows
    }

    pub fn only_chat(&self) -> bool {
        self.workflows.len() == 1 && self.workflows[0].id == CHAT_ID
    }

    /// Insert a plugin workflow. `chat` and duplicate ids fail.
    pub fn register(&mut self, spec: WorkflowSpec) -> Result<(), &'static str> {
        if spec.id == CHAT_ID || !valid_workflow_id(&spec.id) {
            return Err("invalid workflow id");
        }
        if self.workflows.iter().any(|existing| existing.id == spec.id) {
            return Err("duplicate workflow");
        }
        self.workflows.push(spec);
        Ok(())
    }
}

fn valid_workflow_id(id: &str) -> bool {
    if id.is_empty() || id.len() > 128 {
        return false;
    }
    let mut parts = id.split('/');
    let Some(plugin) = parts.next() else {
        return false;
    };
    let Some(name) = parts.next() else {
        return false;
    };
    if parts.next().is_some() {
        return false;
    }
    valid_segment(plugin) && valid_segment(name)
}

fn valid_segment(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::TaskType;

    #[test]
    fn chat_is_always_present() {
        let registry = WorkflowRegistry::with_chat(4);
        assert!(registry.only_chat());
        assert_eq!(registry.chat().id, CHAT_ID);
        assert_eq!(registry.chat().max_rounds, 4);
        assert_eq!(registry.chat().resolve_lane(), "tool_loop");
    }

    #[test]
    fn plugin_workflow_rejects_chat_id() {
        let mut registry = WorkflowRegistry::with_chat(4);
        let err = registry.register(WorkflowSpec {
            id: "chat".to_string(),
            description: "x".to_string(),
            lane: "tool_loop".to_string(),
            tools: Vec::new(),
            task_type: TaskType::Chat,
            verifier_id: "unchecked".to_string(),
            criteria: "x".to_string(),
            prompt: String::new(),
            max_rounds: 4,
            outline: Vec::new(),
        });
        assert_eq!(err, Err("invalid workflow id"));
    }
}
