//! Pluggable verifiers keyed by id.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use super::state::RunState;

/// Built-in verifier that records no check.
pub const UNCHECKED_ID: &str = "unchecked";
pub const CODE_ID: &str = "code";
pub const CONFIG_ID: &str = "config";

/// Shared verifier result shape.
#[derive(Clone, Debug, PartialEq)]
pub struct VerifyResult {
    pub ok: bool,
    pub evidence: String,
    pub confidence: f32,
}

/// One verifier implementation.
#[async_trait]
pub trait Verifier: Send + Sync {
    fn id(&self) -> &'static str;

    async fn verify(
        &self,
        state: &RunState,
        tools: &superglue::tools::ToolRegistry,
    ) -> VerifyResult;
}

/// Registry of verifiers available to the control machine.
#[derive(Clone, Default)]
pub struct VerifierRegistry {
    verifiers: HashMap<String, Arc<dyn Verifier>>,
}

impl VerifierRegistry {
    pub fn with_builtins() -> Self {
        let mut registry = Self::default();
        registry.register(Arc::new(Unchecked));
        registry
    }

    pub fn register(&mut self, verifier: Arc<dyn Verifier>) {
        self.verifiers.insert(verifier.id().to_string(), verifier);
    }

    pub fn get(&self, id: &str) -> Arc<dyn Verifier> {
        self.verifiers
            .get(id)
            .cloned()
            .unwrap_or_else(|| Arc::new(Unchecked) as Arc<dyn Verifier>)
    }
}

/// Verifier that records the criteria and does not check them.
pub struct Unchecked;

#[async_trait]
impl Verifier for Unchecked {
    fn id(&self) -> &'static str {
        UNCHECKED_ID
    }

    async fn verify(
        &self,
        state: &RunState,
        _tools: &superglue::tools::ToolRegistry,
    ) -> VerifyResult {
        VerifyResult {
            ok: true,
            evidence: format!("unchecked against criteria: {}", state.success_criteria),
            confidence: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::{Budget, RunState, TaskType};

    fn sample_state(verifier_id: &str) -> RunState {
        RunState {
            workflow_id: "chat".to_string(),
            lane: "tool_loop".to_string(),
            task_type: TaskType::Chat,
            success_criteria: "done".to_string(),
            verifier_id: verifier_id.to_string(),
            plan: Vec::new(),
            step_idx: 0,
            retries: 0,
            failure_signatures: Vec::new(),
            budget: Budget::tool_loop(),
            pending_approval: None,
            notes: String::new(),
            verified: false,
            verify_evidence: String::new(),
            saved_plan_id: None,
            saved_task_refs: Vec::new(),
            saved_plan_error: String::new(),
        }
    }

    #[tokio::test]
    async fn unchecked_reports_zero_confidence() {
        let registry = VerifierRegistry::with_builtins();
        let tools = superglue::tools::ToolRegistry::new();
        let result = registry
            .get(UNCHECKED_ID)
            .verify(&sample_state(UNCHECKED_ID), &tools)
            .await;
        assert!(result.ok);
        assert_eq!(result.confidence, 0.0);
    }

    #[tokio::test]
    async fn unknown_verifier_uses_unchecked() {
        let registry = VerifierRegistry::with_builtins();
        assert_eq!(registry.get("missing").id(), UNCHECKED_ID);
    }
}
