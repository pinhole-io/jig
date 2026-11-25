//! Build one agent spec, two drivers, and one plugin workflow.
//!
//! This example does not call a model.

use std::sync::Arc;

use jig::control::{UNCHECKED_ID, WorkflowSpec};
use jig::{
    AgentSpec, Budget, CoordinationSource, DoneCheck, EngineDriver, RoundBudget, StaticDriver,
    TaskType, ToolPolicy, ToolSet, WorkflowRegistry,
};

const SPEC: AgentSpec = AgentSpec {
    id: "guide",
    prompt: "You answer in one short paragraph.",
    instructions: "",
    tools: ToolPolicy::fixed(ToolSet::All),
    max_tool_rounds: RoundBudget::Fixed(4),
    coordination: CoordinationSource::Single,
    drivers: &["answer", "tool_loop"],
    worktree: false,
    done_check: DoneCheck::None,
    children: &[],
    swarm: false,
    primary: true,
    awaits_verify: false,
    tools_error: "",
};

fn main() {
    let answer = StaticDriver {
        id: "answer",
        chooser_line: "Answer from the prompt. Do not call a tool.",
        budget: Budget::answer(),
        task: Some(TaskType::Chat),
        allowlist: &[],
        prompt: "",
        rounds: Some(1),
        deny_tools: true,
    };
    let tool_loop = StaticDriver {
        id: "tool_loop",
        chooser_line: "Use a tool when the reply needs a lookup.",
        budget: Budget::tool_loop(),
        task: None,
        allowlist: &[],
        prompt: "",
        rounds: None,
        deny_tools: false,
    };
    let drivers: Vec<Arc<dyn EngineDriver>> = vec![Arc::new(answer), Arc::new(tool_loop)];

    let mut workflows = WorkflowRegistry::with_chat(4);
    workflows
        .register(WorkflowSpec {
            id: "guide/lookup".to_string(),
            description: "Look up one fact, then answer.".to_string(),
            lane: "tool_loop".to_string(),
            tools: Vec::new(),
            task_type: TaskType::Lookup,
            verifier_id: UNCHECKED_ID.to_string(),
            criteria: "the reply names the fact".to_string(),
            prompt: String::new(),
            max_rounds: 4,
            outline: Vec::new(),
        })
        .expect("guide/lookup is a valid plugin id");

    println!("agent: {}", SPEC.id);
    println!("rounds: {}", SPEC.rounds(8, 1));
    println!("drivers: {}", drivers.len());
    for driver in &drivers {
        println!(
            "  {} steps={} deny_tools={}",
            driver.id(),
            driver.budget().steps_left,
            driver.deny_tools()
        );
    }
    println!("workflows:");
    for workflow in workflows.list() {
        println!("  {} lane={}", workflow.id, workflow.resolve_lane());
    }
    println!("two drivers ask the chooser. one driver skips it.");
}
