//! Run one answer turn through SuperGlue.
//!
//! The example exits when `OPENAI_API_KEY` is unset.
//! `OPENAI_MODEL` overrides `gpt-6-luna`.

use std::io::{self, Write};
use std::sync::Arc;

use jig::{
    AgentRunner, AgentSpec, ApiMode, Budget, CoordinationSource, DoneCheck, EngineDriver,
    RoundBudget, StaticDriver, StreamPiece, SuperGlueCompleter, ToolPolicy, ToolSet, TurnInput,
    VerifierRegistry, WorkflowRegistry,
};

const SPEC: AgentSpec = AgentSpec {
    id: "guide",
    prompt: "You answer in one short paragraph.",
    instructions: "",
    tools: ToolPolicy::fixed(ToolSet::All),
    max_tool_rounds: RoundBudget::Fixed(1),
    coordination: CoordinationSource::Single,
    drivers: &["answer"],
    worktree: false,
    done_check: DoneCheck::None,
    children: &[],
    swarm: false,
    primary: true,
    awaits_verify: false,
    tools_error: "",
};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    if key.is_empty() {
        println!("Set OPENAI_API_KEY to run this example.");
        return;
    }

    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-6-luna".to_string());
    let completer = SuperGlueCompleter::try_from_env()
        .expect("http client")
        .with_mode(ApiMode::ChatCompletions);
    let drivers: Vec<Arc<dyn EngineDriver>> = vec![Arc::new(StaticDriver {
        id: "answer",
        chooser_line: "Answer from the prompt. Do not call a tool.",
        budget: Budget::answer(),
        task: None,
        allowlist: &[],
        prompt: "",
        rounds: Some(1),
        deny_tools: true,
    })];
    let workflows = WorkflowRegistry::with_chat(1);
    let verifiers = VerifierRegistry::with_builtins();
    let tools = superglue::tools::ToolRegistry::new();
    let cancel = tokio_util::sync::CancellationToken::new();

    let outcome = AgentRunner::run(
        &completer,
        TurnInput {
            spec: &SPEC,
            drivers: &drivers,
            workflows: &workflows,
            verifiers: &verifiers,
            tools: &tools,
            history: &[],
            transcript: &[],
            user_text: "Name one property of the Rust type system.",
            volatile: &[],
            cancel,
            single: true,
            unattended: false,
            server_rounds: 1,
            swarm_steps: 1,
            model: &model,
            thinking: "none",
            temperature: None,
            steering: None,
        },
        |piece| {
            if let StreamPiece::Text(text) = piece {
                print!("{text}");
                io::stdout().flush().ok();
            }
        },
    )
    .await;

    match outcome {
        Ok(outcome) => {
            println!();
            println!("lane: {}", outcome.lane);
            println!("workflow: {}", outcome.workflow_id);
        }
        Err(err) => {
            eprintln!("turn failed: {err}");
            std::process::exit(1);
        }
    }
}
