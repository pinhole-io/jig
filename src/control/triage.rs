//! Select a workflow and lane for one turn.
//!
//! The chooser reads a structured System One reply. It does not match phrases
//! in the user text.

use super::state::{Budget, RunState, TaskType};
use super::verify::UNCHECKED_ID;
use super::workflow::{CHAT_ID, WorkflowRegistry, WorkflowSpec};

use superglue::systemone::{Choice, Questions, SystemOneResponse, choice_if_confident};

/// Cheap model used only for the control-path chooser.
pub const TRIAGE_MODEL: &str = "typesafe:jev-latest";

/// Minimum System One confidence that keeps a lane or a workflow id.
pub const CHOOSER_MIN_CONFIDENCE: f64 = 0.5;

/// Maximum characters of user text sent to System One.
const CHOOSER_STATE_CHARS: usize = 2_000;

/// One driver the chooser can name. The host fills this from its driver list.
#[derive(Clone, Copy, Debug)]
pub struct LaneSpec {
    pub id: &'static str,
    pub budget: Budget,
    pub task: Option<TaskType>,
}

/// Result of triage before Act.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TriageDecision {
    pub state: RunState,
    pub workflow: WorkflowSpec,
}

/// Structured pick from the Jev chooser.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChooserPick {
    pub workflow_id: Option<String>,
    pub lane: Option<String>,
}

/// Pick a workflow and lane. Unknown chooser output selects `chat`.
///
/// When `pick.lane` is set, it wins over the workflow lane.
pub fn triage(
    registry: &WorkflowRegistry,
    pick: Option<&ChooserPick>,
    allowed: &[&str],
    lanes: &[LaneSpec],
) -> TriageDecision {
    let chosen_id = pick.and_then(|pick| pick.workflow_id.as_deref());
    let workflow = select_workflow(registry, chosen_id).clone();
    let picked = pick.and_then(|pick| pick.lane.as_deref());
    let lane = picked
        .filter(|lane| allowed.is_empty() || allowed.contains(lane))
        .unwrap_or_else(|| workflow.resolve_lane());
    let lane = if allowed.is_empty() || allowed.contains(&lane) {
        lane.to_string()
    } else {
        "tool_loop".to_string()
    };
    let task_type = task_for(lanes, &lane, workflow.task_type);
    let budget = budget_for(lanes, &lane);
    let verifier_id = if workflow.verifier_id.is_empty() {
        UNCHECKED_ID.to_string()
    } else {
        workflow.verifier_id.clone()
    };
    let state = RunState {
        workflow_id: workflow.id.clone(),
        lane,
        task_type,
        success_criteria: workflow.criteria.clone(),
        verifier_id,
        plan: Vec::new(),
        step_idx: 0,
        retries: 0,
        failure_signatures: Vec::new(),
        budget,
        pending_approval: None,
        notes: String::new(),
        verified: false,
        verify_evidence: String::new(),
        saved_plan_id: None,
        saved_task_refs: Vec::new(),
        saved_plan_error: String::new(),
    };
    TriageDecision { state, workflow }
}

pub fn budget_for(lanes: &[LaneSpec], id: &str) -> Budget {
    lanes
        .iter()
        .find(|lane| lane.id == id)
        .map(|lane| lane.budget)
        .unwrap_or_else(Budget::tool_loop)
}

pub fn task_for(lanes: &[LaneSpec], id: &str, fallback: TaskType) -> TaskType {
    lanes
        .iter()
        .find(|lane| lane.id == id)
        .and_then(|lane| lane.task)
        .unwrap_or(fallback)
}

fn select_workflow<'a>(
    registry: &'a WorkflowRegistry,
    chosen_id: Option<&str>,
) -> &'a WorkflowSpec {
    chosen_id
        .and_then(|id| registry.get(id))
        .unwrap_or_else(|| registry.chat())
}

/// Prompt for the Jev control-path chooser.
///
/// The lane lines are the same text as the System One Choice options.
pub fn chooser_prompt(
    registry: &WorkflowRegistry,
    request: &str,
    lanes: &[(&str, &str)],
) -> String {
    let mut lane_names = String::new();
    for (index, (key, _)) in lanes.iter().enumerate() {
        if index > 0 {
            lane_names.push('|');
        }
        lane_names.push_str(key);
    }
    let mut lines = format!(
        "Select the control path for this user request.\nReply with JSON only: {{\"workflow_id\":\"...\",\"lane\":\"{lane_names}\"}}.\nUse workflow_id \"chat\" when unsure.\nLane guide:\n"
    );
    for (key, description) in lanes {
        lines.push_str("- ");
        lines.push_str(key);
        lines.push_str(": ");
        lines.push_str(description);
        lines.push('\n');
    }
    lines.push_str("\nWorkflows:\n");
    for workflow in registry.list() {
        lines.push_str("- ");
        lines.push_str(&workflow.id);
        lines.push_str(": ");
        lines.push_str(&workflow.description);
        lines.push('\n');
    }
    lines.push_str("\nUser request:\n");
    lines.push_str(request);
    lines
}

/// Choice questions for the System One chooser.
///
/// `workflow` is omitted when the registry holds only `chat`.
pub fn chooser_questions(registry: &WorkflowRegistry, lanes: &[(&str, &str)]) -> Questions {
    let mut questions = Questions::new();
    let mut lane = Choice::new("Select the control lane for this user request.");
    for (key, description) in lanes {
        lane = lane.option(*key, *description);
    }
    let _ = questions.insert("lane", lane);
    if registry.list().iter().any(|spec| spec.id != CHAT_ID) {
        let mut workflow = Choice::new("Select the workflow id. Use chat when unsure.");
        for spec in registry.list() {
            let description = if spec.description.is_empty() {
                spec.id.as_str()
            } else {
                spec.description.as_str()
            };
            workflow = workflow.option(spec.id.clone(), description);
        }
        let _ = questions.insert("workflow", workflow);
    }
    questions
}

/// User text clipped for the System One state field.
pub fn chooser_state(request: &str) -> String {
    superglue::systemone::clip_state(request, CHOOSER_STATE_CHARS)
}

/// Map a System One reply onto a [`ChooserPick`].
///
/// A missing answer, a low confidence, an unknown lane, or an unknown workflow id becomes `None`.
pub fn pick_from_answers(
    response: &SystemOneResponse,
    registry: &WorkflowRegistry,
    min_confidence: f64,
    allowed: &[&str],
) -> ChooserPick {
    let lane = response
        .answer("lane")
        .and_then(|answer| choice_if_confident(answer, min_confidence))
        .and_then(parse_lane)
        .filter(|lane| allowed.is_empty() || allowed.iter().any(|id| id == lane));
    let workflow_id = response
        .answer("workflow")
        .and_then(|answer| choice_if_confident(answer, min_confidence))
        .filter(|id| registry.get(id).is_some())
        .map(str::to_string);
    ChooserPick { workflow_id, lane }
}

/// True when a completion request is the control-path chooser.
pub fn is_chooser_prompt(text: &str) -> bool {
    text.contains("Select the control path for this user request.")
}

/// Parse a chooser model reply into a workflow id and optional lane.
pub fn parse_chooser_reply(text: &str) -> ChooserPick {
    let Some(value) = extract_json_object(text) else {
        return ChooserPick {
            workflow_id: None,
            lane: None,
        };
    };
    let workflow_id = value
        .get("workflow_id")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let lane = value
        .get("lane")
        .and_then(|v| v.as_str())
        .and_then(parse_lane);
    ChooserPick { workflow_id, lane }
}

fn extract_json_object(text: &str) -> Option<serde_json::Value> {
    let trimmed = text.trim();
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
        return Some(value);
    }
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    if end <= start {
        return None;
    }
    serde_json::from_str(&trimmed[start..=end]).ok()
}

fn parse_lane(value: &str) -> Option<String> {
    let lane = match value.trim().to_ascii_lowercase().as_str() {
        "answer" => "answer",
        "read" => "read",
        "tool_loop" | "tool-loop" | "toolloop" => "tool_loop",
        "code" => "code",
        "research" => "research",
        "plan_execute" | "plan-execute" | "planexecute" => "plan_execute",
        _ => return None,
    };
    Some(lane.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lanes() -> Vec<LaneSpec> {
        vec![
            LaneSpec {
                id: "answer",
                budget: Budget::answer(),
                task: Some(TaskType::Chat),
            },
            LaneSpec {
                id: "tool_loop",
                budget: Budget::tool_loop(),
                task: None,
            },
            LaneSpec {
                id: "research",
                budget: Budget::research(),
                task: Some(TaskType::Research),
            },
        ]
    }

    #[test]
    fn unknown_id_falls_back_to_chat() {
        let registry = WorkflowRegistry::with_chat(4);
        let pick = ChooserPick {
            workflow_id: Some("missing/thing".to_string()),
            lane: None,
        };
        let decision = triage(&registry, Some(&pick), &[], &lanes());
        assert_eq!(decision.state.workflow_id, CHAT_ID);
        assert_eq!(decision.state.lane, "tool_loop");
    }

    #[test]
    fn parse_chooser_accepts_workflow_and_lane() {
        let pick = parse_chooser_reply("{\"workflow_id\":\"demo/lookup\",\"lane\":\"tool_loop\"}");
        assert_eq!(pick.workflow_id.as_deref(), Some("demo/lookup"));
        assert_eq!(pick.lane, Some("tool_loop".to_string()));
        let empty = parse_chooser_reply("nope");
        assert_eq!(empty.workflow_id, None);
        assert_eq!(empty.lane, None);
    }

    #[test]
    fn research_lane_sets_the_research_budget() {
        let registry = WorkflowRegistry::with_chat(4);
        let pick = ChooserPick {
            workflow_id: Some("chat".to_string()),
            lane: Some("research".to_string()),
        };
        let decision = triage(&registry, Some(&pick), &[], &lanes());
        assert_eq!(decision.state.lane, "research");
        assert_eq!(decision.state.task_type, TaskType::Research);
        assert_eq!(decision.state.budget.steps_left, 4);
    }
}
