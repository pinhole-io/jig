# Library API

`jig` is a library crate. Import the common types from `jig`:

```rust
use jig::{
    AgentRunner, AgentSpec, ApiMode, PromptBuilder, PromptContext, PromptSlots,
    StaticDriver, SuperGlueCompleter, TurnInput, WorkflowRegistry,
};
```

Workflow and verifier types that are not re-exported live in `jig::control`.

Public re-exports are defined in `src/lib.rs`.

## `AgentSpec`

`src/agent.rs`

One agent the caller can run. The spec is `'static` because `AgentRunner::run` borrows it for the turn.

| Field | Meaning |
| --- | --- |
| `id` | Stable name |
| `prompt` | System text. Empty adds no identity block |
| `tools` | `ToolPolicy` for single or orchestrate, attended or unattended |
| `max_tool_rounds` | `RoundBudget::Server`, `Fixed`, or `SwarmSteps` |
| `drivers` | Driver ids the turn may use. One id skips the chooser |
| `children` | Agent ids this agent may start |

`AgentSpec::child` builds a non-primary spec from a `ChildAgent`. `rounds` resolves the budget. `single` reports whether this turn may spawn children.

`ToolPolicy::fixed` uses one `ToolSet` for every mode. `ToolSet::All`, `Only`, and `Except` filter names.

## `EngineDriver` and `StaticDriver`

`src/runner.rs`

| Method | Meaning |
| --- | --- |
| `id` | Lane id. Match this string from `AgentSpec::drivers` |
| `chooser_line` | One line the chooser shows for this lane |
| `budget` | Step and token budget |
| `allowlist` | Tool names to keep. Empty keeps the workflow set |
| `prompt` | Extra volatile text for this lane |
| `deny_tools` | When true, the turn gets an empty tool registry |

`StaticDriver` stores those fields. Use it for a fixed lane such as `answer` or `tool_loop`.

## `WorkflowRegistry`

`src/control/workflow.rs`

`WorkflowRegistry::with_chat` always registers `chat`. `register` accepts a plugin id of the form `plugin/name`. The id `chat` and a duplicate id fail.

`WorkflowSpec::chat` is the built-in bundle. Its lane is `tool_loop`.

## `VerifierRegistry`

`src/control/verify.rs`

`with_builtins` registers `unchecked`. `get` returns that verifier when the id is missing. `Unchecked` records the criteria and does not run a check.

Implement `Verifier` when the caller has a real check. Register it before the turn.

## `PromptBuilder`

`src/prompt.rs`

`PromptSlots::with_identity` puts the spec prompt in a stable slot. `PromptSlot::StablePlugin` stays in the cache. `PromptSlot::DynamicPlugin` is volatile.

`build` adds the clock. `build_with_rules` also adds caller-owned rule text. The byte cap is `MAX_PROMPT_BYTES` (256 KiB).

## `SuperGlueCompleter`

`src/completer.rs`

| Method | Meaning |
| --- | --- |
| `try_from_env` | Build an HTTP client and load credentials from the environment |
| `from_client` | Use a caller-built HTTP client |
| `from_shared` | Use caller-owned hook and guardrail registries |
| `with_mode` | `ApiMode::ChatCompletions` or `ApiMode::Responses` |
| `with_base_url` | Provider origin. The default is `https://api.openai.com` |
| `set_credentials` | Replace credentials for the next call |

Chat Completions is the default path. Responses keeps tool calls when a model runs with high reasoning effort.

## `AgentRunner`

`src/runner.rs`

```rust
AgentRunner::run(&completer, input, |piece| { /* StreamPiece */ }).await?;
```

`TurnInput` borrows the spec, the drivers, the registries, the tools, the history, and the user text. `volatile` holds turn text such as a brief. `server_rounds` and `swarm_steps` feed `AgentSpec::rounds`.

`TurnOutcome` holds `reply`, `reasoning`, `usage`, `lane`, `workflow_id`, `history`, and `transcript`.

`TurnError` is `Cancelled`, `Prompt`, or `Failed`.

## `StreamPiece`

`src/stream.rs`

| Variant | Meaning |
| --- | --- |
| `Text` | Assistant text |
| `Reasoning` | Model reasoning text |
| `Tool` | A tool call, a tool result, or a blocked call |

`DeltaBuf` keeps text until `DELTA_CHARS` (32) or an explicit flush. The runner does not use it. A caller can.
