# Architecture

Jig runs one agent turn. The caller owns the process, the tools, and the credentials.

## Data flow

```text
AgentSpec + drivers
        |
        v
  PromptBuilder
        |
        v
  chooser, when drivers > 1
        |
        v
  tool policy + workflow tools
        |
        v
  SuperGlueCompleter
        |
        v
  StreamPiece  -->  caller
        |
        v
  Verifier
        |
        v
  TurnOutcome
```

## Who owns what

| Piece | Owner | Role |
| --- | --- | --- |
| `AgentSpec` | Caller | Prompt, tool policy, round budget, driver ids |
| `EngineDriver` | Caller | Lane id, budget, tool allowlist, extra prompt |
| `ToolRegistry` | Caller | The tools this process can run |
| `SuperGlueCompleter` | Caller | HTTP client, credentials, API mode |
| `WorkflowRegistry` | Caller | Chat plus any `plugin/name` workflow |
| `AgentRunner` | Jig | One turn |
| Provider HTTP | SuperGlue | The model call |

Jig has no server, sandbox, or built-in tool set.

## One turn

1. Keep the drivers named by `AgentSpec::drivers`.
2. Build the prompt from the spec, the user text, and the volatile blocks.
3. Use the only driver when the list has one id.
4. Ask the chooser when the list has more than one id. A low-confidence pick falls back to chat.
5. Narrow the tool registry with the policy, the workflow tool list, and the driver allowlist.
6. Stream `StreamPiece::Text`, `StreamPiece::Reasoning`, and `StreamPiece::Tool`.
7. Run the verifier named by the selected workflow.
8. Return `TurnOutcome` with the reply, the lane, and the next transcript.

A cancelled turn returns `TurnError::Cancelled`. A prompt that exceeds 256 KiB returns `TurnError::Prompt`.

## Prompt blocks

Stable blocks go in `PromptBlocks::cached`. The clock and other turn text go in `PromptBlocks::volatile`.

Disk rules stay with the caller. Pass their text to `PromptBuilder::build_with_rules`. The runner calls `PromptBuilder::build` and does not read rule files.

## Tool watch

`AgentRunner` sets `watch_tools` on the completion. The completer installs one hook pair named `jig-tool-watch`.

The hook emits a tool piece before the call and after the result. The same tool error twice cancels the turn. The `agents` tool name skips the watch.

## Credentials

`SuperGlueCompleter` stores provider credentials in `secrecy` types. `try_from_env` reads the process environment. `set_credentials` replaces them for the next call.

Error text from the completer drops bearer tokens and key-shaped strings. Do not log the key.
