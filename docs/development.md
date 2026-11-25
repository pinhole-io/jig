# Development

Run these commands from the repository root.

## Format and test

```bash
cargo fmt
cargo test --offline
cargo clippy --all-targets -- -D warnings
```

`cargo test --offline` uses the lockfile and the local Cargo cache. Run `cargo fetch` once when `superglue` is not cached. It does not call a provider.

## Examples

```bash
cargo build --examples
cargo run --example 01_agent_spec
cargo run --example 02_prompt
```

`03_one_turn` calls the provider when `OPENAI_API_KEY` is set. An empty key prints one line and exits 0.

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-6-luna   # optional
cargo run --example 03_one_turn
```

The key stays in the environment. The example prints the reply and the lane. It does not print the key.

## Layout

| Path | Role |
| --- | --- |
| `src/lib.rs` | Crate root and re-exports |
| `src/agent.rs` | `AgentSpec` and tool policy |
| `src/runner.rs` | `AgentRunner` and drivers |
| `src/prompt.rs` | Prompt blocks |
| `src/completer.rs` | SuperGlue completer |
| `src/control/` | Workflows, triage, and verifiers |
| `docs/` | Guides |
| `examples/` | Runnable programs |

## MSRV

`rust-version` is `1.88`. Keep that floor so a caller on 1.88 can build this crate.
