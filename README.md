# Jig

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust edition 2024](https://img.shields.io/badge/rust-edition%202024-orange.svg)](Cargo.toml)
[![Rust 1.88+](https://img.shields.io/badge/rust-1.88%2B-orange.svg)](Cargo.toml)

**A protocol-free agent runtime.** One turn builds a prompt, picks a driver, streams the model, and checks the reply.

Jig does not open a server, a sandbox, or a tool set. The caller supplies the tool registry, the completer, and the drivers. Model calls go through [SuperGlue](https://github.com/pinhole-io/superglue).

## Requirements

- Rust 1.88 or newer. `rust-version` in `Cargo.toml` sets this floor.
- A sibling checkout of [superglue](https://github.com/pinhole-io/superglue). Cargo resolves `path = "../superglue"`.

## Install

There is no crates.io package. Clone the two repositories as siblings.

```text
git clone https://github.com/pinhole-io/jig.git
git clone https://github.com/pinhole-io/superglue.git
```

## Run one turn

1. Build an `AgentSpec` with a prompt, a tool policy, and one or more driver ids.
2. Build a `SuperGlueCompleter` from an HTTP client.
3. Call `AgentRunner::run` with a `TurnInput`.

One driver skips the chooser. More than one driver asks the chooser which driver runs.

## License

MIT. See [LICENSE](LICENSE).
