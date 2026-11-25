# Jig

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust edition 2024](https://img.shields.io/badge/rust-edition%202024-orange.svg)](Cargo.toml)
[![Rust 1.88+](https://img.shields.io/badge/rust-1.88%2B-orange.svg)](Cargo.toml)

**A protocol-free agent runtime.** One turn builds a prompt, picks a driver, streams the model, and checks the reply.

The caller supplies the tool registry, the completer, and the drivers. Model calls go through [SuperGlue](https://github.com/pinhole-io/superglue).

[Quick start](#quick-start) · [Examples](examples/README.md) · [Architecture](docs/architecture.md) · [API](docs/api.md)

## Contents

- [Requirements](#requirements)
- [Install](#install)
- [Quick start](#quick-start)
- [Features](#features)
- [How it fits together](#how-it-fits-together)
- [Documentation](#documentation)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [License](#license)

## Requirements

- Rust 1.88 or newer. `rust-version` in `Cargo.toml` sets this floor.
- A sibling checkout of [superglue](https://github.com/pinhole-io/superglue). Cargo resolves `path = "../superglue"`.
- `OPENAI_API_KEY` when you run `03_one_turn`. The other examples stay on this machine.

## Install

There is no crates.io package. Clone the two repositories as siblings.

```bash
mkdir -p ~/src && cd ~/src
git clone https://github.com/pinhole-io/jig.git
git clone https://github.com/pinhole-io/superglue.git
```

You need this layout:

```text
~/src/
  jig/
  superglue/
```

| Surface | Where it runs | How to build |
| --- | --- | --- |
| Rust library | This repository | `cargo test` |
| Examples | `examples/` | `cargo run --example <name>` |

## Quick start

`01_agent_spec` and `02_prompt` do not call a model. The sibling checkout is still required.

```bash
cd jig
cargo test
cargo run --example 01_agent_spec
cargo run --example 02_prompt
```

## Features

| Feature | What you get |
| --- | --- |
| Spec | `AgentSpec` holds the prompt, the tool policy, the round budget, and the driver ids |
| Drivers | `StaticDriver` is one lane. One driver skips the chooser |
| Prompt | `PromptBuilder` splits stable blocks from the clock and other volatile text |
| Turn | `AgentRunner::run` streams text, reasoning, and tool pieces |
| Completer | `SuperGlueCompleter` calls Chat Completions or Responses |
| Check | `VerifierRegistry` records the workflow check after the reply |

## How it fits together

```text
Caller
  spec, drivers, tools, completer
            |
            v
          Jig
            |
            v
       SuperGlue
            |
            v
        Provider
```

- The caller owns the tools, the credentials, and the driver list.
- Jig owns one turn: prompt, lane, stream, and verifier.
- SuperGlue owns the provider HTTP call.

## Documentation

| Goal | Document |
| --- | --- |
| See one turn from spec to reply | [Architecture](docs/architecture.md) |
| Call the Rust types | [Library API](docs/api.md) |
| Build, test, and run examples | [Development](docs/development.md) |
| Pick an example | [Examples](examples/README.md) |

## Examples

| Example | Calls a provider | What it shows |
| --- | --- | --- |
| `01_agent_spec` | no | One spec, two drivers, and a plugin workflow |
| `02_prompt` | no | Stable blocks, a rule, and the clock line |
| `03_one_turn` | yes | One answer turn. Exits when `OPENAI_API_KEY` is unset |

```bash
cargo run --example 03_one_turn
```

Keep the API key in the environment. Do not commit it.

## Troubleshooting

| What you see | What to do |
| --- | --- |
| `can't find crate superglue` | Clone SuperGlue next to this repository. See [Install](#install). |
| `01_agent_spec` fails to build | The example does not call a model. The path dependency is still required. |
| `03_one_turn` prints `Set OPENAI_API_KEY` | Export the key, then run the example again. |
| The chooser runs on a one-lane agent | Put one id in `AgentSpec::drivers`. More than one id asks the chooser. |
| A plugin workflow returns `invalid workflow id` | Use `plugin/name`. Lowercase letters, digits, and `-` only. `chat` is reserved. |

## Development

```bash
cargo fmt
cargo test --offline
cargo build --examples
cargo clippy --all-targets -- -D warnings
```

Details: [Development](docs/development.md).

## License

The license is [MIT](LICENSE).

Keep the copyright notice and the permission notice in every copy.
