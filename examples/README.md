# Jig examples

Run an example from the repository root:

```bash
cargo run --example 01_agent_spec
```

Omit the `.rs` suffix.

`01_agent_spec` and `02_prompt` do not call a provider. They still need the sibling SuperGlue checkout.

`03_one_turn` reads `OPENAI_API_KEY`. It exits when that variable is empty. `OPENAI_MODEL` overrides `gpt-6-luna`.

| Example | Provider | What it shows |
| --- | --- | --- |
| `01_agent_spec` | no | One spec, an answer driver, a tool-loop driver, and a `guide/lookup` workflow |
| `02_prompt` | no | Stable blocks, one rule, and the clock line |
| `03_one_turn` | yes | `AgentRunner::run` with one driver and an empty tool registry |
