//! Context windows used when a turn must stay inside the model limit.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Context and output limits used when the provider does not expose metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelBudget {
    pub context_tokens: u32,
    pub output_reserve: u32,
}

/// Published context windows from the model catalog.
#[derive(Clone, Default)]
pub struct ContextWindows {
    inner: Arc<Mutex<HashMap<String, u32>>>,
}

impl ContextWindows {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the map with the latest catalog windows.
    pub fn replace(&self, windows: HashMap<String, u32>) {
        match self.inner.lock() {
            Ok(mut guard) => *guard = windows,
            Err(poisoned) => *poisoned.into_inner() = windows,
        }
    }

    /// Known window for `model`, when the catalog published one.
    pub fn get(&self, model: &str) -> Option<u32> {
        let guard = match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.get(model).copied().filter(|n| *n > 0)
    }
}

/// Budget for compaction. A known catalog window replaces the name guess.
pub fn model_budget(model: &str, known: Option<u32>) -> ModelBudget {
    let context_tokens = known
        .filter(|n| *n > 0)
        .unwrap_or_else(|| guess_context(model));
    ModelBudget {
        context_tokens,
        output_reserve: context_tokens.min(16_384),
    }
}

/// Name-based window when the catalog omits `context_length`.
pub fn guess_context(model: &str) -> u32 {
    let name = model
        .rsplit(':')
        .next()
        .unwrap_or(model)
        .to_ascii_lowercase();
    if name.contains("claude") {
        200_000
    } else if name.contains("gemini") {
        1_000_000
    } else if name.contains("gpt-4") || name.contains("gpt-5") || name.contains("gpt-6") {
        128_000
    } else {
        64_000
    }
}

#[cfg(test)]
mod tests {
    use super::model_budget;

    #[test]
    fn known_window_wins_over_name_guess() {
        let budget = model_budget("openrouter:deepseek/deepseek-v4.1-flash", Some(1_048_576));
        assert_eq!(budget.context_tokens, 1_048_576);
        assert_eq!(budget.output_reserve, 16_384);
    }

    #[test]
    fn missing_window_uses_name_guess() {
        let budget = model_budget("openrouter:deepseek/deepseek-v4.1-flash", None);
        assert_eq!(budget.context_tokens, 64_000);
        let gpt = model_budget("openai:gpt-6-luna", None);
        assert_eq!(gpt.context_tokens, 128_000);
    }
}
