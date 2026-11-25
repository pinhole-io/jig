//! Cache-aware system prompt blocks.
//!
//! Stable blocks stay in the provider cache. Volatile blocks, including the
//! clock and a research brief, sit after the conversation.

use jiff::Timestamp;

use crate::clock;

pub const MAX_PROMPT_BYTES: usize = 256 * 1024;
const MAX_CONTRIBUTIONS: usize = 64;
pub const PROMPT_TOO_LARGE: &str = "prompt too large";

/// Values that shape one model turn's system prompt.
pub struct PromptContext<'a> {
    pub user_text: &'a str,
    pub system_note: &'a str,
    pub current_time: Timestamp,
}

/// A fixed prompt location that plugins can populate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptSlot {
    /// Stable instructions contributed by a plugin.
    StablePlugin,
    /// Turn-specific instructions contributed by a plugin.
    DynamicPlugin,
}

/// Bounded contributions collected before the prompt is rendered.
pub struct PromptSlots {
    stable_plugin: Vec<String>,
    dynamic_plugin: Vec<String>,
    cached_core: Vec<String>,
    core_volatile: Vec<String>,
    contribution_count: usize,
    total_bytes: usize,
}

impl PromptSlots {
    /// Create empty slots. An empty identity adds no block.
    pub fn new() -> Self {
        Self::with_identity("")
    }

    /// Create slots with `identity` as the first stable block.
    pub fn with_identity(identity: &str) -> Self {
        let mut slots = Self {
            stable_plugin: Vec::new(),
            dynamic_plugin: Vec::new(),
            cached_core: Vec::new(),
            core_volatile: Vec::new(),
            contribution_count: 0,
            total_bytes: 0,
        };
        if identity.is_empty() {
            return slots;
        }
        match slots.add(PromptSlot::StablePlugin, identity) {
            Ok(()) => slots,
            Err(_) => unreachable!("identity prompt fits the prompt limit"),
        }
    }

    /// Add bounded text to a fixed slot.
    pub fn add(&mut self, slot: PromptSlot, text: &str) -> Result<(), &'static str> {
        if text.is_empty() {
            return Ok(());
        }
        let next_bytes = self.total_bytes.saturating_add(text.len());
        if text.len() > MAX_PROMPT_BYTES
            || text.as_bytes().contains(&0)
            || self.contribution_count >= MAX_CONTRIBUTIONS
            || next_bytes > MAX_PROMPT_BYTES
        {
            return Err(PROMPT_TOO_LARGE);
        }
        self.contribution_count += 1;
        self.total_bytes = next_bytes;
        match slot {
            PromptSlot::StablePlugin => self.stable_plugin.push(text.to_string()),
            PromptSlot::DynamicPlugin => self.dynamic_plugin.push(text.to_string()),
        }
        Ok(())
    }

    /// Bytes still available under the total prompt contribution limit.
    pub fn remaining_bytes(&self) -> usize {
        MAX_PROMPT_BYTES.saturating_sub(self.total_bytes)
    }

    pub(crate) fn add_cached(&mut self, text: &str) -> Result<(), &'static str> {
        self.charge(text)?;
        if !text.is_empty() {
            self.cached_core.push(text.to_string());
        }
        Ok(())
    }

    pub(crate) fn add_core(&mut self, text: &str) -> Result<(), &'static str> {
        self.charge(text)?;
        if !text.is_empty() {
            self.core_volatile.push(text.to_string());
        }
        Ok(())
    }

    fn charge(&mut self, text: &str) -> Result<(), &'static str> {
        if text.is_empty() {
            return Ok(());
        }
        let next_bytes = self.total_bytes.saturating_add(text.len());
        if text.len() > MAX_PROMPT_BYTES
            || text.as_bytes().contains(&0)
            || next_bytes > MAX_PROMPT_BYTES
        {
            return Err(PROMPT_TOO_LARGE);
        }
        self.total_bytes = next_bytes;
        Ok(())
    }
}

impl Default for PromptSlots {
    fn default() -> Self {
        Self::new()
    }
}

/// Structured system prompt blocks for the model provider.
#[derive(Clone)]
pub struct PromptBlocks {
    pub cached: Vec<String>,
    pub volatile: Vec<String>,
}

impl PromptBlocks {
    /// Build a prompt with one cached block.
    pub fn cached_only(text: &str) -> Self {
        Self {
            cached: vec![text.to_string()],
            volatile: Vec::new(),
        }
    }

    /// Join blocks for tests and diagnostics.
    pub fn joined(&self) -> String {
        self.cached
            .iter()
            .chain(self.volatile.iter())
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// Renders prompt slots into stable and volatile provider blocks.
///
/// Disk rules stay with the host. Pass their text to [`PromptBuilder::build_with_rules`].
pub struct PromptBuilder<'a> {
    context: PromptContext<'a>,
    slots: PromptSlots,
}

impl<'a> PromptBuilder<'a> {
    /// Create a builder from core context and plugin contributions.
    pub fn new(context: PromptContext<'a>, slots: PromptSlots) -> Self {
        Self { context, slots }
    }

    pub fn user_text(&self) -> &'a str {
        self.context.user_text
    }

    /// Render stable rule text and matched rule text, then the clock.
    pub fn build_with_rules(
        self,
        stable_rules: Option<&str>,
        matched_rules: Option<&str>,
    ) -> Result<PromptBlocks, &'static str> {
        self.finish(stable_rules, matched_rules)
    }

    /// Render slots and the clock. This does not read rule files.
    pub fn build(self) -> Result<PromptBlocks, &'static str> {
        self.finish(None, None)
    }

    fn finish(
        mut self,
        stable_rules: Option<&str>,
        matched_rules: Option<&str>,
    ) -> Result<PromptBlocks, &'static str> {
        if let Some(stable) = stable_rules {
            self.slots.add_cached(stable)?;
        }
        self.slots.add_cached(self.context.system_note)?;
        if let Some(matched) = matched_rules {
            self.slots.add_core(matched)?;
        }
        let current_time =
            clock::clock_line(self.context.current_time, &jiff::tz::TimeZone::system());
        self.slots.add_core(&current_time)?;
        let mut cached = std::mem::take(&mut self.slots.stable_plugin);
        cached.extend(std::mem::take(&mut self.slots.cached_core));
        let mut volatile = std::mem::take(&mut self.slots.dynamic_plugin);
        volatile.extend(std::mem::take(&mut self.slots.core_volatile));
        Ok(PromptBlocks { cached, volatile })
    }
}
