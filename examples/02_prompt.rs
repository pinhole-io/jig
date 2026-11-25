//! Render stable prompt blocks and the clock line.
//!
//! This example does not call a model.

use jiff::Timestamp;
use jig::{PromptBuilder, PromptContext, PromptSlot, PromptSlots};

fn main() {
    let mut slots = PromptSlots::with_identity("You answer in one short paragraph.");
    slots
        .add(
            PromptSlot::StablePlugin,
            "Prefer the tools the caller registered.",
        )
        .expect("stable text fits");
    slots
        .add(PromptSlot::DynamicPlugin, "The user asked about the clock.")
        .expect("volatile text fits");

    let blocks = PromptBuilder::new(
        PromptContext {
            user_text: "What time is it?",
            system_note: "",
            current_time: Timestamp::now(),
        },
        slots,
    )
    .build_with_rules(Some("Keep answers short."), None)
    .expect("prompt fits");

    println!("cached blocks: {}", blocks.cached.len());
    println!("volatile blocks: {}", blocks.volatile.len());
    println!();
    println!("{}", blocks.joined());
}
