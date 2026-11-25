//! Streaming pieces for one model turn.
//!
//! A host can map these pieces onto its own event stream.

pub const DELTA_CHARS: usize = 32;

pub enum StreamPiece {
    /// Visible reply text.
    Text(String),
    /// Model reasoning. This is not part of the reply.
    Reasoning(String),
    /// A tool call, result, or block.
    Tool(ToolPiece),
}

/// One tool notice on the turn lane.
#[derive(Clone)]
pub struct ToolPiece {
    /// Tool name.
    pub name: String,
    /// Call, result, or blocked.
    pub point: ToolPoint,
    /// `true` when `point` is a successful result.
    pub ok: bool,
    /// Short reason when the call is blocked or the result is not ok.
    pub error: String,
    /// JSON arguments the model passed.
    pub arguments: String,
    /// One id for the call and the result of the same invocation.
    pub call_id: String,
    /// Tool return value. Empty on a call or a block.
    pub result: String,
}

/// Where a tool notice sits in the call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolPoint {
    Call,
    Result,
    Blocked,
}

/// Ordered pieces for one turn. The turn loop is the only publisher.
pub struct PieceLane {
    pieces: std::sync::Mutex<Vec<StreamPiece>>,
    notify: tokio::sync::Notify,
}

impl PieceLane {
    pub fn new() -> Self {
        Self {
            pieces: std::sync::Mutex::new(Vec::new()),
            notify: tokio::sync::Notify::new(),
        }
    }

    /// Append `piece` and wake the turn loop.
    pub fn push(&self, piece: StreamPiece) {
        match self.pieces.lock() {
            Ok(mut guard) => guard.push(piece),
            Err(poisoned) => poisoned.into_inner().push(piece),
        }
        self.notify.notify_one();
    }

    /// Take every piece pushed since the last take.
    pub fn take(&self) -> Vec<StreamPiece> {
        match self.pieces.lock() {
            Ok(mut guard) => std::mem::take(&mut *guard),
            Err(poisoned) => std::mem::take(&mut *poisoned.into_inner()),
        }
    }

    pub fn notified(&self) -> tokio::sync::futures::Notified<'_> {
        self.notify.notified()
    }
}

impl Default for PieceLane {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Default)]
pub struct DeltaBuf {
    pending: String,
}

impl DeltaBuf {
    pub fn push(&mut self, chunk: &str) -> Vec<String> {
        self.pending.push_str(chunk);
        let mut chars = 0usize;
        let mut end = 0usize;
        for (idx, ch) in self.pending.char_indices() {
            chars += 1;
            if chars.is_multiple_of(DELTA_CHARS) {
                end = idx + ch.len_utf8();
            }
        }
        if end == 0 {
            return Vec::new();
        }
        let taken = self.pending.drain(..end).collect::<String>();
        let mut out = Vec::new();
        let mut start = 0usize;
        let mut seen = 0usize;
        for (idx, _) in taken.char_indices() {
            if seen == DELTA_CHARS {
                out.push(taken[start..idx].to_string());
                start = idx;
                seen = 0;
            }
            seen += 1;
        }
        if seen == DELTA_CHARS {
            out.push(taken[start..].to_string());
        }
        out
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    pub fn flush(&mut self) -> Option<String> {
        if self.pending.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.pending))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_keeps_a_chunk_past_the_old_reply_limit() {
        let mut buf = DeltaBuf::default();
        let chunk = "a".repeat(64 * 1024 + 8);
        let pieces = buf.push(&chunk);
        let kept = pieces.iter().map(String::len).sum::<usize>() + buf.pending.len();
        assert_eq!(kept, chunk.len());
    }
}
