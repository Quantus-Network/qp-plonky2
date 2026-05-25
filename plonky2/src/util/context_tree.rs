#[cfg(not(feature = "std"))]
use alloc::{
    string::{String, ToString},
    vec,
    vec::Vec,
};

use log::{log, Level};

pub(crate) const MAX_CONTEXT_NAME_LEN: usize = 256;
pub(crate) const MAX_CONTEXT_DEPTH: usize = 64;

/// The hierarchy of contexts, and the gate count contributed by each one. Useful for debugging.
#[derive(Debug)]
pub(crate) struct ContextTree {
    /// The name of this scope.
    name: String,
    /// The level at which to log this scope and its children.
    level: log::Level,
    /// The gate count when this scope was created.
    enter_gate_count: usize,
    /// The gate count when this scope was destroyed, or None if it has not yet been destroyed.
    exit_gate_count: Option<usize>,
    /// Any child contexts.
    children: Vec<ContextTree>,
    /// The currently-open non-root context names. Maintained only on the root tree.
    open_stack: Vec<String>,
    /// Cached display form of the open context stack.
    open_stack_display: String,
}

impl ContextTree {
    pub fn new() -> Self {
        Self {
            name: "root".to_string(),
            level: Level::Debug,
            enter_gate_count: 0,
            exit_gate_count: None,
            children: vec![],
            open_stack: vec![],
            open_stack_display: "root".to_string(),
        }
    }

    /// Whether this context is still in scope.
    const fn is_open(&self) -> bool {
        self.exit_gate_count.is_none()
    }

    /// A description of the stack of currently-open scopes.
    pub fn open_stack(&self) -> String {
        self.open_stack_display.clone()
    }

    fn bounded_context_name(ctx: &str) -> String {
        if ctx.len() <= MAX_CONTEXT_NAME_LEN {
            return ctx.to_string();
        }

        let mut end = MAX_CONTEXT_NAME_LEN;
        while !ctx.is_char_boundary(end) {
            end -= 1;
        }
        ctx[..end].to_string()
    }

    fn refresh_open_stack_display(&mut self) {
        self.open_stack_display = "root".to_string();
        for ctx in &self.open_stack {
            self.open_stack_display.push_str(" > ");
            self.open_stack_display.push_str(ctx);
        }
    }

    pub fn try_push(
        &mut self,
        ctx: &str,
        level: log::Level,
        current_gate_count: usize,
    ) -> Result<(), &'static str> {
        if self.open_stack.len() >= MAX_CONTEXT_DEPTH {
            return Err("context metadata depth limit exceeded");
        }

        let ctx = Self::bounded_context_name(ctx);
        self.push_node(&ctx, level, current_gate_count);
        self.open_stack.push(ctx);
        self.refresh_open_stack_display();
        Ok(())
    }

    pub fn push(&mut self, ctx: &str, level: log::Level, current_gate_count: usize) {
        self.try_push(ctx, level, current_gate_count)
            .expect("context metadata depth limit exceeded")
    }

    fn push_node(&mut self, ctx: &str, mut level: log::Level, current_gate_count: usize) {
        assert!(self.is_open());

        // We don't want a scope's log level to be stronger than that of its parent.
        level = level.max(self.level);

        if let Some(last_child) = self.children.last_mut() {
            if last_child.is_open() {
                last_child.push_node(ctx, level, current_gate_count);
                return;
            }
        }

        self.children.push(ContextTree {
            name: ctx.to_string(),
            level,
            enter_gate_count: current_gate_count,
            exit_gate_count: None,
            children: vec![],
            open_stack: vec![],
            open_stack_display: String::new(),
        })
    }

    /// Close the deepest open context from this tree.
    pub fn pop(&mut self, current_gate_count: usize) {
        assert!(
            !self.open_stack.is_empty(),
            "attempted to pop context metadata with no open context"
        );
        self.pop_node(current_gate_count);
        self.open_stack.pop();
        self.refresh_open_stack_display();
    }

    fn pop_node(&mut self, current_gate_count: usize) {
        assert!(self.is_open());

        if let Some(last_child) = self.children.last_mut() {
            if last_child.is_open() {
                last_child.pop_node(current_gate_count);
                return;
            }
        }

        self.exit_gate_count = Some(current_gate_count);
    }

    fn gate_count_delta(&self, current_gate_count: usize) -> usize {
        self.exit_gate_count.unwrap_or(current_gate_count) - self.enter_gate_count
    }

    /// Filter out children with a low gate count.
    pub fn filter(&self, current_gate_count: usize, min_delta: usize) -> Self {
        Self {
            name: self.name.clone(),
            level: self.level,
            enter_gate_count: self.enter_gate_count,
            exit_gate_count: self.exit_gate_count,
            children: self
                .children
                .iter()
                .filter(|c| c.gate_count_delta(current_gate_count) >= min_delta)
                .map(|c| c.filter(current_gate_count, min_delta))
                .collect(),
            open_stack: vec![],
            open_stack_display: String::new(),
        }
    }

    pub fn print(&self, current_gate_count: usize) {
        self.print_helper(current_gate_count, 0);
    }

    fn print_helper(&self, current_gate_count: usize, depth: usize) {
        let prefix = "| ".repeat(depth);
        log!(
            self.level,
            "{}{} gates to {}",
            prefix,
            self.gate_count_delta(current_gate_count),
            self.name
        );
        for child in &self.children {
            child.print_helper(current_gate_count, depth + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_names_are_truncated_in_cached_stack() {
        let mut tree = ContextTree::new();
        let long_name = "x".repeat(MAX_CONTEXT_NAME_LEN + 1024);

        tree.try_push(&long_name, Level::Debug, 0).unwrap();
        let stack = tree.open_stack();

        assert!(stack.starts_with("root > "));
        assert_eq!(stack.len(), "root > ".len() + MAX_CONTEXT_NAME_LEN);
    }

    #[test]
    fn context_depth_is_bounded() {
        let mut tree = ContextTree::new();
        for _ in 0..MAX_CONTEXT_DEPTH {
            tree.try_push("ctx", Level::Debug, 0).unwrap();
        }

        assert!(tree.try_push("ctx", Level::Debug, 0).is_err());
    }

    #[test]
    fn cached_stack_updates_after_pop() {
        let mut tree = ContextTree::new();
        tree.try_push("outer", Level::Debug, 0).unwrap();
        tree.try_push("inner", Level::Debug, 0).unwrap();
        assert_eq!(tree.open_stack(), "root > outer > inner");

        tree.pop(0);
        assert_eq!(tree.open_stack(), "root > outer");
        tree.pop(0);
        assert_eq!(tree.open_stack(), "root");
    }
}

/// Creates a named scope; useful for debugging.
#[macro_export]
macro_rules! with_context {
    ($builder:expr, $level:expr, $ctx:expr, $exp:expr) => {{
        $builder.push_context($level, $ctx);
        let res = $exp;
        $builder.pop_context();
        res
    }};
    // If no context is specified, default to Debug.
    ($builder:expr, $ctx:expr, $exp:expr) => {{
        $builder.push_context(log::Level::Debug, $ctx);
        let res = $exp;
        $builder.pop_context();
        res
    }};
}
