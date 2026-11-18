//! An attribute-like procedural macro for unrolling for loops with integer literal bounds.
//!
//! This is a vendored fork of the [`unroll`](https://crates.io/crates/unroll) crate (v0.1.5,
//! MIT OR Apache-2.0, by Egor Larionov), ported to `syn` 2. The upstream crate has had no
//! release since 2020 and was the last `syn` 1 dependency in the workspace.
//!
//! Just add `#[unroll_for_loops]` above the function whose for loops you would like to unroll.
//! All for loops with integer literal bounds will be unrolled; the loop variable becomes a
//! `const` in each unrolled copy of the body, so `if i < N { .. }` guards fold away. The macro
//! does not look inside closures.
//!
//! ```rust
//! use qp_plonky2_unroll::unroll_for_loops;
//!
//! #[unroll_for_loops]
//! fn mtx_vec_mul(mtx: &[[f64; 5]; 5], vec: &[f64; 5]) -> [f64; 5] {
//!     let mut out = [0.0; 5];
//!     for col in 0..5 {
//!         for row in 0..5 {
//!             out[row] += mtx[col][row] * vec[col];
//!         }
//!     }
//!     out
//! }
//! ```

use proc_macro::TokenStream;
use quote::quote;
use syn::token::Brace;
use syn::{
    parse_quote, Block, Expr, ExprBlock, ExprForLoop, ExprIf, ExprLet, ExprLit, ExprRange, Item,
    ItemFn, Lit, Pat, PatIdent, RangeLimits, Stmt,
};

/// Attribute used to unroll for loops found inside a function block.
#[proc_macro_attribute]
pub fn unroll_for_loops(_meta: TokenStream, input: TokenStream) -> TokenStream {
    let item: Item = syn::parse(input).expect("Failed to parse input.");

    if let Item::Fn(item_fn) = item {
        let new_block = unroll_in_block(&item_fn.block);
        let new_item = Item::Fn(ItemFn {
            block: Box::new(new_block),
            ..item_fn
        });
        quote!( #new_item ).into()
    } else {
        quote!( #item ).into()
    }
}

/// Routine to unroll for loops within a block.
fn unroll_in_block(block: &Block) -> Block {
    let Block { brace_token, stmts } = block;
    let new_stmts = stmts
        .iter()
        .map(|stmt| match stmt {
            Stmt::Expr(expr, semi) => Stmt::Expr(unroll(expr), *semi),
            other => other.clone(),
        })
        .collect();
    Block {
        brace_token: *brace_token,
        stmts: new_stmts,
    }
}

/// Parses an integer literal bound, or `None` if the expression is not one.
fn literal_bound(expr: &Expr) -> Option<usize> {
    if let Expr::Lit(ExprLit {
        lit: Lit::Int(lit_int),
        ..
    }) = expr
    {
        Some(lit_int.base10_parse::<usize>().unwrap())
    } else {
        None
    }
}

/// Routine to unroll a for loop expression, or return the expression (with any nested loops
/// unrolled) unchanged if it is not an unrollable for loop.
fn unroll(expr: &Expr) -> Expr {
    match expr {
        Expr::ForLoop(for_loop) => {
            let ExprForLoop {
                attrs,
                label,
                pat,
                expr: range_expr,
                body,
                ..
            } = for_loop;

            let new_body = unroll_in_block(body);

            let forloop_with_body = |body| {
                Expr::ForLoop(ExprForLoop {
                    body,
                    ..for_loop.clone()
                })
            };

            let Pat::Ident(PatIdent {
                by_ref,
                mutability,
                ident: idx,
                subpat,
                ..
            }) = &**pat
            else {
                return forloop_with_body(new_body);
            };
            // Don't know how to deal with these so skip and return the original.
            if by_ref.is_some() || mutability.is_some() || subpat.is_some() {
                return forloop_with_body(new_body);
            }

            let Expr::Range(ExprRange {
                start, limits, end, ..
            }) = &**range_expr
            else {
                return forloop_with_body(new_body);
            };

            let begin = match start {
                Some(start) => match literal_bound(start) {
                    Some(b) => b,
                    None => return forloop_with_body(new_body),
                },
                None => 0,
            };
            // We need to know where the limit is to know how much to unroll by.
            let end = match end.as_deref().and_then(literal_bound) {
                Some(e) => e,
                None => return forloop_with_body(new_body),
            } + match limits {
                RangeLimits::Closed(_) => 1,
                RangeLimits::HalfOpen(_) => 0,
            };

            let stmts = (begin..end)
                .map(|i| {
                    let declare_i: Stmt = parse_quote! {
                        #[allow(non_upper_case_globals)]
                        const #idx: usize = #i;
                    };
                    let mut augmented_body = new_body.clone();
                    augmented_body.stmts.insert(0, declare_i);
                    parse_quote! { #augmented_body }
                })
                .collect();
            Expr::Block(ExprBlock {
                attrs: attrs.clone(),
                label: label.clone(),
                block: Block {
                    brace_token: Brace::default(),
                    stmts,
                },
            })
        }
        Expr::If(if_expr) => {
            let ExprIf {
                cond,
                then_branch,
                else_branch,
                ..
            } = if_expr;
            Expr::If(ExprIf {
                cond: Box::new(unroll(cond)),
                then_branch: unroll_in_block(then_branch),
                else_branch: else_branch
                    .as_ref()
                    .map(|(else_token, expr)| (*else_token, Box::new(unroll(expr)))),
                ..if_expr.clone()
            })
        }
        Expr::Let(let_expr) => Expr::Let(ExprLet {
            expr: Box::new(unroll(&let_expr.expr)),
            ..let_expr.clone()
        }),
        Expr::Block(expr_block) => Expr::Block(ExprBlock {
            block: unroll_in_block(&expr_block.block),
            ..expr_block.clone()
        }),
        other => other.clone(),
    }
}
