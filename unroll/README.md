# qp-plonky2-unroll

`#[unroll_for_loops]`: an attribute macro that unrolls `for` loops with integer
literal bounds at compile time.

This is a vendored fork of [`unroll`](https://crates.io/crates/unroll) v0.1.5
(MIT OR Apache-2.0, Egor Larionov), ported to `syn` 2 and kept in this
workspace so the Poseidon/FFT hot paths can keep using it without depending on
an unmaintained crate. The original license files are included alongside.
