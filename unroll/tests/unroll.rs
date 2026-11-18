use qp_plonky2_unroll::unroll_for_loops;

#[unroll_for_loops]
fn sum_half_open() -> usize {
    let mut acc = 0;
    for i in 0..5 {
        acc += i;
    }
    acc
}

#[unroll_for_loops]
fn sum_closed_with_start() -> usize {
    let mut acc = 0;
    for i in 2..=4 {
        acc += i;
    }
    acc
}

/// The loop variable is a `const` in each unrolled body, so it can be used in
/// const positions and `if i < N` guards fold away. This is the pattern the
/// Poseidon code relies on to unroll loops over a generic width.
#[unroll_for_loops]
fn guarded<const N: usize>(input: &[u64; N]) -> u64 {
    let mut acc = 0u64;
    for i in 0..12 {
        if i < N {
            let _: [u8; i] = [0; i];
            acc = acc.wrapping_mul(31).wrapping_add(input[i]);
        }
    }
    acc
}

#[unroll_for_loops]
fn nested(mtx: &[[u32; 3]; 3], vec: &[u32; 3]) -> [u32; 3] {
    let mut out = [0; 3];
    for col in 0..3 {
        for row in 0..3 {
            out[row] += mtx[col][row] * vec[col];
        }
    }
    out
}

/// Non-literal bounds are left alone (still a loop, still correct).
#[unroll_for_loops]
fn dynamic_bound(n: usize) -> usize {
    let mut acc = 0;
    for i in 0..n {
        acc += i;
    }
    acc
}

#[test]
fn unrolled_loops_compute_the_same_values() {
    assert_eq!(sum_half_open(), 0 + 1 + 2 + 3 + 4);
    assert_eq!(sum_closed_with_start(), 2 + 3 + 4);

    let input = [7u64, 11, 13];
    let mut expected = 0u64;
    for x in input {
        expected = expected.wrapping_mul(31).wrapping_add(x);
    }
    assert_eq!(guarded(&input), expected);

    let mtx = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let vec = [1, 0, 2];
    assert_eq!(nested(&mtx, &vec), [1 + 14, 2 + 16, 3 + 18]);

    assert_eq!(dynamic_bound(10), 45);
}
