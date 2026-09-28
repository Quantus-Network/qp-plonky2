/-
  Step 8f — the recorded `n_inner = 2` public-batch wrapper, landed on `RPublicBatch`.

  `Generated/PublicBatchWrapper2.lean` proves, from `Satisfies` on the wiring the real
  `CircuitBuilder` emitted, the meaning of each of the 122 gadget calls
  `build_public_batch_constraints` made over two `2`-leaf private batches. This module reads
  the inner outputs and the aggregated output off the named targets and public inputs and
  discharges every hypothesis of `public_batch_val` (`Plonky2Bridge/PublicBatch.lean`), so
  `public_batch_end_to_end` is restated on the wiring with `private_batch_proof_sound` as
  its only axiom (`public_batch_end_to_end_wired`).
-/
import Plonky2Bridge.PublicBatch
import Plonky2Spec.Generated.PublicBatchWrapper2

namespace Plonky2Bridge.PublicWrapper2

open Plonky2Spec (bselect band bnot bor scanStep)
open Plonky2Spec.Wiring
open Plonky2Spec.Generated (publicBatchWrapper2 publicBatchWrapper2_decode)
open WormholeSpec (Digest Felt PrivateBatchOutput PublicBatchOutput ExitSlot RandomOracle RPublicBatch)

variable {p : ℕ} [Fact p.Prime]

/-! ### Reading the spec objects off the wiring -/

/-- The 52 public inputs of inner proof `i` (`inner_pis_i`), at the private-batch
    `aggregated_output` layout. -/
def innerPis : Fin 2 → Fin 52 → Target
  | 0 => publicBatchWrapper2.inner_pis_0
  | 1 => publicBatchWrapper2.inner_pis_1

/-- The field wires of inner `i` the wrapper reads, with the `is_equal` witnesses of its
    `bytes_digest_eq(block_hash, 0)` dummy check (calls 0–3 / 7–10). -/
def row (a : Assignment p) : Fin 2 → InnerRow p
  | 0 =>
    { blockHash := fun j => a (innerPis 0 ⟨3 + j, by omega⟩)
      dummyEq := ![a (.virt 19080), a (.virt 19082), a (.virt 19084), a (.virt 19086)]
      dummyInv := ![a (.virt 19081), a (.virt 19083), a (.virt 19085), a (.virt 19087)]
      blockNumber := a (innerPis 0 7), assetId := a (innerPis 0 1), fee := a (innerPis 0 2)
      slots := [(a (innerPis 0 8), fun j => a (innerPis 0 ⟨9 + j, by omega⟩)),
        (a (innerPis 0 13), fun j => a (innerPis 0 ⟨14 + j, by omega⟩)),
        (a (innerPis 0 18), fun j => a (innerPis 0 ⟨19 + j, by omega⟩)),
        (a (innerPis 0 23), fun j => a (innerPis 0 ⟨24 + j, by omega⟩))]
      nulls := [fun j => a (innerPis 0 ⟨28 + j, by omega⟩), fun j => a (innerPis 0 ⟨32 + j, by omega⟩)] }
  | 1 =>
    { blockHash := fun j => a (innerPis 1 ⟨3 + j, by omega⟩)
      dummyEq := ![a (.virt 19088), a (.virt 19090), a (.virt 19092), a (.virt 19094)]
      dummyInv := ![a (.virt 19089), a (.virt 19091), a (.virt 19093), a (.virt 19095)]
      blockNumber := a (innerPis 1 7), assetId := a (innerPis 1 1), fee := a (innerPis 1 2)
      slots := [(a (innerPis 1 8), fun j => a (innerPis 1 ⟨9 + j, by omega⟩)),
        (a (innerPis 1 13), fun j => a (innerPis 1 ⟨14 + j, by omega⟩)),
        (a (innerPis 1 18), fun j => a (innerPis 1 ⟨19 + j, by omega⟩)),
        (a (innerPis 1 23), fun j => a (innerPis 1 ⟨24 + j, by omega⟩))]
      nulls := [fun j => a (innerPis 1 ⟨28 + j, by omega⟩), fun j => a (innerPis 1 ⟨32 + j, by omega⟩)] }

/-- Inner `i`'s private-batch output, decoded through `.val`. -/
def inner (a : Assignment p) (i : Fin 2) : PrivateBatchOutput :=
  { numExitSlots := (a (innerPis i 0)).val, assetId := (a (innerPis i 1)).val,
    volumeFeeBps := (a (innerPis i 2)).val, blockHash := valDigest (row a i).blockHash,
    blockNumber := (a (innerPis i 7)).val, exitSlots := (row a i).slots.map valSlot,
    nullifiers := (row a i).nulls.map valDigest }

def rows (a : Assignment p) : List (InnerPair p) := [(row a 0, inner a 0), (row a 1, inner a 1)]
def inners (a : Assignment p) : List PrivateBatchOutput := [inner a 0, inner a 1]

omit [Fact p.Prime] in
theorem rows_inners (a : Assignment p) : (rows a).map Prod.snd = inners a := rfl

/-- The `aggregator_address` witness, decoded. -/
def addr (a : Assignment p) : Digest := valDigest fun j => a (publicBatchWrapper2.aggregator_address j)

/-- Public input `k` of the wrapper, decoded through `.val`. -/
def pv (a : Assignment p) (k : ℕ) : Felt :=
  (a ((publicBatchWrapper2 p).publicInputs.getD k (.virt 0))).val

/-- The aggregated output at the public-batch layout: `[aggregator_address(4), asset_id,
    volume_fee_bps, block_hash(4), block_number, total_exit_slots]`, `2 · 4` forwarded exit
    slots `[sum, account(4)]`, `2 · 2` forwarded nullifiers. -/
def out (a : Assignment p) : PublicBatchOutput :=
  { aggregatorAddress := ⟨pv a 0, pv a 1, pv a 2, pv a 3⟩, assetId := pv a 4, volumeFeeBps := pv a 5,
    blockHash := ⟨pv a 6, pv a 7, pv a 8, pv a 9⟩, blockNumber := pv a 10, totalExitSlots := pv a 11,
    exitSlots := [⟨(a (.wire 7 71)).val, ⟨(a (.wire 7 79)).val, (a (.wire 9 7)).val, (a (.wire 9 15)).val, (a (.wire 9 23)).val⟩⟩,
      ⟨(a (.wire 9 31)).val, ⟨(a (.wire 9 39)).val, (a (.wire 9 47)).val, (a (.wire 9 55)).val, (a (.wire 9 63)).val⟩⟩,
      ⟨(a (.wire 9 71)).val, ⟨(a (.wire 9 79)).val, (a (.wire 10 7)).val, (a (.wire 10 15)).val, (a (.wire 10 23)).val⟩⟩,
      ⟨(a (.wire 10 31)).val, ⟨(a (.wire 10 39)).val, (a (.wire 10 47)).val, (a (.wire 10 55)).val, (a (.wire 10 63)).val⟩⟩,
      ⟨(a (.wire 10 71)).val, ⟨(a (.wire 10 79)).val, (a (.wire 11 7)).val, (a (.wire 11 15)).val, (a (.wire 11 23)).val⟩⟩,
      ⟨(a (.wire 11 31)).val, ⟨(a (.wire 11 39)).val, (a (.wire 11 47)).val, (a (.wire 11 55)).val, (a (.wire 11 63)).val⟩⟩,
      ⟨(a (.wire 11 71)).val, ⟨(a (.wire 11 79)).val, (a (.wire 12 7)).val, (a (.wire 12 15)).val, (a (.wire 12 23)).val⟩⟩,
      ⟨(a (.wire 12 31)).val, ⟨(a (.wire 12 39)).val, (a (.wire 12 47)).val, (a (.wire 12 55)).val, (a (.wire 12 63)).val⟩⟩],
    nullifiers := [⟨(a (.wire 12 71)).val, (a (.wire 12 79)).val, (a (.wire 13 7)).val, (a (.wire 13 15)).val⟩,
      ⟨(a (.wire 13 23)).val, (a (.wire 13 31)).val, (a (.wire 13 39)).val, (a (.wire 13 47)).val⟩,
      ⟨(a (.wire 13 55)).val, (a (.wire 13 63)).val, (a (.wire 13 71)).val, (a (.wire 13 79)).val⟩,
      ⟨(a (.wire 14 7)).val, (a (.wire 14 15)).val, (a (.wire 14 23)).val, (a (.wire 14 31)).val⟩] }

/-! ### The two-row scan -/

theorem bnot_zero : bnot (0 : ZMod p) = 1 := by simp [bnot]
theorem bor_zero_left (x : ZMod p) : bor 0 x = x := by simp [bor]
theorem band_one_right (x : ZMod p) : band x 1 = x := by simp [band]

/-- `scanRef` over two rows, as the circuit's two `select`s: the first `take` is `is_real_0`
    (`and(is_real_0, not(false))` folds), the second is `and(is_real_1, not(is_real_0))`. -/
theorem scanRef_two (t0 t1 : InnerPair p) (f : InnerPair p → ZMod p) :
    scanRef [t0, t1] f
      = bselect (band (bnot t1.1.isDummy) (bnot (bnot t0.1.isDummy))) (f t1)
          (bselect (bnot t0.1.isDummy) (f t0) 0) := by
  simp only [scanRef, List.map_cons, List.map_nil, List.foldl_cons, List.foldl_nil, scanStep,
    bor_zero_left, bnot_zero, band_one_right]

/-! ### Composition -/

theorem consts (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 19078) = 1 ∧ a (.virt 19079) = 0 ∧ a (.virt 19120) = 8 := by
  have hc := h.2.2
  simp only [publicBatchWrapper2, List.forall_mem_cons, List.not_mem_nil, false_implies,
    implies_true, and_true] at hc
  exact hc

/-- **The recorded public-batch wrapper satisfies `RPublicBatch`.** Every satisfying
    assignment decodes to an `RPublicBatch` instance on the two inner outputs, the
    aggregator address and the aggregated output; the oracle is irrelevant (the wrapper
    hashes nothing). -/
theorem sound (ro : RandomOracle) (hpg : WormholeSpec.goldilocks ≤ p)
    (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    RPublicBatch ro (inners a) (addr a) (out a) := by
  obtain ⟨kone, kzero, keight⟩ := consts a h
  have hf := publicBatchWrapper2_decode a h
  have e0_0 := hf.1.1
  have e0_1 := hf.1.2.1
  have e0_2 := hf.1.2.2.1
  have e0_3 := hf.1.2.2.2.1
  have a0a := hf.1.2.2.2.2.1
  have a0b := hf.1.2.2.2.2.2.1
  have d0 := hf.1.2.2.2.2.2.2.1
  have e1_0 := hf.1.2.2.2.2.2.2.2.1
  have e1_1 := hf.1.2.2.2.2.2.2.2.2.1
  have e1_2 := hf.1.2.2.2.2.2.2.2.2.2.1
  have e1_3 := hf.1.2.2.2.2.2.2.2.2.2.2.1
  have a1a := hf.1.2.2.2.2.2.2.2.2.2.2.2.1
  have a1b := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.1
  have d1 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have r0 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sb0_0 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sb0_1 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sb0_2 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sb0_3 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sn0 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sa0 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sf0 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have r1 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have nf := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have tk1 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sb1_0 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sb1_1 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sb1_2 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sb1_3 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have sn1 := hf.2.1.1
  have sa1 := hf.2.1.2.1
  have sf1 := hf.2.1.2.2.1
  have ca0 := hf.2.1.2.2.2.2.1
  have cao0 := hf.2.1.2.2.2.2.2.1
  have cak0 := hf.2.1.2.2.2.2.2.2.1
  have cf0 := hf.2.1.2.2.2.2.2.2.2.1
  have cfo0 := hf.2.1.2.2.2.2.2.2.2.2.1
  have cfk0 := hf.2.1.2.2.2.2.2.2.2.2.2.1
  have cb0_0 := hf.2.1.2.2.2.2.2.2.2.2.2.2.1
  have cb0_1 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.1
  have cb0_2 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cb0_3 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cba0 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cbb0 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cbc0 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cbo0 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cbk0 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have ca1 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cao1 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cak1 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cf1 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cfo1 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cfk1 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cb1_0 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cb1_1 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cb1_2 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cb1_3 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cba1 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cbb1 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cbc1 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have cbo1 := hf.2.2.1.1
  have cbk1 := hf.2.2.1.2.1
  have fs0 := hf.2.2.1.2.2.1
  have fs1 := hf.2.2.1.2.2.2.1
  have fs2 := hf.2.2.1.2.2.2.2.1
  have fs3 := hf.2.2.1.2.2.2.2.2.1
  have fs4 := hf.2.2.1.2.2.2.2.2.2.1
  have fs5 := hf.2.2.1.2.2.2.2.2.2.2.1
  have fs6 := hf.2.2.1.2.2.2.2.2.2.2.2.1
  have fs7 := hf.2.2.1.2.2.2.2.2.2.2.2.2.1
  have fs8 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.1
  have fs9 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.1
  have fs10 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fs11 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fs12 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fs13 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fs14 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fs15 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fs16 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fs17 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fs18 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fs19 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fs20 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fs21 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fs22 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fs23 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fs24 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fs25 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fs26 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fs27 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fs28 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fs29 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have fs30 := hf.2.2.2.1
  have fs31 := hf.2.2.2.2.1
  have fs32 := hf.2.2.2.2.2.1
  have fs33 := hf.2.2.2.2.2.2.1
  have fs34 := hf.2.2.2.2.2.2.2.1
  have fs35 := hf.2.2.2.2.2.2.2.2.1
  have fs36 := hf.2.2.2.2.2.2.2.2.2.1
  have fs37 := hf.2.2.2.2.2.2.2.2.2.2.1
  have fs38 := hf.2.2.2.2.2.2.2.2.2.2.2.1
  have fs39 := hf.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fn0 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fn1 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fn2 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fn3 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fn4 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fn5 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fn6 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fn7 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fn8 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fn9 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fn10 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fn11 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fn12 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fn13 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fn14 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fn15 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2

  rw [kzero] at e0_0 e0_1 e0_2 e0_3 e1_0 e1_1 e1_2 e1_3
  have hd0 : a (.wire 1 43) = (row a 0).isDummy := by rw [d0, a0a, a0b]; rfl
  have hd1 : a (.wire 2 7) = (row a 1).isDummy := by rw [d1, a1a, a1b]; rfl
  have hsel : ∀ x y z : ZMod p, bselect (a (.wire 2 11)) x (bselect (a (.wire 0 67)) y z)
      = bselect (band (bnot (row a 1).isDummy) (bnot (bnot (row a 0).isDummy))) x
          (bselect (bnot (row a 0).isDummy) y z) := by
    intro x y z
    rw [tk1, r1, nf, r0, hd0, hd1]
  have hblk : ∀ j, a (![Target.wire 3 31, .wire 3 39, .wire 3 47, .wire 3 55] j)
      = blockRef (rows a) j := by
    intro j
    simp only [blockRef, rows, scanRef_two]
    fin_cases j
    · show a (.wire 3 31) = _
      rw [sb1_0, sb0_0, kzero, hsel]; rfl
    · show a (.wire 3 39) = _
      rw [sb1_1, sb0_1, kzero, hsel]; rfl
    · show a (.wire 3 47) = _
      rw [sb1_2, sb0_2, kzero, hsel]; rfl
    · show a (.wire 3 55) = _
      rw [sb1_3, sb0_3, kzero, hsel]; rfl
  have hb0 : a (.wire 3 31) = blockRef (rows a) 0 := hblk 0
  have hb1 : a (.wire 3 39) = blockRef (rows a) 1 := hblk 1
  have hb2 : a (.wire 3 47) = blockRef (rows a) 2 := hblk 2
  have hb3 : a (.wire 3 55) = blockRef (rows a) 3 := hblk 3
  have hnumR : a (.wire 3 63) = scanRef (rows a) fun t => t.1.blockNumber := by
    rw [rows, scanRef_two, sn1, sn0, kzero, hsel]; rfl
  have hassetR : a (.wire 3 71) = scanRef (rows a) fun t => t.1.assetId := by
    rw [rows, scanRef_two, sa1, sa0, kzero, hsel]; rfl
  have hfeeR : a (.wire 3 79) = scanRef (rows a) fun t => t.1.fee := by
    rw [rows, scanRef_two, sf1, sf0, kzero, hsel]; rfl
  have hmem : ∀ t ∈ rows a, t = (row a 0, inner a 0) ∨ t = (row a 1, inner a 1) := by
    intro t ht
    simpa only [rows, List.mem_cons, List.not_mem_nil, or_false] using ht
  have hdec : ∀ (P : InnerPair p → Prop), P (row a 0, inner a 0) → P (row a 1, inner a 1) →
      ∀ t ∈ rows a, P t := by
    intro P h0 h1 t ht
    rcases hmem t ht with rfl | rfl
    · exact h0
    · exact h1
  refine public_batch_val ro (rows a) (k := 4) rfl ?_ (hdec _ rfl rfl) (hdec _ rfl rfl)
    (hdec _ rfl rfl) (hdec _ rfl rfl) (hdec _ rfl rfl) (hdec _ rfl rfl) ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
    ?_ ?_
  · -- dummy checks
    intro t ht
    rcases hmem t ht with rfl | rfl
    · intro j; fin_cases j <;> assumption
    · intro j; fin_cases j <;> assumption
  · -- header: block hash
    show (⟨(a (.wire 3 31)).val, (a (.wire 3 39)).val, (a (.wire 3 47)).val, (a (.wire 3 55)).val⟩ : Digest)
      = ⟨(blockRef (rows a) 0).val, (blockRef (rows a) 1).val, (blockRef (rows a) 2).val,
          (blockRef (rows a) 3).val⟩
    rw [hb0, hb1, hb2, hb3]
  · show (a (.wire 3 63)).val = _
    rw [hnumR]
  · show (a (.wire 3 71)).val = _
    rw [hassetR]
  · show (a (.wire 3 79)).val = _
    rw [hfeeR]
  · -- asset consistency
    intro t ht
    rcases hmem t ht with rfl | rfl
    · exact ⟨_, _, hassetR ▸ ca0, by rw [← hd0, ← cao0, cak0, kone]⟩
    · exact ⟨_, _, hassetR ▸ ca1, by rw [← hd1, ← cao1, cak1, kone]⟩
  · -- fee consistency
    intro t ht
    rcases hmem t ht with rfl | rfl
    · exact ⟨_, _, hfeeR ▸ cf0, by rw [← hd0, ← cfo0, cfk0, kone]⟩
    · exact ⟨_, _, hfeeR ▸ cf1, by rw [← hd1, ← cfo1, cfk1, kone]⟩
  · -- block consistency
    intro t ht
    rcases hmem t ht with rfl | rfl
    · refine ⟨![a (.virt 19100), a (.virt 19102), a (.virt 19104), a (.virt 19106)],
        ![a (.virt 19101), a (.virt 19103), a (.virt 19105), a (.virt 19107)], ?_, ?_⟩
      · intro j; rw [← hblk j]; fin_cases j <;> assumption
      · show bor (row a 0).isDummy (band (band (a (.virt 19100)) (a (.virt 19102)))
          (band (a (.virt 19104)) (a (.virt 19106)))) = 1
        rw [← hd0, ← cba0, ← cbb0, ← cbc0, ← cbo0, cbk0, kone]
    · refine ⟨![a (.virt 19112), a (.virt 19114), a (.virt 19116), a (.virt 19118)],
        ![a (.virt 19113), a (.virt 19115), a (.virt 19117), a (.virt 19119)], ?_, ?_⟩
      · intro j; rw [← hblk j]; fin_cases j <;> assumption
      · show bor (row a 1).isDummy (band (band (a (.virt 19112)) (a (.virt 19114)))
          (band (a (.virt 19116)) (a (.virt 19118)))) = 1
        rw [← hd1, ← cba1, ← cbb1, ← cbc1, ← cbo1, cbk1, kone]
  · -- forwarded exit slots
    show ([⟨(a (.wire 7 71)).val, ⟨(a (.wire 7 79)).val, (a (.wire 9 7)).val, (a (.wire 9 15)).val, (a (.wire 9 23)).val⟩⟩,
      ⟨(a (.wire 9 31)).val, ⟨(a (.wire 9 39)).val, (a (.wire 9 47)).val, (a (.wire 9 55)).val, (a (.wire 9 63)).val⟩⟩,
      ⟨(a (.wire 9 71)).val, ⟨(a (.wire 9 79)).val, (a (.wire 10 7)).val, (a (.wire 10 15)).val, (a (.wire 10 23)).val⟩⟩,
      ⟨(a (.wire 10 31)).val, ⟨(a (.wire 10 39)).val, (a (.wire 10 47)).val, (a (.wire 10 55)).val, (a (.wire 10 63)).val⟩⟩,
      ⟨(a (.wire 10 71)).val, ⟨(a (.wire 10 79)).val, (a (.wire 11 7)).val, (a (.wire 11 15)).val, (a (.wire 11 23)).val⟩⟩,
      ⟨(a (.wire 11 31)).val, ⟨(a (.wire 11 39)).val, (a (.wire 11 47)).val, (a (.wire 11 55)).val, (a (.wire 11 63)).val⟩⟩,
      ⟨(a (.wire 11 71)).val, ⟨(a (.wire 11 79)).val, (a (.wire 12 7)).val, (a (.wire 12 15)).val, (a (.wire 12 23)).val⟩⟩,
      ⟨(a (.wire 12 31)).val, ⟨(a (.wire 12 39)).val, (a (.wire 12 47)).val, (a (.wire 12 55)).val, (a (.wire 12 63)).val⟩⟩] : List ExitSlot) = _
    rw [fs0, fs1, fs2, fs3, fs4, fs5, fs6, fs7, fs8, fs9, fs10, fs11, fs12, fs13, fs14, fs15, fs16, fs17, fs18, fs19, fs20, fs21, fs22, fs23, fs24, fs25, fs26, fs27, fs28, fs29, fs30, fs31, fs32, fs33, fs34, fs35, fs36, fs37, fs38, fs39, kzero, hd0, hd1]
    rfl
  · -- forwarded nullifiers
    show ([⟨(a (.wire 12 71)).val, (a (.wire 12 79)).val, (a (.wire 13 7)).val, (a (.wire 13 15)).val⟩,
      ⟨(a (.wire 13 23)).val, (a (.wire 13 31)).val, (a (.wire 13 39)).val, (a (.wire 13 47)).val⟩,
      ⟨(a (.wire 13 55)).val, (a (.wire 13 63)).val, (a (.wire 13 71)).val, (a (.wire 13 79)).val⟩,
      ⟨(a (.wire 14 7)).val, (a (.wire 14 15)).val, (a (.wire 14 23)).val, (a (.wire 14 31)).val⟩] : List Digest) = _
    rw [fn0, fn1, fn2, fn3, fn4, fn5, fn6, fn7, fn8, fn9, fn10, fn11, fn12, fn13, fn14, fn15, kzero, hd0, hd1]
    rfl
  · intro t ht
    rcases hmem t ht with rfl | rfl <;> rfl
  · show (a (.virt 19120)).val = 2 * 4
    rw [keight]
    have h8 : (8 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
    rw [← Nat.cast_ofNat, ZMod.val_natCast_of_lt h8]

end Plonky2Bridge.PublicWrapper2

namespace Plonky2Bridge

open Plonky2Spec.Wiring
open Plonky2Spec.Generated (publicBatchWrapper2)
open WormholeSpec (RandomOracle RPublicBatch RPrivateBatch PrivateBatchProofAccepted
  private_batch_proof_sound RPublicBatch_totalExitSlots)

variable {p : ℕ} [Fact p.Prime]

/-- **The public-batch capstone on the recorded wiring.** `public_batch_end_to_end` with its
    decode hypotheses discharged by `PublicWrapper2.sound`: a satisfying assignment of the
    `n_inner = 2` public-batch wrapper whose recursion gadgets accepted both inner
    private-batch proofs (i) satisfies `RPublicBatch`, (ii) has the slot-count header equal
    to the sum of the inners' slot counts and (iii) attests every inner's `RPrivateBatch` —
    through `private_batch_proof_sound`, the only axiom. -/
theorem public_batch_end_to_end_wired (ro : RandomOracle) (hpg : WormholeSpec.goldilocks ≤ p)
    (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a)
    (hacc : ∀ o ∈ PublicWrapper2.inners a, PrivateBatchProofAccepted ro o) :
    RPublicBatch ro (PublicWrapper2.inners a) (PublicWrapper2.addr a) (PublicWrapper2.out a)
      ∧ (PublicWrapper2.out a).totalExitSlots
          = ((PublicWrapper2.inners a).map fun o => o.exitSlots.length).sum
      ∧ ∀ o ∈ PublicWrapper2.inners a, ∃ leaves us, RPrivateBatch ro leaves us o := by
  have hR := PublicWrapper2.sound ro hpg a h
  exact ⟨hR, RPublicBatch_totalExitSlots hR, fun o ho => private_batch_proof_sound ro o (hacc o ho)⟩

end Plonky2Bridge
