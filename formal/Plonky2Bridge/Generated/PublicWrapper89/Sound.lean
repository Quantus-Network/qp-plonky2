import Plonky2Bridge.Generated.PublicWrapper89.Limb0
import Plonky2Bridge.Generated.PublicWrapper89.Limb1
import Plonky2Bridge.Generated.PublicWrapper89.Limb2
import Plonky2Bridge.Generated.PublicWrapper89.Limb3
import Plonky2Bridge.Generated.PublicWrapper89.Scalar
import Plonky2Bridge.Generated.PublicWrapper89.Tree
import Plonky2Bridge.Generated.PublicWrapper89.Dispatch
import Plonky2Bridge.Generated.PublicWrapper89.FwdJoin
import Plonky2Bridge.Generated.PublicWrapper89.Fwd

namespace Plonky2Bridge.PublicWrapper89

open Plonky2Spec (bselect band bnot bor scanStep)
open Plonky2Spec.Wiring
open Plonky2Spec.Generated
open Plonky2Spec.Poseidon2 (St)
open WormholeSpec (Digest Felt PrivateBatchOutput PublicBatchOutput ExitSlot RandomOracle RPublicBatch
  RPrivateBatch RPublicBatch_totalExitSlots)

variable {p : ℕ} [Fact p.Prime]

set_option maxRecDepth 200000
set_option maxHeartbeats 0

set_option maxHeartbeats 20000000 in
/-- **The recorded public-batch wrapper satisfies `RPublicBatch`.** Every satisfying
    assignment decodes to an `RPublicBatch` instance on the 89 inner outputs, the
    aggregator address and the aggregated output; the oracle is irrelevant (the wrapper
    hashes nothing). -/
theorem sound (ro : RandomOracle) (hpg : WormholeSpec.goldilocks ≤ p)
    (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    RPublicBatch ro (inners a) (addr a) (out a) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j
    fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  have hb0 : a (.wire 169 63) = blockRef (rows a) 0 := hblk 0
  have hb1 : a (.wire 169 71) = blockRef (rows a) 1 := hblk 1
  have hb2 : a (.wire 169 79) = blockRef (rows a) 2 := hblk 2
  have hb3 : a (.wire 170 7) = blockRef (rows a) 3 := hblk 3
  have hsnR := scan_sn a h
  have hsaR := scan_sa a h
  have hsfR := scan_sf a h
  refine public_batch_val ro (rows a) (k := 32) rfl
    (dummyAll a h) (decBlock a) (decNum a) (decAsset a) (decFee a)
    (decSlots a) (decNulls a) ?_ ?_ ?_ ?_
    (assetAll a h) (feeAll a h) (blockAll a h)
    (exitFwd a h) (nullFwd a h) (shapeAll a) (countAll a h hpg) ?_
  · -- header: block hash
    show (⟨(a (.wire 169 63)).val, (a (.wire 169 71)).val, (a (.wire 169 79)).val, (a (.wire 170 7)).val⟩ : Digest)
      = ⟨(blockRef (rows a) 0).val, (blockRef (rows a) 1).val, (blockRef (rows a) 2).val,
          (blockRef (rows a) 3).val⟩
    rw [hb0, hb1, hb2, hb3]
  · show (a (.wire 170 15)).val = _
    rw [hsnR]
  · show (a (.wire 170 23)).val = _
    rw [hsaR]
  · show (a (.wire 170 31)).val = _
    rw [hsfR]
  · show (a (.virt 19587)).val = 89 * 32
    rw [ktot]
    have hc : (2848 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
    rw [← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


/-- **The public-batch capstone on the recorded wiring.** `public_batch_end_to_end` with its
    decode hypotheses discharged by `sound`: a satisfying assignment of the `n_inner = 89`
    public-batch wrapper whose recursion gadgets accepted every inner private-batch proof
    (i) satisfies `RPublicBatch`, (ii) has the slot-count header equal to the sum of the
    inners' slot counts and (iii) attests every inner's `RPrivateBatch` — through
    `proof_sound` on the good events `hgw` for the inner wrapper's proofs and `hgl` for the
    leaf proofs, and the wired wrapper bridges below it. -/
theorem end_to_end_wired (perm : St p → St p) (hpg : WormholeSpec.goldilocks ≤ p)
    (hgw : GoodEvents perm Wrapper8.tree) (hgl : GoodEvents perm LeafCircuit.tree)
    (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hacc : ∀ i : Fin 89, ProofAccepted perm Wrapper8.tree ((List.ofFn (innerPis i)).map a)) :
    RPublicBatch (spongeRO perm) (inners a) (addr a) (out a)
      ∧ (out a).totalExitSlots = ((inners a).map fun o => o.exitSlots.length).sum
      ∧ ∀ o ∈ inners a, ∃ leaves us, RPrivateBatch (spongeRO perm) leaves us o := by
  have hR := sound (spongeRO perm) hpg a h
  refine ⟨hR, RPublicBatch_totalExitSlots hR, ?_⟩
  intro o ho
  rw [show inners a = List.ofFn (inner a) from rfl] at ho
  obtain ⟨i, rfl⟩ := List.mem_ofFn.mp ho
  rw [inner_eq]
  exact Wrapper8.accepted_sound perm hpg hgw hgl (hacc i)
end Plonky2Bridge.PublicWrapper89
