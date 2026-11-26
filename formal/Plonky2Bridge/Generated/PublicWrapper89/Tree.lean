import Plonky2Bridge.Generated.PublicWrapper89.Read

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

/-! ### The wrapper as a recursion tree (PLAN.md Step 10) -/

/-- The wrapper with its recursion gadgets: `verify_proof` under the `8`-leaf private-batch
    verifier key on each `inner_pis_i`. -/
noncomputable def tree : Recursive p :=
  .node "public_batch_wrapper_n89" (publicBatchWrapper89 p)
    [(Wrapper8.tree, List.ofFn (innerPis 0)),
     (Wrapper8.tree, List.ofFn (innerPis 1)),
     (Wrapper8.tree, List.ofFn (innerPis 2)),
     (Wrapper8.tree, List.ofFn (innerPis 3)),
     (Wrapper8.tree, List.ofFn (innerPis 4)),
     (Wrapper8.tree, List.ofFn (innerPis 5)),
     (Wrapper8.tree, List.ofFn (innerPis 6)),
     (Wrapper8.tree, List.ofFn (innerPis 7)),
     (Wrapper8.tree, List.ofFn (innerPis 8)),
     (Wrapper8.tree, List.ofFn (innerPis 9)),
     (Wrapper8.tree, List.ofFn (innerPis 10)),
     (Wrapper8.tree, List.ofFn (innerPis 11)),
     (Wrapper8.tree, List.ofFn (innerPis 12)),
     (Wrapper8.tree, List.ofFn (innerPis 13)),
     (Wrapper8.tree, List.ofFn (innerPis 14)),
     (Wrapper8.tree, List.ofFn (innerPis 15)),
     (Wrapper8.tree, List.ofFn (innerPis 16)),
     (Wrapper8.tree, List.ofFn (innerPis 17)),
     (Wrapper8.tree, List.ofFn (innerPis 18)),
     (Wrapper8.tree, List.ofFn (innerPis 19)),
     (Wrapper8.tree, List.ofFn (innerPis 20)),
     (Wrapper8.tree, List.ofFn (innerPis 21)),
     (Wrapper8.tree, List.ofFn (innerPis 22)),
     (Wrapper8.tree, List.ofFn (innerPis 23)),
     (Wrapper8.tree, List.ofFn (innerPis 24)),
     (Wrapper8.tree, List.ofFn (innerPis 25)),
     (Wrapper8.tree, List.ofFn (innerPis 26)),
     (Wrapper8.tree, List.ofFn (innerPis 27)),
     (Wrapper8.tree, List.ofFn (innerPis 28)),
     (Wrapper8.tree, List.ofFn (innerPis 29)),
     (Wrapper8.tree, List.ofFn (innerPis 30)),
     (Wrapper8.tree, List.ofFn (innerPis 31)),
     (Wrapper8.tree, List.ofFn (innerPis 32)),
     (Wrapper8.tree, List.ofFn (innerPis 33)),
     (Wrapper8.tree, List.ofFn (innerPis 34)),
     (Wrapper8.tree, List.ofFn (innerPis 35)),
     (Wrapper8.tree, List.ofFn (innerPis 36)),
     (Wrapper8.tree, List.ofFn (innerPis 37)),
     (Wrapper8.tree, List.ofFn (innerPis 38)),
     (Wrapper8.tree, List.ofFn (innerPis 39)),
     (Wrapper8.tree, List.ofFn (innerPis 40)),
     (Wrapper8.tree, List.ofFn (innerPis 41)),
     (Wrapper8.tree, List.ofFn (innerPis 42)),
     (Wrapper8.tree, List.ofFn (innerPis 43)),
     (Wrapper8.tree, List.ofFn (innerPis 44)),
     (Wrapper8.tree, List.ofFn (innerPis 45)),
     (Wrapper8.tree, List.ofFn (innerPis 46)),
     (Wrapper8.tree, List.ofFn (innerPis 47)),
     (Wrapper8.tree, List.ofFn (innerPis 48)),
     (Wrapper8.tree, List.ofFn (innerPis 49)),
     (Wrapper8.tree, List.ofFn (innerPis 50)),
     (Wrapper8.tree, List.ofFn (innerPis 51)),
     (Wrapper8.tree, List.ofFn (innerPis 52)),
     (Wrapper8.tree, List.ofFn (innerPis 53)),
     (Wrapper8.tree, List.ofFn (innerPis 54)),
     (Wrapper8.tree, List.ofFn (innerPis 55)),
     (Wrapper8.tree, List.ofFn (innerPis 56)),
     (Wrapper8.tree, List.ofFn (innerPis 57)),
     (Wrapper8.tree, List.ofFn (innerPis 58)),
     (Wrapper8.tree, List.ofFn (innerPis 59)),
     (Wrapper8.tree, List.ofFn (innerPis 60)),
     (Wrapper8.tree, List.ofFn (innerPis 61)),
     (Wrapper8.tree, List.ofFn (innerPis 62)),
     (Wrapper8.tree, List.ofFn (innerPis 63)),
     (Wrapper8.tree, List.ofFn (innerPis 64)),
     (Wrapper8.tree, List.ofFn (innerPis 65)),
     (Wrapper8.tree, List.ofFn (innerPis 66)),
     (Wrapper8.tree, List.ofFn (innerPis 67)),
     (Wrapper8.tree, List.ofFn (innerPis 68)),
     (Wrapper8.tree, List.ofFn (innerPis 69)),
     (Wrapper8.tree, List.ofFn (innerPis 70)),
     (Wrapper8.tree, List.ofFn (innerPis 71)),
     (Wrapper8.tree, List.ofFn (innerPis 72)),
     (Wrapper8.tree, List.ofFn (innerPis 73)),
     (Wrapper8.tree, List.ofFn (innerPis 74)),
     (Wrapper8.tree, List.ofFn (innerPis 75)),
     (Wrapper8.tree, List.ofFn (innerPis 76)),
     (Wrapper8.tree, List.ofFn (innerPis 77)),
     (Wrapper8.tree, List.ofFn (innerPis 78)),
     (Wrapper8.tree, List.ofFn (innerPis 79)),
     (Wrapper8.tree, List.ofFn (innerPis 80)),
     (Wrapper8.tree, List.ofFn (innerPis 81)),
     (Wrapper8.tree, List.ofFn (innerPis 82)),
     (Wrapper8.tree, List.ofFn (innerPis 83)),
     (Wrapper8.tree, List.ofFn (innerPis 84)),
     (Wrapper8.tree, List.ofFn (innerPis 85)),
     (Wrapper8.tree, List.ofFn (innerPis 86)),
     (Wrapper8.tree, List.ofFn (innerPis 87)),
     (Wrapper8.tree, List.ofFn (innerPis 88))]


theorem pisEq_0 :
    List.ofFn (innerPis 0) = (publicBatchWrapper89.verifiers[0]'(by decide)).2 := by
  rfl


theorem pisEq_1 :
    List.ofFn (innerPis 1) = (publicBatchWrapper89.verifiers[1]'(by decide)).2 := by
  rfl


theorem pisEq_2 :
    List.ofFn (innerPis 2) = (publicBatchWrapper89.verifiers[2]'(by decide)).2 := by
  rfl


theorem pisEq_3 :
    List.ofFn (innerPis 3) = (publicBatchWrapper89.verifiers[3]'(by decide)).2 := by
  rfl


theorem pisEq_4 :
    List.ofFn (innerPis 4) = (publicBatchWrapper89.verifiers[4]'(by decide)).2 := by
  rfl


theorem pisEq_5 :
    List.ofFn (innerPis 5) = (publicBatchWrapper89.verifiers[5]'(by decide)).2 := by
  rfl


theorem pisEq_6 :
    List.ofFn (innerPis 6) = (publicBatchWrapper89.verifiers[6]'(by decide)).2 := by
  rfl


theorem pisEq_7 :
    List.ofFn (innerPis 7) = (publicBatchWrapper89.verifiers[7]'(by decide)).2 := by
  rfl


theorem pisEq_8 :
    List.ofFn (innerPis 8) = (publicBatchWrapper89.verifiers[8]'(by decide)).2 := by
  rfl


theorem pisEq_9 :
    List.ofFn (innerPis 9) = (publicBatchWrapper89.verifiers[9]'(by decide)).2 := by
  rfl


theorem pisEq_10 :
    List.ofFn (innerPis 10) = (publicBatchWrapper89.verifiers[10]'(by decide)).2 := by
  rfl


theorem pisEq_11 :
    List.ofFn (innerPis 11) = (publicBatchWrapper89.verifiers[11]'(by decide)).2 := by
  rfl


theorem pisEq_12 :
    List.ofFn (innerPis 12) = (publicBatchWrapper89.verifiers[12]'(by decide)).2 := by
  rfl


theorem pisEq_13 :
    List.ofFn (innerPis 13) = (publicBatchWrapper89.verifiers[13]'(by decide)).2 := by
  rfl


theorem pisEq_14 :
    List.ofFn (innerPis 14) = (publicBatchWrapper89.verifiers[14]'(by decide)).2 := by
  rfl


theorem pisEq_15 :
    List.ofFn (innerPis 15) = (publicBatchWrapper89.verifiers[15]'(by decide)).2 := by
  rfl


theorem pisEq_16 :
    List.ofFn (innerPis 16) = (publicBatchWrapper89.verifiers[16]'(by decide)).2 := by
  rfl


theorem pisEq_17 :
    List.ofFn (innerPis 17) = (publicBatchWrapper89.verifiers[17]'(by decide)).2 := by
  rfl


theorem pisEq_18 :
    List.ofFn (innerPis 18) = (publicBatchWrapper89.verifiers[18]'(by decide)).2 := by
  rfl


theorem pisEq_19 :
    List.ofFn (innerPis 19) = (publicBatchWrapper89.verifiers[19]'(by decide)).2 := by
  rfl


theorem pisEq_20 :
    List.ofFn (innerPis 20) = (publicBatchWrapper89.verifiers[20]'(by decide)).2 := by
  rfl


theorem pisEq_21 :
    List.ofFn (innerPis 21) = (publicBatchWrapper89.verifiers[21]'(by decide)).2 := by
  rfl


theorem pisEq_22 :
    List.ofFn (innerPis 22) = (publicBatchWrapper89.verifiers[22]'(by decide)).2 := by
  rfl


theorem pisEq_23 :
    List.ofFn (innerPis 23) = (publicBatchWrapper89.verifiers[23]'(by decide)).2 := by
  rfl


theorem pisEq_24 :
    List.ofFn (innerPis 24) = (publicBatchWrapper89.verifiers[24]'(by decide)).2 := by
  rfl


theorem pisEq_25 :
    List.ofFn (innerPis 25) = (publicBatchWrapper89.verifiers[25]'(by decide)).2 := by
  rfl


theorem pisEq_26 :
    List.ofFn (innerPis 26) = (publicBatchWrapper89.verifiers[26]'(by decide)).2 := by
  rfl


theorem pisEq_27 :
    List.ofFn (innerPis 27) = (publicBatchWrapper89.verifiers[27]'(by decide)).2 := by
  rfl


theorem pisEq_28 :
    List.ofFn (innerPis 28) = (publicBatchWrapper89.verifiers[28]'(by decide)).2 := by
  rfl


theorem pisEq_29 :
    List.ofFn (innerPis 29) = (publicBatchWrapper89.verifiers[29]'(by decide)).2 := by
  rfl


theorem pisEq_30 :
    List.ofFn (innerPis 30) = (publicBatchWrapper89.verifiers[30]'(by decide)).2 := by
  rfl


theorem pisEq_31 :
    List.ofFn (innerPis 31) = (publicBatchWrapper89.verifiers[31]'(by decide)).2 := by
  rfl


theorem pisEq_32 :
    List.ofFn (innerPis 32) = (publicBatchWrapper89.verifiers[32]'(by decide)).2 := by
  rfl


theorem pisEq_33 :
    List.ofFn (innerPis 33) = (publicBatchWrapper89.verifiers[33]'(by decide)).2 := by
  rfl


theorem pisEq_34 :
    List.ofFn (innerPis 34) = (publicBatchWrapper89.verifiers[34]'(by decide)).2 := by
  rfl


theorem pisEq_35 :
    List.ofFn (innerPis 35) = (publicBatchWrapper89.verifiers[35]'(by decide)).2 := by
  rfl


theorem pisEq_36 :
    List.ofFn (innerPis 36) = (publicBatchWrapper89.verifiers[36]'(by decide)).2 := by
  rfl


theorem pisEq_37 :
    List.ofFn (innerPis 37) = (publicBatchWrapper89.verifiers[37]'(by decide)).2 := by
  rfl


theorem pisEq_38 :
    List.ofFn (innerPis 38) = (publicBatchWrapper89.verifiers[38]'(by decide)).2 := by
  rfl


theorem pisEq_39 :
    List.ofFn (innerPis 39) = (publicBatchWrapper89.verifiers[39]'(by decide)).2 := by
  rfl


theorem pisEq_40 :
    List.ofFn (innerPis 40) = (publicBatchWrapper89.verifiers[40]'(by decide)).2 := by
  rfl


theorem pisEq_41 :
    List.ofFn (innerPis 41) = (publicBatchWrapper89.verifiers[41]'(by decide)).2 := by
  rfl


theorem pisEq_42 :
    List.ofFn (innerPis 42) = (publicBatchWrapper89.verifiers[42]'(by decide)).2 := by
  rfl


theorem pisEq_43 :
    List.ofFn (innerPis 43) = (publicBatchWrapper89.verifiers[43]'(by decide)).2 := by
  rfl


theorem pisEq_44 :
    List.ofFn (innerPis 44) = (publicBatchWrapper89.verifiers[44]'(by decide)).2 := by
  rfl


theorem pisEq_45 :
    List.ofFn (innerPis 45) = (publicBatchWrapper89.verifiers[45]'(by decide)).2 := by
  rfl


theorem pisEq_46 :
    List.ofFn (innerPis 46) = (publicBatchWrapper89.verifiers[46]'(by decide)).2 := by
  rfl


theorem pisEq_47 :
    List.ofFn (innerPis 47) = (publicBatchWrapper89.verifiers[47]'(by decide)).2 := by
  rfl


theorem pisEq_48 :
    List.ofFn (innerPis 48) = (publicBatchWrapper89.verifiers[48]'(by decide)).2 := by
  rfl


theorem pisEq_49 :
    List.ofFn (innerPis 49) = (publicBatchWrapper89.verifiers[49]'(by decide)).2 := by
  rfl


theorem pisEq_50 :
    List.ofFn (innerPis 50) = (publicBatchWrapper89.verifiers[50]'(by decide)).2 := by
  rfl


theorem pisEq_51 :
    List.ofFn (innerPis 51) = (publicBatchWrapper89.verifiers[51]'(by decide)).2 := by
  rfl


theorem pisEq_52 :
    List.ofFn (innerPis 52) = (publicBatchWrapper89.verifiers[52]'(by decide)).2 := by
  rfl


theorem pisEq_53 :
    List.ofFn (innerPis 53) = (publicBatchWrapper89.verifiers[53]'(by decide)).2 := by
  rfl


theorem pisEq_54 :
    List.ofFn (innerPis 54) = (publicBatchWrapper89.verifiers[54]'(by decide)).2 := by
  rfl


theorem pisEq_55 :
    List.ofFn (innerPis 55) = (publicBatchWrapper89.verifiers[55]'(by decide)).2 := by
  rfl


theorem pisEq_56 :
    List.ofFn (innerPis 56) = (publicBatchWrapper89.verifiers[56]'(by decide)).2 := by
  rfl


theorem pisEq_57 :
    List.ofFn (innerPis 57) = (publicBatchWrapper89.verifiers[57]'(by decide)).2 := by
  rfl


theorem pisEq_58 :
    List.ofFn (innerPis 58) = (publicBatchWrapper89.verifiers[58]'(by decide)).2 := by
  rfl


theorem pisEq_59 :
    List.ofFn (innerPis 59) = (publicBatchWrapper89.verifiers[59]'(by decide)).2 := by
  rfl


theorem pisEq_60 :
    List.ofFn (innerPis 60) = (publicBatchWrapper89.verifiers[60]'(by decide)).2 := by
  rfl


theorem pisEq_61 :
    List.ofFn (innerPis 61) = (publicBatchWrapper89.verifiers[61]'(by decide)).2 := by
  rfl


theorem pisEq_62 :
    List.ofFn (innerPis 62) = (publicBatchWrapper89.verifiers[62]'(by decide)).2 := by
  rfl


theorem pisEq_63 :
    List.ofFn (innerPis 63) = (publicBatchWrapper89.verifiers[63]'(by decide)).2 := by
  rfl


theorem pisEq_64 :
    List.ofFn (innerPis 64) = (publicBatchWrapper89.verifiers[64]'(by decide)).2 := by
  rfl


theorem pisEq_65 :
    List.ofFn (innerPis 65) = (publicBatchWrapper89.verifiers[65]'(by decide)).2 := by
  rfl


theorem pisEq_66 :
    List.ofFn (innerPis 66) = (publicBatchWrapper89.verifiers[66]'(by decide)).2 := by
  rfl


theorem pisEq_67 :
    List.ofFn (innerPis 67) = (publicBatchWrapper89.verifiers[67]'(by decide)).2 := by
  rfl


theorem pisEq_68 :
    List.ofFn (innerPis 68) = (publicBatchWrapper89.verifiers[68]'(by decide)).2 := by
  rfl


theorem pisEq_69 :
    List.ofFn (innerPis 69) = (publicBatchWrapper89.verifiers[69]'(by decide)).2 := by
  rfl


theorem pisEq_70 :
    List.ofFn (innerPis 70) = (publicBatchWrapper89.verifiers[70]'(by decide)).2 := by
  rfl


theorem pisEq_71 :
    List.ofFn (innerPis 71) = (publicBatchWrapper89.verifiers[71]'(by decide)).2 := by
  rfl


theorem pisEq_72 :
    List.ofFn (innerPis 72) = (publicBatchWrapper89.verifiers[72]'(by decide)).2 := by
  rfl


theorem pisEq_73 :
    List.ofFn (innerPis 73) = (publicBatchWrapper89.verifiers[73]'(by decide)).2 := by
  rfl


theorem pisEq_74 :
    List.ofFn (innerPis 74) = (publicBatchWrapper89.verifiers[74]'(by decide)).2 := by
  rfl


theorem pisEq_75 :
    List.ofFn (innerPis 75) = (publicBatchWrapper89.verifiers[75]'(by decide)).2 := by
  rfl


theorem pisEq_76 :
    List.ofFn (innerPis 76) = (publicBatchWrapper89.verifiers[76]'(by decide)).2 := by
  rfl


theorem pisEq_77 :
    List.ofFn (innerPis 77) = (publicBatchWrapper89.verifiers[77]'(by decide)).2 := by
  rfl


theorem pisEq_78 :
    List.ofFn (innerPis 78) = (publicBatchWrapper89.verifiers[78]'(by decide)).2 := by
  rfl


theorem pisEq_79 :
    List.ofFn (innerPis 79) = (publicBatchWrapper89.verifiers[79]'(by decide)).2 := by
  rfl


theorem pisEq_80 :
    List.ofFn (innerPis 80) = (publicBatchWrapper89.verifiers[80]'(by decide)).2 := by
  rfl


theorem pisEq_81 :
    List.ofFn (innerPis 81) = (publicBatchWrapper89.verifiers[81]'(by decide)).2 := by
  rfl


theorem pisEq_82 :
    List.ofFn (innerPis 82) = (publicBatchWrapper89.verifiers[82]'(by decide)).2 := by
  rfl


theorem pisEq_83 :
    List.ofFn (innerPis 83) = (publicBatchWrapper89.verifiers[83]'(by decide)).2 := by
  rfl


theorem pisEq_84 :
    List.ofFn (innerPis 84) = (publicBatchWrapper89.verifiers[84]'(by decide)).2 := by
  rfl


theorem pisEq_85 :
    List.ofFn (innerPis 85) = (publicBatchWrapper89.verifiers[85]'(by decide)).2 := by
  rfl


theorem pisEq_86 :
    List.ofFn (innerPis 86) = (publicBatchWrapper89.verifiers[86]'(by decide)).2 := by
  rfl


theorem pisEq_87 :
    List.ofFn (innerPis 87) = (publicBatchWrapper89.verifiers[87]'(by decide)).2 := by
  rfl


theorem pisEq_88 :
    List.ofFn (innerPis 88) = (publicBatchWrapper89.verifiers[88]'(by decide)).2 := by
  rfl


/-- The tree's gadgets are the recorded ones. -/
theorem tree_verifiers : (tree (p := p)).verifiers = publicBatchWrapper89.verifiers := by
  have htag : Wrapper8.tree.tag (p := p) = "private_batch_wrapper_n8" := by
    simp only [Recursive.tag, Wrapper8.tree]
  rw [Recursive.verifiers, tree, Recursive.children]
  rw [List.map_cons, htag, pisEq_0]
  rw [List.map_cons, htag, pisEq_1]
  rw [List.map_cons, htag, pisEq_2]
  rw [List.map_cons, htag, pisEq_3]
  rw [List.map_cons, htag, pisEq_4]
  rw [List.map_cons, htag, pisEq_5]
  rw [List.map_cons, htag, pisEq_6]
  rw [List.map_cons, htag, pisEq_7]
  rw [List.map_cons, htag, pisEq_8]
  rw [List.map_cons, htag, pisEq_9]
  rw [List.map_cons, htag, pisEq_10]
  rw [List.map_cons, htag, pisEq_11]
  rw [List.map_cons, htag, pisEq_12]
  rw [List.map_cons, htag, pisEq_13]
  rw [List.map_cons, htag, pisEq_14]
  rw [List.map_cons, htag, pisEq_15]
  rw [List.map_cons, htag, pisEq_16]
  rw [List.map_cons, htag, pisEq_17]
  rw [List.map_cons, htag, pisEq_18]
  rw [List.map_cons, htag, pisEq_19]
  rw [List.map_cons, htag, pisEq_20]
  rw [List.map_cons, htag, pisEq_21]
  rw [List.map_cons, htag, pisEq_22]
  rw [List.map_cons, htag, pisEq_23]
  rw [List.map_cons, htag, pisEq_24]
  rw [List.map_cons, htag, pisEq_25]
  rw [List.map_cons, htag, pisEq_26]
  rw [List.map_cons, htag, pisEq_27]
  rw [List.map_cons, htag, pisEq_28]
  rw [List.map_cons, htag, pisEq_29]
  rw [List.map_cons, htag, pisEq_30]
  rw [List.map_cons, htag, pisEq_31]
  rw [List.map_cons, htag, pisEq_32]
  rw [List.map_cons, htag, pisEq_33]
  rw [List.map_cons, htag, pisEq_34]
  rw [List.map_cons, htag, pisEq_35]
  rw [List.map_cons, htag, pisEq_36]
  rw [List.map_cons, htag, pisEq_37]
  rw [List.map_cons, htag, pisEq_38]
  rw [List.map_cons, htag, pisEq_39]
  rw [List.map_cons, htag, pisEq_40]
  rw [List.map_cons, htag, pisEq_41]
  rw [List.map_cons, htag, pisEq_42]
  rw [List.map_cons, htag, pisEq_43]
  rw [List.map_cons, htag, pisEq_44]
  rw [List.map_cons, htag, pisEq_45]
  rw [List.map_cons, htag, pisEq_46]
  rw [List.map_cons, htag, pisEq_47]
  rw [List.map_cons, htag, pisEq_48]
  rw [List.map_cons, htag, pisEq_49]
  rw [List.map_cons, htag, pisEq_50]
  rw [List.map_cons, htag, pisEq_51]
  rw [List.map_cons, htag, pisEq_52]
  rw [List.map_cons, htag, pisEq_53]
  rw [List.map_cons, htag, pisEq_54]
  rw [List.map_cons, htag, pisEq_55]
  rw [List.map_cons, htag, pisEq_56]
  rw [List.map_cons, htag, pisEq_57]
  rw [List.map_cons, htag, pisEq_58]
  rw [List.map_cons, htag, pisEq_59]
  rw [List.map_cons, htag, pisEq_60]
  rw [List.map_cons, htag, pisEq_61]
  rw [List.map_cons, htag, pisEq_62]
  rw [List.map_cons, htag, pisEq_63]
  rw [List.map_cons, htag, pisEq_64]
  rw [List.map_cons, htag, pisEq_65]
  rw [List.map_cons, htag, pisEq_66]
  rw [List.map_cons, htag, pisEq_67]
  rw [List.map_cons, htag, pisEq_68]
  rw [List.map_cons, htag, pisEq_69]
  rw [List.map_cons, htag, pisEq_70]
  rw [List.map_cons, htag, pisEq_71]
  rw [List.map_cons, htag, pisEq_72]
  rw [List.map_cons, htag, pisEq_73]
  rw [List.map_cons, htag, pisEq_74]
  rw [List.map_cons, htag, pisEq_75]
  rw [List.map_cons, htag, pisEq_76]
  rw [List.map_cons, htag, pisEq_77]
  rw [List.map_cons, htag, pisEq_78]
  rw [List.map_cons, htag, pisEq_79]
  rw [List.map_cons, htag, pisEq_80]
  rw [List.map_cons, htag, pisEq_81]
  rw [List.map_cons, htag, pisEq_82]
  rw [List.map_cons, htag, pisEq_83]
  rw [List.map_cons, htag, pisEq_84]
  rw [List.map_cons, htag, pisEq_85]
  rw [List.map_cons, htag, pisEq_86]
  rw [List.map_cons, htag, pisEq_87]
  rw [List.map_cons, htag, pisEq_88]
  rw [List.map_nil]
  rfl


omit [Fact p.Prime] in
set_option maxRecDepth 8192 in
/-- Inner `i`'s public inputs, as the wrapper reads them, are the private-batch wrapper's
    `outOf`. -/
theorem inner_eq (a : Assignment p) (i : Fin 89) :
    inner a i = Wrapper8.outOf (a ∘ innerPis i) := by
  fin_cases i <;> rfl

end Plonky2Bridge.PublicWrapper89
