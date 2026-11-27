import Plonky2Bridge.Generated.PublicWrapper86.Fwd

namespace Plonky2Bridge.PublicWrapper86

open Plonky2Spec (bselect band bnot bor scanStep)
open Plonky2Spec.Wiring
open Plonky2Spec.Generated
open Plonky2Spec.Poseidon2 (St)
open WormholeSpec (Digest Felt PrivateBatchOutput PublicBatchOutput ExitSlot RandomOracle RPublicBatch
  RPrivateBatch RPublicBatch_totalExitSlots)

variable {p : ℕ} [Fact p.Prime]

set_option maxRecDepth 200000
set_option maxHeartbeats 0

theorem exitFwd (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    (out a).exitSlots = (forwardedSlotsF (rows a)).map valSlot := by
  simp only [out, forwardedSlotsF, rows, List.map_cons, List.map_nil,
    List.flatten_cons, List.flatten_nil]
  simp only [exitChunk_eq_0 a h, exitChunk_eq_1 a h, exitChunk_eq_2 a h, exitChunk_eq_3 a h, exitChunk_eq_4 a h, exitChunk_eq_5 a h, exitChunk_eq_6 a h, exitChunk_eq_7 a h, exitChunk_eq_8 a h, exitChunk_eq_9 a h, exitChunk_eq_10 a h, exitChunk_eq_11 a h, exitChunk_eq_12 a h, exitChunk_eq_13 a h, exitChunk_eq_14 a h, exitChunk_eq_15 a h, exitChunk_eq_16 a h, exitChunk_eq_17 a h, exitChunk_eq_18 a h, exitChunk_eq_19 a h, exitChunk_eq_20 a h, exitChunk_eq_21 a h, exitChunk_eq_22 a h, exitChunk_eq_23 a h, exitChunk_eq_24 a h, exitChunk_eq_25 a h, exitChunk_eq_26 a h, exitChunk_eq_27 a h, exitChunk_eq_28 a h, exitChunk_eq_29 a h, exitChunk_eq_30 a h, exitChunk_eq_31 a h, exitChunk_eq_32 a h, exitChunk_eq_33 a h, exitChunk_eq_34 a h, exitChunk_eq_35 a h, exitChunk_eq_36 a h, exitChunk_eq_37 a h, exitChunk_eq_38 a h, exitChunk_eq_39 a h, exitChunk_eq_40 a h, exitChunk_eq_41 a h, exitChunk_eq_42 a h, exitChunk_eq_43 a h, exitChunk_eq_44 a h, exitChunk_eq_45 a h, exitChunk_eq_46 a h, exitChunk_eq_47 a h, exitChunk_eq_48 a h, exitChunk_eq_49 a h, exitChunk_eq_50 a h, exitChunk_eq_51 a h, exitChunk_eq_52 a h, exitChunk_eq_53 a h, exitChunk_eq_54 a h, exitChunk_eq_55 a h, exitChunk_eq_56 a h, exitChunk_eq_57 a h, exitChunk_eq_58 a h, exitChunk_eq_59 a h, exitChunk_eq_60 a h, exitChunk_eq_61 a h, exitChunk_eq_62 a h, exitChunk_eq_63 a h, exitChunk_eq_64 a h, exitChunk_eq_65 a h, exitChunk_eq_66 a h, exitChunk_eq_67 a h, exitChunk_eq_68 a h, exitChunk_eq_69 a h, exitChunk_eq_70 a h, exitChunk_eq_71 a h, exitChunk_eq_72 a h, exitChunk_eq_73 a h, exitChunk_eq_74 a h, exitChunk_eq_75 a h, exitChunk_eq_76 a h, exitChunk_eq_77 a h, exitChunk_eq_78 a h, exitChunk_eq_79 a h, exitChunk_eq_80 a h, exitChunk_eq_81 a h, exitChunk_eq_82 a h, exitChunk_eq_83 a h, exitChunk_eq_84 a h, exitChunk_eq_85 a h]
  rfl


theorem nullFwd (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    (out a).nullifiers = (forwardedNullsF (rows a)).map valDigest := by
  simp only [out, forwardedNullsF, rows, List.map_cons, List.map_nil,
    List.flatten_cons, List.flatten_nil]
  simp only [nullChunk_eq_0 a h, nullChunk_eq_1 a h, nullChunk_eq_2 a h, nullChunk_eq_3 a h, nullChunk_eq_4 a h, nullChunk_eq_5 a h, nullChunk_eq_6 a h, nullChunk_eq_7 a h, nullChunk_eq_8 a h, nullChunk_eq_9 a h, nullChunk_eq_10 a h, nullChunk_eq_11 a h, nullChunk_eq_12 a h, nullChunk_eq_13 a h, nullChunk_eq_14 a h, nullChunk_eq_15 a h, nullChunk_eq_16 a h, nullChunk_eq_17 a h, nullChunk_eq_18 a h, nullChunk_eq_19 a h, nullChunk_eq_20 a h, nullChunk_eq_21 a h, nullChunk_eq_22 a h, nullChunk_eq_23 a h, nullChunk_eq_24 a h, nullChunk_eq_25 a h, nullChunk_eq_26 a h, nullChunk_eq_27 a h, nullChunk_eq_28 a h, nullChunk_eq_29 a h, nullChunk_eq_30 a h, nullChunk_eq_31 a h, nullChunk_eq_32 a h, nullChunk_eq_33 a h, nullChunk_eq_34 a h, nullChunk_eq_35 a h, nullChunk_eq_36 a h, nullChunk_eq_37 a h, nullChunk_eq_38 a h, nullChunk_eq_39 a h, nullChunk_eq_40 a h, nullChunk_eq_41 a h, nullChunk_eq_42 a h, nullChunk_eq_43 a h, nullChunk_eq_44 a h, nullChunk_eq_45 a h, nullChunk_eq_46 a h, nullChunk_eq_47 a h, nullChunk_eq_48 a h, nullChunk_eq_49 a h, nullChunk_eq_50 a h, nullChunk_eq_51 a h, nullChunk_eq_52 a h, nullChunk_eq_53 a h, nullChunk_eq_54 a h, nullChunk_eq_55 a h, nullChunk_eq_56 a h, nullChunk_eq_57 a h, nullChunk_eq_58 a h, nullChunk_eq_59 a h, nullChunk_eq_60 a h, nullChunk_eq_61 a h, nullChunk_eq_62 a h, nullChunk_eq_63 a h, nullChunk_eq_64 a h, nullChunk_eq_65 a h, nullChunk_eq_66 a h, nullChunk_eq_67 a h, nullChunk_eq_68 a h, nullChunk_eq_69 a h, nullChunk_eq_70 a h, nullChunk_eq_71 a h, nullChunk_eq_72 a h, nullChunk_eq_73 a h, nullChunk_eq_74 a h, nullChunk_eq_75 a h, nullChunk_eq_76 a h, nullChunk_eq_77 a h, nullChunk_eq_78 a h, nullChunk_eq_79 a h, nullChunk_eq_80 a h, nullChunk_eq_81 a h, nullChunk_eq_82 a h, nullChunk_eq_83 a h, nullChunk_eq_84 a h, nullChunk_eq_85 a h]
  rfl

end Plonky2Bridge.PublicWrapper86
