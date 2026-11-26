import Plonky2Bridge.Generated.PublicWrapper86.Rel

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

theorem dummyAll (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ∀ t ∈ rows a, DummyCheck t.1 := by
  intro t ht
  rw [show rows a = List.ofFn (fun i => (row a i, inner a i)) from rfl] at ht
  obtain ⟨i, rfl⟩ := List.mem_ofFn.mp ht
  fin_cases i
  · exact dummyAt_0 a h
  · exact dummyAt_1 a h
  · exact dummyAt_2 a h
  · exact dummyAt_3 a h
  · exact dummyAt_4 a h
  · exact dummyAt_5 a h
  · exact dummyAt_6 a h
  · exact dummyAt_7 a h
  · exact dummyAt_8 a h
  · exact dummyAt_9 a h
  · exact dummyAt_10 a h
  · exact dummyAt_11 a h
  · exact dummyAt_12 a h
  · exact dummyAt_13 a h
  · exact dummyAt_14 a h
  · exact dummyAt_15 a h
  · exact dummyAt_16 a h
  · exact dummyAt_17 a h
  · exact dummyAt_18 a h
  · exact dummyAt_19 a h
  · exact dummyAt_20 a h
  · exact dummyAt_21 a h
  · exact dummyAt_22 a h
  · exact dummyAt_23 a h
  · exact dummyAt_24 a h
  · exact dummyAt_25 a h
  · exact dummyAt_26 a h
  · exact dummyAt_27 a h
  · exact dummyAt_28 a h
  · exact dummyAt_29 a h
  · exact dummyAt_30 a h
  · exact dummyAt_31 a h
  · exact dummyAt_32 a h
  · exact dummyAt_33 a h
  · exact dummyAt_34 a h
  · exact dummyAt_35 a h
  · exact dummyAt_36 a h
  · exact dummyAt_37 a h
  · exact dummyAt_38 a h
  · exact dummyAt_39 a h
  · exact dummyAt_40 a h
  · exact dummyAt_41 a h
  · exact dummyAt_42 a h
  · exact dummyAt_43 a h
  · exact dummyAt_44 a h
  · exact dummyAt_45 a h
  · exact dummyAt_46 a h
  · exact dummyAt_47 a h
  · exact dummyAt_48 a h
  · exact dummyAt_49 a h
  · exact dummyAt_50 a h
  · exact dummyAt_51 a h
  · exact dummyAt_52 a h
  · exact dummyAt_53 a h
  · exact dummyAt_54 a h
  · exact dummyAt_55 a h
  · exact dummyAt_56 a h
  · exact dummyAt_57 a h
  · exact dummyAt_58 a h
  · exact dummyAt_59 a h
  · exact dummyAt_60 a h
  · exact dummyAt_61 a h
  · exact dummyAt_62 a h
  · exact dummyAt_63 a h
  · exact dummyAt_64 a h
  · exact dummyAt_65 a h
  · exact dummyAt_66 a h
  · exact dummyAt_67 a h
  · exact dummyAt_68 a h
  · exact dummyAt_69 a h
  · exact dummyAt_70 a h
  · exact dummyAt_71 a h
  · exact dummyAt_72 a h
  · exact dummyAt_73 a h
  · exact dummyAt_74 a h
  · exact dummyAt_75 a h
  · exact dummyAt_76 a h
  · exact dummyAt_77 a h
  · exact dummyAt_78 a h
  · exact dummyAt_79 a h
  · exact dummyAt_80 a h
  · exact dummyAt_81 a h
  · exact dummyAt_82 a h
  · exact dummyAt_83 a h
  · exact dummyAt_84 a h
  · exact dummyAt_85 a h


theorem assetAll (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ∀ t ∈ rows a, ConsistencyCheck t.1.isDummy t.1.assetId (scanRef (rows a) fun t => t.1.assetId) := by
  intro t ht
  rw [show rows a = List.ofFn (fun i => (row a i, inner a i)) from rfl] at ht
  obtain ⟨i, rfl⟩ := List.mem_ofFn.mp ht
  fin_cases i
  · exact assetAt_0 a h
  · exact assetAt_1 a h
  · exact assetAt_2 a h
  · exact assetAt_3 a h
  · exact assetAt_4 a h
  · exact assetAt_5 a h
  · exact assetAt_6 a h
  · exact assetAt_7 a h
  · exact assetAt_8 a h
  · exact assetAt_9 a h
  · exact assetAt_10 a h
  · exact assetAt_11 a h
  · exact assetAt_12 a h
  · exact assetAt_13 a h
  · exact assetAt_14 a h
  · exact assetAt_15 a h
  · exact assetAt_16 a h
  · exact assetAt_17 a h
  · exact assetAt_18 a h
  · exact assetAt_19 a h
  · exact assetAt_20 a h
  · exact assetAt_21 a h
  · exact assetAt_22 a h
  · exact assetAt_23 a h
  · exact assetAt_24 a h
  · exact assetAt_25 a h
  · exact assetAt_26 a h
  · exact assetAt_27 a h
  · exact assetAt_28 a h
  · exact assetAt_29 a h
  · exact assetAt_30 a h
  · exact assetAt_31 a h
  · exact assetAt_32 a h
  · exact assetAt_33 a h
  · exact assetAt_34 a h
  · exact assetAt_35 a h
  · exact assetAt_36 a h
  · exact assetAt_37 a h
  · exact assetAt_38 a h
  · exact assetAt_39 a h
  · exact assetAt_40 a h
  · exact assetAt_41 a h
  · exact assetAt_42 a h
  · exact assetAt_43 a h
  · exact assetAt_44 a h
  · exact assetAt_45 a h
  · exact assetAt_46 a h
  · exact assetAt_47 a h
  · exact assetAt_48 a h
  · exact assetAt_49 a h
  · exact assetAt_50 a h
  · exact assetAt_51 a h
  · exact assetAt_52 a h
  · exact assetAt_53 a h
  · exact assetAt_54 a h
  · exact assetAt_55 a h
  · exact assetAt_56 a h
  · exact assetAt_57 a h
  · exact assetAt_58 a h
  · exact assetAt_59 a h
  · exact assetAt_60 a h
  · exact assetAt_61 a h
  · exact assetAt_62 a h
  · exact assetAt_63 a h
  · exact assetAt_64 a h
  · exact assetAt_65 a h
  · exact assetAt_66 a h
  · exact assetAt_67 a h
  · exact assetAt_68 a h
  · exact assetAt_69 a h
  · exact assetAt_70 a h
  · exact assetAt_71 a h
  · exact assetAt_72 a h
  · exact assetAt_73 a h
  · exact assetAt_74 a h
  · exact assetAt_75 a h
  · exact assetAt_76 a h
  · exact assetAt_77 a h
  · exact assetAt_78 a h
  · exact assetAt_79 a h
  · exact assetAt_80 a h
  · exact assetAt_81 a h
  · exact assetAt_82 a h
  · exact assetAt_83 a h
  · exact assetAt_84 a h
  · exact assetAt_85 a h


theorem feeAll (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ∀ t ∈ rows a, ConsistencyCheck t.1.isDummy t.1.fee (scanRef (rows a) fun t => t.1.fee) := by
  intro t ht
  rw [show rows a = List.ofFn (fun i => (row a i, inner a i)) from rfl] at ht
  obtain ⟨i, rfl⟩ := List.mem_ofFn.mp ht
  fin_cases i
  · exact feeAt_0 a h
  · exact feeAt_1 a h
  · exact feeAt_2 a h
  · exact feeAt_3 a h
  · exact feeAt_4 a h
  · exact feeAt_5 a h
  · exact feeAt_6 a h
  · exact feeAt_7 a h
  · exact feeAt_8 a h
  · exact feeAt_9 a h
  · exact feeAt_10 a h
  · exact feeAt_11 a h
  · exact feeAt_12 a h
  · exact feeAt_13 a h
  · exact feeAt_14 a h
  · exact feeAt_15 a h
  · exact feeAt_16 a h
  · exact feeAt_17 a h
  · exact feeAt_18 a h
  · exact feeAt_19 a h
  · exact feeAt_20 a h
  · exact feeAt_21 a h
  · exact feeAt_22 a h
  · exact feeAt_23 a h
  · exact feeAt_24 a h
  · exact feeAt_25 a h
  · exact feeAt_26 a h
  · exact feeAt_27 a h
  · exact feeAt_28 a h
  · exact feeAt_29 a h
  · exact feeAt_30 a h
  · exact feeAt_31 a h
  · exact feeAt_32 a h
  · exact feeAt_33 a h
  · exact feeAt_34 a h
  · exact feeAt_35 a h
  · exact feeAt_36 a h
  · exact feeAt_37 a h
  · exact feeAt_38 a h
  · exact feeAt_39 a h
  · exact feeAt_40 a h
  · exact feeAt_41 a h
  · exact feeAt_42 a h
  · exact feeAt_43 a h
  · exact feeAt_44 a h
  · exact feeAt_45 a h
  · exact feeAt_46 a h
  · exact feeAt_47 a h
  · exact feeAt_48 a h
  · exact feeAt_49 a h
  · exact feeAt_50 a h
  · exact feeAt_51 a h
  · exact feeAt_52 a h
  · exact feeAt_53 a h
  · exact feeAt_54 a h
  · exact feeAt_55 a h
  · exact feeAt_56 a h
  · exact feeAt_57 a h
  · exact feeAt_58 a h
  · exact feeAt_59 a h
  · exact feeAt_60 a h
  · exact feeAt_61 a h
  · exact feeAt_62 a h
  · exact feeAt_63 a h
  · exact feeAt_64 a h
  · exact feeAt_65 a h
  · exact feeAt_66 a h
  · exact feeAt_67 a h
  · exact feeAt_68 a h
  · exact feeAt_69 a h
  · exact feeAt_70 a h
  · exact feeAt_71 a h
  · exact feeAt_72 a h
  · exact feeAt_73 a h
  · exact feeAt_74 a h
  · exact feeAt_75 a h
  · exact feeAt_76 a h
  · exact feeAt_77 a h
  · exact feeAt_78 a h
  · exact feeAt_79 a h
  · exact feeAt_80 a h
  · exact feeAt_81 a h
  · exact feeAt_82 a h
  · exact feeAt_83 a h
  · exact feeAt_84 a h
  · exact feeAt_85 a h


theorem blockAll (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ∀ t ∈ rows a, DigestConsistencyCheck t.1.isDummy t.1.blockHash (blockRef (rows a)) := by
  intro t ht
  rw [show rows a = List.ofFn (fun i => (row a i, inner a i)) from rfl] at ht
  obtain ⟨i, rfl⟩ := List.mem_ofFn.mp ht
  fin_cases i
  · exact blockAt_0 a h
  · exact blockAt_1 a h
  · exact blockAt_2 a h
  · exact blockAt_3 a h
  · exact blockAt_4 a h
  · exact blockAt_5 a h
  · exact blockAt_6 a h
  · exact blockAt_7 a h
  · exact blockAt_8 a h
  · exact blockAt_9 a h
  · exact blockAt_10 a h
  · exact blockAt_11 a h
  · exact blockAt_12 a h
  · exact blockAt_13 a h
  · exact blockAt_14 a h
  · exact blockAt_15 a h
  · exact blockAt_16 a h
  · exact blockAt_17 a h
  · exact blockAt_18 a h
  · exact blockAt_19 a h
  · exact blockAt_20 a h
  · exact blockAt_21 a h
  · exact blockAt_22 a h
  · exact blockAt_23 a h
  · exact blockAt_24 a h
  · exact blockAt_25 a h
  · exact blockAt_26 a h
  · exact blockAt_27 a h
  · exact blockAt_28 a h
  · exact blockAt_29 a h
  · exact blockAt_30 a h
  · exact blockAt_31 a h
  · exact blockAt_32 a h
  · exact blockAt_33 a h
  · exact blockAt_34 a h
  · exact blockAt_35 a h
  · exact blockAt_36 a h
  · exact blockAt_37 a h
  · exact blockAt_38 a h
  · exact blockAt_39 a h
  · exact blockAt_40 a h
  · exact blockAt_41 a h
  · exact blockAt_42 a h
  · exact blockAt_43 a h
  · exact blockAt_44 a h
  · exact blockAt_45 a h
  · exact blockAt_46 a h
  · exact blockAt_47 a h
  · exact blockAt_48 a h
  · exact blockAt_49 a h
  · exact blockAt_50 a h
  · exact blockAt_51 a h
  · exact blockAt_52 a h
  · exact blockAt_53 a h
  · exact blockAt_54 a h
  · exact blockAt_55 a h
  · exact blockAt_56 a h
  · exact blockAt_57 a h
  · exact blockAt_58 a h
  · exact blockAt_59 a h
  · exact blockAt_60 a h
  · exact blockAt_61 a h
  · exact blockAt_62 a h
  · exact blockAt_63 a h
  · exact blockAt_64 a h
  · exact blockAt_65 a h
  · exact blockAt_66 a h
  · exact blockAt_67 a h
  · exact blockAt_68 a h
  · exact blockAt_69 a h
  · exact blockAt_70 a h
  · exact blockAt_71 a h
  · exact blockAt_72 a h
  · exact blockAt_73 a h
  · exact blockAt_74 a h
  · exact blockAt_75 a h
  · exact blockAt_76 a h
  · exact blockAt_77 a h
  · exact blockAt_78 a h
  · exact blockAt_79 a h
  · exact blockAt_80 a h
  · exact blockAt_81 a h
  · exact blockAt_82 a h
  · exact blockAt_83 a h
  · exact blockAt_84 a h
  · exact blockAt_85 a h


theorem countAll (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) (hpg : WormholeSpec.goldilocks ≤ p) :
    ∀ t ∈ rows a, t.2.numExitSlots = 32 := by
  intro t ht
  rw [show rows a = List.ofFn (fun i => (row a i, inner a i)) from rfl] at ht
  obtain ⟨i, rfl⟩ := List.mem_ofFn.mp ht
  fin_cases i
  · exact countAt_0 a h hpg
  · exact countAt_1 a h hpg
  · exact countAt_2 a h hpg
  · exact countAt_3 a h hpg
  · exact countAt_4 a h hpg
  · exact countAt_5 a h hpg
  · exact countAt_6 a h hpg
  · exact countAt_7 a h hpg
  · exact countAt_8 a h hpg
  · exact countAt_9 a h hpg
  · exact countAt_10 a h hpg
  · exact countAt_11 a h hpg
  · exact countAt_12 a h hpg
  · exact countAt_13 a h hpg
  · exact countAt_14 a h hpg
  · exact countAt_15 a h hpg
  · exact countAt_16 a h hpg
  · exact countAt_17 a h hpg
  · exact countAt_18 a h hpg
  · exact countAt_19 a h hpg
  · exact countAt_20 a h hpg
  · exact countAt_21 a h hpg
  · exact countAt_22 a h hpg
  · exact countAt_23 a h hpg
  · exact countAt_24 a h hpg
  · exact countAt_25 a h hpg
  · exact countAt_26 a h hpg
  · exact countAt_27 a h hpg
  · exact countAt_28 a h hpg
  · exact countAt_29 a h hpg
  · exact countAt_30 a h hpg
  · exact countAt_31 a h hpg
  · exact countAt_32 a h hpg
  · exact countAt_33 a h hpg
  · exact countAt_34 a h hpg
  · exact countAt_35 a h hpg
  · exact countAt_36 a h hpg
  · exact countAt_37 a h hpg
  · exact countAt_38 a h hpg
  · exact countAt_39 a h hpg
  · exact countAt_40 a h hpg
  · exact countAt_41 a h hpg
  · exact countAt_42 a h hpg
  · exact countAt_43 a h hpg
  · exact countAt_44 a h hpg
  · exact countAt_45 a h hpg
  · exact countAt_46 a h hpg
  · exact countAt_47 a h hpg
  · exact countAt_48 a h hpg
  · exact countAt_49 a h hpg
  · exact countAt_50 a h hpg
  · exact countAt_51 a h hpg
  · exact countAt_52 a h hpg
  · exact countAt_53 a h hpg
  · exact countAt_54 a h hpg
  · exact countAt_55 a h hpg
  · exact countAt_56 a h hpg
  · exact countAt_57 a h hpg
  · exact countAt_58 a h hpg
  · exact countAt_59 a h hpg
  · exact countAt_60 a h hpg
  · exact countAt_61 a h hpg
  · exact countAt_62 a h hpg
  · exact countAt_63 a h hpg
  · exact countAt_64 a h hpg
  · exact countAt_65 a h hpg
  · exact countAt_66 a h hpg
  · exact countAt_67 a h hpg
  · exact countAt_68 a h hpg
  · exact countAt_69 a h hpg
  · exact countAt_70 a h hpg
  · exact countAt_71 a h hpg
  · exact countAt_72 a h hpg
  · exact countAt_73 a h hpg
  · exact countAt_74 a h hpg
  · exact countAt_75 a h hpg
  · exact countAt_76 a h hpg
  · exact countAt_77 a h hpg
  · exact countAt_78 a h hpg
  · exact countAt_79 a h hpg
  · exact countAt_80 a h hpg
  · exact countAt_81 a h hpg
  · exact countAt_82 a h hpg
  · exact countAt_83 a h hpg
  · exact countAt_84 a h hpg
  · exact countAt_85 a h hpg


theorem decBlock (a : Assignment p) :
    ∀ t ∈ rows a, valDigest t.1.blockHash = t.2.blockHash := by
  intro t ht
  rw [show rows a = List.ofFn (fun i => (row a i, inner a i)) from rfl] at ht
  obtain ⟨i, rfl⟩ := List.mem_ofFn.mp ht
  rfl


theorem decSlots (a : Assignment p) :
    ∀ t ∈ rows a, t.1.slots.map valSlot = t.2.exitSlots := by
  intro t ht
  rw [show rows a = List.ofFn (fun i => (row a i, inner a i)) from rfl] at ht
  obtain ⟨i, rfl⟩ := List.mem_ofFn.mp ht
  rfl


theorem decNulls (a : Assignment p) :
    ∀ t ∈ rows a, t.1.nulls.map valDigest = t.2.nullifiers := by
  intro t ht
  rw [show rows a = List.ofFn (fun i => (row a i, inner a i)) from rfl] at ht
  obtain ⟨i, rfl⟩ := List.mem_ofFn.mp ht
  rfl


theorem decNum (a : Assignment p) :
    ∀ t ∈ rows a, t.1.blockNumber.val = t.2.blockNumber := by
  intro t ht
  rw [show rows a = List.ofFn (fun i => (row a i, inner a i)) from rfl] at ht
  obtain ⟨i, rfl⟩ := List.mem_ofFn.mp ht
  fin_cases i <;> rfl


theorem decAsset (a : Assignment p) :
    ∀ t ∈ rows a, t.1.assetId.val = t.2.assetId := by
  intro t ht
  rw [show rows a = List.ofFn (fun i => (row a i, inner a i)) from rfl] at ht
  obtain ⟨i, rfl⟩ := List.mem_ofFn.mp ht
  fin_cases i <;> rfl


theorem decFee (a : Assignment p) :
    ∀ t ∈ rows a, t.1.fee.val = t.2.volumeFeeBps := by
  intro t ht
  rw [show rows a = List.ofFn (fun i => (row a i, inner a i)) from rfl] at ht
  obtain ⟨i, rfl⟩ := List.mem_ofFn.mp ht
  fin_cases i <;> rfl


theorem shapeAll (a : Assignment p) :
    ∀ t ∈ rows a, t.1.slots.length = 32 := by
  intro t ht
  rw [show rows a = List.ofFn (fun i => (row a i, inner a i)) from rfl] at ht
  obtain ⟨i, rfl⟩ := List.mem_ofFn.mp ht
  fin_cases i <;> rfl

end Plonky2Bridge.PublicWrapper86
