import Plonky2Bridge.Generated.PublicWrapper86.Scan
import Plonky2Bridge.Generated.PublicWrapper86.Take

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

theorem blockLimb_1 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 165 23) = blockRef (rows a) 1 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  rw [(publicBatchWrapper86_f1627 a h)]
  rw [(publicBatchWrapper86_f1616 a h)]
  rw [(publicBatchWrapper86_f1605 a h)]
  rw [(publicBatchWrapper86_f1594 a h)]
  rw [(publicBatchWrapper86_f1583 a h)]
  rw [(publicBatchWrapper86_f1572 a h)]
  rw [(publicBatchWrapper86_f1561 a h)]
  rw [(publicBatchWrapper86_f1550 a h)]
  rw [(publicBatchWrapper86_f1539 a h)]
  rw [(publicBatchWrapper86_f1528 a h)]
  rw [(publicBatchWrapper86_f1517 a h)]
  rw [(publicBatchWrapper86_f1506 a h)]
  rw [(publicBatchWrapper86_f1495 a h)]
  rw [(publicBatchWrapper86_f1484 a h)]
  rw [(publicBatchWrapper86_f1473 a h)]
  rw [(publicBatchWrapper86_f1462 a h)]
  rw [(publicBatchWrapper86_f1451 a h)]
  rw [(publicBatchWrapper86_f1440 a h)]
  rw [(publicBatchWrapper86_f1429 a h)]
  rw [(publicBatchWrapper86_f1418 a h)]
  rw [(publicBatchWrapper86_f1407 a h)]
  rw [(publicBatchWrapper86_f1396 a h)]
  rw [(publicBatchWrapper86_f1385 a h)]
  rw [(publicBatchWrapper86_f1374 a h)]
  rw [(publicBatchWrapper86_f1363 a h)]
  rw [(publicBatchWrapper86_f1352 a h)]
  rw [(publicBatchWrapper86_f1341 a h)]
  rw [(publicBatchWrapper86_f1330 a h)]
  rw [(publicBatchWrapper86_f1319 a h)]
  rw [(publicBatchWrapper86_f1308 a h)]
  rw [(publicBatchWrapper86_f1297 a h)]
  rw [(publicBatchWrapper86_f1286 a h)]
  rw [(publicBatchWrapper86_f1275 a h)]
  rw [(publicBatchWrapper86_f1264 a h)]
  rw [(publicBatchWrapper86_f1253 a h)]
  rw [(publicBatchWrapper86_f1242 a h)]
  rw [(publicBatchWrapper86_f1231 a h)]
  rw [(publicBatchWrapper86_f1220 a h)]
  rw [(publicBatchWrapper86_f1209 a h)]
  rw [(publicBatchWrapper86_f1198 a h)]
  rw [(publicBatchWrapper86_f1187 a h)]
  rw [(publicBatchWrapper86_f1176 a h)]
  rw [(publicBatchWrapper86_f1165 a h)]
  rw [(publicBatchWrapper86_f1154 a h)]
  rw [(publicBatchWrapper86_f1143 a h)]
  rw [(publicBatchWrapper86_f1132 a h)]
  rw [(publicBatchWrapper86_f1121 a h)]
  rw [(publicBatchWrapper86_f1110 a h)]
  rw [(publicBatchWrapper86_f1099 a h)]
  rw [(publicBatchWrapper86_f1088 a h)]
  rw [(publicBatchWrapper86_f1077 a h)]
  rw [(publicBatchWrapper86_f1066 a h)]
  rw [(publicBatchWrapper86_f1055 a h)]
  rw [(publicBatchWrapper86_f1044 a h)]
  rw [(publicBatchWrapper86_f1033 a h)]
  rw [(publicBatchWrapper86_f1022 a h)]
  rw [(publicBatchWrapper86_f1011 a h)]
  rw [(publicBatchWrapper86_f1000 a h)]
  rw [(publicBatchWrapper86_f989 a h)]
  rw [(publicBatchWrapper86_f978 a h)]
  rw [(publicBatchWrapper86_f967 a h)]
  rw [(publicBatchWrapper86_f956 a h)]
  rw [(publicBatchWrapper86_f945 a h)]
  rw [(publicBatchWrapper86_f934 a h)]
  rw [(publicBatchWrapper86_f923 a h)]
  rw [(publicBatchWrapper86_f912 a h)]
  rw [(publicBatchWrapper86_f901 a h)]
  rw [(publicBatchWrapper86_f890 a h)]
  rw [(publicBatchWrapper86_f879 a h)]
  rw [(publicBatchWrapper86_f868 a h)]
  rw [(publicBatchWrapper86_f857 a h)]
  rw [(publicBatchWrapper86_f846 a h)]
  rw [(publicBatchWrapper86_f835 a h)]
  rw [(publicBatchWrapper86_f824 a h)]
  rw [(publicBatchWrapper86_f813 a h)]
  rw [(publicBatchWrapper86_f802 a h)]
  rw [(publicBatchWrapper86_f791 a h)]
  rw [(publicBatchWrapper86_f780 a h)]
  rw [(publicBatchWrapper86_f769 a h)]
  rw [(publicBatchWrapper86_f758 a h)]
  rw [(publicBatchWrapper86_f747 a h)]
  rw [(publicBatchWrapper86_f736 a h)]
  rw [(publicBatchWrapper86_f725 a h)]
  rw [(publicBatchWrapper86_f714 a h)]
  rw [(publicBatchWrapper86_f703 a h)]
  rw [(publicBatchWrapper86_f692 a h)]
  rw [kzero, selChain_eq a h]
  rw [blockRef, rows, scanRef_86]
  rfl

end Plonky2Bridge.PublicWrapper86
