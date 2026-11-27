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

theorem blockLimb_0 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 165 15) = blockRef (rows a) 0 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  rw [(publicBatchWrapper86_f1626 a h)]
  rw [(publicBatchWrapper86_f1615 a h)]
  rw [(publicBatchWrapper86_f1604 a h)]
  rw [(publicBatchWrapper86_f1593 a h)]
  rw [(publicBatchWrapper86_f1582 a h)]
  rw [(publicBatchWrapper86_f1571 a h)]
  rw [(publicBatchWrapper86_f1560 a h)]
  rw [(publicBatchWrapper86_f1549 a h)]
  rw [(publicBatchWrapper86_f1538 a h)]
  rw [(publicBatchWrapper86_f1527 a h)]
  rw [(publicBatchWrapper86_f1516 a h)]
  rw [(publicBatchWrapper86_f1505 a h)]
  rw [(publicBatchWrapper86_f1494 a h)]
  rw [(publicBatchWrapper86_f1483 a h)]
  rw [(publicBatchWrapper86_f1472 a h)]
  rw [(publicBatchWrapper86_f1461 a h)]
  rw [(publicBatchWrapper86_f1450 a h)]
  rw [(publicBatchWrapper86_f1439 a h)]
  rw [(publicBatchWrapper86_f1428 a h)]
  rw [(publicBatchWrapper86_f1417 a h)]
  rw [(publicBatchWrapper86_f1406 a h)]
  rw [(publicBatchWrapper86_f1395 a h)]
  rw [(publicBatchWrapper86_f1384 a h)]
  rw [(publicBatchWrapper86_f1373 a h)]
  rw [(publicBatchWrapper86_f1362 a h)]
  rw [(publicBatchWrapper86_f1351 a h)]
  rw [(publicBatchWrapper86_f1340 a h)]
  rw [(publicBatchWrapper86_f1329 a h)]
  rw [(publicBatchWrapper86_f1318 a h)]
  rw [(publicBatchWrapper86_f1307 a h)]
  rw [(publicBatchWrapper86_f1296 a h)]
  rw [(publicBatchWrapper86_f1285 a h)]
  rw [(publicBatchWrapper86_f1274 a h)]
  rw [(publicBatchWrapper86_f1263 a h)]
  rw [(publicBatchWrapper86_f1252 a h)]
  rw [(publicBatchWrapper86_f1241 a h)]
  rw [(publicBatchWrapper86_f1230 a h)]
  rw [(publicBatchWrapper86_f1219 a h)]
  rw [(publicBatchWrapper86_f1208 a h)]
  rw [(publicBatchWrapper86_f1197 a h)]
  rw [(publicBatchWrapper86_f1186 a h)]
  rw [(publicBatchWrapper86_f1175 a h)]
  rw [(publicBatchWrapper86_f1164 a h)]
  rw [(publicBatchWrapper86_f1153 a h)]
  rw [(publicBatchWrapper86_f1142 a h)]
  rw [(publicBatchWrapper86_f1131 a h)]
  rw [(publicBatchWrapper86_f1120 a h)]
  rw [(publicBatchWrapper86_f1109 a h)]
  rw [(publicBatchWrapper86_f1098 a h)]
  rw [(publicBatchWrapper86_f1087 a h)]
  rw [(publicBatchWrapper86_f1076 a h)]
  rw [(publicBatchWrapper86_f1065 a h)]
  rw [(publicBatchWrapper86_f1054 a h)]
  rw [(publicBatchWrapper86_f1043 a h)]
  rw [(publicBatchWrapper86_f1032 a h)]
  rw [(publicBatchWrapper86_f1021 a h)]
  rw [(publicBatchWrapper86_f1010 a h)]
  rw [(publicBatchWrapper86_f999 a h)]
  rw [(publicBatchWrapper86_f988 a h)]
  rw [(publicBatchWrapper86_f977 a h)]
  rw [(publicBatchWrapper86_f966 a h)]
  rw [(publicBatchWrapper86_f955 a h)]
  rw [(publicBatchWrapper86_f944 a h)]
  rw [(publicBatchWrapper86_f933 a h)]
  rw [(publicBatchWrapper86_f922 a h)]
  rw [(publicBatchWrapper86_f911 a h)]
  rw [(publicBatchWrapper86_f900 a h)]
  rw [(publicBatchWrapper86_f889 a h)]
  rw [(publicBatchWrapper86_f878 a h)]
  rw [(publicBatchWrapper86_f867 a h)]
  rw [(publicBatchWrapper86_f856 a h)]
  rw [(publicBatchWrapper86_f845 a h)]
  rw [(publicBatchWrapper86_f834 a h)]
  rw [(publicBatchWrapper86_f823 a h)]
  rw [(publicBatchWrapper86_f812 a h)]
  rw [(publicBatchWrapper86_f801 a h)]
  rw [(publicBatchWrapper86_f790 a h)]
  rw [(publicBatchWrapper86_f779 a h)]
  rw [(publicBatchWrapper86_f768 a h)]
  rw [(publicBatchWrapper86_f757 a h)]
  rw [(publicBatchWrapper86_f746 a h)]
  rw [(publicBatchWrapper86_f735 a h)]
  rw [(publicBatchWrapper86_f724 a h)]
  rw [(publicBatchWrapper86_f713 a h)]
  rw [(publicBatchWrapper86_f702 a h)]
  rw [(publicBatchWrapper86_f691 a h)]
  rw [kzero, selChain_eq a h]
  rw [blockRef, rows, scanRef_86]
  rfl

end Plonky2Bridge.PublicWrapper86
