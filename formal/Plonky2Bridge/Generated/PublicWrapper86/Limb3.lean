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

theorem blockLimb_3 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 165 39) = blockRef (rows a) 3 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  rw [(publicBatchWrapper86_f1629 a h)]
  rw [(publicBatchWrapper86_f1618 a h)]
  rw [(publicBatchWrapper86_f1607 a h)]
  rw [(publicBatchWrapper86_f1596 a h)]
  rw [(publicBatchWrapper86_f1585 a h)]
  rw [(publicBatchWrapper86_f1574 a h)]
  rw [(publicBatchWrapper86_f1563 a h)]
  rw [(publicBatchWrapper86_f1552 a h)]
  rw [(publicBatchWrapper86_f1541 a h)]
  rw [(publicBatchWrapper86_f1530 a h)]
  rw [(publicBatchWrapper86_f1519 a h)]
  rw [(publicBatchWrapper86_f1508 a h)]
  rw [(publicBatchWrapper86_f1497 a h)]
  rw [(publicBatchWrapper86_f1486 a h)]
  rw [(publicBatchWrapper86_f1475 a h)]
  rw [(publicBatchWrapper86_f1464 a h)]
  rw [(publicBatchWrapper86_f1453 a h)]
  rw [(publicBatchWrapper86_f1442 a h)]
  rw [(publicBatchWrapper86_f1431 a h)]
  rw [(publicBatchWrapper86_f1420 a h)]
  rw [(publicBatchWrapper86_f1409 a h)]
  rw [(publicBatchWrapper86_f1398 a h)]
  rw [(publicBatchWrapper86_f1387 a h)]
  rw [(publicBatchWrapper86_f1376 a h)]
  rw [(publicBatchWrapper86_f1365 a h)]
  rw [(publicBatchWrapper86_f1354 a h)]
  rw [(publicBatchWrapper86_f1343 a h)]
  rw [(publicBatchWrapper86_f1332 a h)]
  rw [(publicBatchWrapper86_f1321 a h)]
  rw [(publicBatchWrapper86_f1310 a h)]
  rw [(publicBatchWrapper86_f1299 a h)]
  rw [(publicBatchWrapper86_f1288 a h)]
  rw [(publicBatchWrapper86_f1277 a h)]
  rw [(publicBatchWrapper86_f1266 a h)]
  rw [(publicBatchWrapper86_f1255 a h)]
  rw [(publicBatchWrapper86_f1244 a h)]
  rw [(publicBatchWrapper86_f1233 a h)]
  rw [(publicBatchWrapper86_f1222 a h)]
  rw [(publicBatchWrapper86_f1211 a h)]
  rw [(publicBatchWrapper86_f1200 a h)]
  rw [(publicBatchWrapper86_f1189 a h)]
  rw [(publicBatchWrapper86_f1178 a h)]
  rw [(publicBatchWrapper86_f1167 a h)]
  rw [(publicBatchWrapper86_f1156 a h)]
  rw [(publicBatchWrapper86_f1145 a h)]
  rw [(publicBatchWrapper86_f1134 a h)]
  rw [(publicBatchWrapper86_f1123 a h)]
  rw [(publicBatchWrapper86_f1112 a h)]
  rw [(publicBatchWrapper86_f1101 a h)]
  rw [(publicBatchWrapper86_f1090 a h)]
  rw [(publicBatchWrapper86_f1079 a h)]
  rw [(publicBatchWrapper86_f1068 a h)]
  rw [(publicBatchWrapper86_f1057 a h)]
  rw [(publicBatchWrapper86_f1046 a h)]
  rw [(publicBatchWrapper86_f1035 a h)]
  rw [(publicBatchWrapper86_f1024 a h)]
  rw [(publicBatchWrapper86_f1013 a h)]
  rw [(publicBatchWrapper86_f1002 a h)]
  rw [(publicBatchWrapper86_f991 a h)]
  rw [(publicBatchWrapper86_f980 a h)]
  rw [(publicBatchWrapper86_f969 a h)]
  rw [(publicBatchWrapper86_f958 a h)]
  rw [(publicBatchWrapper86_f947 a h)]
  rw [(publicBatchWrapper86_f936 a h)]
  rw [(publicBatchWrapper86_f925 a h)]
  rw [(publicBatchWrapper86_f914 a h)]
  rw [(publicBatchWrapper86_f903 a h)]
  rw [(publicBatchWrapper86_f892 a h)]
  rw [(publicBatchWrapper86_f881 a h)]
  rw [(publicBatchWrapper86_f870 a h)]
  rw [(publicBatchWrapper86_f859 a h)]
  rw [(publicBatchWrapper86_f848 a h)]
  rw [(publicBatchWrapper86_f837 a h)]
  rw [(publicBatchWrapper86_f826 a h)]
  rw [(publicBatchWrapper86_f815 a h)]
  rw [(publicBatchWrapper86_f804 a h)]
  rw [(publicBatchWrapper86_f793 a h)]
  rw [(publicBatchWrapper86_f782 a h)]
  rw [(publicBatchWrapper86_f771 a h)]
  rw [(publicBatchWrapper86_f760 a h)]
  rw [(publicBatchWrapper86_f749 a h)]
  rw [(publicBatchWrapper86_f738 a h)]
  rw [(publicBatchWrapper86_f727 a h)]
  rw [(publicBatchWrapper86_f716 a h)]
  rw [(publicBatchWrapper86_f705 a h)]
  rw [(publicBatchWrapper86_f694 a h)]
  rw [kzero, selChain_eq a h]
  rw [blockRef, rows, scanRef_86]
  rfl

end Plonky2Bridge.PublicWrapper86
