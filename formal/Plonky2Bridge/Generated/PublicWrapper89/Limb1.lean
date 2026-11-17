import Plonky2Bridge.Generated.PublicWrapper89.Scan
import Plonky2Bridge.Generated.PublicWrapper89.Take

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

theorem blockLimb_1 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 169 71) = blockRef (rows a) 1 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  rw [(publicBatchWrapper89_f1684 a h)]
  rw [(publicBatchWrapper89_f1673 a h)]
  rw [(publicBatchWrapper89_f1662 a h)]
  rw [(publicBatchWrapper89_f1651 a h)]
  rw [(publicBatchWrapper89_f1640 a h)]
  rw [(publicBatchWrapper89_f1629 a h)]
  rw [(publicBatchWrapper89_f1618 a h)]
  rw [(publicBatchWrapper89_f1607 a h)]
  rw [(publicBatchWrapper89_f1596 a h)]
  rw [(publicBatchWrapper89_f1585 a h)]
  rw [(publicBatchWrapper89_f1574 a h)]
  rw [(publicBatchWrapper89_f1563 a h)]
  rw [(publicBatchWrapper89_f1552 a h)]
  rw [(publicBatchWrapper89_f1541 a h)]
  rw [(publicBatchWrapper89_f1530 a h)]
  rw [(publicBatchWrapper89_f1519 a h)]
  rw [(publicBatchWrapper89_f1508 a h)]
  rw [(publicBatchWrapper89_f1497 a h)]
  rw [(publicBatchWrapper89_f1486 a h)]
  rw [(publicBatchWrapper89_f1475 a h)]
  rw [(publicBatchWrapper89_f1464 a h)]
  rw [(publicBatchWrapper89_f1453 a h)]
  rw [(publicBatchWrapper89_f1442 a h)]
  rw [(publicBatchWrapper89_f1431 a h)]
  rw [(publicBatchWrapper89_f1420 a h)]
  rw [(publicBatchWrapper89_f1409 a h)]
  rw [(publicBatchWrapper89_f1398 a h)]
  rw [(publicBatchWrapper89_f1387 a h)]
  rw [(publicBatchWrapper89_f1376 a h)]
  rw [(publicBatchWrapper89_f1365 a h)]
  rw [(publicBatchWrapper89_f1354 a h)]
  rw [(publicBatchWrapper89_f1343 a h)]
  rw [(publicBatchWrapper89_f1332 a h)]
  rw [(publicBatchWrapper89_f1321 a h)]
  rw [(publicBatchWrapper89_f1310 a h)]
  rw [(publicBatchWrapper89_f1299 a h)]
  rw [(publicBatchWrapper89_f1288 a h)]
  rw [(publicBatchWrapper89_f1277 a h)]
  rw [(publicBatchWrapper89_f1266 a h)]
  rw [(publicBatchWrapper89_f1255 a h)]
  rw [(publicBatchWrapper89_f1244 a h)]
  rw [(publicBatchWrapper89_f1233 a h)]
  rw [(publicBatchWrapper89_f1222 a h)]
  rw [(publicBatchWrapper89_f1211 a h)]
  rw [(publicBatchWrapper89_f1200 a h)]
  rw [(publicBatchWrapper89_f1189 a h)]
  rw [(publicBatchWrapper89_f1178 a h)]
  rw [(publicBatchWrapper89_f1167 a h)]
  rw [(publicBatchWrapper89_f1156 a h)]
  rw [(publicBatchWrapper89_f1145 a h)]
  rw [(publicBatchWrapper89_f1134 a h)]
  rw [(publicBatchWrapper89_f1123 a h)]
  rw [(publicBatchWrapper89_f1112 a h)]
  rw [(publicBatchWrapper89_f1101 a h)]
  rw [(publicBatchWrapper89_f1090 a h)]
  rw [(publicBatchWrapper89_f1079 a h)]
  rw [(publicBatchWrapper89_f1068 a h)]
  rw [(publicBatchWrapper89_f1057 a h)]
  rw [(publicBatchWrapper89_f1046 a h)]
  rw [(publicBatchWrapper89_f1035 a h)]
  rw [(publicBatchWrapper89_f1024 a h)]
  rw [(publicBatchWrapper89_f1013 a h)]
  rw [(publicBatchWrapper89_f1002 a h)]
  rw [(publicBatchWrapper89_f991 a h)]
  rw [(publicBatchWrapper89_f980 a h)]
  rw [(publicBatchWrapper89_f969 a h)]
  rw [(publicBatchWrapper89_f958 a h)]
  rw [(publicBatchWrapper89_f947 a h)]
  rw [(publicBatchWrapper89_f936 a h)]
  rw [(publicBatchWrapper89_f925 a h)]
  rw [(publicBatchWrapper89_f914 a h)]
  rw [(publicBatchWrapper89_f903 a h)]
  rw [(publicBatchWrapper89_f892 a h)]
  rw [(publicBatchWrapper89_f881 a h)]
  rw [(publicBatchWrapper89_f870 a h)]
  rw [(publicBatchWrapper89_f859 a h)]
  rw [(publicBatchWrapper89_f848 a h)]
  rw [(publicBatchWrapper89_f837 a h)]
  rw [(publicBatchWrapper89_f826 a h)]
  rw [(publicBatchWrapper89_f815 a h)]
  rw [(publicBatchWrapper89_f804 a h)]
  rw [(publicBatchWrapper89_f793 a h)]
  rw [(publicBatchWrapper89_f782 a h)]
  rw [(publicBatchWrapper89_f771 a h)]
  rw [(publicBatchWrapper89_f760 a h)]
  rw [(publicBatchWrapper89_f749 a h)]
  rw [(publicBatchWrapper89_f738 a h)]
  rw [(publicBatchWrapper89_f727 a h)]
  rw [(publicBatchWrapper89_f716 a h)]
  rw [kzero, selChain_eq a h]
  rw [blockRef, rows, scanRef_89]
  rfl

end Plonky2Bridge.PublicWrapper89
