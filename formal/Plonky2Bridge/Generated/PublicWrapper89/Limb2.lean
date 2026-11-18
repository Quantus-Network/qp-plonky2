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

theorem blockLimb_2 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 169 79) = blockRef (rows a) 2 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  rw [(publicBatchWrapper89_f1685 a h)]
  rw [(publicBatchWrapper89_f1674 a h)]
  rw [(publicBatchWrapper89_f1663 a h)]
  rw [(publicBatchWrapper89_f1652 a h)]
  rw [(publicBatchWrapper89_f1641 a h)]
  rw [(publicBatchWrapper89_f1630 a h)]
  rw [(publicBatchWrapper89_f1619 a h)]
  rw [(publicBatchWrapper89_f1608 a h)]
  rw [(publicBatchWrapper89_f1597 a h)]
  rw [(publicBatchWrapper89_f1586 a h)]
  rw [(publicBatchWrapper89_f1575 a h)]
  rw [(publicBatchWrapper89_f1564 a h)]
  rw [(publicBatchWrapper89_f1553 a h)]
  rw [(publicBatchWrapper89_f1542 a h)]
  rw [(publicBatchWrapper89_f1531 a h)]
  rw [(publicBatchWrapper89_f1520 a h)]
  rw [(publicBatchWrapper89_f1509 a h)]
  rw [(publicBatchWrapper89_f1498 a h)]
  rw [(publicBatchWrapper89_f1487 a h)]
  rw [(publicBatchWrapper89_f1476 a h)]
  rw [(publicBatchWrapper89_f1465 a h)]
  rw [(publicBatchWrapper89_f1454 a h)]
  rw [(publicBatchWrapper89_f1443 a h)]
  rw [(publicBatchWrapper89_f1432 a h)]
  rw [(publicBatchWrapper89_f1421 a h)]
  rw [(publicBatchWrapper89_f1410 a h)]
  rw [(publicBatchWrapper89_f1399 a h)]
  rw [(publicBatchWrapper89_f1388 a h)]
  rw [(publicBatchWrapper89_f1377 a h)]
  rw [(publicBatchWrapper89_f1366 a h)]
  rw [(publicBatchWrapper89_f1355 a h)]
  rw [(publicBatchWrapper89_f1344 a h)]
  rw [(publicBatchWrapper89_f1333 a h)]
  rw [(publicBatchWrapper89_f1322 a h)]
  rw [(publicBatchWrapper89_f1311 a h)]
  rw [(publicBatchWrapper89_f1300 a h)]
  rw [(publicBatchWrapper89_f1289 a h)]
  rw [(publicBatchWrapper89_f1278 a h)]
  rw [(publicBatchWrapper89_f1267 a h)]
  rw [(publicBatchWrapper89_f1256 a h)]
  rw [(publicBatchWrapper89_f1245 a h)]
  rw [(publicBatchWrapper89_f1234 a h)]
  rw [(publicBatchWrapper89_f1223 a h)]
  rw [(publicBatchWrapper89_f1212 a h)]
  rw [(publicBatchWrapper89_f1201 a h)]
  rw [(publicBatchWrapper89_f1190 a h)]
  rw [(publicBatchWrapper89_f1179 a h)]
  rw [(publicBatchWrapper89_f1168 a h)]
  rw [(publicBatchWrapper89_f1157 a h)]
  rw [(publicBatchWrapper89_f1146 a h)]
  rw [(publicBatchWrapper89_f1135 a h)]
  rw [(publicBatchWrapper89_f1124 a h)]
  rw [(publicBatchWrapper89_f1113 a h)]
  rw [(publicBatchWrapper89_f1102 a h)]
  rw [(publicBatchWrapper89_f1091 a h)]
  rw [(publicBatchWrapper89_f1080 a h)]
  rw [(publicBatchWrapper89_f1069 a h)]
  rw [(publicBatchWrapper89_f1058 a h)]
  rw [(publicBatchWrapper89_f1047 a h)]
  rw [(publicBatchWrapper89_f1036 a h)]
  rw [(publicBatchWrapper89_f1025 a h)]
  rw [(publicBatchWrapper89_f1014 a h)]
  rw [(publicBatchWrapper89_f1003 a h)]
  rw [(publicBatchWrapper89_f992 a h)]
  rw [(publicBatchWrapper89_f981 a h)]
  rw [(publicBatchWrapper89_f970 a h)]
  rw [(publicBatchWrapper89_f959 a h)]
  rw [(publicBatchWrapper89_f948 a h)]
  rw [(publicBatchWrapper89_f937 a h)]
  rw [(publicBatchWrapper89_f926 a h)]
  rw [(publicBatchWrapper89_f915 a h)]
  rw [(publicBatchWrapper89_f904 a h)]
  rw [(publicBatchWrapper89_f893 a h)]
  rw [(publicBatchWrapper89_f882 a h)]
  rw [(publicBatchWrapper89_f871 a h)]
  rw [(publicBatchWrapper89_f860 a h)]
  rw [(publicBatchWrapper89_f849 a h)]
  rw [(publicBatchWrapper89_f838 a h)]
  rw [(publicBatchWrapper89_f827 a h)]
  rw [(publicBatchWrapper89_f816 a h)]
  rw [(publicBatchWrapper89_f805 a h)]
  rw [(publicBatchWrapper89_f794 a h)]
  rw [(publicBatchWrapper89_f783 a h)]
  rw [(publicBatchWrapper89_f772 a h)]
  rw [(publicBatchWrapper89_f761 a h)]
  rw [(publicBatchWrapper89_f750 a h)]
  rw [(publicBatchWrapper89_f739 a h)]
  rw [(publicBatchWrapper89_f728 a h)]
  rw [(publicBatchWrapper89_f717 a h)]
  rw [kzero, selChain_eq a h]
  rw [blockRef, rows, scanRef_89]
  rfl

end Plonky2Bridge.PublicWrapper89
