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

theorem blockLimb_0 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 169 63) = blockRef (rows a) 0 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  rw [(publicBatchWrapper89_f1683 a h)]
  rw [(publicBatchWrapper89_f1672 a h)]
  rw [(publicBatchWrapper89_f1661 a h)]
  rw [(publicBatchWrapper89_f1650 a h)]
  rw [(publicBatchWrapper89_f1639 a h)]
  rw [(publicBatchWrapper89_f1628 a h)]
  rw [(publicBatchWrapper89_f1617 a h)]
  rw [(publicBatchWrapper89_f1606 a h)]
  rw [(publicBatchWrapper89_f1595 a h)]
  rw [(publicBatchWrapper89_f1584 a h)]
  rw [(publicBatchWrapper89_f1573 a h)]
  rw [(publicBatchWrapper89_f1562 a h)]
  rw [(publicBatchWrapper89_f1551 a h)]
  rw [(publicBatchWrapper89_f1540 a h)]
  rw [(publicBatchWrapper89_f1529 a h)]
  rw [(publicBatchWrapper89_f1518 a h)]
  rw [(publicBatchWrapper89_f1507 a h)]
  rw [(publicBatchWrapper89_f1496 a h)]
  rw [(publicBatchWrapper89_f1485 a h)]
  rw [(publicBatchWrapper89_f1474 a h)]
  rw [(publicBatchWrapper89_f1463 a h)]
  rw [(publicBatchWrapper89_f1452 a h)]
  rw [(publicBatchWrapper89_f1441 a h)]
  rw [(publicBatchWrapper89_f1430 a h)]
  rw [(publicBatchWrapper89_f1419 a h)]
  rw [(publicBatchWrapper89_f1408 a h)]
  rw [(publicBatchWrapper89_f1397 a h)]
  rw [(publicBatchWrapper89_f1386 a h)]
  rw [(publicBatchWrapper89_f1375 a h)]
  rw [(publicBatchWrapper89_f1364 a h)]
  rw [(publicBatchWrapper89_f1353 a h)]
  rw [(publicBatchWrapper89_f1342 a h)]
  rw [(publicBatchWrapper89_f1331 a h)]
  rw [(publicBatchWrapper89_f1320 a h)]
  rw [(publicBatchWrapper89_f1309 a h)]
  rw [(publicBatchWrapper89_f1298 a h)]
  rw [(publicBatchWrapper89_f1287 a h)]
  rw [(publicBatchWrapper89_f1276 a h)]
  rw [(publicBatchWrapper89_f1265 a h)]
  rw [(publicBatchWrapper89_f1254 a h)]
  rw [(publicBatchWrapper89_f1243 a h)]
  rw [(publicBatchWrapper89_f1232 a h)]
  rw [(publicBatchWrapper89_f1221 a h)]
  rw [(publicBatchWrapper89_f1210 a h)]
  rw [(publicBatchWrapper89_f1199 a h)]
  rw [(publicBatchWrapper89_f1188 a h)]
  rw [(publicBatchWrapper89_f1177 a h)]
  rw [(publicBatchWrapper89_f1166 a h)]
  rw [(publicBatchWrapper89_f1155 a h)]
  rw [(publicBatchWrapper89_f1144 a h)]
  rw [(publicBatchWrapper89_f1133 a h)]
  rw [(publicBatchWrapper89_f1122 a h)]
  rw [(publicBatchWrapper89_f1111 a h)]
  rw [(publicBatchWrapper89_f1100 a h)]
  rw [(publicBatchWrapper89_f1089 a h)]
  rw [(publicBatchWrapper89_f1078 a h)]
  rw [(publicBatchWrapper89_f1067 a h)]
  rw [(publicBatchWrapper89_f1056 a h)]
  rw [(publicBatchWrapper89_f1045 a h)]
  rw [(publicBatchWrapper89_f1034 a h)]
  rw [(publicBatchWrapper89_f1023 a h)]
  rw [(publicBatchWrapper89_f1012 a h)]
  rw [(publicBatchWrapper89_f1001 a h)]
  rw [(publicBatchWrapper89_f990 a h)]
  rw [(publicBatchWrapper89_f979 a h)]
  rw [(publicBatchWrapper89_f968 a h)]
  rw [(publicBatchWrapper89_f957 a h)]
  rw [(publicBatchWrapper89_f946 a h)]
  rw [(publicBatchWrapper89_f935 a h)]
  rw [(publicBatchWrapper89_f924 a h)]
  rw [(publicBatchWrapper89_f913 a h)]
  rw [(publicBatchWrapper89_f902 a h)]
  rw [(publicBatchWrapper89_f891 a h)]
  rw [(publicBatchWrapper89_f880 a h)]
  rw [(publicBatchWrapper89_f869 a h)]
  rw [(publicBatchWrapper89_f858 a h)]
  rw [(publicBatchWrapper89_f847 a h)]
  rw [(publicBatchWrapper89_f836 a h)]
  rw [(publicBatchWrapper89_f825 a h)]
  rw [(publicBatchWrapper89_f814 a h)]
  rw [(publicBatchWrapper89_f803 a h)]
  rw [(publicBatchWrapper89_f792 a h)]
  rw [(publicBatchWrapper89_f781 a h)]
  rw [(publicBatchWrapper89_f770 a h)]
  rw [(publicBatchWrapper89_f759 a h)]
  rw [(publicBatchWrapper89_f748 a h)]
  rw [(publicBatchWrapper89_f737 a h)]
  rw [(publicBatchWrapper89_f726 a h)]
  rw [(publicBatchWrapper89_f715 a h)]
  rw [kzero, selChain_eq a h]
  rw [blockRef, rows, scanRef_89]
  rfl

end Plonky2Bridge.PublicWrapper89
