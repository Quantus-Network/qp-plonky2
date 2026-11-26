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

theorem blockLimb_2 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 165 31) = blockRef (rows a) 2 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  rw [(publicBatchWrapper86_f1628 a h)]
  rw [(publicBatchWrapper86_f1617 a h)]
  rw [(publicBatchWrapper86_f1606 a h)]
  rw [(publicBatchWrapper86_f1595 a h)]
  rw [(publicBatchWrapper86_f1584 a h)]
  rw [(publicBatchWrapper86_f1573 a h)]
  rw [(publicBatchWrapper86_f1562 a h)]
  rw [(publicBatchWrapper86_f1551 a h)]
  rw [(publicBatchWrapper86_f1540 a h)]
  rw [(publicBatchWrapper86_f1529 a h)]
  rw [(publicBatchWrapper86_f1518 a h)]
  rw [(publicBatchWrapper86_f1507 a h)]
  rw [(publicBatchWrapper86_f1496 a h)]
  rw [(publicBatchWrapper86_f1485 a h)]
  rw [(publicBatchWrapper86_f1474 a h)]
  rw [(publicBatchWrapper86_f1463 a h)]
  rw [(publicBatchWrapper86_f1452 a h)]
  rw [(publicBatchWrapper86_f1441 a h)]
  rw [(publicBatchWrapper86_f1430 a h)]
  rw [(publicBatchWrapper86_f1419 a h)]
  rw [(publicBatchWrapper86_f1408 a h)]
  rw [(publicBatchWrapper86_f1397 a h)]
  rw [(publicBatchWrapper86_f1386 a h)]
  rw [(publicBatchWrapper86_f1375 a h)]
  rw [(publicBatchWrapper86_f1364 a h)]
  rw [(publicBatchWrapper86_f1353 a h)]
  rw [(publicBatchWrapper86_f1342 a h)]
  rw [(publicBatchWrapper86_f1331 a h)]
  rw [(publicBatchWrapper86_f1320 a h)]
  rw [(publicBatchWrapper86_f1309 a h)]
  rw [(publicBatchWrapper86_f1298 a h)]
  rw [(publicBatchWrapper86_f1287 a h)]
  rw [(publicBatchWrapper86_f1276 a h)]
  rw [(publicBatchWrapper86_f1265 a h)]
  rw [(publicBatchWrapper86_f1254 a h)]
  rw [(publicBatchWrapper86_f1243 a h)]
  rw [(publicBatchWrapper86_f1232 a h)]
  rw [(publicBatchWrapper86_f1221 a h)]
  rw [(publicBatchWrapper86_f1210 a h)]
  rw [(publicBatchWrapper86_f1199 a h)]
  rw [(publicBatchWrapper86_f1188 a h)]
  rw [(publicBatchWrapper86_f1177 a h)]
  rw [(publicBatchWrapper86_f1166 a h)]
  rw [(publicBatchWrapper86_f1155 a h)]
  rw [(publicBatchWrapper86_f1144 a h)]
  rw [(publicBatchWrapper86_f1133 a h)]
  rw [(publicBatchWrapper86_f1122 a h)]
  rw [(publicBatchWrapper86_f1111 a h)]
  rw [(publicBatchWrapper86_f1100 a h)]
  rw [(publicBatchWrapper86_f1089 a h)]
  rw [(publicBatchWrapper86_f1078 a h)]
  rw [(publicBatchWrapper86_f1067 a h)]
  rw [(publicBatchWrapper86_f1056 a h)]
  rw [(publicBatchWrapper86_f1045 a h)]
  rw [(publicBatchWrapper86_f1034 a h)]
  rw [(publicBatchWrapper86_f1023 a h)]
  rw [(publicBatchWrapper86_f1012 a h)]
  rw [(publicBatchWrapper86_f1001 a h)]
  rw [(publicBatchWrapper86_f990 a h)]
  rw [(publicBatchWrapper86_f979 a h)]
  rw [(publicBatchWrapper86_f968 a h)]
  rw [(publicBatchWrapper86_f957 a h)]
  rw [(publicBatchWrapper86_f946 a h)]
  rw [(publicBatchWrapper86_f935 a h)]
  rw [(publicBatchWrapper86_f924 a h)]
  rw [(publicBatchWrapper86_f913 a h)]
  rw [(publicBatchWrapper86_f902 a h)]
  rw [(publicBatchWrapper86_f891 a h)]
  rw [(publicBatchWrapper86_f880 a h)]
  rw [(publicBatchWrapper86_f869 a h)]
  rw [(publicBatchWrapper86_f858 a h)]
  rw [(publicBatchWrapper86_f847 a h)]
  rw [(publicBatchWrapper86_f836 a h)]
  rw [(publicBatchWrapper86_f825 a h)]
  rw [(publicBatchWrapper86_f814 a h)]
  rw [(publicBatchWrapper86_f803 a h)]
  rw [(publicBatchWrapper86_f792 a h)]
  rw [(publicBatchWrapper86_f781 a h)]
  rw [(publicBatchWrapper86_f770 a h)]
  rw [(publicBatchWrapper86_f759 a h)]
  rw [(publicBatchWrapper86_f748 a h)]
  rw [(publicBatchWrapper86_f737 a h)]
  rw [(publicBatchWrapper86_f726 a h)]
  rw [(publicBatchWrapper86_f715 a h)]
  rw [(publicBatchWrapper86_f704 a h)]
  rw [(publicBatchWrapper86_f693 a h)]
  rw [kzero, selChain_eq a h]
  rw [blockRef, rows, scanRef_86]
  rfl

end Plonky2Bridge.PublicWrapper86
