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

theorem blockLimb_3 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 170 7) = blockRef (rows a) 3 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  rw [(publicBatchWrapper89_f1686 a h)]
  rw [(publicBatchWrapper89_f1675 a h)]
  rw [(publicBatchWrapper89_f1664 a h)]
  rw [(publicBatchWrapper89_f1653 a h)]
  rw [(publicBatchWrapper89_f1642 a h)]
  rw [(publicBatchWrapper89_f1631 a h)]
  rw [(publicBatchWrapper89_f1620 a h)]
  rw [(publicBatchWrapper89_f1609 a h)]
  rw [(publicBatchWrapper89_f1598 a h)]
  rw [(publicBatchWrapper89_f1587 a h)]
  rw [(publicBatchWrapper89_f1576 a h)]
  rw [(publicBatchWrapper89_f1565 a h)]
  rw [(publicBatchWrapper89_f1554 a h)]
  rw [(publicBatchWrapper89_f1543 a h)]
  rw [(publicBatchWrapper89_f1532 a h)]
  rw [(publicBatchWrapper89_f1521 a h)]
  rw [(publicBatchWrapper89_f1510 a h)]
  rw [(publicBatchWrapper89_f1499 a h)]
  rw [(publicBatchWrapper89_f1488 a h)]
  rw [(publicBatchWrapper89_f1477 a h)]
  rw [(publicBatchWrapper89_f1466 a h)]
  rw [(publicBatchWrapper89_f1455 a h)]
  rw [(publicBatchWrapper89_f1444 a h)]
  rw [(publicBatchWrapper89_f1433 a h)]
  rw [(publicBatchWrapper89_f1422 a h)]
  rw [(publicBatchWrapper89_f1411 a h)]
  rw [(publicBatchWrapper89_f1400 a h)]
  rw [(publicBatchWrapper89_f1389 a h)]
  rw [(publicBatchWrapper89_f1378 a h)]
  rw [(publicBatchWrapper89_f1367 a h)]
  rw [(publicBatchWrapper89_f1356 a h)]
  rw [(publicBatchWrapper89_f1345 a h)]
  rw [(publicBatchWrapper89_f1334 a h)]
  rw [(publicBatchWrapper89_f1323 a h)]
  rw [(publicBatchWrapper89_f1312 a h)]
  rw [(publicBatchWrapper89_f1301 a h)]
  rw [(publicBatchWrapper89_f1290 a h)]
  rw [(publicBatchWrapper89_f1279 a h)]
  rw [(publicBatchWrapper89_f1268 a h)]
  rw [(publicBatchWrapper89_f1257 a h)]
  rw [(publicBatchWrapper89_f1246 a h)]
  rw [(publicBatchWrapper89_f1235 a h)]
  rw [(publicBatchWrapper89_f1224 a h)]
  rw [(publicBatchWrapper89_f1213 a h)]
  rw [(publicBatchWrapper89_f1202 a h)]
  rw [(publicBatchWrapper89_f1191 a h)]
  rw [(publicBatchWrapper89_f1180 a h)]
  rw [(publicBatchWrapper89_f1169 a h)]
  rw [(publicBatchWrapper89_f1158 a h)]
  rw [(publicBatchWrapper89_f1147 a h)]
  rw [(publicBatchWrapper89_f1136 a h)]
  rw [(publicBatchWrapper89_f1125 a h)]
  rw [(publicBatchWrapper89_f1114 a h)]
  rw [(publicBatchWrapper89_f1103 a h)]
  rw [(publicBatchWrapper89_f1092 a h)]
  rw [(publicBatchWrapper89_f1081 a h)]
  rw [(publicBatchWrapper89_f1070 a h)]
  rw [(publicBatchWrapper89_f1059 a h)]
  rw [(publicBatchWrapper89_f1048 a h)]
  rw [(publicBatchWrapper89_f1037 a h)]
  rw [(publicBatchWrapper89_f1026 a h)]
  rw [(publicBatchWrapper89_f1015 a h)]
  rw [(publicBatchWrapper89_f1004 a h)]
  rw [(publicBatchWrapper89_f993 a h)]
  rw [(publicBatchWrapper89_f982 a h)]
  rw [(publicBatchWrapper89_f971 a h)]
  rw [(publicBatchWrapper89_f960 a h)]
  rw [(publicBatchWrapper89_f949 a h)]
  rw [(publicBatchWrapper89_f938 a h)]
  rw [(publicBatchWrapper89_f927 a h)]
  rw [(publicBatchWrapper89_f916 a h)]
  rw [(publicBatchWrapper89_f905 a h)]
  rw [(publicBatchWrapper89_f894 a h)]
  rw [(publicBatchWrapper89_f883 a h)]
  rw [(publicBatchWrapper89_f872 a h)]
  rw [(publicBatchWrapper89_f861 a h)]
  rw [(publicBatchWrapper89_f850 a h)]
  rw [(publicBatchWrapper89_f839 a h)]
  rw [(publicBatchWrapper89_f828 a h)]
  rw [(publicBatchWrapper89_f817 a h)]
  rw [(publicBatchWrapper89_f806 a h)]
  rw [(publicBatchWrapper89_f795 a h)]
  rw [(publicBatchWrapper89_f784 a h)]
  rw [(publicBatchWrapper89_f773 a h)]
  rw [(publicBatchWrapper89_f762 a h)]
  rw [(publicBatchWrapper89_f751 a h)]
  rw [(publicBatchWrapper89_f740 a h)]
  rw [(publicBatchWrapper89_f729 a h)]
  rw [(publicBatchWrapper89_f718 a h)]
  rw [kzero, selChain_eq a h]
  rw [blockRef, rows, scanRef_89]
  rfl

end Plonky2Bridge.PublicWrapper89
