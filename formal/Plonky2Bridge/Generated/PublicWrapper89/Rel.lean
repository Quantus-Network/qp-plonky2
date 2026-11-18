import Plonky2Bridge.Generated.PublicWrapper89.Limb0
import Plonky2Bridge.Generated.PublicWrapper89.Limb1
import Plonky2Bridge.Generated.PublicWrapper89.Limb2
import Plonky2Bridge.Generated.PublicWrapper89.Limb3
import Plonky2Bridge.Generated.PublicWrapper89.Scalar

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

theorem dummyAt_0 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 0) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f89 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f90 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f91 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f92 a h); rw [kzero] at e; exact e


theorem assetAt_0 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 0).isDummy (row a 0).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 1 43) = (row a 0).isDummy := by rw [(publicBatchWrapper89_f95 a h), (publicBatchWrapper89_f93 a h), (publicBatchWrapper89_f94 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1691 a h), by rw [← hd, ← (publicBatchWrapper89_f1692 a h), (publicBatchWrapper89_f1693 a h), kone]⟩


theorem feeAt_0 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 0).isDummy (row a 0).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 1 43) = (row a 0).isDummy := by rw [(publicBatchWrapper89_f95 a h), (publicBatchWrapper89_f93 a h), (publicBatchWrapper89_f94 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1694 a h), by rw [← hd, ← (publicBatchWrapper89_f1695 a h), (publicBatchWrapper89_f1696 a h), kone]⟩


theorem blockAt_0 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 0).isDummy (row a 0).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 1 43) = (row a 0).isDummy := by rw [(publicBatchWrapper89_f95 a h), (publicBatchWrapper89_f93 a h), (publicBatchWrapper89_f94 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605478), a (.virt 605480), a (.virt 605482), a (.virt 605484)],
    ![a (.virt 605479), a (.virt 605481), a (.virt 605483), a (.virt 605485)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1697 a h)
    · exact (publicBatchWrapper89_f1698 a h)
    · exact (publicBatchWrapper89_f1699 a h)
    · exact (publicBatchWrapper89_f1700 a h)
  · show bor (row a 0).isDummy (band (band (a (.virt 605478)) (a (.virt 605480)))
      (band (a (.virt 605482)) (a (.virt 605484)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1701 a h), ← (publicBatchWrapper89_f1702 a h), ← (publicBatchWrapper89_f1703 a h), ← (publicBatchWrapper89_f1704 a h), (publicBatchWrapper89_f1705 a h), kone]


theorem countAt_0 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 0).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 6595)).val = 32
  rw [(publicBatchWrapper89_f0 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_1 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 1) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f96 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f97 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f98 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f99 a h); rw [kzero] at e; exact e


theorem assetAt_1 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 1).isDummy (row a 1).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 2 7) = (row a 1).isDummy := by rw [(publicBatchWrapper89_f102 a h), (publicBatchWrapper89_f100 a h), (publicBatchWrapper89_f101 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1706 a h), by rw [← hd, ← (publicBatchWrapper89_f1707 a h), (publicBatchWrapper89_f1708 a h), kone]⟩


theorem feeAt_1 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 1).isDummy (row a 1).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 2 7) = (row a 1).isDummy := by rw [(publicBatchWrapper89_f102 a h), (publicBatchWrapper89_f100 a h), (publicBatchWrapper89_f101 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1709 a h), by rw [← hd, ← (publicBatchWrapper89_f1710 a h), (publicBatchWrapper89_f1711 a h), kone]⟩


theorem blockAt_1 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 1).isDummy (row a 1).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 2 7) = (row a 1).isDummy := by rw [(publicBatchWrapper89_f102 a h), (publicBatchWrapper89_f100 a h), (publicBatchWrapper89_f101 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605490), a (.virt 605492), a (.virt 605494), a (.virt 605496)],
    ![a (.virt 605491), a (.virt 605493), a (.virt 605495), a (.virt 605497)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1712 a h)
    · exact (publicBatchWrapper89_f1713 a h)
    · exact (publicBatchWrapper89_f1714 a h)
    · exact (publicBatchWrapper89_f1715 a h)
  · show bor (row a 1).isDummy (band (band (a (.virt 605490)) (a (.virt 605492)))
      (band (a (.virt 605494)) (a (.virt 605496)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1716 a h), ← (publicBatchWrapper89_f1717 a h), ← (publicBatchWrapper89_f1718 a h), ← (publicBatchWrapper89_f1719 a h), (publicBatchWrapper89_f1720 a h), kone]


theorem countAt_1 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 1).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 13390)).val = 32
  rw [(publicBatchWrapper89_f1 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_2 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 2) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f103 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f104 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f105 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f106 a h); rw [kzero] at e; exact e


theorem assetAt_2 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 2).isDummy (row a 2).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 2 51) = (row a 2).isDummy := by rw [(publicBatchWrapper89_f109 a h), (publicBatchWrapper89_f107 a h), (publicBatchWrapper89_f108 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1721 a h), by rw [← hd, ← (publicBatchWrapper89_f1722 a h), (publicBatchWrapper89_f1723 a h), kone]⟩


theorem feeAt_2 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 2).isDummy (row a 2).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 2 51) = (row a 2).isDummy := by rw [(publicBatchWrapper89_f109 a h), (publicBatchWrapper89_f107 a h), (publicBatchWrapper89_f108 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1724 a h), by rw [← hd, ← (publicBatchWrapper89_f1725 a h), (publicBatchWrapper89_f1726 a h), kone]⟩


theorem blockAt_2 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 2).isDummy (row a 2).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 2 51) = (row a 2).isDummy := by rw [(publicBatchWrapper89_f109 a h), (publicBatchWrapper89_f107 a h), (publicBatchWrapper89_f108 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605502), a (.virt 605504), a (.virt 605506), a (.virt 605508)],
    ![a (.virt 605503), a (.virt 605505), a (.virt 605507), a (.virt 605509)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1727 a h)
    · exact (publicBatchWrapper89_f1728 a h)
    · exact (publicBatchWrapper89_f1729 a h)
    · exact (publicBatchWrapper89_f1730 a h)
  · show bor (row a 2).isDummy (band (band (a (.virt 605502)) (a (.virt 605504)))
      (band (a (.virt 605506)) (a (.virt 605508)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1731 a h), ← (publicBatchWrapper89_f1732 a h), ← (publicBatchWrapper89_f1733 a h), ← (publicBatchWrapper89_f1734 a h), (publicBatchWrapper89_f1735 a h), kone]


theorem countAt_2 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 2).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 20185)).val = 32
  rw [(publicBatchWrapper89_f2 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_3 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 3) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f110 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f111 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f112 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f113 a h); rw [kzero] at e; exact e


theorem assetAt_3 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 3).isDummy (row a 3).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 4 15) = (row a 3).isDummy := by rw [(publicBatchWrapper89_f116 a h), (publicBatchWrapper89_f114 a h), (publicBatchWrapper89_f115 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1736 a h), by rw [← hd, ← (publicBatchWrapper89_f1737 a h), (publicBatchWrapper89_f1738 a h), kone]⟩


theorem feeAt_3 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 3).isDummy (row a 3).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 4 15) = (row a 3).isDummy := by rw [(publicBatchWrapper89_f116 a h), (publicBatchWrapper89_f114 a h), (publicBatchWrapper89_f115 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1739 a h), by rw [← hd, ← (publicBatchWrapper89_f1740 a h), (publicBatchWrapper89_f1741 a h), kone]⟩


theorem blockAt_3 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 3).isDummy (row a 3).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 4 15) = (row a 3).isDummy := by rw [(publicBatchWrapper89_f116 a h), (publicBatchWrapper89_f114 a h), (publicBatchWrapper89_f115 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605514), a (.virt 605516), a (.virt 605518), a (.virt 605520)],
    ![a (.virt 605515), a (.virt 605517), a (.virt 605519), a (.virt 605521)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1742 a h)
    · exact (publicBatchWrapper89_f1743 a h)
    · exact (publicBatchWrapper89_f1744 a h)
    · exact (publicBatchWrapper89_f1745 a h)
  · show bor (row a 3).isDummy (band (band (a (.virt 605514)) (a (.virt 605516)))
      (band (a (.virt 605518)) (a (.virt 605520)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1746 a h), ← (publicBatchWrapper89_f1747 a h), ← (publicBatchWrapper89_f1748 a h), ← (publicBatchWrapper89_f1749 a h), (publicBatchWrapper89_f1750 a h), kone]


theorem countAt_3 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 3).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 26980)).val = 32
  rw [(publicBatchWrapper89_f3 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_4 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 4) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f117 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f118 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f119 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f120 a h); rw [kzero] at e; exact e


theorem assetAt_4 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 4).isDummy (row a 4).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 4 59) = (row a 4).isDummy := by rw [(publicBatchWrapper89_f123 a h), (publicBatchWrapper89_f121 a h), (publicBatchWrapper89_f122 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1751 a h), by rw [← hd, ← (publicBatchWrapper89_f1752 a h), (publicBatchWrapper89_f1753 a h), kone]⟩


theorem feeAt_4 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 4).isDummy (row a 4).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 4 59) = (row a 4).isDummy := by rw [(publicBatchWrapper89_f123 a h), (publicBatchWrapper89_f121 a h), (publicBatchWrapper89_f122 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1754 a h), by rw [← hd, ← (publicBatchWrapper89_f1755 a h), (publicBatchWrapper89_f1756 a h), kone]⟩


theorem blockAt_4 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 4).isDummy (row a 4).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 4 59) = (row a 4).isDummy := by rw [(publicBatchWrapper89_f123 a h), (publicBatchWrapper89_f121 a h), (publicBatchWrapper89_f122 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605526), a (.virt 605528), a (.virt 605530), a (.virt 605532)],
    ![a (.virt 605527), a (.virt 605529), a (.virt 605531), a (.virt 605533)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1757 a h)
    · exact (publicBatchWrapper89_f1758 a h)
    · exact (publicBatchWrapper89_f1759 a h)
    · exact (publicBatchWrapper89_f1760 a h)
  · show bor (row a 4).isDummy (band (band (a (.virt 605526)) (a (.virt 605528)))
      (band (a (.virt 605530)) (a (.virt 605532)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1761 a h), ← (publicBatchWrapper89_f1762 a h), ← (publicBatchWrapper89_f1763 a h), ← (publicBatchWrapper89_f1764 a h), (publicBatchWrapper89_f1765 a h), kone]


theorem countAt_4 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 4).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 33775)).val = 32
  rw [(publicBatchWrapper89_f4 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_5 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 5) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f124 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f125 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f126 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f127 a h); rw [kzero] at e; exact e


theorem assetAt_5 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 5).isDummy (row a 5).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 6 23) = (row a 5).isDummy := by rw [(publicBatchWrapper89_f130 a h), (publicBatchWrapper89_f128 a h), (publicBatchWrapper89_f129 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1766 a h), by rw [← hd, ← (publicBatchWrapper89_f1767 a h), (publicBatchWrapper89_f1768 a h), kone]⟩


theorem feeAt_5 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 5).isDummy (row a 5).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 6 23) = (row a 5).isDummy := by rw [(publicBatchWrapper89_f130 a h), (publicBatchWrapper89_f128 a h), (publicBatchWrapper89_f129 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1769 a h), by rw [← hd, ← (publicBatchWrapper89_f1770 a h), (publicBatchWrapper89_f1771 a h), kone]⟩


theorem blockAt_5 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 5).isDummy (row a 5).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 6 23) = (row a 5).isDummy := by rw [(publicBatchWrapper89_f130 a h), (publicBatchWrapper89_f128 a h), (publicBatchWrapper89_f129 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605538), a (.virt 605540), a (.virt 605542), a (.virt 605544)],
    ![a (.virt 605539), a (.virt 605541), a (.virt 605543), a (.virt 605545)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1772 a h)
    · exact (publicBatchWrapper89_f1773 a h)
    · exact (publicBatchWrapper89_f1774 a h)
    · exact (publicBatchWrapper89_f1775 a h)
  · show bor (row a 5).isDummy (band (band (a (.virt 605538)) (a (.virt 605540)))
      (band (a (.virt 605542)) (a (.virt 605544)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1776 a h), ← (publicBatchWrapper89_f1777 a h), ← (publicBatchWrapper89_f1778 a h), ← (publicBatchWrapper89_f1779 a h), (publicBatchWrapper89_f1780 a h), kone]


theorem countAt_5 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 5).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 40570)).val = 32
  rw [(publicBatchWrapper89_f5 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_6 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 6) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f131 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f132 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f133 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f134 a h); rw [kzero] at e; exact e


theorem assetAt_6 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 6).isDummy (row a 6).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 6 67) = (row a 6).isDummy := by rw [(publicBatchWrapper89_f137 a h), (publicBatchWrapper89_f135 a h), (publicBatchWrapper89_f136 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1781 a h), by rw [← hd, ← (publicBatchWrapper89_f1782 a h), (publicBatchWrapper89_f1783 a h), kone]⟩


theorem feeAt_6 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 6).isDummy (row a 6).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 6 67) = (row a 6).isDummy := by rw [(publicBatchWrapper89_f137 a h), (publicBatchWrapper89_f135 a h), (publicBatchWrapper89_f136 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1784 a h), by rw [← hd, ← (publicBatchWrapper89_f1785 a h), (publicBatchWrapper89_f1786 a h), kone]⟩


theorem blockAt_6 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 6).isDummy (row a 6).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 6 67) = (row a 6).isDummy := by rw [(publicBatchWrapper89_f137 a h), (publicBatchWrapper89_f135 a h), (publicBatchWrapper89_f136 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605550), a (.virt 605552), a (.virt 605554), a (.virt 605556)],
    ![a (.virt 605551), a (.virt 605553), a (.virt 605555), a (.virt 605557)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1787 a h)
    · exact (publicBatchWrapper89_f1788 a h)
    · exact (publicBatchWrapper89_f1789 a h)
    · exact (publicBatchWrapper89_f1790 a h)
  · show bor (row a 6).isDummy (band (band (a (.virt 605550)) (a (.virt 605552)))
      (band (a (.virt 605554)) (a (.virt 605556)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1791 a h), ← (publicBatchWrapper89_f1792 a h), ← (publicBatchWrapper89_f1793 a h), ← (publicBatchWrapper89_f1794 a h), (publicBatchWrapper89_f1795 a h), kone]


theorem countAt_6 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 6).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 47365)).val = 32
  rw [(publicBatchWrapper89_f6 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_7 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 7) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f138 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f139 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f140 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f141 a h); rw [kzero] at e; exact e


theorem assetAt_7 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 7).isDummy (row a 7).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 7 31) = (row a 7).isDummy := by rw [(publicBatchWrapper89_f144 a h), (publicBatchWrapper89_f142 a h), (publicBatchWrapper89_f143 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1796 a h), by rw [← hd, ← (publicBatchWrapper89_f1797 a h), (publicBatchWrapper89_f1798 a h), kone]⟩


theorem feeAt_7 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 7).isDummy (row a 7).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 7 31) = (row a 7).isDummy := by rw [(publicBatchWrapper89_f144 a h), (publicBatchWrapper89_f142 a h), (publicBatchWrapper89_f143 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1799 a h), by rw [← hd, ← (publicBatchWrapper89_f1800 a h), (publicBatchWrapper89_f1801 a h), kone]⟩


theorem blockAt_7 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 7).isDummy (row a 7).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 7 31) = (row a 7).isDummy := by rw [(publicBatchWrapper89_f144 a h), (publicBatchWrapper89_f142 a h), (publicBatchWrapper89_f143 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605562), a (.virt 605564), a (.virt 605566), a (.virt 605568)],
    ![a (.virt 605563), a (.virt 605565), a (.virt 605567), a (.virt 605569)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1802 a h)
    · exact (publicBatchWrapper89_f1803 a h)
    · exact (publicBatchWrapper89_f1804 a h)
    · exact (publicBatchWrapper89_f1805 a h)
  · show bor (row a 7).isDummy (band (band (a (.virt 605562)) (a (.virt 605564)))
      (band (a (.virt 605566)) (a (.virt 605568)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1806 a h), ← (publicBatchWrapper89_f1807 a h), ← (publicBatchWrapper89_f1808 a h), ← (publicBatchWrapper89_f1809 a h), (publicBatchWrapper89_f1810 a h), kone]


theorem countAt_7 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 7).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 54160)).val = 32
  rw [(publicBatchWrapper89_f7 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_8 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 8) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f145 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f146 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f147 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f148 a h); rw [kzero] at e; exact e


theorem assetAt_8 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 8).isDummy (row a 8).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 7 75) = (row a 8).isDummy := by rw [(publicBatchWrapper89_f151 a h), (publicBatchWrapper89_f149 a h), (publicBatchWrapper89_f150 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1811 a h), by rw [← hd, ← (publicBatchWrapper89_f1812 a h), (publicBatchWrapper89_f1813 a h), kone]⟩


theorem feeAt_8 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 8).isDummy (row a 8).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 7 75) = (row a 8).isDummy := by rw [(publicBatchWrapper89_f151 a h), (publicBatchWrapper89_f149 a h), (publicBatchWrapper89_f150 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1814 a h), by rw [← hd, ← (publicBatchWrapper89_f1815 a h), (publicBatchWrapper89_f1816 a h), kone]⟩


theorem blockAt_8 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 8).isDummy (row a 8).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 7 75) = (row a 8).isDummy := by rw [(publicBatchWrapper89_f151 a h), (publicBatchWrapper89_f149 a h), (publicBatchWrapper89_f150 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605574), a (.virt 605576), a (.virt 605578), a (.virt 605580)],
    ![a (.virt 605575), a (.virt 605577), a (.virt 605579), a (.virt 605581)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1817 a h)
    · exact (publicBatchWrapper89_f1818 a h)
    · exact (publicBatchWrapper89_f1819 a h)
    · exact (publicBatchWrapper89_f1820 a h)
  · show bor (row a 8).isDummy (band (band (a (.virt 605574)) (a (.virt 605576)))
      (band (a (.virt 605578)) (a (.virt 605580)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1821 a h), ← (publicBatchWrapper89_f1822 a h), ← (publicBatchWrapper89_f1823 a h), ← (publicBatchWrapper89_f1824 a h), (publicBatchWrapper89_f1825 a h), kone]


theorem countAt_8 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 8).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 60955)).val = 32
  rw [(publicBatchWrapper89_f8 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_9 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 9) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f152 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f153 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f154 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f155 a h); rw [kzero] at e; exact e


theorem assetAt_9 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 9).isDummy (row a 9).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 9 39) = (row a 9).isDummy := by rw [(publicBatchWrapper89_f158 a h), (publicBatchWrapper89_f156 a h), (publicBatchWrapper89_f157 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1826 a h), by rw [← hd, ← (publicBatchWrapper89_f1827 a h), (publicBatchWrapper89_f1828 a h), kone]⟩


theorem feeAt_9 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 9).isDummy (row a 9).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 9 39) = (row a 9).isDummy := by rw [(publicBatchWrapper89_f158 a h), (publicBatchWrapper89_f156 a h), (publicBatchWrapper89_f157 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1829 a h), by rw [← hd, ← (publicBatchWrapper89_f1830 a h), (publicBatchWrapper89_f1831 a h), kone]⟩


theorem blockAt_9 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 9).isDummy (row a 9).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 9 39) = (row a 9).isDummy := by rw [(publicBatchWrapper89_f158 a h), (publicBatchWrapper89_f156 a h), (publicBatchWrapper89_f157 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605586), a (.virt 605588), a (.virt 605590), a (.virt 605592)],
    ![a (.virt 605587), a (.virt 605589), a (.virt 605591), a (.virt 605593)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1832 a h)
    · exact (publicBatchWrapper89_f1833 a h)
    · exact (publicBatchWrapper89_f1834 a h)
    · exact (publicBatchWrapper89_f1835 a h)
  · show bor (row a 9).isDummy (band (band (a (.virt 605586)) (a (.virt 605588)))
      (band (a (.virt 605590)) (a (.virt 605592)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1836 a h), ← (publicBatchWrapper89_f1837 a h), ← (publicBatchWrapper89_f1838 a h), ← (publicBatchWrapper89_f1839 a h), (publicBatchWrapper89_f1840 a h), kone]


theorem countAt_9 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 9).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 67750)).val = 32
  rw [(publicBatchWrapper89_f9 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_10 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 10) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f159 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f160 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f161 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f162 a h); rw [kzero] at e; exact e


theorem assetAt_10 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 10).isDummy (row a 10).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 11 3) = (row a 10).isDummy := by rw [(publicBatchWrapper89_f165 a h), (publicBatchWrapper89_f163 a h), (publicBatchWrapper89_f164 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1841 a h), by rw [← hd, ← (publicBatchWrapper89_f1842 a h), (publicBatchWrapper89_f1843 a h), kone]⟩


theorem feeAt_10 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 10).isDummy (row a 10).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 11 3) = (row a 10).isDummy := by rw [(publicBatchWrapper89_f165 a h), (publicBatchWrapper89_f163 a h), (publicBatchWrapper89_f164 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1844 a h), by rw [← hd, ← (publicBatchWrapper89_f1845 a h), (publicBatchWrapper89_f1846 a h), kone]⟩


theorem blockAt_10 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 10).isDummy (row a 10).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 11 3) = (row a 10).isDummy := by rw [(publicBatchWrapper89_f165 a h), (publicBatchWrapper89_f163 a h), (publicBatchWrapper89_f164 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605598), a (.virt 605600), a (.virt 605602), a (.virt 605604)],
    ![a (.virt 605599), a (.virt 605601), a (.virt 605603), a (.virt 605605)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1847 a h)
    · exact (publicBatchWrapper89_f1848 a h)
    · exact (publicBatchWrapper89_f1849 a h)
    · exact (publicBatchWrapper89_f1850 a h)
  · show bor (row a 10).isDummy (band (band (a (.virt 605598)) (a (.virt 605600)))
      (band (a (.virt 605602)) (a (.virt 605604)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1851 a h), ← (publicBatchWrapper89_f1852 a h), ← (publicBatchWrapper89_f1853 a h), ← (publicBatchWrapper89_f1854 a h), (publicBatchWrapper89_f1855 a h), kone]


theorem countAt_10 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 10).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 74545)).val = 32
  rw [(publicBatchWrapper89_f10 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_11 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 11) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f166 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f167 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f168 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f169 a h); rw [kzero] at e; exact e


theorem assetAt_11 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 11).isDummy (row a 11).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 11 47) = (row a 11).isDummy := by rw [(publicBatchWrapper89_f172 a h), (publicBatchWrapper89_f170 a h), (publicBatchWrapper89_f171 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1856 a h), by rw [← hd, ← (publicBatchWrapper89_f1857 a h), (publicBatchWrapper89_f1858 a h), kone]⟩


theorem feeAt_11 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 11).isDummy (row a 11).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 11 47) = (row a 11).isDummy := by rw [(publicBatchWrapper89_f172 a h), (publicBatchWrapper89_f170 a h), (publicBatchWrapper89_f171 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1859 a h), by rw [← hd, ← (publicBatchWrapper89_f1860 a h), (publicBatchWrapper89_f1861 a h), kone]⟩


theorem blockAt_11 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 11).isDummy (row a 11).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 11 47) = (row a 11).isDummy := by rw [(publicBatchWrapper89_f172 a h), (publicBatchWrapper89_f170 a h), (publicBatchWrapper89_f171 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605610), a (.virt 605612), a (.virt 605614), a (.virt 605616)],
    ![a (.virt 605611), a (.virt 605613), a (.virt 605615), a (.virt 605617)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1862 a h)
    · exact (publicBatchWrapper89_f1863 a h)
    · exact (publicBatchWrapper89_f1864 a h)
    · exact (publicBatchWrapper89_f1865 a h)
  · show bor (row a 11).isDummy (band (band (a (.virt 605610)) (a (.virt 605612)))
      (band (a (.virt 605614)) (a (.virt 605616)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1866 a h), ← (publicBatchWrapper89_f1867 a h), ← (publicBatchWrapper89_f1868 a h), ← (publicBatchWrapper89_f1869 a h), (publicBatchWrapper89_f1870 a h), kone]


theorem countAt_11 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 11).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 81340)).val = 32
  rw [(publicBatchWrapper89_f11 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_12 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 12) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f173 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f174 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f175 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f176 a h); rw [kzero] at e; exact e


theorem assetAt_12 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 12).isDummy (row a 12).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 13 11) = (row a 12).isDummy := by rw [(publicBatchWrapper89_f179 a h), (publicBatchWrapper89_f177 a h), (publicBatchWrapper89_f178 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1871 a h), by rw [← hd, ← (publicBatchWrapper89_f1872 a h), (publicBatchWrapper89_f1873 a h), kone]⟩


theorem feeAt_12 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 12).isDummy (row a 12).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 13 11) = (row a 12).isDummy := by rw [(publicBatchWrapper89_f179 a h), (publicBatchWrapper89_f177 a h), (publicBatchWrapper89_f178 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1874 a h), by rw [← hd, ← (publicBatchWrapper89_f1875 a h), (publicBatchWrapper89_f1876 a h), kone]⟩


theorem blockAt_12 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 12).isDummy (row a 12).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 13 11) = (row a 12).isDummy := by rw [(publicBatchWrapper89_f179 a h), (publicBatchWrapper89_f177 a h), (publicBatchWrapper89_f178 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605622), a (.virt 605624), a (.virt 605626), a (.virt 605628)],
    ![a (.virt 605623), a (.virt 605625), a (.virt 605627), a (.virt 605629)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1877 a h)
    · exact (publicBatchWrapper89_f1878 a h)
    · exact (publicBatchWrapper89_f1879 a h)
    · exact (publicBatchWrapper89_f1880 a h)
  · show bor (row a 12).isDummy (band (band (a (.virt 605622)) (a (.virt 605624)))
      (band (a (.virt 605626)) (a (.virt 605628)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1881 a h), ← (publicBatchWrapper89_f1882 a h), ← (publicBatchWrapper89_f1883 a h), ← (publicBatchWrapper89_f1884 a h), (publicBatchWrapper89_f1885 a h), kone]


theorem countAt_12 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 12).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 88135)).val = 32
  rw [(publicBatchWrapper89_f12 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_13 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 13) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f180 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f181 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f182 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f183 a h); rw [kzero] at e; exact e


theorem assetAt_13 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 13).isDummy (row a 13).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 13 55) = (row a 13).isDummy := by rw [(publicBatchWrapper89_f186 a h), (publicBatchWrapper89_f184 a h), (publicBatchWrapper89_f185 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1886 a h), by rw [← hd, ← (publicBatchWrapper89_f1887 a h), (publicBatchWrapper89_f1888 a h), kone]⟩


theorem feeAt_13 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 13).isDummy (row a 13).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 13 55) = (row a 13).isDummy := by rw [(publicBatchWrapper89_f186 a h), (publicBatchWrapper89_f184 a h), (publicBatchWrapper89_f185 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1889 a h), by rw [← hd, ← (publicBatchWrapper89_f1890 a h), (publicBatchWrapper89_f1891 a h), kone]⟩


theorem blockAt_13 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 13).isDummy (row a 13).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 13 55) = (row a 13).isDummy := by rw [(publicBatchWrapper89_f186 a h), (publicBatchWrapper89_f184 a h), (publicBatchWrapper89_f185 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605634), a (.virt 605636), a (.virt 605638), a (.virt 605640)],
    ![a (.virt 605635), a (.virt 605637), a (.virt 605639), a (.virt 605641)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1892 a h)
    · exact (publicBatchWrapper89_f1893 a h)
    · exact (publicBatchWrapper89_f1894 a h)
    · exact (publicBatchWrapper89_f1895 a h)
  · show bor (row a 13).isDummy (band (band (a (.virt 605634)) (a (.virt 605636)))
      (band (a (.virt 605638)) (a (.virt 605640)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1896 a h), ← (publicBatchWrapper89_f1897 a h), ← (publicBatchWrapper89_f1898 a h), ← (publicBatchWrapper89_f1899 a h), (publicBatchWrapper89_f1900 a h), kone]


theorem countAt_13 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 13).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 94930)).val = 32
  rw [(publicBatchWrapper89_f13 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_14 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 14) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f187 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f188 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f189 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f190 a h); rw [kzero] at e; exact e


theorem assetAt_14 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 14).isDummy (row a 14).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 14 19) = (row a 14).isDummy := by rw [(publicBatchWrapper89_f193 a h), (publicBatchWrapper89_f191 a h), (publicBatchWrapper89_f192 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1901 a h), by rw [← hd, ← (publicBatchWrapper89_f1902 a h), (publicBatchWrapper89_f1903 a h), kone]⟩


theorem feeAt_14 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 14).isDummy (row a 14).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 14 19) = (row a 14).isDummy := by rw [(publicBatchWrapper89_f193 a h), (publicBatchWrapper89_f191 a h), (publicBatchWrapper89_f192 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1904 a h), by rw [← hd, ← (publicBatchWrapper89_f1905 a h), (publicBatchWrapper89_f1906 a h), kone]⟩


theorem blockAt_14 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 14).isDummy (row a 14).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 14 19) = (row a 14).isDummy := by rw [(publicBatchWrapper89_f193 a h), (publicBatchWrapper89_f191 a h), (publicBatchWrapper89_f192 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605646), a (.virt 605648), a (.virt 605650), a (.virt 605652)],
    ![a (.virt 605647), a (.virt 605649), a (.virt 605651), a (.virt 605653)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1907 a h)
    · exact (publicBatchWrapper89_f1908 a h)
    · exact (publicBatchWrapper89_f1909 a h)
    · exact (publicBatchWrapper89_f1910 a h)
  · show bor (row a 14).isDummy (band (band (a (.virt 605646)) (a (.virt 605648)))
      (band (a (.virt 605650)) (a (.virt 605652)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1911 a h), ← (publicBatchWrapper89_f1912 a h), ← (publicBatchWrapper89_f1913 a h), ← (publicBatchWrapper89_f1914 a h), (publicBatchWrapper89_f1915 a h), kone]


theorem countAt_14 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 14).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 101725)).val = 32
  rw [(publicBatchWrapper89_f14 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_15 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 15) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f194 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f195 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f196 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f197 a h); rw [kzero] at e; exact e


theorem assetAt_15 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 15).isDummy (row a 15).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 14 63) = (row a 15).isDummy := by rw [(publicBatchWrapper89_f200 a h), (publicBatchWrapper89_f198 a h), (publicBatchWrapper89_f199 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1916 a h), by rw [← hd, ← (publicBatchWrapper89_f1917 a h), (publicBatchWrapper89_f1918 a h), kone]⟩


theorem feeAt_15 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 15).isDummy (row a 15).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 14 63) = (row a 15).isDummy := by rw [(publicBatchWrapper89_f200 a h), (publicBatchWrapper89_f198 a h), (publicBatchWrapper89_f199 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1919 a h), by rw [← hd, ← (publicBatchWrapper89_f1920 a h), (publicBatchWrapper89_f1921 a h), kone]⟩


theorem blockAt_15 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 15).isDummy (row a 15).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 14 63) = (row a 15).isDummy := by rw [(publicBatchWrapper89_f200 a h), (publicBatchWrapper89_f198 a h), (publicBatchWrapper89_f199 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605658), a (.virt 605660), a (.virt 605662), a (.virt 605664)],
    ![a (.virt 605659), a (.virt 605661), a (.virt 605663), a (.virt 605665)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1922 a h)
    · exact (publicBatchWrapper89_f1923 a h)
    · exact (publicBatchWrapper89_f1924 a h)
    · exact (publicBatchWrapper89_f1925 a h)
  · show bor (row a 15).isDummy (band (band (a (.virt 605658)) (a (.virt 605660)))
      (band (a (.virt 605662)) (a (.virt 605664)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1926 a h), ← (publicBatchWrapper89_f1927 a h), ← (publicBatchWrapper89_f1928 a h), ← (publicBatchWrapper89_f1929 a h), (publicBatchWrapper89_f1930 a h), kone]


theorem countAt_15 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 15).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 108520)).val = 32
  rw [(publicBatchWrapper89_f15 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_16 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 16) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f201 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f202 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f203 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f204 a h); rw [kzero] at e; exact e


theorem assetAt_16 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 16).isDummy (row a 16).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 16 27) = (row a 16).isDummy := by rw [(publicBatchWrapper89_f207 a h), (publicBatchWrapper89_f205 a h), (publicBatchWrapper89_f206 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1931 a h), by rw [← hd, ← (publicBatchWrapper89_f1932 a h), (publicBatchWrapper89_f1933 a h), kone]⟩


theorem feeAt_16 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 16).isDummy (row a 16).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 16 27) = (row a 16).isDummy := by rw [(publicBatchWrapper89_f207 a h), (publicBatchWrapper89_f205 a h), (publicBatchWrapper89_f206 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1934 a h), by rw [← hd, ← (publicBatchWrapper89_f1935 a h), (publicBatchWrapper89_f1936 a h), kone]⟩


theorem blockAt_16 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 16).isDummy (row a 16).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 16 27) = (row a 16).isDummy := by rw [(publicBatchWrapper89_f207 a h), (publicBatchWrapper89_f205 a h), (publicBatchWrapper89_f206 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605670), a (.virt 605672), a (.virt 605674), a (.virt 605676)],
    ![a (.virt 605671), a (.virt 605673), a (.virt 605675), a (.virt 605677)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1937 a h)
    · exact (publicBatchWrapper89_f1938 a h)
    · exact (publicBatchWrapper89_f1939 a h)
    · exact (publicBatchWrapper89_f1940 a h)
  · show bor (row a 16).isDummy (band (band (a (.virt 605670)) (a (.virt 605672)))
      (band (a (.virt 605674)) (a (.virt 605676)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1941 a h), ← (publicBatchWrapper89_f1942 a h), ← (publicBatchWrapper89_f1943 a h), ← (publicBatchWrapper89_f1944 a h), (publicBatchWrapper89_f1945 a h), kone]


theorem countAt_16 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 16).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 115315)).val = 32
  rw [(publicBatchWrapper89_f16 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_17 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 17) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f208 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f209 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f210 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f211 a h); rw [kzero] at e; exact e


theorem assetAt_17 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 17).isDummy (row a 17).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 16 71) = (row a 17).isDummy := by rw [(publicBatchWrapper89_f214 a h), (publicBatchWrapper89_f212 a h), (publicBatchWrapper89_f213 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1946 a h), by rw [← hd, ← (publicBatchWrapper89_f1947 a h), (publicBatchWrapper89_f1948 a h), kone]⟩


theorem feeAt_17 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 17).isDummy (row a 17).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 16 71) = (row a 17).isDummy := by rw [(publicBatchWrapper89_f214 a h), (publicBatchWrapper89_f212 a h), (publicBatchWrapper89_f213 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1949 a h), by rw [← hd, ← (publicBatchWrapper89_f1950 a h), (publicBatchWrapper89_f1951 a h), kone]⟩


theorem blockAt_17 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 17).isDummy (row a 17).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 16 71) = (row a 17).isDummy := by rw [(publicBatchWrapper89_f214 a h), (publicBatchWrapper89_f212 a h), (publicBatchWrapper89_f213 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605682), a (.virt 605684), a (.virt 605686), a (.virt 605688)],
    ![a (.virt 605683), a (.virt 605685), a (.virt 605687), a (.virt 605689)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1952 a h)
    · exact (publicBatchWrapper89_f1953 a h)
    · exact (publicBatchWrapper89_f1954 a h)
    · exact (publicBatchWrapper89_f1955 a h)
  · show bor (row a 17).isDummy (band (band (a (.virt 605682)) (a (.virt 605684)))
      (band (a (.virt 605686)) (a (.virt 605688)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1956 a h), ← (publicBatchWrapper89_f1957 a h), ← (publicBatchWrapper89_f1958 a h), ← (publicBatchWrapper89_f1959 a h), (publicBatchWrapper89_f1960 a h), kone]


theorem countAt_17 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 17).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 122110)).val = 32
  rw [(publicBatchWrapper89_f17 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_18 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 18) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f215 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f216 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f217 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f218 a h); rw [kzero] at e; exact e


theorem assetAt_18 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 18).isDummy (row a 18).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 18 35) = (row a 18).isDummy := by rw [(publicBatchWrapper89_f221 a h), (publicBatchWrapper89_f219 a h), (publicBatchWrapper89_f220 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1961 a h), by rw [← hd, ← (publicBatchWrapper89_f1962 a h), (publicBatchWrapper89_f1963 a h), kone]⟩


theorem feeAt_18 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 18).isDummy (row a 18).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 18 35) = (row a 18).isDummy := by rw [(publicBatchWrapper89_f221 a h), (publicBatchWrapper89_f219 a h), (publicBatchWrapper89_f220 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1964 a h), by rw [← hd, ← (publicBatchWrapper89_f1965 a h), (publicBatchWrapper89_f1966 a h), kone]⟩


theorem blockAt_18 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 18).isDummy (row a 18).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 18 35) = (row a 18).isDummy := by rw [(publicBatchWrapper89_f221 a h), (publicBatchWrapper89_f219 a h), (publicBatchWrapper89_f220 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605694), a (.virt 605696), a (.virt 605698), a (.virt 605700)],
    ![a (.virt 605695), a (.virt 605697), a (.virt 605699), a (.virt 605701)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1967 a h)
    · exact (publicBatchWrapper89_f1968 a h)
    · exact (publicBatchWrapper89_f1969 a h)
    · exact (publicBatchWrapper89_f1970 a h)
  · show bor (row a 18).isDummy (band (band (a (.virt 605694)) (a (.virt 605696)))
      (band (a (.virt 605698)) (a (.virt 605700)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1971 a h), ← (publicBatchWrapper89_f1972 a h), ← (publicBatchWrapper89_f1973 a h), ← (publicBatchWrapper89_f1974 a h), (publicBatchWrapper89_f1975 a h), kone]


theorem countAt_18 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 18).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 128905)).val = 32
  rw [(publicBatchWrapper89_f18 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_19 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 19) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f222 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f223 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f224 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f225 a h); rw [kzero] at e; exact e


theorem assetAt_19 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 19).isDummy (row a 19).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 18 79) = (row a 19).isDummy := by rw [(publicBatchWrapper89_f228 a h), (publicBatchWrapper89_f226 a h), (publicBatchWrapper89_f227 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1976 a h), by rw [← hd, ← (publicBatchWrapper89_f1977 a h), (publicBatchWrapper89_f1978 a h), kone]⟩


theorem feeAt_19 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 19).isDummy (row a 19).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 18 79) = (row a 19).isDummy := by rw [(publicBatchWrapper89_f228 a h), (publicBatchWrapper89_f226 a h), (publicBatchWrapper89_f227 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1979 a h), by rw [← hd, ← (publicBatchWrapper89_f1980 a h), (publicBatchWrapper89_f1981 a h), kone]⟩


theorem blockAt_19 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 19).isDummy (row a 19).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 18 79) = (row a 19).isDummy := by rw [(publicBatchWrapper89_f228 a h), (publicBatchWrapper89_f226 a h), (publicBatchWrapper89_f227 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605706), a (.virt 605708), a (.virt 605710), a (.virt 605712)],
    ![a (.virt 605707), a (.virt 605709), a (.virt 605711), a (.virt 605713)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1982 a h)
    · exact (publicBatchWrapper89_f1983 a h)
    · exact (publicBatchWrapper89_f1984 a h)
    · exact (publicBatchWrapper89_f1985 a h)
  · show bor (row a 19).isDummy (band (band (a (.virt 605706)) (a (.virt 605708)))
      (band (a (.virt 605710)) (a (.virt 605712)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1986 a h), ← (publicBatchWrapper89_f1987 a h), ← (publicBatchWrapper89_f1988 a h), ← (publicBatchWrapper89_f1989 a h), (publicBatchWrapper89_f1990 a h), kone]


theorem countAt_19 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 19).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 135700)).val = 32
  rw [(publicBatchWrapper89_f19 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_20 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 20) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f229 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f230 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f231 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f232 a h); rw [kzero] at e; exact e


theorem assetAt_20 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 20).isDummy (row a 20).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 20 43) = (row a 20).isDummy := by rw [(publicBatchWrapper89_f235 a h), (publicBatchWrapper89_f233 a h), (publicBatchWrapper89_f234 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1991 a h), by rw [← hd, ← (publicBatchWrapper89_f1992 a h), (publicBatchWrapper89_f1993 a h), kone]⟩


theorem feeAt_20 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 20).isDummy (row a 20).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 20 43) = (row a 20).isDummy := by rw [(publicBatchWrapper89_f235 a h), (publicBatchWrapper89_f233 a h), (publicBatchWrapper89_f234 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1994 a h), by rw [← hd, ← (publicBatchWrapper89_f1995 a h), (publicBatchWrapper89_f1996 a h), kone]⟩


theorem blockAt_20 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 20).isDummy (row a 20).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 20 43) = (row a 20).isDummy := by rw [(publicBatchWrapper89_f235 a h), (publicBatchWrapper89_f233 a h), (publicBatchWrapper89_f234 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605718), a (.virt 605720), a (.virt 605722), a (.virt 605724)],
    ![a (.virt 605719), a (.virt 605721), a (.virt 605723), a (.virt 605725)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1997 a h)
    · exact (publicBatchWrapper89_f1998 a h)
    · exact (publicBatchWrapper89_f1999 a h)
    · exact (publicBatchWrapper89_f2000 a h)
  · show bor (row a 20).isDummy (band (band (a (.virt 605718)) (a (.virt 605720)))
      (band (a (.virt 605722)) (a (.virt 605724)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2001 a h), ← (publicBatchWrapper89_f2002 a h), ← (publicBatchWrapper89_f2003 a h), ← (publicBatchWrapper89_f2004 a h), (publicBatchWrapper89_f2005 a h), kone]


theorem countAt_20 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 20).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 142495)).val = 32
  rw [(publicBatchWrapper89_f20 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_21 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 21) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f236 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f237 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f238 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f239 a h); rw [kzero] at e; exact e


theorem assetAt_21 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 21).isDummy (row a 21).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 21 7) = (row a 21).isDummy := by rw [(publicBatchWrapper89_f242 a h), (publicBatchWrapper89_f240 a h), (publicBatchWrapper89_f241 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2006 a h), by rw [← hd, ← (publicBatchWrapper89_f2007 a h), (publicBatchWrapper89_f2008 a h), kone]⟩


theorem feeAt_21 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 21).isDummy (row a 21).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 21 7) = (row a 21).isDummy := by rw [(publicBatchWrapper89_f242 a h), (publicBatchWrapper89_f240 a h), (publicBatchWrapper89_f241 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2009 a h), by rw [← hd, ← (publicBatchWrapper89_f2010 a h), (publicBatchWrapper89_f2011 a h), kone]⟩


theorem blockAt_21 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 21).isDummy (row a 21).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 21 7) = (row a 21).isDummy := by rw [(publicBatchWrapper89_f242 a h), (publicBatchWrapper89_f240 a h), (publicBatchWrapper89_f241 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605730), a (.virt 605732), a (.virt 605734), a (.virt 605736)],
    ![a (.virt 605731), a (.virt 605733), a (.virt 605735), a (.virt 605737)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2012 a h)
    · exact (publicBatchWrapper89_f2013 a h)
    · exact (publicBatchWrapper89_f2014 a h)
    · exact (publicBatchWrapper89_f2015 a h)
  · show bor (row a 21).isDummy (band (band (a (.virt 605730)) (a (.virt 605732)))
      (band (a (.virt 605734)) (a (.virt 605736)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2016 a h), ← (publicBatchWrapper89_f2017 a h), ← (publicBatchWrapper89_f2018 a h), ← (publicBatchWrapper89_f2019 a h), (publicBatchWrapper89_f2020 a h), kone]


theorem countAt_21 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 21).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 149290)).val = 32
  rw [(publicBatchWrapper89_f21 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_22 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 22) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f243 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f244 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f245 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f246 a h); rw [kzero] at e; exact e


theorem assetAt_22 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 22).isDummy (row a 22).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 21 51) = (row a 22).isDummy := by rw [(publicBatchWrapper89_f249 a h), (publicBatchWrapper89_f247 a h), (publicBatchWrapper89_f248 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2021 a h), by rw [← hd, ← (publicBatchWrapper89_f2022 a h), (publicBatchWrapper89_f2023 a h), kone]⟩


theorem feeAt_22 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 22).isDummy (row a 22).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 21 51) = (row a 22).isDummy := by rw [(publicBatchWrapper89_f249 a h), (publicBatchWrapper89_f247 a h), (publicBatchWrapper89_f248 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2024 a h), by rw [← hd, ← (publicBatchWrapper89_f2025 a h), (publicBatchWrapper89_f2026 a h), kone]⟩


theorem blockAt_22 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 22).isDummy (row a 22).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 21 51) = (row a 22).isDummy := by rw [(publicBatchWrapper89_f249 a h), (publicBatchWrapper89_f247 a h), (publicBatchWrapper89_f248 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605742), a (.virt 605744), a (.virt 605746), a (.virt 605748)],
    ![a (.virt 605743), a (.virt 605745), a (.virt 605747), a (.virt 605749)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2027 a h)
    · exact (publicBatchWrapper89_f2028 a h)
    · exact (publicBatchWrapper89_f2029 a h)
    · exact (publicBatchWrapper89_f2030 a h)
  · show bor (row a 22).isDummy (band (band (a (.virt 605742)) (a (.virt 605744)))
      (band (a (.virt 605746)) (a (.virt 605748)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2031 a h), ← (publicBatchWrapper89_f2032 a h), ← (publicBatchWrapper89_f2033 a h), ← (publicBatchWrapper89_f2034 a h), (publicBatchWrapper89_f2035 a h), kone]


theorem countAt_22 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 22).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 156085)).val = 32
  rw [(publicBatchWrapper89_f22 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_23 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 23) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f250 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f251 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f252 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f253 a h); rw [kzero] at e; exact e


theorem assetAt_23 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 23).isDummy (row a 23).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 23 15) = (row a 23).isDummy := by rw [(publicBatchWrapper89_f256 a h), (publicBatchWrapper89_f254 a h), (publicBatchWrapper89_f255 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2036 a h), by rw [← hd, ← (publicBatchWrapper89_f2037 a h), (publicBatchWrapper89_f2038 a h), kone]⟩


theorem feeAt_23 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 23).isDummy (row a 23).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 23 15) = (row a 23).isDummy := by rw [(publicBatchWrapper89_f256 a h), (publicBatchWrapper89_f254 a h), (publicBatchWrapper89_f255 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2039 a h), by rw [← hd, ← (publicBatchWrapper89_f2040 a h), (publicBatchWrapper89_f2041 a h), kone]⟩


theorem blockAt_23 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 23).isDummy (row a 23).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 23 15) = (row a 23).isDummy := by rw [(publicBatchWrapper89_f256 a h), (publicBatchWrapper89_f254 a h), (publicBatchWrapper89_f255 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605754), a (.virt 605756), a (.virt 605758), a (.virt 605760)],
    ![a (.virt 605755), a (.virt 605757), a (.virt 605759), a (.virt 605761)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2042 a h)
    · exact (publicBatchWrapper89_f2043 a h)
    · exact (publicBatchWrapper89_f2044 a h)
    · exact (publicBatchWrapper89_f2045 a h)
  · show bor (row a 23).isDummy (band (band (a (.virt 605754)) (a (.virt 605756)))
      (band (a (.virt 605758)) (a (.virt 605760)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2046 a h), ← (publicBatchWrapper89_f2047 a h), ← (publicBatchWrapper89_f2048 a h), ← (publicBatchWrapper89_f2049 a h), (publicBatchWrapper89_f2050 a h), kone]


theorem countAt_23 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 23).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 162880)).val = 32
  rw [(publicBatchWrapper89_f23 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_24 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 24) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f257 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f258 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f259 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f260 a h); rw [kzero] at e; exact e


theorem assetAt_24 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 24).isDummy (row a 24).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 23 59) = (row a 24).isDummy := by rw [(publicBatchWrapper89_f263 a h), (publicBatchWrapper89_f261 a h), (publicBatchWrapper89_f262 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2051 a h), by rw [← hd, ← (publicBatchWrapper89_f2052 a h), (publicBatchWrapper89_f2053 a h), kone]⟩


theorem feeAt_24 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 24).isDummy (row a 24).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 23 59) = (row a 24).isDummy := by rw [(publicBatchWrapper89_f263 a h), (publicBatchWrapper89_f261 a h), (publicBatchWrapper89_f262 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2054 a h), by rw [← hd, ← (publicBatchWrapper89_f2055 a h), (publicBatchWrapper89_f2056 a h), kone]⟩


theorem blockAt_24 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 24).isDummy (row a 24).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 23 59) = (row a 24).isDummy := by rw [(publicBatchWrapper89_f263 a h), (publicBatchWrapper89_f261 a h), (publicBatchWrapper89_f262 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605766), a (.virt 605768), a (.virt 605770), a (.virt 605772)],
    ![a (.virt 605767), a (.virt 605769), a (.virt 605771), a (.virt 605773)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2057 a h)
    · exact (publicBatchWrapper89_f2058 a h)
    · exact (publicBatchWrapper89_f2059 a h)
    · exact (publicBatchWrapper89_f2060 a h)
  · show bor (row a 24).isDummy (band (band (a (.virt 605766)) (a (.virt 605768)))
      (band (a (.virt 605770)) (a (.virt 605772)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2061 a h), ← (publicBatchWrapper89_f2062 a h), ← (publicBatchWrapper89_f2063 a h), ← (publicBatchWrapper89_f2064 a h), (publicBatchWrapper89_f2065 a h), kone]


theorem countAt_24 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 24).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 169675)).val = 32
  rw [(publicBatchWrapper89_f24 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_25 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 25) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f264 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f265 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f266 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f267 a h); rw [kzero] at e; exact e


theorem assetAt_25 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 25).isDummy (row a 25).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 25 23) = (row a 25).isDummy := by rw [(publicBatchWrapper89_f270 a h), (publicBatchWrapper89_f268 a h), (publicBatchWrapper89_f269 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2066 a h), by rw [← hd, ← (publicBatchWrapper89_f2067 a h), (publicBatchWrapper89_f2068 a h), kone]⟩


theorem feeAt_25 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 25).isDummy (row a 25).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 25 23) = (row a 25).isDummy := by rw [(publicBatchWrapper89_f270 a h), (publicBatchWrapper89_f268 a h), (publicBatchWrapper89_f269 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2069 a h), by rw [← hd, ← (publicBatchWrapper89_f2070 a h), (publicBatchWrapper89_f2071 a h), kone]⟩


theorem blockAt_25 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 25).isDummy (row a 25).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 25 23) = (row a 25).isDummy := by rw [(publicBatchWrapper89_f270 a h), (publicBatchWrapper89_f268 a h), (publicBatchWrapper89_f269 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605778), a (.virt 605780), a (.virt 605782), a (.virt 605784)],
    ![a (.virt 605779), a (.virt 605781), a (.virt 605783), a (.virt 605785)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2072 a h)
    · exact (publicBatchWrapper89_f2073 a h)
    · exact (publicBatchWrapper89_f2074 a h)
    · exact (publicBatchWrapper89_f2075 a h)
  · show bor (row a 25).isDummy (band (band (a (.virt 605778)) (a (.virt 605780)))
      (band (a (.virt 605782)) (a (.virt 605784)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2076 a h), ← (publicBatchWrapper89_f2077 a h), ← (publicBatchWrapper89_f2078 a h), ← (publicBatchWrapper89_f2079 a h), (publicBatchWrapper89_f2080 a h), kone]


theorem countAt_25 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 25).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 176470)).val = 32
  rw [(publicBatchWrapper89_f25 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_26 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 26) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f271 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f272 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f273 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f274 a h); rw [kzero] at e; exact e


theorem assetAt_26 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 26).isDummy (row a 26).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 25 67) = (row a 26).isDummy := by rw [(publicBatchWrapper89_f277 a h), (publicBatchWrapper89_f275 a h), (publicBatchWrapper89_f276 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2081 a h), by rw [← hd, ← (publicBatchWrapper89_f2082 a h), (publicBatchWrapper89_f2083 a h), kone]⟩


theorem feeAt_26 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 26).isDummy (row a 26).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 25 67) = (row a 26).isDummy := by rw [(publicBatchWrapper89_f277 a h), (publicBatchWrapper89_f275 a h), (publicBatchWrapper89_f276 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2084 a h), by rw [← hd, ← (publicBatchWrapper89_f2085 a h), (publicBatchWrapper89_f2086 a h), kone]⟩


theorem blockAt_26 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 26).isDummy (row a 26).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 25 67) = (row a 26).isDummy := by rw [(publicBatchWrapper89_f277 a h), (publicBatchWrapper89_f275 a h), (publicBatchWrapper89_f276 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605790), a (.virt 605792), a (.virt 605794), a (.virt 605796)],
    ![a (.virt 605791), a (.virt 605793), a (.virt 605795), a (.virt 605797)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2087 a h)
    · exact (publicBatchWrapper89_f2088 a h)
    · exact (publicBatchWrapper89_f2089 a h)
    · exact (publicBatchWrapper89_f2090 a h)
  · show bor (row a 26).isDummy (band (band (a (.virt 605790)) (a (.virt 605792)))
      (band (a (.virt 605794)) (a (.virt 605796)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2091 a h), ← (publicBatchWrapper89_f2092 a h), ← (publicBatchWrapper89_f2093 a h), ← (publicBatchWrapper89_f2094 a h), (publicBatchWrapper89_f2095 a h), kone]


theorem countAt_26 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 26).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 183265)).val = 32
  rw [(publicBatchWrapper89_f26 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_27 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 27) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f278 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f279 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f280 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f281 a h); rw [kzero] at e; exact e


theorem assetAt_27 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 27).isDummy (row a 27).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 26 31) = (row a 27).isDummy := by rw [(publicBatchWrapper89_f284 a h), (publicBatchWrapper89_f282 a h), (publicBatchWrapper89_f283 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2096 a h), by rw [← hd, ← (publicBatchWrapper89_f2097 a h), (publicBatchWrapper89_f2098 a h), kone]⟩


theorem feeAt_27 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 27).isDummy (row a 27).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 26 31) = (row a 27).isDummy := by rw [(publicBatchWrapper89_f284 a h), (publicBatchWrapper89_f282 a h), (publicBatchWrapper89_f283 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2099 a h), by rw [← hd, ← (publicBatchWrapper89_f2100 a h), (publicBatchWrapper89_f2101 a h), kone]⟩


theorem blockAt_27 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 27).isDummy (row a 27).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 26 31) = (row a 27).isDummy := by rw [(publicBatchWrapper89_f284 a h), (publicBatchWrapper89_f282 a h), (publicBatchWrapper89_f283 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605802), a (.virt 605804), a (.virt 605806), a (.virt 605808)],
    ![a (.virt 605803), a (.virt 605805), a (.virt 605807), a (.virt 605809)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2102 a h)
    · exact (publicBatchWrapper89_f2103 a h)
    · exact (publicBatchWrapper89_f2104 a h)
    · exact (publicBatchWrapper89_f2105 a h)
  · show bor (row a 27).isDummy (band (band (a (.virt 605802)) (a (.virt 605804)))
      (band (a (.virt 605806)) (a (.virt 605808)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2106 a h), ← (publicBatchWrapper89_f2107 a h), ← (publicBatchWrapper89_f2108 a h), ← (publicBatchWrapper89_f2109 a h), (publicBatchWrapper89_f2110 a h), kone]


theorem countAt_27 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 27).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 190060)).val = 32
  rw [(publicBatchWrapper89_f27 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_28 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 28) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f285 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f286 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f287 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f288 a h); rw [kzero] at e; exact e


theorem assetAt_28 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 28).isDummy (row a 28).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 26 75) = (row a 28).isDummy := by rw [(publicBatchWrapper89_f291 a h), (publicBatchWrapper89_f289 a h), (publicBatchWrapper89_f290 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2111 a h), by rw [← hd, ← (publicBatchWrapper89_f2112 a h), (publicBatchWrapper89_f2113 a h), kone]⟩


theorem feeAt_28 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 28).isDummy (row a 28).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 26 75) = (row a 28).isDummy := by rw [(publicBatchWrapper89_f291 a h), (publicBatchWrapper89_f289 a h), (publicBatchWrapper89_f290 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2114 a h), by rw [← hd, ← (publicBatchWrapper89_f2115 a h), (publicBatchWrapper89_f2116 a h), kone]⟩


theorem blockAt_28 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 28).isDummy (row a 28).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 26 75) = (row a 28).isDummy := by rw [(publicBatchWrapper89_f291 a h), (publicBatchWrapper89_f289 a h), (publicBatchWrapper89_f290 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605814), a (.virt 605816), a (.virt 605818), a (.virt 605820)],
    ![a (.virt 605815), a (.virt 605817), a (.virt 605819), a (.virt 605821)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2117 a h)
    · exact (publicBatchWrapper89_f2118 a h)
    · exact (publicBatchWrapper89_f2119 a h)
    · exact (publicBatchWrapper89_f2120 a h)
  · show bor (row a 28).isDummy (band (band (a (.virt 605814)) (a (.virt 605816)))
      (band (a (.virt 605818)) (a (.virt 605820)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2121 a h), ← (publicBatchWrapper89_f2122 a h), ← (publicBatchWrapper89_f2123 a h), ← (publicBatchWrapper89_f2124 a h), (publicBatchWrapper89_f2125 a h), kone]


theorem countAt_28 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 28).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 196855)).val = 32
  rw [(publicBatchWrapper89_f28 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_29 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 29) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f292 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f293 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f294 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f295 a h); rw [kzero] at e; exact e


theorem assetAt_29 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 29).isDummy (row a 29).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 28 39) = (row a 29).isDummy := by rw [(publicBatchWrapper89_f298 a h), (publicBatchWrapper89_f296 a h), (publicBatchWrapper89_f297 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2126 a h), by rw [← hd, ← (publicBatchWrapper89_f2127 a h), (publicBatchWrapper89_f2128 a h), kone]⟩


theorem feeAt_29 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 29).isDummy (row a 29).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 28 39) = (row a 29).isDummy := by rw [(publicBatchWrapper89_f298 a h), (publicBatchWrapper89_f296 a h), (publicBatchWrapper89_f297 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2129 a h), by rw [← hd, ← (publicBatchWrapper89_f2130 a h), (publicBatchWrapper89_f2131 a h), kone]⟩


theorem blockAt_29 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 29).isDummy (row a 29).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 28 39) = (row a 29).isDummy := by rw [(publicBatchWrapper89_f298 a h), (publicBatchWrapper89_f296 a h), (publicBatchWrapper89_f297 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605826), a (.virt 605828), a (.virt 605830), a (.virt 605832)],
    ![a (.virt 605827), a (.virt 605829), a (.virt 605831), a (.virt 605833)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2132 a h)
    · exact (publicBatchWrapper89_f2133 a h)
    · exact (publicBatchWrapper89_f2134 a h)
    · exact (publicBatchWrapper89_f2135 a h)
  · show bor (row a 29).isDummy (band (band (a (.virt 605826)) (a (.virt 605828)))
      (band (a (.virt 605830)) (a (.virt 605832)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2136 a h), ← (publicBatchWrapper89_f2137 a h), ← (publicBatchWrapper89_f2138 a h), ← (publicBatchWrapper89_f2139 a h), (publicBatchWrapper89_f2140 a h), kone]


theorem countAt_29 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 29).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 203650)).val = 32
  rw [(publicBatchWrapper89_f29 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_30 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 30) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f299 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f300 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f301 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f302 a h); rw [kzero] at e; exact e


theorem assetAt_30 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 30).isDummy (row a 30).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 30 3) = (row a 30).isDummy := by rw [(publicBatchWrapper89_f305 a h), (publicBatchWrapper89_f303 a h), (publicBatchWrapper89_f304 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2141 a h), by rw [← hd, ← (publicBatchWrapper89_f2142 a h), (publicBatchWrapper89_f2143 a h), kone]⟩


theorem feeAt_30 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 30).isDummy (row a 30).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 30 3) = (row a 30).isDummy := by rw [(publicBatchWrapper89_f305 a h), (publicBatchWrapper89_f303 a h), (publicBatchWrapper89_f304 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2144 a h), by rw [← hd, ← (publicBatchWrapper89_f2145 a h), (publicBatchWrapper89_f2146 a h), kone]⟩


theorem blockAt_30 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 30).isDummy (row a 30).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 30 3) = (row a 30).isDummy := by rw [(publicBatchWrapper89_f305 a h), (publicBatchWrapper89_f303 a h), (publicBatchWrapper89_f304 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605838), a (.virt 605840), a (.virt 605842), a (.virt 605844)],
    ![a (.virt 605839), a (.virt 605841), a (.virt 605843), a (.virt 605845)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2147 a h)
    · exact (publicBatchWrapper89_f2148 a h)
    · exact (publicBatchWrapper89_f2149 a h)
    · exact (publicBatchWrapper89_f2150 a h)
  · show bor (row a 30).isDummy (band (band (a (.virt 605838)) (a (.virt 605840)))
      (band (a (.virt 605842)) (a (.virt 605844)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2151 a h), ← (publicBatchWrapper89_f2152 a h), ← (publicBatchWrapper89_f2153 a h), ← (publicBatchWrapper89_f2154 a h), (publicBatchWrapper89_f2155 a h), kone]


theorem countAt_30 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 30).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 210445)).val = 32
  rw [(publicBatchWrapper89_f30 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_31 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 31) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f306 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f307 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f308 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f309 a h); rw [kzero] at e; exact e


theorem assetAt_31 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 31).isDummy (row a 31).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 30 47) = (row a 31).isDummy := by rw [(publicBatchWrapper89_f312 a h), (publicBatchWrapper89_f310 a h), (publicBatchWrapper89_f311 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2156 a h), by rw [← hd, ← (publicBatchWrapper89_f2157 a h), (publicBatchWrapper89_f2158 a h), kone]⟩


theorem feeAt_31 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 31).isDummy (row a 31).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 30 47) = (row a 31).isDummy := by rw [(publicBatchWrapper89_f312 a h), (publicBatchWrapper89_f310 a h), (publicBatchWrapper89_f311 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2159 a h), by rw [← hd, ← (publicBatchWrapper89_f2160 a h), (publicBatchWrapper89_f2161 a h), kone]⟩


theorem blockAt_31 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 31).isDummy (row a 31).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 30 47) = (row a 31).isDummy := by rw [(publicBatchWrapper89_f312 a h), (publicBatchWrapper89_f310 a h), (publicBatchWrapper89_f311 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605850), a (.virt 605852), a (.virt 605854), a (.virt 605856)],
    ![a (.virt 605851), a (.virt 605853), a (.virt 605855), a (.virt 605857)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2162 a h)
    · exact (publicBatchWrapper89_f2163 a h)
    · exact (publicBatchWrapper89_f2164 a h)
    · exact (publicBatchWrapper89_f2165 a h)
  · show bor (row a 31).isDummy (band (band (a (.virt 605850)) (a (.virt 605852)))
      (band (a (.virt 605854)) (a (.virt 605856)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2166 a h), ← (publicBatchWrapper89_f2167 a h), ← (publicBatchWrapper89_f2168 a h), ← (publicBatchWrapper89_f2169 a h), (publicBatchWrapper89_f2170 a h), kone]


theorem countAt_31 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 31).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 217240)).val = 32
  rw [(publicBatchWrapper89_f31 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_32 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 32) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f313 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f314 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f315 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f316 a h); rw [kzero] at e; exact e


theorem assetAt_32 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 32).isDummy (row a 32).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 32 11) = (row a 32).isDummy := by rw [(publicBatchWrapper89_f319 a h), (publicBatchWrapper89_f317 a h), (publicBatchWrapper89_f318 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2171 a h), by rw [← hd, ← (publicBatchWrapper89_f2172 a h), (publicBatchWrapper89_f2173 a h), kone]⟩


theorem feeAt_32 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 32).isDummy (row a 32).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 32 11) = (row a 32).isDummy := by rw [(publicBatchWrapper89_f319 a h), (publicBatchWrapper89_f317 a h), (publicBatchWrapper89_f318 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2174 a h), by rw [← hd, ← (publicBatchWrapper89_f2175 a h), (publicBatchWrapper89_f2176 a h), kone]⟩


theorem blockAt_32 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 32).isDummy (row a 32).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 32 11) = (row a 32).isDummy := by rw [(publicBatchWrapper89_f319 a h), (publicBatchWrapper89_f317 a h), (publicBatchWrapper89_f318 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605862), a (.virt 605864), a (.virt 605866), a (.virt 605868)],
    ![a (.virt 605863), a (.virt 605865), a (.virt 605867), a (.virt 605869)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2177 a h)
    · exact (publicBatchWrapper89_f2178 a h)
    · exact (publicBatchWrapper89_f2179 a h)
    · exact (publicBatchWrapper89_f2180 a h)
  · show bor (row a 32).isDummy (band (band (a (.virt 605862)) (a (.virt 605864)))
      (band (a (.virt 605866)) (a (.virt 605868)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2181 a h), ← (publicBatchWrapper89_f2182 a h), ← (publicBatchWrapper89_f2183 a h), ← (publicBatchWrapper89_f2184 a h), (publicBatchWrapper89_f2185 a h), kone]


theorem countAt_32 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 32).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 224035)).val = 32
  rw [(publicBatchWrapper89_f32 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_33 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 33) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f320 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f321 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f322 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f323 a h); rw [kzero] at e; exact e


theorem assetAt_33 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 33).isDummy (row a 33).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 32 55) = (row a 33).isDummy := by rw [(publicBatchWrapper89_f326 a h), (publicBatchWrapper89_f324 a h), (publicBatchWrapper89_f325 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2186 a h), by rw [← hd, ← (publicBatchWrapper89_f2187 a h), (publicBatchWrapper89_f2188 a h), kone]⟩


theorem feeAt_33 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 33).isDummy (row a 33).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 32 55) = (row a 33).isDummy := by rw [(publicBatchWrapper89_f326 a h), (publicBatchWrapper89_f324 a h), (publicBatchWrapper89_f325 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2189 a h), by rw [← hd, ← (publicBatchWrapper89_f2190 a h), (publicBatchWrapper89_f2191 a h), kone]⟩


theorem blockAt_33 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 33).isDummy (row a 33).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 32 55) = (row a 33).isDummy := by rw [(publicBatchWrapper89_f326 a h), (publicBatchWrapper89_f324 a h), (publicBatchWrapper89_f325 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605874), a (.virt 605876), a (.virt 605878), a (.virt 605880)],
    ![a (.virt 605875), a (.virt 605877), a (.virt 605879), a (.virt 605881)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2192 a h)
    · exact (publicBatchWrapper89_f2193 a h)
    · exact (publicBatchWrapper89_f2194 a h)
    · exact (publicBatchWrapper89_f2195 a h)
  · show bor (row a 33).isDummy (band (band (a (.virt 605874)) (a (.virt 605876)))
      (band (a (.virt 605878)) (a (.virt 605880)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2196 a h), ← (publicBatchWrapper89_f2197 a h), ← (publicBatchWrapper89_f2198 a h), ← (publicBatchWrapper89_f2199 a h), (publicBatchWrapper89_f2200 a h), kone]


theorem countAt_33 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 33).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 230830)).val = 32
  rw [(publicBatchWrapper89_f33 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_34 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 34) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f327 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f328 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f329 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f330 a h); rw [kzero] at e; exact e


theorem assetAt_34 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 34).isDummy (row a 34).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 33 19) = (row a 34).isDummy := by rw [(publicBatchWrapper89_f333 a h), (publicBatchWrapper89_f331 a h), (publicBatchWrapper89_f332 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2201 a h), by rw [← hd, ← (publicBatchWrapper89_f2202 a h), (publicBatchWrapper89_f2203 a h), kone]⟩


theorem feeAt_34 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 34).isDummy (row a 34).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 33 19) = (row a 34).isDummy := by rw [(publicBatchWrapper89_f333 a h), (publicBatchWrapper89_f331 a h), (publicBatchWrapper89_f332 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2204 a h), by rw [← hd, ← (publicBatchWrapper89_f2205 a h), (publicBatchWrapper89_f2206 a h), kone]⟩


theorem blockAt_34 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 34).isDummy (row a 34).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 33 19) = (row a 34).isDummy := by rw [(publicBatchWrapper89_f333 a h), (publicBatchWrapper89_f331 a h), (publicBatchWrapper89_f332 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605886), a (.virt 605888), a (.virt 605890), a (.virt 605892)],
    ![a (.virt 605887), a (.virt 605889), a (.virt 605891), a (.virt 605893)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2207 a h)
    · exact (publicBatchWrapper89_f2208 a h)
    · exact (publicBatchWrapper89_f2209 a h)
    · exact (publicBatchWrapper89_f2210 a h)
  · show bor (row a 34).isDummy (band (band (a (.virt 605886)) (a (.virt 605888)))
      (band (a (.virt 605890)) (a (.virt 605892)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2211 a h), ← (publicBatchWrapper89_f2212 a h), ← (publicBatchWrapper89_f2213 a h), ← (publicBatchWrapper89_f2214 a h), (publicBatchWrapper89_f2215 a h), kone]


theorem countAt_34 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 34).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 237625)).val = 32
  rw [(publicBatchWrapper89_f34 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_35 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 35) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f334 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f335 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f336 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f337 a h); rw [kzero] at e; exact e


theorem assetAt_35 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 35).isDummy (row a 35).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 33 63) = (row a 35).isDummy := by rw [(publicBatchWrapper89_f340 a h), (publicBatchWrapper89_f338 a h), (publicBatchWrapper89_f339 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2216 a h), by rw [← hd, ← (publicBatchWrapper89_f2217 a h), (publicBatchWrapper89_f2218 a h), kone]⟩


theorem feeAt_35 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 35).isDummy (row a 35).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 33 63) = (row a 35).isDummy := by rw [(publicBatchWrapper89_f340 a h), (publicBatchWrapper89_f338 a h), (publicBatchWrapper89_f339 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2219 a h), by rw [← hd, ← (publicBatchWrapper89_f2220 a h), (publicBatchWrapper89_f2221 a h), kone]⟩


theorem blockAt_35 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 35).isDummy (row a 35).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 33 63) = (row a 35).isDummy := by rw [(publicBatchWrapper89_f340 a h), (publicBatchWrapper89_f338 a h), (publicBatchWrapper89_f339 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605898), a (.virt 605900), a (.virt 605902), a (.virt 605904)],
    ![a (.virt 605899), a (.virt 605901), a (.virt 605903), a (.virt 605905)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2222 a h)
    · exact (publicBatchWrapper89_f2223 a h)
    · exact (publicBatchWrapper89_f2224 a h)
    · exact (publicBatchWrapper89_f2225 a h)
  · show bor (row a 35).isDummy (band (band (a (.virt 605898)) (a (.virt 605900)))
      (band (a (.virt 605902)) (a (.virt 605904)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2226 a h), ← (publicBatchWrapper89_f2227 a h), ← (publicBatchWrapper89_f2228 a h), ← (publicBatchWrapper89_f2229 a h), (publicBatchWrapper89_f2230 a h), kone]


theorem countAt_35 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 35).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 244420)).val = 32
  rw [(publicBatchWrapper89_f35 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_36 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 36) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f341 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f342 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f343 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f344 a h); rw [kzero] at e; exact e


theorem assetAt_36 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 36).isDummy (row a 36).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 35 27) = (row a 36).isDummy := by rw [(publicBatchWrapper89_f347 a h), (publicBatchWrapper89_f345 a h), (publicBatchWrapper89_f346 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2231 a h), by rw [← hd, ← (publicBatchWrapper89_f2232 a h), (publicBatchWrapper89_f2233 a h), kone]⟩


theorem feeAt_36 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 36).isDummy (row a 36).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 35 27) = (row a 36).isDummy := by rw [(publicBatchWrapper89_f347 a h), (publicBatchWrapper89_f345 a h), (publicBatchWrapper89_f346 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2234 a h), by rw [← hd, ← (publicBatchWrapper89_f2235 a h), (publicBatchWrapper89_f2236 a h), kone]⟩


theorem blockAt_36 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 36).isDummy (row a 36).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 35 27) = (row a 36).isDummy := by rw [(publicBatchWrapper89_f347 a h), (publicBatchWrapper89_f345 a h), (publicBatchWrapper89_f346 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605910), a (.virt 605912), a (.virt 605914), a (.virt 605916)],
    ![a (.virt 605911), a (.virt 605913), a (.virt 605915), a (.virt 605917)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2237 a h)
    · exact (publicBatchWrapper89_f2238 a h)
    · exact (publicBatchWrapper89_f2239 a h)
    · exact (publicBatchWrapper89_f2240 a h)
  · show bor (row a 36).isDummy (band (band (a (.virt 605910)) (a (.virt 605912)))
      (band (a (.virt 605914)) (a (.virt 605916)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2241 a h), ← (publicBatchWrapper89_f2242 a h), ← (publicBatchWrapper89_f2243 a h), ← (publicBatchWrapper89_f2244 a h), (publicBatchWrapper89_f2245 a h), kone]


theorem countAt_36 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 36).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 251215)).val = 32
  rw [(publicBatchWrapper89_f36 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_37 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 37) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f348 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f349 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f350 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f351 a h); rw [kzero] at e; exact e


theorem assetAt_37 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 37).isDummy (row a 37).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 35 71) = (row a 37).isDummy := by rw [(publicBatchWrapper89_f354 a h), (publicBatchWrapper89_f352 a h), (publicBatchWrapper89_f353 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2246 a h), by rw [← hd, ← (publicBatchWrapper89_f2247 a h), (publicBatchWrapper89_f2248 a h), kone]⟩


theorem feeAt_37 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 37).isDummy (row a 37).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 35 71) = (row a 37).isDummy := by rw [(publicBatchWrapper89_f354 a h), (publicBatchWrapper89_f352 a h), (publicBatchWrapper89_f353 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2249 a h), by rw [← hd, ← (publicBatchWrapper89_f2250 a h), (publicBatchWrapper89_f2251 a h), kone]⟩


theorem blockAt_37 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 37).isDummy (row a 37).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 35 71) = (row a 37).isDummy := by rw [(publicBatchWrapper89_f354 a h), (publicBatchWrapper89_f352 a h), (publicBatchWrapper89_f353 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605922), a (.virt 605924), a (.virt 605926), a (.virt 605928)],
    ![a (.virt 605923), a (.virt 605925), a (.virt 605927), a (.virt 605929)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2252 a h)
    · exact (publicBatchWrapper89_f2253 a h)
    · exact (publicBatchWrapper89_f2254 a h)
    · exact (publicBatchWrapper89_f2255 a h)
  · show bor (row a 37).isDummy (band (band (a (.virt 605922)) (a (.virt 605924)))
      (band (a (.virt 605926)) (a (.virt 605928)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2256 a h), ← (publicBatchWrapper89_f2257 a h), ← (publicBatchWrapper89_f2258 a h), ← (publicBatchWrapper89_f2259 a h), (publicBatchWrapper89_f2260 a h), kone]


theorem countAt_37 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 37).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 258010)).val = 32
  rw [(publicBatchWrapper89_f37 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_38 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 38) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f355 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f356 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f357 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f358 a h); rw [kzero] at e; exact e


theorem assetAt_38 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 38).isDummy (row a 38).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 37 35) = (row a 38).isDummy := by rw [(publicBatchWrapper89_f361 a h), (publicBatchWrapper89_f359 a h), (publicBatchWrapper89_f360 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2261 a h), by rw [← hd, ← (publicBatchWrapper89_f2262 a h), (publicBatchWrapper89_f2263 a h), kone]⟩


theorem feeAt_38 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 38).isDummy (row a 38).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 37 35) = (row a 38).isDummy := by rw [(publicBatchWrapper89_f361 a h), (publicBatchWrapper89_f359 a h), (publicBatchWrapper89_f360 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2264 a h), by rw [← hd, ← (publicBatchWrapper89_f2265 a h), (publicBatchWrapper89_f2266 a h), kone]⟩


theorem blockAt_38 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 38).isDummy (row a 38).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 37 35) = (row a 38).isDummy := by rw [(publicBatchWrapper89_f361 a h), (publicBatchWrapper89_f359 a h), (publicBatchWrapper89_f360 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605934), a (.virt 605936), a (.virt 605938), a (.virt 605940)],
    ![a (.virt 605935), a (.virt 605937), a (.virt 605939), a (.virt 605941)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2267 a h)
    · exact (publicBatchWrapper89_f2268 a h)
    · exact (publicBatchWrapper89_f2269 a h)
    · exact (publicBatchWrapper89_f2270 a h)
  · show bor (row a 38).isDummy (band (band (a (.virt 605934)) (a (.virt 605936)))
      (band (a (.virt 605938)) (a (.virt 605940)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2271 a h), ← (publicBatchWrapper89_f2272 a h), ← (publicBatchWrapper89_f2273 a h), ← (publicBatchWrapper89_f2274 a h), (publicBatchWrapper89_f2275 a h), kone]


theorem countAt_38 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 38).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 264805)).val = 32
  rw [(publicBatchWrapper89_f38 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_39 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 39) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f362 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f363 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f364 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f365 a h); rw [kzero] at e; exact e


theorem assetAt_39 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 39).isDummy (row a 39).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 37 79) = (row a 39).isDummy := by rw [(publicBatchWrapper89_f368 a h), (publicBatchWrapper89_f366 a h), (publicBatchWrapper89_f367 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2276 a h), by rw [← hd, ← (publicBatchWrapper89_f2277 a h), (publicBatchWrapper89_f2278 a h), kone]⟩


theorem feeAt_39 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 39).isDummy (row a 39).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 37 79) = (row a 39).isDummy := by rw [(publicBatchWrapper89_f368 a h), (publicBatchWrapper89_f366 a h), (publicBatchWrapper89_f367 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2279 a h), by rw [← hd, ← (publicBatchWrapper89_f2280 a h), (publicBatchWrapper89_f2281 a h), kone]⟩


theorem blockAt_39 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 39).isDummy (row a 39).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 37 79) = (row a 39).isDummy := by rw [(publicBatchWrapper89_f368 a h), (publicBatchWrapper89_f366 a h), (publicBatchWrapper89_f367 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605946), a (.virt 605948), a (.virt 605950), a (.virt 605952)],
    ![a (.virt 605947), a (.virt 605949), a (.virt 605951), a (.virt 605953)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2282 a h)
    · exact (publicBatchWrapper89_f2283 a h)
    · exact (publicBatchWrapper89_f2284 a h)
    · exact (publicBatchWrapper89_f2285 a h)
  · show bor (row a 39).isDummy (band (band (a (.virt 605946)) (a (.virt 605948)))
      (band (a (.virt 605950)) (a (.virt 605952)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2286 a h), ← (publicBatchWrapper89_f2287 a h), ← (publicBatchWrapper89_f2288 a h), ← (publicBatchWrapper89_f2289 a h), (publicBatchWrapper89_f2290 a h), kone]


theorem countAt_39 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 39).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 271600)).val = 32
  rw [(publicBatchWrapper89_f39 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_40 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 40) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f369 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f370 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f371 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f372 a h); rw [kzero] at e; exact e


theorem assetAt_40 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 40).isDummy (row a 40).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 39 43) = (row a 40).isDummy := by rw [(publicBatchWrapper89_f375 a h), (publicBatchWrapper89_f373 a h), (publicBatchWrapper89_f374 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2291 a h), by rw [← hd, ← (publicBatchWrapper89_f2292 a h), (publicBatchWrapper89_f2293 a h), kone]⟩


theorem feeAt_40 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 40).isDummy (row a 40).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 39 43) = (row a 40).isDummy := by rw [(publicBatchWrapper89_f375 a h), (publicBatchWrapper89_f373 a h), (publicBatchWrapper89_f374 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2294 a h), by rw [← hd, ← (publicBatchWrapper89_f2295 a h), (publicBatchWrapper89_f2296 a h), kone]⟩


theorem blockAt_40 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 40).isDummy (row a 40).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 39 43) = (row a 40).isDummy := by rw [(publicBatchWrapper89_f375 a h), (publicBatchWrapper89_f373 a h), (publicBatchWrapper89_f374 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605958), a (.virt 605960), a (.virt 605962), a (.virt 605964)],
    ![a (.virt 605959), a (.virt 605961), a (.virt 605963), a (.virt 605965)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2297 a h)
    · exact (publicBatchWrapper89_f2298 a h)
    · exact (publicBatchWrapper89_f2299 a h)
    · exact (publicBatchWrapper89_f2300 a h)
  · show bor (row a 40).isDummy (band (band (a (.virt 605958)) (a (.virt 605960)))
      (band (a (.virt 605962)) (a (.virt 605964)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2301 a h), ← (publicBatchWrapper89_f2302 a h), ← (publicBatchWrapper89_f2303 a h), ← (publicBatchWrapper89_f2304 a h), (publicBatchWrapper89_f2305 a h), kone]


theorem countAt_40 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 40).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 278395)).val = 32
  rw [(publicBatchWrapper89_f40 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_41 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 41) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f376 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f377 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f378 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f379 a h); rw [kzero] at e; exact e


theorem assetAt_41 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 41).isDummy (row a 41).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 40 7) = (row a 41).isDummy := by rw [(publicBatchWrapper89_f382 a h), (publicBatchWrapper89_f380 a h), (publicBatchWrapper89_f381 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2306 a h), by rw [← hd, ← (publicBatchWrapper89_f2307 a h), (publicBatchWrapper89_f2308 a h), kone]⟩


theorem feeAt_41 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 41).isDummy (row a 41).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 40 7) = (row a 41).isDummy := by rw [(publicBatchWrapper89_f382 a h), (publicBatchWrapper89_f380 a h), (publicBatchWrapper89_f381 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2309 a h), by rw [← hd, ← (publicBatchWrapper89_f2310 a h), (publicBatchWrapper89_f2311 a h), kone]⟩


theorem blockAt_41 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 41).isDummy (row a 41).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 40 7) = (row a 41).isDummy := by rw [(publicBatchWrapper89_f382 a h), (publicBatchWrapper89_f380 a h), (publicBatchWrapper89_f381 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605970), a (.virt 605972), a (.virt 605974), a (.virt 605976)],
    ![a (.virt 605971), a (.virt 605973), a (.virt 605975), a (.virt 605977)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2312 a h)
    · exact (publicBatchWrapper89_f2313 a h)
    · exact (publicBatchWrapper89_f2314 a h)
    · exact (publicBatchWrapper89_f2315 a h)
  · show bor (row a 41).isDummy (band (band (a (.virt 605970)) (a (.virt 605972)))
      (band (a (.virt 605974)) (a (.virt 605976)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2316 a h), ← (publicBatchWrapper89_f2317 a h), ← (publicBatchWrapper89_f2318 a h), ← (publicBatchWrapper89_f2319 a h), (publicBatchWrapper89_f2320 a h), kone]


theorem countAt_41 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 41).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 285190)).val = 32
  rw [(publicBatchWrapper89_f41 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_42 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 42) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f383 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f384 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f385 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f386 a h); rw [kzero] at e; exact e


theorem assetAt_42 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 42).isDummy (row a 42).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 40 51) = (row a 42).isDummy := by rw [(publicBatchWrapper89_f389 a h), (publicBatchWrapper89_f387 a h), (publicBatchWrapper89_f388 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2321 a h), by rw [← hd, ← (publicBatchWrapper89_f2322 a h), (publicBatchWrapper89_f2323 a h), kone]⟩


theorem feeAt_42 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 42).isDummy (row a 42).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 40 51) = (row a 42).isDummy := by rw [(publicBatchWrapper89_f389 a h), (publicBatchWrapper89_f387 a h), (publicBatchWrapper89_f388 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2324 a h), by rw [← hd, ← (publicBatchWrapper89_f2325 a h), (publicBatchWrapper89_f2326 a h), kone]⟩


theorem blockAt_42 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 42).isDummy (row a 42).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 40 51) = (row a 42).isDummy := by rw [(publicBatchWrapper89_f389 a h), (publicBatchWrapper89_f387 a h), (publicBatchWrapper89_f388 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605982), a (.virt 605984), a (.virt 605986), a (.virt 605988)],
    ![a (.virt 605983), a (.virt 605985), a (.virt 605987), a (.virt 605989)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2327 a h)
    · exact (publicBatchWrapper89_f2328 a h)
    · exact (publicBatchWrapper89_f2329 a h)
    · exact (publicBatchWrapper89_f2330 a h)
  · show bor (row a 42).isDummy (band (band (a (.virt 605982)) (a (.virt 605984)))
      (band (a (.virt 605986)) (a (.virt 605988)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2331 a h), ← (publicBatchWrapper89_f2332 a h), ← (publicBatchWrapper89_f2333 a h), ← (publicBatchWrapper89_f2334 a h), (publicBatchWrapper89_f2335 a h), kone]


theorem countAt_42 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 42).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 291985)).val = 32
  rw [(publicBatchWrapper89_f42 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_43 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 43) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f390 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f391 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f392 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f393 a h); rw [kzero] at e; exact e


theorem assetAt_43 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 43).isDummy (row a 43).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 42 15) = (row a 43).isDummy := by rw [(publicBatchWrapper89_f396 a h), (publicBatchWrapper89_f394 a h), (publicBatchWrapper89_f395 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2336 a h), by rw [← hd, ← (publicBatchWrapper89_f2337 a h), (publicBatchWrapper89_f2338 a h), kone]⟩


theorem feeAt_43 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 43).isDummy (row a 43).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 42 15) = (row a 43).isDummy := by rw [(publicBatchWrapper89_f396 a h), (publicBatchWrapper89_f394 a h), (publicBatchWrapper89_f395 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2339 a h), by rw [← hd, ← (publicBatchWrapper89_f2340 a h), (publicBatchWrapper89_f2341 a h), kone]⟩


theorem blockAt_43 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 43).isDummy (row a 43).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 42 15) = (row a 43).isDummy := by rw [(publicBatchWrapper89_f396 a h), (publicBatchWrapper89_f394 a h), (publicBatchWrapper89_f395 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 605994), a (.virt 605996), a (.virt 605998), a (.virt 606000)],
    ![a (.virt 605995), a (.virt 605997), a (.virt 605999), a (.virt 606001)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2342 a h)
    · exact (publicBatchWrapper89_f2343 a h)
    · exact (publicBatchWrapper89_f2344 a h)
    · exact (publicBatchWrapper89_f2345 a h)
  · show bor (row a 43).isDummy (band (band (a (.virt 605994)) (a (.virt 605996)))
      (band (a (.virt 605998)) (a (.virt 606000)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2346 a h), ← (publicBatchWrapper89_f2347 a h), ← (publicBatchWrapper89_f2348 a h), ← (publicBatchWrapper89_f2349 a h), (publicBatchWrapper89_f2350 a h), kone]


theorem countAt_43 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 43).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 298780)).val = 32
  rw [(publicBatchWrapper89_f43 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_44 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 44) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f397 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f398 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f399 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f400 a h); rw [kzero] at e; exact e


theorem assetAt_44 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 44).isDummy (row a 44).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 42 59) = (row a 44).isDummy := by rw [(publicBatchWrapper89_f403 a h), (publicBatchWrapper89_f401 a h), (publicBatchWrapper89_f402 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2351 a h), by rw [← hd, ← (publicBatchWrapper89_f2352 a h), (publicBatchWrapper89_f2353 a h), kone]⟩


theorem feeAt_44 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 44).isDummy (row a 44).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 42 59) = (row a 44).isDummy := by rw [(publicBatchWrapper89_f403 a h), (publicBatchWrapper89_f401 a h), (publicBatchWrapper89_f402 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2354 a h), by rw [← hd, ← (publicBatchWrapper89_f2355 a h), (publicBatchWrapper89_f2356 a h), kone]⟩


theorem blockAt_44 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 44).isDummy (row a 44).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 42 59) = (row a 44).isDummy := by rw [(publicBatchWrapper89_f403 a h), (publicBatchWrapper89_f401 a h), (publicBatchWrapper89_f402 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606006), a (.virt 606008), a (.virt 606010), a (.virt 606012)],
    ![a (.virt 606007), a (.virt 606009), a (.virt 606011), a (.virt 606013)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2357 a h)
    · exact (publicBatchWrapper89_f2358 a h)
    · exact (publicBatchWrapper89_f2359 a h)
    · exact (publicBatchWrapper89_f2360 a h)
  · show bor (row a 44).isDummy (band (band (a (.virt 606006)) (a (.virt 606008)))
      (band (a (.virt 606010)) (a (.virt 606012)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2361 a h), ← (publicBatchWrapper89_f2362 a h), ← (publicBatchWrapper89_f2363 a h), ← (publicBatchWrapper89_f2364 a h), (publicBatchWrapper89_f2365 a h), kone]


theorem countAt_44 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 44).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 305575)).val = 32
  rw [(publicBatchWrapper89_f44 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_45 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 45) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f404 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f405 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f406 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f407 a h); rw [kzero] at e; exact e


theorem assetAt_45 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 45).isDummy (row a 45).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 44 23) = (row a 45).isDummy := by rw [(publicBatchWrapper89_f410 a h), (publicBatchWrapper89_f408 a h), (publicBatchWrapper89_f409 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2366 a h), by rw [← hd, ← (publicBatchWrapper89_f2367 a h), (publicBatchWrapper89_f2368 a h), kone]⟩


theorem feeAt_45 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 45).isDummy (row a 45).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 44 23) = (row a 45).isDummy := by rw [(publicBatchWrapper89_f410 a h), (publicBatchWrapper89_f408 a h), (publicBatchWrapper89_f409 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2369 a h), by rw [← hd, ← (publicBatchWrapper89_f2370 a h), (publicBatchWrapper89_f2371 a h), kone]⟩


theorem blockAt_45 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 45).isDummy (row a 45).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 44 23) = (row a 45).isDummy := by rw [(publicBatchWrapper89_f410 a h), (publicBatchWrapper89_f408 a h), (publicBatchWrapper89_f409 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606018), a (.virt 606020), a (.virt 606022), a (.virt 606024)],
    ![a (.virt 606019), a (.virt 606021), a (.virt 606023), a (.virt 606025)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2372 a h)
    · exact (publicBatchWrapper89_f2373 a h)
    · exact (publicBatchWrapper89_f2374 a h)
    · exact (publicBatchWrapper89_f2375 a h)
  · show bor (row a 45).isDummy (band (band (a (.virt 606018)) (a (.virt 606020)))
      (band (a (.virt 606022)) (a (.virt 606024)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2376 a h), ← (publicBatchWrapper89_f2377 a h), ← (publicBatchWrapper89_f2378 a h), ← (publicBatchWrapper89_f2379 a h), (publicBatchWrapper89_f2380 a h), kone]


theorem countAt_45 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 45).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 312370)).val = 32
  rw [(publicBatchWrapper89_f45 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_46 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 46) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f411 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f412 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f413 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f414 a h); rw [kzero] at e; exact e


theorem assetAt_46 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 46).isDummy (row a 46).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 44 67) = (row a 46).isDummy := by rw [(publicBatchWrapper89_f417 a h), (publicBatchWrapper89_f415 a h), (publicBatchWrapper89_f416 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2381 a h), by rw [← hd, ← (publicBatchWrapper89_f2382 a h), (publicBatchWrapper89_f2383 a h), kone]⟩


theorem feeAt_46 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 46).isDummy (row a 46).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 44 67) = (row a 46).isDummy := by rw [(publicBatchWrapper89_f417 a h), (publicBatchWrapper89_f415 a h), (publicBatchWrapper89_f416 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2384 a h), by rw [← hd, ← (publicBatchWrapper89_f2385 a h), (publicBatchWrapper89_f2386 a h), kone]⟩


theorem blockAt_46 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 46).isDummy (row a 46).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 44 67) = (row a 46).isDummy := by rw [(publicBatchWrapper89_f417 a h), (publicBatchWrapper89_f415 a h), (publicBatchWrapper89_f416 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606030), a (.virt 606032), a (.virt 606034), a (.virt 606036)],
    ![a (.virt 606031), a (.virt 606033), a (.virt 606035), a (.virt 606037)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2387 a h)
    · exact (publicBatchWrapper89_f2388 a h)
    · exact (publicBatchWrapper89_f2389 a h)
    · exact (publicBatchWrapper89_f2390 a h)
  · show bor (row a 46).isDummy (band (band (a (.virt 606030)) (a (.virt 606032)))
      (band (a (.virt 606034)) (a (.virt 606036)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2391 a h), ← (publicBatchWrapper89_f2392 a h), ← (publicBatchWrapper89_f2393 a h), ← (publicBatchWrapper89_f2394 a h), (publicBatchWrapper89_f2395 a h), kone]


theorem countAt_46 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 46).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 319165)).val = 32
  rw [(publicBatchWrapper89_f46 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_47 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 47) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f418 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f419 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f420 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f421 a h); rw [kzero] at e; exact e


theorem assetAt_47 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 47).isDummy (row a 47).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 45 31) = (row a 47).isDummy := by rw [(publicBatchWrapper89_f424 a h), (publicBatchWrapper89_f422 a h), (publicBatchWrapper89_f423 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2396 a h), by rw [← hd, ← (publicBatchWrapper89_f2397 a h), (publicBatchWrapper89_f2398 a h), kone]⟩


theorem feeAt_47 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 47).isDummy (row a 47).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 45 31) = (row a 47).isDummy := by rw [(publicBatchWrapper89_f424 a h), (publicBatchWrapper89_f422 a h), (publicBatchWrapper89_f423 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2399 a h), by rw [← hd, ← (publicBatchWrapper89_f2400 a h), (publicBatchWrapper89_f2401 a h), kone]⟩


theorem blockAt_47 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 47).isDummy (row a 47).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 45 31) = (row a 47).isDummy := by rw [(publicBatchWrapper89_f424 a h), (publicBatchWrapper89_f422 a h), (publicBatchWrapper89_f423 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606042), a (.virt 606044), a (.virt 606046), a (.virt 606048)],
    ![a (.virt 606043), a (.virt 606045), a (.virt 606047), a (.virt 606049)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2402 a h)
    · exact (publicBatchWrapper89_f2403 a h)
    · exact (publicBatchWrapper89_f2404 a h)
    · exact (publicBatchWrapper89_f2405 a h)
  · show bor (row a 47).isDummy (band (band (a (.virt 606042)) (a (.virt 606044)))
      (band (a (.virt 606046)) (a (.virt 606048)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2406 a h), ← (publicBatchWrapper89_f2407 a h), ← (publicBatchWrapper89_f2408 a h), ← (publicBatchWrapper89_f2409 a h), (publicBatchWrapper89_f2410 a h), kone]


theorem countAt_47 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 47).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 325960)).val = 32
  rw [(publicBatchWrapper89_f47 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_48 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 48) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f425 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f426 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f427 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f428 a h); rw [kzero] at e; exact e


theorem assetAt_48 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 48).isDummy (row a 48).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 45 75) = (row a 48).isDummy := by rw [(publicBatchWrapper89_f431 a h), (publicBatchWrapper89_f429 a h), (publicBatchWrapper89_f430 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2411 a h), by rw [← hd, ← (publicBatchWrapper89_f2412 a h), (publicBatchWrapper89_f2413 a h), kone]⟩


theorem feeAt_48 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 48).isDummy (row a 48).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 45 75) = (row a 48).isDummy := by rw [(publicBatchWrapper89_f431 a h), (publicBatchWrapper89_f429 a h), (publicBatchWrapper89_f430 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2414 a h), by rw [← hd, ← (publicBatchWrapper89_f2415 a h), (publicBatchWrapper89_f2416 a h), kone]⟩


theorem blockAt_48 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 48).isDummy (row a 48).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 45 75) = (row a 48).isDummy := by rw [(publicBatchWrapper89_f431 a h), (publicBatchWrapper89_f429 a h), (publicBatchWrapper89_f430 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606054), a (.virt 606056), a (.virt 606058), a (.virt 606060)],
    ![a (.virt 606055), a (.virt 606057), a (.virt 606059), a (.virt 606061)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2417 a h)
    · exact (publicBatchWrapper89_f2418 a h)
    · exact (publicBatchWrapper89_f2419 a h)
    · exact (publicBatchWrapper89_f2420 a h)
  · show bor (row a 48).isDummy (band (band (a (.virt 606054)) (a (.virt 606056)))
      (band (a (.virt 606058)) (a (.virt 606060)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2421 a h), ← (publicBatchWrapper89_f2422 a h), ← (publicBatchWrapper89_f2423 a h), ← (publicBatchWrapper89_f2424 a h), (publicBatchWrapper89_f2425 a h), kone]


theorem countAt_48 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 48).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 332755)).val = 32
  rw [(publicBatchWrapper89_f48 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_49 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 49) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f432 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f433 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f434 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f435 a h); rw [kzero] at e; exact e


theorem assetAt_49 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 49).isDummy (row a 49).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 47 39) = (row a 49).isDummy := by rw [(publicBatchWrapper89_f438 a h), (publicBatchWrapper89_f436 a h), (publicBatchWrapper89_f437 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2426 a h), by rw [← hd, ← (publicBatchWrapper89_f2427 a h), (publicBatchWrapper89_f2428 a h), kone]⟩


theorem feeAt_49 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 49).isDummy (row a 49).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 47 39) = (row a 49).isDummy := by rw [(publicBatchWrapper89_f438 a h), (publicBatchWrapper89_f436 a h), (publicBatchWrapper89_f437 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2429 a h), by rw [← hd, ← (publicBatchWrapper89_f2430 a h), (publicBatchWrapper89_f2431 a h), kone]⟩


theorem blockAt_49 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 49).isDummy (row a 49).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 47 39) = (row a 49).isDummy := by rw [(publicBatchWrapper89_f438 a h), (publicBatchWrapper89_f436 a h), (publicBatchWrapper89_f437 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606066), a (.virt 606068), a (.virt 606070), a (.virt 606072)],
    ![a (.virt 606067), a (.virt 606069), a (.virt 606071), a (.virt 606073)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2432 a h)
    · exact (publicBatchWrapper89_f2433 a h)
    · exact (publicBatchWrapper89_f2434 a h)
    · exact (publicBatchWrapper89_f2435 a h)
  · show bor (row a 49).isDummy (band (band (a (.virt 606066)) (a (.virt 606068)))
      (band (a (.virt 606070)) (a (.virt 606072)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2436 a h), ← (publicBatchWrapper89_f2437 a h), ← (publicBatchWrapper89_f2438 a h), ← (publicBatchWrapper89_f2439 a h), (publicBatchWrapper89_f2440 a h), kone]


theorem countAt_49 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 49).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 339550)).val = 32
  rw [(publicBatchWrapper89_f49 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_50 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 50) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f439 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f440 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f441 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f442 a h); rw [kzero] at e; exact e


theorem assetAt_50 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 50).isDummy (row a 50).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 49 3) = (row a 50).isDummy := by rw [(publicBatchWrapper89_f445 a h), (publicBatchWrapper89_f443 a h), (publicBatchWrapper89_f444 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2441 a h), by rw [← hd, ← (publicBatchWrapper89_f2442 a h), (publicBatchWrapper89_f2443 a h), kone]⟩


theorem feeAt_50 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 50).isDummy (row a 50).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 49 3) = (row a 50).isDummy := by rw [(publicBatchWrapper89_f445 a h), (publicBatchWrapper89_f443 a h), (publicBatchWrapper89_f444 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2444 a h), by rw [← hd, ← (publicBatchWrapper89_f2445 a h), (publicBatchWrapper89_f2446 a h), kone]⟩


theorem blockAt_50 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 50).isDummy (row a 50).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 49 3) = (row a 50).isDummy := by rw [(publicBatchWrapper89_f445 a h), (publicBatchWrapper89_f443 a h), (publicBatchWrapper89_f444 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606078), a (.virt 606080), a (.virt 606082), a (.virt 606084)],
    ![a (.virt 606079), a (.virt 606081), a (.virt 606083), a (.virt 606085)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2447 a h)
    · exact (publicBatchWrapper89_f2448 a h)
    · exact (publicBatchWrapper89_f2449 a h)
    · exact (publicBatchWrapper89_f2450 a h)
  · show bor (row a 50).isDummy (band (band (a (.virt 606078)) (a (.virt 606080)))
      (band (a (.virt 606082)) (a (.virt 606084)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2451 a h), ← (publicBatchWrapper89_f2452 a h), ← (publicBatchWrapper89_f2453 a h), ← (publicBatchWrapper89_f2454 a h), (publicBatchWrapper89_f2455 a h), kone]


theorem countAt_50 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 50).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 346345)).val = 32
  rw [(publicBatchWrapper89_f50 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_51 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 51) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f446 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f447 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f448 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f449 a h); rw [kzero] at e; exact e


theorem assetAt_51 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 51).isDummy (row a 51).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 49 47) = (row a 51).isDummy := by rw [(publicBatchWrapper89_f452 a h), (publicBatchWrapper89_f450 a h), (publicBatchWrapper89_f451 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2456 a h), by rw [← hd, ← (publicBatchWrapper89_f2457 a h), (publicBatchWrapper89_f2458 a h), kone]⟩


theorem feeAt_51 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 51).isDummy (row a 51).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 49 47) = (row a 51).isDummy := by rw [(publicBatchWrapper89_f452 a h), (publicBatchWrapper89_f450 a h), (publicBatchWrapper89_f451 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2459 a h), by rw [← hd, ← (publicBatchWrapper89_f2460 a h), (publicBatchWrapper89_f2461 a h), kone]⟩


theorem blockAt_51 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 51).isDummy (row a 51).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 49 47) = (row a 51).isDummy := by rw [(publicBatchWrapper89_f452 a h), (publicBatchWrapper89_f450 a h), (publicBatchWrapper89_f451 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606090), a (.virt 606092), a (.virt 606094), a (.virt 606096)],
    ![a (.virt 606091), a (.virt 606093), a (.virt 606095), a (.virt 606097)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2462 a h)
    · exact (publicBatchWrapper89_f2463 a h)
    · exact (publicBatchWrapper89_f2464 a h)
    · exact (publicBatchWrapper89_f2465 a h)
  · show bor (row a 51).isDummy (band (band (a (.virt 606090)) (a (.virt 606092)))
      (band (a (.virt 606094)) (a (.virt 606096)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2466 a h), ← (publicBatchWrapper89_f2467 a h), ← (publicBatchWrapper89_f2468 a h), ← (publicBatchWrapper89_f2469 a h), (publicBatchWrapper89_f2470 a h), kone]


theorem countAt_51 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 51).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 353140)).val = 32
  rw [(publicBatchWrapper89_f51 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_52 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 52) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f453 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f454 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f455 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f456 a h); rw [kzero] at e; exact e


theorem assetAt_52 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 52).isDummy (row a 52).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 51 11) = (row a 52).isDummy := by rw [(publicBatchWrapper89_f459 a h), (publicBatchWrapper89_f457 a h), (publicBatchWrapper89_f458 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2471 a h), by rw [← hd, ← (publicBatchWrapper89_f2472 a h), (publicBatchWrapper89_f2473 a h), kone]⟩


theorem feeAt_52 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 52).isDummy (row a 52).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 51 11) = (row a 52).isDummy := by rw [(publicBatchWrapper89_f459 a h), (publicBatchWrapper89_f457 a h), (publicBatchWrapper89_f458 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2474 a h), by rw [← hd, ← (publicBatchWrapper89_f2475 a h), (publicBatchWrapper89_f2476 a h), kone]⟩


theorem blockAt_52 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 52).isDummy (row a 52).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 51 11) = (row a 52).isDummy := by rw [(publicBatchWrapper89_f459 a h), (publicBatchWrapper89_f457 a h), (publicBatchWrapper89_f458 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606102), a (.virt 606104), a (.virt 606106), a (.virt 606108)],
    ![a (.virt 606103), a (.virt 606105), a (.virt 606107), a (.virt 606109)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2477 a h)
    · exact (publicBatchWrapper89_f2478 a h)
    · exact (publicBatchWrapper89_f2479 a h)
    · exact (publicBatchWrapper89_f2480 a h)
  · show bor (row a 52).isDummy (band (band (a (.virt 606102)) (a (.virt 606104)))
      (band (a (.virt 606106)) (a (.virt 606108)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2481 a h), ← (publicBatchWrapper89_f2482 a h), ← (publicBatchWrapper89_f2483 a h), ← (publicBatchWrapper89_f2484 a h), (publicBatchWrapper89_f2485 a h), kone]


theorem countAt_52 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 52).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 359935)).val = 32
  rw [(publicBatchWrapper89_f52 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_53 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 53) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f460 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f461 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f462 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f463 a h); rw [kzero] at e; exact e


theorem assetAt_53 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 53).isDummy (row a 53).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 51 55) = (row a 53).isDummy := by rw [(publicBatchWrapper89_f466 a h), (publicBatchWrapper89_f464 a h), (publicBatchWrapper89_f465 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2486 a h), by rw [← hd, ← (publicBatchWrapper89_f2487 a h), (publicBatchWrapper89_f2488 a h), kone]⟩


theorem feeAt_53 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 53).isDummy (row a 53).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 51 55) = (row a 53).isDummy := by rw [(publicBatchWrapper89_f466 a h), (publicBatchWrapper89_f464 a h), (publicBatchWrapper89_f465 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2489 a h), by rw [← hd, ← (publicBatchWrapper89_f2490 a h), (publicBatchWrapper89_f2491 a h), kone]⟩


theorem blockAt_53 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 53).isDummy (row a 53).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 51 55) = (row a 53).isDummy := by rw [(publicBatchWrapper89_f466 a h), (publicBatchWrapper89_f464 a h), (publicBatchWrapper89_f465 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606114), a (.virt 606116), a (.virt 606118), a (.virt 606120)],
    ![a (.virt 606115), a (.virt 606117), a (.virt 606119), a (.virt 606121)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2492 a h)
    · exact (publicBatchWrapper89_f2493 a h)
    · exact (publicBatchWrapper89_f2494 a h)
    · exact (publicBatchWrapper89_f2495 a h)
  · show bor (row a 53).isDummy (band (band (a (.virt 606114)) (a (.virt 606116)))
      (band (a (.virt 606118)) (a (.virt 606120)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2496 a h), ← (publicBatchWrapper89_f2497 a h), ← (publicBatchWrapper89_f2498 a h), ← (publicBatchWrapper89_f2499 a h), (publicBatchWrapper89_f2500 a h), kone]


theorem countAt_53 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 53).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 366730)).val = 32
  rw [(publicBatchWrapper89_f53 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_54 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 54) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f467 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f468 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f469 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f470 a h); rw [kzero] at e; exact e


theorem assetAt_54 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 54).isDummy (row a 54).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 52 19) = (row a 54).isDummy := by rw [(publicBatchWrapper89_f473 a h), (publicBatchWrapper89_f471 a h), (publicBatchWrapper89_f472 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2501 a h), by rw [← hd, ← (publicBatchWrapper89_f2502 a h), (publicBatchWrapper89_f2503 a h), kone]⟩


theorem feeAt_54 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 54).isDummy (row a 54).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 52 19) = (row a 54).isDummy := by rw [(publicBatchWrapper89_f473 a h), (publicBatchWrapper89_f471 a h), (publicBatchWrapper89_f472 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2504 a h), by rw [← hd, ← (publicBatchWrapper89_f2505 a h), (publicBatchWrapper89_f2506 a h), kone]⟩


theorem blockAt_54 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 54).isDummy (row a 54).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 52 19) = (row a 54).isDummy := by rw [(publicBatchWrapper89_f473 a h), (publicBatchWrapper89_f471 a h), (publicBatchWrapper89_f472 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606126), a (.virt 606128), a (.virt 606130), a (.virt 606132)],
    ![a (.virt 606127), a (.virt 606129), a (.virt 606131), a (.virt 606133)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2507 a h)
    · exact (publicBatchWrapper89_f2508 a h)
    · exact (publicBatchWrapper89_f2509 a h)
    · exact (publicBatchWrapper89_f2510 a h)
  · show bor (row a 54).isDummy (band (band (a (.virt 606126)) (a (.virt 606128)))
      (band (a (.virt 606130)) (a (.virt 606132)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2511 a h), ← (publicBatchWrapper89_f2512 a h), ← (publicBatchWrapper89_f2513 a h), ← (publicBatchWrapper89_f2514 a h), (publicBatchWrapper89_f2515 a h), kone]


theorem countAt_54 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 54).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 373525)).val = 32
  rw [(publicBatchWrapper89_f54 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_55 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 55) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f474 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f475 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f476 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f477 a h); rw [kzero] at e; exact e


theorem assetAt_55 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 55).isDummy (row a 55).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 52 63) = (row a 55).isDummy := by rw [(publicBatchWrapper89_f480 a h), (publicBatchWrapper89_f478 a h), (publicBatchWrapper89_f479 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2516 a h), by rw [← hd, ← (publicBatchWrapper89_f2517 a h), (publicBatchWrapper89_f2518 a h), kone]⟩


theorem feeAt_55 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 55).isDummy (row a 55).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 52 63) = (row a 55).isDummy := by rw [(publicBatchWrapper89_f480 a h), (publicBatchWrapper89_f478 a h), (publicBatchWrapper89_f479 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2519 a h), by rw [← hd, ← (publicBatchWrapper89_f2520 a h), (publicBatchWrapper89_f2521 a h), kone]⟩


theorem blockAt_55 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 55).isDummy (row a 55).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 52 63) = (row a 55).isDummy := by rw [(publicBatchWrapper89_f480 a h), (publicBatchWrapper89_f478 a h), (publicBatchWrapper89_f479 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606138), a (.virt 606140), a (.virt 606142), a (.virt 606144)],
    ![a (.virt 606139), a (.virt 606141), a (.virt 606143), a (.virt 606145)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2522 a h)
    · exact (publicBatchWrapper89_f2523 a h)
    · exact (publicBatchWrapper89_f2524 a h)
    · exact (publicBatchWrapper89_f2525 a h)
  · show bor (row a 55).isDummy (band (band (a (.virt 606138)) (a (.virt 606140)))
      (band (a (.virt 606142)) (a (.virt 606144)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2526 a h), ← (publicBatchWrapper89_f2527 a h), ← (publicBatchWrapper89_f2528 a h), ← (publicBatchWrapper89_f2529 a h), (publicBatchWrapper89_f2530 a h), kone]


theorem countAt_55 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 55).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 380320)).val = 32
  rw [(publicBatchWrapper89_f55 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_56 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 56) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f481 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f482 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f483 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f484 a h); rw [kzero] at e; exact e


theorem assetAt_56 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 56).isDummy (row a 56).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 54 27) = (row a 56).isDummy := by rw [(publicBatchWrapper89_f487 a h), (publicBatchWrapper89_f485 a h), (publicBatchWrapper89_f486 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2531 a h), by rw [← hd, ← (publicBatchWrapper89_f2532 a h), (publicBatchWrapper89_f2533 a h), kone]⟩


theorem feeAt_56 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 56).isDummy (row a 56).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 54 27) = (row a 56).isDummy := by rw [(publicBatchWrapper89_f487 a h), (publicBatchWrapper89_f485 a h), (publicBatchWrapper89_f486 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2534 a h), by rw [← hd, ← (publicBatchWrapper89_f2535 a h), (publicBatchWrapper89_f2536 a h), kone]⟩


theorem blockAt_56 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 56).isDummy (row a 56).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 54 27) = (row a 56).isDummy := by rw [(publicBatchWrapper89_f487 a h), (publicBatchWrapper89_f485 a h), (publicBatchWrapper89_f486 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606150), a (.virt 606152), a (.virt 606154), a (.virt 606156)],
    ![a (.virt 606151), a (.virt 606153), a (.virt 606155), a (.virt 606157)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2537 a h)
    · exact (publicBatchWrapper89_f2538 a h)
    · exact (publicBatchWrapper89_f2539 a h)
    · exact (publicBatchWrapper89_f2540 a h)
  · show bor (row a 56).isDummy (band (band (a (.virt 606150)) (a (.virt 606152)))
      (band (a (.virt 606154)) (a (.virt 606156)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2541 a h), ← (publicBatchWrapper89_f2542 a h), ← (publicBatchWrapper89_f2543 a h), ← (publicBatchWrapper89_f2544 a h), (publicBatchWrapper89_f2545 a h), kone]


theorem countAt_56 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 56).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 387115)).val = 32
  rw [(publicBatchWrapper89_f56 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_57 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 57) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f488 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f489 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f490 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f491 a h); rw [kzero] at e; exact e


theorem assetAt_57 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 57).isDummy (row a 57).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 54 71) = (row a 57).isDummy := by rw [(publicBatchWrapper89_f494 a h), (publicBatchWrapper89_f492 a h), (publicBatchWrapper89_f493 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2546 a h), by rw [← hd, ← (publicBatchWrapper89_f2547 a h), (publicBatchWrapper89_f2548 a h), kone]⟩


theorem feeAt_57 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 57).isDummy (row a 57).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 54 71) = (row a 57).isDummy := by rw [(publicBatchWrapper89_f494 a h), (publicBatchWrapper89_f492 a h), (publicBatchWrapper89_f493 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2549 a h), by rw [← hd, ← (publicBatchWrapper89_f2550 a h), (publicBatchWrapper89_f2551 a h), kone]⟩


theorem blockAt_57 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 57).isDummy (row a 57).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 54 71) = (row a 57).isDummy := by rw [(publicBatchWrapper89_f494 a h), (publicBatchWrapper89_f492 a h), (publicBatchWrapper89_f493 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606162), a (.virt 606164), a (.virt 606166), a (.virt 606168)],
    ![a (.virt 606163), a (.virt 606165), a (.virt 606167), a (.virt 606169)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2552 a h)
    · exact (publicBatchWrapper89_f2553 a h)
    · exact (publicBatchWrapper89_f2554 a h)
    · exact (publicBatchWrapper89_f2555 a h)
  · show bor (row a 57).isDummy (band (band (a (.virt 606162)) (a (.virt 606164)))
      (band (a (.virt 606166)) (a (.virt 606168)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2556 a h), ← (publicBatchWrapper89_f2557 a h), ← (publicBatchWrapper89_f2558 a h), ← (publicBatchWrapper89_f2559 a h), (publicBatchWrapper89_f2560 a h), kone]


theorem countAt_57 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 57).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 393910)).val = 32
  rw [(publicBatchWrapper89_f57 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_58 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 58) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f495 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f496 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f497 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f498 a h); rw [kzero] at e; exact e


theorem assetAt_58 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 58).isDummy (row a 58).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 56 35) = (row a 58).isDummy := by rw [(publicBatchWrapper89_f501 a h), (publicBatchWrapper89_f499 a h), (publicBatchWrapper89_f500 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2561 a h), by rw [← hd, ← (publicBatchWrapper89_f2562 a h), (publicBatchWrapper89_f2563 a h), kone]⟩


theorem feeAt_58 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 58).isDummy (row a 58).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 56 35) = (row a 58).isDummy := by rw [(publicBatchWrapper89_f501 a h), (publicBatchWrapper89_f499 a h), (publicBatchWrapper89_f500 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2564 a h), by rw [← hd, ← (publicBatchWrapper89_f2565 a h), (publicBatchWrapper89_f2566 a h), kone]⟩


theorem blockAt_58 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 58).isDummy (row a 58).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 56 35) = (row a 58).isDummy := by rw [(publicBatchWrapper89_f501 a h), (publicBatchWrapper89_f499 a h), (publicBatchWrapper89_f500 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606174), a (.virt 606176), a (.virt 606178), a (.virt 606180)],
    ![a (.virt 606175), a (.virt 606177), a (.virt 606179), a (.virt 606181)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2567 a h)
    · exact (publicBatchWrapper89_f2568 a h)
    · exact (publicBatchWrapper89_f2569 a h)
    · exact (publicBatchWrapper89_f2570 a h)
  · show bor (row a 58).isDummy (band (band (a (.virt 606174)) (a (.virt 606176)))
      (band (a (.virt 606178)) (a (.virt 606180)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2571 a h), ← (publicBatchWrapper89_f2572 a h), ← (publicBatchWrapper89_f2573 a h), ← (publicBatchWrapper89_f2574 a h), (publicBatchWrapper89_f2575 a h), kone]


theorem countAt_58 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 58).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 400705)).val = 32
  rw [(publicBatchWrapper89_f58 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_59 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 59) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f502 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f503 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f504 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f505 a h); rw [kzero] at e; exact e


theorem assetAt_59 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 59).isDummy (row a 59).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 56 79) = (row a 59).isDummy := by rw [(publicBatchWrapper89_f508 a h), (publicBatchWrapper89_f506 a h), (publicBatchWrapper89_f507 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2576 a h), by rw [← hd, ← (publicBatchWrapper89_f2577 a h), (publicBatchWrapper89_f2578 a h), kone]⟩


theorem feeAt_59 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 59).isDummy (row a 59).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 56 79) = (row a 59).isDummy := by rw [(publicBatchWrapper89_f508 a h), (publicBatchWrapper89_f506 a h), (publicBatchWrapper89_f507 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2579 a h), by rw [← hd, ← (publicBatchWrapper89_f2580 a h), (publicBatchWrapper89_f2581 a h), kone]⟩


theorem blockAt_59 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 59).isDummy (row a 59).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 56 79) = (row a 59).isDummy := by rw [(publicBatchWrapper89_f508 a h), (publicBatchWrapper89_f506 a h), (publicBatchWrapper89_f507 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606186), a (.virt 606188), a (.virt 606190), a (.virt 606192)],
    ![a (.virt 606187), a (.virt 606189), a (.virt 606191), a (.virt 606193)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2582 a h)
    · exact (publicBatchWrapper89_f2583 a h)
    · exact (publicBatchWrapper89_f2584 a h)
    · exact (publicBatchWrapper89_f2585 a h)
  · show bor (row a 59).isDummy (band (band (a (.virt 606186)) (a (.virt 606188)))
      (band (a (.virt 606190)) (a (.virt 606192)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2586 a h), ← (publicBatchWrapper89_f2587 a h), ← (publicBatchWrapper89_f2588 a h), ← (publicBatchWrapper89_f2589 a h), (publicBatchWrapper89_f2590 a h), kone]


theorem countAt_59 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 59).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 407500)).val = 32
  rw [(publicBatchWrapper89_f59 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_60 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 60) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f509 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f510 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f511 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f512 a h); rw [kzero] at e; exact e


theorem assetAt_60 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 60).isDummy (row a 60).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 58 43) = (row a 60).isDummy := by rw [(publicBatchWrapper89_f515 a h), (publicBatchWrapper89_f513 a h), (publicBatchWrapper89_f514 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2591 a h), by rw [← hd, ← (publicBatchWrapper89_f2592 a h), (publicBatchWrapper89_f2593 a h), kone]⟩


theorem feeAt_60 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 60).isDummy (row a 60).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 58 43) = (row a 60).isDummy := by rw [(publicBatchWrapper89_f515 a h), (publicBatchWrapper89_f513 a h), (publicBatchWrapper89_f514 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2594 a h), by rw [← hd, ← (publicBatchWrapper89_f2595 a h), (publicBatchWrapper89_f2596 a h), kone]⟩


theorem blockAt_60 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 60).isDummy (row a 60).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 58 43) = (row a 60).isDummy := by rw [(publicBatchWrapper89_f515 a h), (publicBatchWrapper89_f513 a h), (publicBatchWrapper89_f514 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606198), a (.virt 606200), a (.virt 606202), a (.virt 606204)],
    ![a (.virt 606199), a (.virt 606201), a (.virt 606203), a (.virt 606205)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2597 a h)
    · exact (publicBatchWrapper89_f2598 a h)
    · exact (publicBatchWrapper89_f2599 a h)
    · exact (publicBatchWrapper89_f2600 a h)
  · show bor (row a 60).isDummy (band (band (a (.virt 606198)) (a (.virt 606200)))
      (band (a (.virt 606202)) (a (.virt 606204)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2601 a h), ← (publicBatchWrapper89_f2602 a h), ← (publicBatchWrapper89_f2603 a h), ← (publicBatchWrapper89_f2604 a h), (publicBatchWrapper89_f2605 a h), kone]


theorem countAt_60 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 60).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 414295)).val = 32
  rw [(publicBatchWrapper89_f60 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_61 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 61) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f516 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f517 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f518 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f519 a h); rw [kzero] at e; exact e


theorem assetAt_61 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 61).isDummy (row a 61).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 59 7) = (row a 61).isDummy := by rw [(publicBatchWrapper89_f522 a h), (publicBatchWrapper89_f520 a h), (publicBatchWrapper89_f521 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2606 a h), by rw [← hd, ← (publicBatchWrapper89_f2607 a h), (publicBatchWrapper89_f2608 a h), kone]⟩


theorem feeAt_61 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 61).isDummy (row a 61).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 59 7) = (row a 61).isDummy := by rw [(publicBatchWrapper89_f522 a h), (publicBatchWrapper89_f520 a h), (publicBatchWrapper89_f521 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2609 a h), by rw [← hd, ← (publicBatchWrapper89_f2610 a h), (publicBatchWrapper89_f2611 a h), kone]⟩


theorem blockAt_61 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 61).isDummy (row a 61).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 59 7) = (row a 61).isDummy := by rw [(publicBatchWrapper89_f522 a h), (publicBatchWrapper89_f520 a h), (publicBatchWrapper89_f521 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606210), a (.virt 606212), a (.virt 606214), a (.virt 606216)],
    ![a (.virt 606211), a (.virt 606213), a (.virt 606215), a (.virt 606217)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2612 a h)
    · exact (publicBatchWrapper89_f2613 a h)
    · exact (publicBatchWrapper89_f2614 a h)
    · exact (publicBatchWrapper89_f2615 a h)
  · show bor (row a 61).isDummy (band (band (a (.virt 606210)) (a (.virt 606212)))
      (band (a (.virt 606214)) (a (.virt 606216)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2616 a h), ← (publicBatchWrapper89_f2617 a h), ← (publicBatchWrapper89_f2618 a h), ← (publicBatchWrapper89_f2619 a h), (publicBatchWrapper89_f2620 a h), kone]


theorem countAt_61 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 61).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 421090)).val = 32
  rw [(publicBatchWrapper89_f61 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_62 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 62) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f523 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f524 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f525 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f526 a h); rw [kzero] at e; exact e


theorem assetAt_62 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 62).isDummy (row a 62).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 59 51) = (row a 62).isDummy := by rw [(publicBatchWrapper89_f529 a h), (publicBatchWrapper89_f527 a h), (publicBatchWrapper89_f528 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2621 a h), by rw [← hd, ← (publicBatchWrapper89_f2622 a h), (publicBatchWrapper89_f2623 a h), kone]⟩


theorem feeAt_62 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 62).isDummy (row a 62).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 59 51) = (row a 62).isDummy := by rw [(publicBatchWrapper89_f529 a h), (publicBatchWrapper89_f527 a h), (publicBatchWrapper89_f528 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2624 a h), by rw [← hd, ← (publicBatchWrapper89_f2625 a h), (publicBatchWrapper89_f2626 a h), kone]⟩


theorem blockAt_62 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 62).isDummy (row a 62).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 59 51) = (row a 62).isDummy := by rw [(publicBatchWrapper89_f529 a h), (publicBatchWrapper89_f527 a h), (publicBatchWrapper89_f528 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606222), a (.virt 606224), a (.virt 606226), a (.virt 606228)],
    ![a (.virt 606223), a (.virt 606225), a (.virt 606227), a (.virt 606229)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2627 a h)
    · exact (publicBatchWrapper89_f2628 a h)
    · exact (publicBatchWrapper89_f2629 a h)
    · exact (publicBatchWrapper89_f2630 a h)
  · show bor (row a 62).isDummy (band (band (a (.virt 606222)) (a (.virt 606224)))
      (band (a (.virt 606226)) (a (.virt 606228)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2631 a h), ← (publicBatchWrapper89_f2632 a h), ← (publicBatchWrapper89_f2633 a h), ← (publicBatchWrapper89_f2634 a h), (publicBatchWrapper89_f2635 a h), kone]


theorem countAt_62 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 62).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 427885)).val = 32
  rw [(publicBatchWrapper89_f62 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_63 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 63) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f530 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f531 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f532 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f533 a h); rw [kzero] at e; exact e


theorem assetAt_63 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 63).isDummy (row a 63).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 61 15) = (row a 63).isDummy := by rw [(publicBatchWrapper89_f536 a h), (publicBatchWrapper89_f534 a h), (publicBatchWrapper89_f535 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2636 a h), by rw [← hd, ← (publicBatchWrapper89_f2637 a h), (publicBatchWrapper89_f2638 a h), kone]⟩


theorem feeAt_63 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 63).isDummy (row a 63).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 61 15) = (row a 63).isDummy := by rw [(publicBatchWrapper89_f536 a h), (publicBatchWrapper89_f534 a h), (publicBatchWrapper89_f535 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2639 a h), by rw [← hd, ← (publicBatchWrapper89_f2640 a h), (publicBatchWrapper89_f2641 a h), kone]⟩


theorem blockAt_63 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 63).isDummy (row a 63).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 61 15) = (row a 63).isDummy := by rw [(publicBatchWrapper89_f536 a h), (publicBatchWrapper89_f534 a h), (publicBatchWrapper89_f535 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606234), a (.virt 606236), a (.virt 606238), a (.virt 606240)],
    ![a (.virt 606235), a (.virt 606237), a (.virt 606239), a (.virt 606241)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2642 a h)
    · exact (publicBatchWrapper89_f2643 a h)
    · exact (publicBatchWrapper89_f2644 a h)
    · exact (publicBatchWrapper89_f2645 a h)
  · show bor (row a 63).isDummy (band (band (a (.virt 606234)) (a (.virt 606236)))
      (band (a (.virt 606238)) (a (.virt 606240)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2646 a h), ← (publicBatchWrapper89_f2647 a h), ← (publicBatchWrapper89_f2648 a h), ← (publicBatchWrapper89_f2649 a h), (publicBatchWrapper89_f2650 a h), kone]


theorem countAt_63 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 63).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 434680)).val = 32
  rw [(publicBatchWrapper89_f63 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_64 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 64) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f537 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f538 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f539 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f540 a h); rw [kzero] at e; exact e


theorem assetAt_64 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 64).isDummy (row a 64).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 61 59) = (row a 64).isDummy := by rw [(publicBatchWrapper89_f543 a h), (publicBatchWrapper89_f541 a h), (publicBatchWrapper89_f542 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2651 a h), by rw [← hd, ← (publicBatchWrapper89_f2652 a h), (publicBatchWrapper89_f2653 a h), kone]⟩


theorem feeAt_64 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 64).isDummy (row a 64).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 61 59) = (row a 64).isDummy := by rw [(publicBatchWrapper89_f543 a h), (publicBatchWrapper89_f541 a h), (publicBatchWrapper89_f542 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2654 a h), by rw [← hd, ← (publicBatchWrapper89_f2655 a h), (publicBatchWrapper89_f2656 a h), kone]⟩


theorem blockAt_64 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 64).isDummy (row a 64).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 61 59) = (row a 64).isDummy := by rw [(publicBatchWrapper89_f543 a h), (publicBatchWrapper89_f541 a h), (publicBatchWrapper89_f542 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606246), a (.virt 606248), a (.virt 606250), a (.virt 606252)],
    ![a (.virt 606247), a (.virt 606249), a (.virt 606251), a (.virt 606253)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2657 a h)
    · exact (publicBatchWrapper89_f2658 a h)
    · exact (publicBatchWrapper89_f2659 a h)
    · exact (publicBatchWrapper89_f2660 a h)
  · show bor (row a 64).isDummy (band (band (a (.virt 606246)) (a (.virt 606248)))
      (band (a (.virt 606250)) (a (.virt 606252)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2661 a h), ← (publicBatchWrapper89_f2662 a h), ← (publicBatchWrapper89_f2663 a h), ← (publicBatchWrapper89_f2664 a h), (publicBatchWrapper89_f2665 a h), kone]


theorem countAt_64 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 64).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 441475)).val = 32
  rw [(publicBatchWrapper89_f64 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_65 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 65) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f544 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f545 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f546 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f547 a h); rw [kzero] at e; exact e


theorem assetAt_65 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 65).isDummy (row a 65).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 63 23) = (row a 65).isDummy := by rw [(publicBatchWrapper89_f550 a h), (publicBatchWrapper89_f548 a h), (publicBatchWrapper89_f549 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2666 a h), by rw [← hd, ← (publicBatchWrapper89_f2667 a h), (publicBatchWrapper89_f2668 a h), kone]⟩


theorem feeAt_65 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 65).isDummy (row a 65).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 63 23) = (row a 65).isDummy := by rw [(publicBatchWrapper89_f550 a h), (publicBatchWrapper89_f548 a h), (publicBatchWrapper89_f549 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2669 a h), by rw [← hd, ← (publicBatchWrapper89_f2670 a h), (publicBatchWrapper89_f2671 a h), kone]⟩


theorem blockAt_65 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 65).isDummy (row a 65).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 63 23) = (row a 65).isDummy := by rw [(publicBatchWrapper89_f550 a h), (publicBatchWrapper89_f548 a h), (publicBatchWrapper89_f549 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606258), a (.virt 606260), a (.virt 606262), a (.virt 606264)],
    ![a (.virt 606259), a (.virt 606261), a (.virt 606263), a (.virt 606265)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2672 a h)
    · exact (publicBatchWrapper89_f2673 a h)
    · exact (publicBatchWrapper89_f2674 a h)
    · exact (publicBatchWrapper89_f2675 a h)
  · show bor (row a 65).isDummy (band (band (a (.virt 606258)) (a (.virt 606260)))
      (band (a (.virt 606262)) (a (.virt 606264)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2676 a h), ← (publicBatchWrapper89_f2677 a h), ← (publicBatchWrapper89_f2678 a h), ← (publicBatchWrapper89_f2679 a h), (publicBatchWrapper89_f2680 a h), kone]


theorem countAt_65 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 65).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 448270)).val = 32
  rw [(publicBatchWrapper89_f65 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_66 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 66) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f551 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f552 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f553 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f554 a h); rw [kzero] at e; exact e


theorem assetAt_66 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 66).isDummy (row a 66).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 63 67) = (row a 66).isDummy := by rw [(publicBatchWrapper89_f557 a h), (publicBatchWrapper89_f555 a h), (publicBatchWrapper89_f556 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2681 a h), by rw [← hd, ← (publicBatchWrapper89_f2682 a h), (publicBatchWrapper89_f2683 a h), kone]⟩


theorem feeAt_66 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 66).isDummy (row a 66).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 63 67) = (row a 66).isDummy := by rw [(publicBatchWrapper89_f557 a h), (publicBatchWrapper89_f555 a h), (publicBatchWrapper89_f556 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2684 a h), by rw [← hd, ← (publicBatchWrapper89_f2685 a h), (publicBatchWrapper89_f2686 a h), kone]⟩


theorem blockAt_66 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 66).isDummy (row a 66).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 63 67) = (row a 66).isDummy := by rw [(publicBatchWrapper89_f557 a h), (publicBatchWrapper89_f555 a h), (publicBatchWrapper89_f556 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606270), a (.virt 606272), a (.virt 606274), a (.virt 606276)],
    ![a (.virt 606271), a (.virt 606273), a (.virt 606275), a (.virt 606277)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2687 a h)
    · exact (publicBatchWrapper89_f2688 a h)
    · exact (publicBatchWrapper89_f2689 a h)
    · exact (publicBatchWrapper89_f2690 a h)
  · show bor (row a 66).isDummy (band (band (a (.virt 606270)) (a (.virt 606272)))
      (band (a (.virt 606274)) (a (.virt 606276)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2691 a h), ← (publicBatchWrapper89_f2692 a h), ← (publicBatchWrapper89_f2693 a h), ← (publicBatchWrapper89_f2694 a h), (publicBatchWrapper89_f2695 a h), kone]


theorem countAt_66 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 66).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 455065)).val = 32
  rw [(publicBatchWrapper89_f66 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_67 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 67) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f558 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f559 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f560 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f561 a h); rw [kzero] at e; exact e


theorem assetAt_67 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 67).isDummy (row a 67).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 64 31) = (row a 67).isDummy := by rw [(publicBatchWrapper89_f564 a h), (publicBatchWrapper89_f562 a h), (publicBatchWrapper89_f563 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2696 a h), by rw [← hd, ← (publicBatchWrapper89_f2697 a h), (publicBatchWrapper89_f2698 a h), kone]⟩


theorem feeAt_67 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 67).isDummy (row a 67).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 64 31) = (row a 67).isDummy := by rw [(publicBatchWrapper89_f564 a h), (publicBatchWrapper89_f562 a h), (publicBatchWrapper89_f563 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2699 a h), by rw [← hd, ← (publicBatchWrapper89_f2700 a h), (publicBatchWrapper89_f2701 a h), kone]⟩


theorem blockAt_67 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 67).isDummy (row a 67).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 64 31) = (row a 67).isDummy := by rw [(publicBatchWrapper89_f564 a h), (publicBatchWrapper89_f562 a h), (publicBatchWrapper89_f563 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606282), a (.virt 606284), a (.virt 606286), a (.virt 606288)],
    ![a (.virt 606283), a (.virt 606285), a (.virt 606287), a (.virt 606289)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2702 a h)
    · exact (publicBatchWrapper89_f2703 a h)
    · exact (publicBatchWrapper89_f2704 a h)
    · exact (publicBatchWrapper89_f2705 a h)
  · show bor (row a 67).isDummy (band (band (a (.virt 606282)) (a (.virt 606284)))
      (band (a (.virt 606286)) (a (.virt 606288)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2706 a h), ← (publicBatchWrapper89_f2707 a h), ← (publicBatchWrapper89_f2708 a h), ← (publicBatchWrapper89_f2709 a h), (publicBatchWrapper89_f2710 a h), kone]


theorem countAt_67 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 67).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 461860)).val = 32
  rw [(publicBatchWrapper89_f67 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_68 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 68) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f565 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f566 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f567 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f568 a h); rw [kzero] at e; exact e


theorem assetAt_68 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 68).isDummy (row a 68).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 64 75) = (row a 68).isDummy := by rw [(publicBatchWrapper89_f571 a h), (publicBatchWrapper89_f569 a h), (publicBatchWrapper89_f570 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2711 a h), by rw [← hd, ← (publicBatchWrapper89_f2712 a h), (publicBatchWrapper89_f2713 a h), kone]⟩


theorem feeAt_68 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 68).isDummy (row a 68).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 64 75) = (row a 68).isDummy := by rw [(publicBatchWrapper89_f571 a h), (publicBatchWrapper89_f569 a h), (publicBatchWrapper89_f570 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2714 a h), by rw [← hd, ← (publicBatchWrapper89_f2715 a h), (publicBatchWrapper89_f2716 a h), kone]⟩


theorem blockAt_68 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 68).isDummy (row a 68).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 64 75) = (row a 68).isDummy := by rw [(publicBatchWrapper89_f571 a h), (publicBatchWrapper89_f569 a h), (publicBatchWrapper89_f570 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606294), a (.virt 606296), a (.virt 606298), a (.virt 606300)],
    ![a (.virt 606295), a (.virt 606297), a (.virt 606299), a (.virt 606301)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2717 a h)
    · exact (publicBatchWrapper89_f2718 a h)
    · exact (publicBatchWrapper89_f2719 a h)
    · exact (publicBatchWrapper89_f2720 a h)
  · show bor (row a 68).isDummy (band (band (a (.virt 606294)) (a (.virt 606296)))
      (band (a (.virt 606298)) (a (.virt 606300)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2721 a h), ← (publicBatchWrapper89_f2722 a h), ← (publicBatchWrapper89_f2723 a h), ← (publicBatchWrapper89_f2724 a h), (publicBatchWrapper89_f2725 a h), kone]


theorem countAt_68 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 68).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 468655)).val = 32
  rw [(publicBatchWrapper89_f68 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_69 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 69) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f572 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f573 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f574 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f575 a h); rw [kzero] at e; exact e


theorem assetAt_69 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 69).isDummy (row a 69).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 66 39) = (row a 69).isDummy := by rw [(publicBatchWrapper89_f578 a h), (publicBatchWrapper89_f576 a h), (publicBatchWrapper89_f577 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2726 a h), by rw [← hd, ← (publicBatchWrapper89_f2727 a h), (publicBatchWrapper89_f2728 a h), kone]⟩


theorem feeAt_69 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 69).isDummy (row a 69).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 66 39) = (row a 69).isDummy := by rw [(publicBatchWrapper89_f578 a h), (publicBatchWrapper89_f576 a h), (publicBatchWrapper89_f577 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2729 a h), by rw [← hd, ← (publicBatchWrapper89_f2730 a h), (publicBatchWrapper89_f2731 a h), kone]⟩


theorem blockAt_69 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 69).isDummy (row a 69).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 66 39) = (row a 69).isDummy := by rw [(publicBatchWrapper89_f578 a h), (publicBatchWrapper89_f576 a h), (publicBatchWrapper89_f577 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606306), a (.virt 606308), a (.virt 606310), a (.virt 606312)],
    ![a (.virt 606307), a (.virt 606309), a (.virt 606311), a (.virt 606313)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2732 a h)
    · exact (publicBatchWrapper89_f2733 a h)
    · exact (publicBatchWrapper89_f2734 a h)
    · exact (publicBatchWrapper89_f2735 a h)
  · show bor (row a 69).isDummy (band (band (a (.virt 606306)) (a (.virt 606308)))
      (band (a (.virt 606310)) (a (.virt 606312)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2736 a h), ← (publicBatchWrapper89_f2737 a h), ← (publicBatchWrapper89_f2738 a h), ← (publicBatchWrapper89_f2739 a h), (publicBatchWrapper89_f2740 a h), kone]


theorem countAt_69 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 69).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 475450)).val = 32
  rw [(publicBatchWrapper89_f69 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_70 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 70) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f579 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f580 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f581 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f582 a h); rw [kzero] at e; exact e


theorem assetAt_70 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 70).isDummy (row a 70).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 68 3) = (row a 70).isDummy := by rw [(publicBatchWrapper89_f585 a h), (publicBatchWrapper89_f583 a h), (publicBatchWrapper89_f584 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2741 a h), by rw [← hd, ← (publicBatchWrapper89_f2742 a h), (publicBatchWrapper89_f2743 a h), kone]⟩


theorem feeAt_70 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 70).isDummy (row a 70).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 68 3) = (row a 70).isDummy := by rw [(publicBatchWrapper89_f585 a h), (publicBatchWrapper89_f583 a h), (publicBatchWrapper89_f584 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2744 a h), by rw [← hd, ← (publicBatchWrapper89_f2745 a h), (publicBatchWrapper89_f2746 a h), kone]⟩


theorem blockAt_70 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 70).isDummy (row a 70).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 68 3) = (row a 70).isDummy := by rw [(publicBatchWrapper89_f585 a h), (publicBatchWrapper89_f583 a h), (publicBatchWrapper89_f584 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606318), a (.virt 606320), a (.virt 606322), a (.virt 606324)],
    ![a (.virt 606319), a (.virt 606321), a (.virt 606323), a (.virt 606325)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2747 a h)
    · exact (publicBatchWrapper89_f2748 a h)
    · exact (publicBatchWrapper89_f2749 a h)
    · exact (publicBatchWrapper89_f2750 a h)
  · show bor (row a 70).isDummy (band (band (a (.virt 606318)) (a (.virt 606320)))
      (band (a (.virt 606322)) (a (.virt 606324)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2751 a h), ← (publicBatchWrapper89_f2752 a h), ← (publicBatchWrapper89_f2753 a h), ← (publicBatchWrapper89_f2754 a h), (publicBatchWrapper89_f2755 a h), kone]


theorem countAt_70 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 70).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 482245)).val = 32
  rw [(publicBatchWrapper89_f70 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_71 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 71) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f586 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f587 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f588 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f589 a h); rw [kzero] at e; exact e


theorem assetAt_71 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 71).isDummy (row a 71).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 68 47) = (row a 71).isDummy := by rw [(publicBatchWrapper89_f592 a h), (publicBatchWrapper89_f590 a h), (publicBatchWrapper89_f591 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2756 a h), by rw [← hd, ← (publicBatchWrapper89_f2757 a h), (publicBatchWrapper89_f2758 a h), kone]⟩


theorem feeAt_71 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 71).isDummy (row a 71).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 68 47) = (row a 71).isDummy := by rw [(publicBatchWrapper89_f592 a h), (publicBatchWrapper89_f590 a h), (publicBatchWrapper89_f591 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2759 a h), by rw [← hd, ← (publicBatchWrapper89_f2760 a h), (publicBatchWrapper89_f2761 a h), kone]⟩


theorem blockAt_71 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 71).isDummy (row a 71).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 68 47) = (row a 71).isDummy := by rw [(publicBatchWrapper89_f592 a h), (publicBatchWrapper89_f590 a h), (publicBatchWrapper89_f591 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606330), a (.virt 606332), a (.virt 606334), a (.virt 606336)],
    ![a (.virt 606331), a (.virt 606333), a (.virt 606335), a (.virt 606337)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2762 a h)
    · exact (publicBatchWrapper89_f2763 a h)
    · exact (publicBatchWrapper89_f2764 a h)
    · exact (publicBatchWrapper89_f2765 a h)
  · show bor (row a 71).isDummy (band (band (a (.virt 606330)) (a (.virt 606332)))
      (band (a (.virt 606334)) (a (.virt 606336)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2766 a h), ← (publicBatchWrapper89_f2767 a h), ← (publicBatchWrapper89_f2768 a h), ← (publicBatchWrapper89_f2769 a h), (publicBatchWrapper89_f2770 a h), kone]


theorem countAt_71 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 71).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 489040)).val = 32
  rw [(publicBatchWrapper89_f71 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_72 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 72) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f593 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f594 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f595 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f596 a h); rw [kzero] at e; exact e


theorem assetAt_72 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 72).isDummy (row a 72).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 70 11) = (row a 72).isDummy := by rw [(publicBatchWrapper89_f599 a h), (publicBatchWrapper89_f597 a h), (publicBatchWrapper89_f598 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2771 a h), by rw [← hd, ← (publicBatchWrapper89_f2772 a h), (publicBatchWrapper89_f2773 a h), kone]⟩


theorem feeAt_72 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 72).isDummy (row a 72).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 70 11) = (row a 72).isDummy := by rw [(publicBatchWrapper89_f599 a h), (publicBatchWrapper89_f597 a h), (publicBatchWrapper89_f598 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2774 a h), by rw [← hd, ← (publicBatchWrapper89_f2775 a h), (publicBatchWrapper89_f2776 a h), kone]⟩


theorem blockAt_72 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 72).isDummy (row a 72).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 70 11) = (row a 72).isDummy := by rw [(publicBatchWrapper89_f599 a h), (publicBatchWrapper89_f597 a h), (publicBatchWrapper89_f598 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606342), a (.virt 606344), a (.virt 606346), a (.virt 606348)],
    ![a (.virt 606343), a (.virt 606345), a (.virt 606347), a (.virt 606349)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2777 a h)
    · exact (publicBatchWrapper89_f2778 a h)
    · exact (publicBatchWrapper89_f2779 a h)
    · exact (publicBatchWrapper89_f2780 a h)
  · show bor (row a 72).isDummy (band (band (a (.virt 606342)) (a (.virt 606344)))
      (band (a (.virt 606346)) (a (.virt 606348)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2781 a h), ← (publicBatchWrapper89_f2782 a h), ← (publicBatchWrapper89_f2783 a h), ← (publicBatchWrapper89_f2784 a h), (publicBatchWrapper89_f2785 a h), kone]


theorem countAt_72 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 72).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 495835)).val = 32
  rw [(publicBatchWrapper89_f72 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_73 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 73) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f600 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f601 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f602 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f603 a h); rw [kzero] at e; exact e


theorem assetAt_73 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 73).isDummy (row a 73).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 70 55) = (row a 73).isDummy := by rw [(publicBatchWrapper89_f606 a h), (publicBatchWrapper89_f604 a h), (publicBatchWrapper89_f605 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2786 a h), by rw [← hd, ← (publicBatchWrapper89_f2787 a h), (publicBatchWrapper89_f2788 a h), kone]⟩


theorem feeAt_73 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 73).isDummy (row a 73).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 70 55) = (row a 73).isDummy := by rw [(publicBatchWrapper89_f606 a h), (publicBatchWrapper89_f604 a h), (publicBatchWrapper89_f605 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2789 a h), by rw [← hd, ← (publicBatchWrapper89_f2790 a h), (publicBatchWrapper89_f2791 a h), kone]⟩


theorem blockAt_73 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 73).isDummy (row a 73).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 70 55) = (row a 73).isDummy := by rw [(publicBatchWrapper89_f606 a h), (publicBatchWrapper89_f604 a h), (publicBatchWrapper89_f605 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606354), a (.virt 606356), a (.virt 606358), a (.virt 606360)],
    ![a (.virt 606355), a (.virt 606357), a (.virt 606359), a (.virt 606361)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2792 a h)
    · exact (publicBatchWrapper89_f2793 a h)
    · exact (publicBatchWrapper89_f2794 a h)
    · exact (publicBatchWrapper89_f2795 a h)
  · show bor (row a 73).isDummy (band (band (a (.virt 606354)) (a (.virt 606356)))
      (band (a (.virt 606358)) (a (.virt 606360)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2796 a h), ← (publicBatchWrapper89_f2797 a h), ← (publicBatchWrapper89_f2798 a h), ← (publicBatchWrapper89_f2799 a h), (publicBatchWrapper89_f2800 a h), kone]


theorem countAt_73 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 73).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 502630)).val = 32
  rw [(publicBatchWrapper89_f73 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_74 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 74) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f607 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f608 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f609 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f610 a h); rw [kzero] at e; exact e


theorem assetAt_74 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 74).isDummy (row a 74).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 71 19) = (row a 74).isDummy := by rw [(publicBatchWrapper89_f613 a h), (publicBatchWrapper89_f611 a h), (publicBatchWrapper89_f612 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2801 a h), by rw [← hd, ← (publicBatchWrapper89_f2802 a h), (publicBatchWrapper89_f2803 a h), kone]⟩


theorem feeAt_74 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 74).isDummy (row a 74).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 71 19) = (row a 74).isDummy := by rw [(publicBatchWrapper89_f613 a h), (publicBatchWrapper89_f611 a h), (publicBatchWrapper89_f612 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2804 a h), by rw [← hd, ← (publicBatchWrapper89_f2805 a h), (publicBatchWrapper89_f2806 a h), kone]⟩


theorem blockAt_74 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 74).isDummy (row a 74).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 71 19) = (row a 74).isDummy := by rw [(publicBatchWrapper89_f613 a h), (publicBatchWrapper89_f611 a h), (publicBatchWrapper89_f612 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606366), a (.virt 606368), a (.virt 606370), a (.virt 606372)],
    ![a (.virt 606367), a (.virt 606369), a (.virt 606371), a (.virt 606373)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2807 a h)
    · exact (publicBatchWrapper89_f2808 a h)
    · exact (publicBatchWrapper89_f2809 a h)
    · exact (publicBatchWrapper89_f2810 a h)
  · show bor (row a 74).isDummy (band (band (a (.virt 606366)) (a (.virt 606368)))
      (band (a (.virt 606370)) (a (.virt 606372)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2811 a h), ← (publicBatchWrapper89_f2812 a h), ← (publicBatchWrapper89_f2813 a h), ← (publicBatchWrapper89_f2814 a h), (publicBatchWrapper89_f2815 a h), kone]


theorem countAt_74 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 74).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 509425)).val = 32
  rw [(publicBatchWrapper89_f74 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_75 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 75) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f614 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f615 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f616 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f617 a h); rw [kzero] at e; exact e


theorem assetAt_75 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 75).isDummy (row a 75).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 71 63) = (row a 75).isDummy := by rw [(publicBatchWrapper89_f620 a h), (publicBatchWrapper89_f618 a h), (publicBatchWrapper89_f619 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2816 a h), by rw [← hd, ← (publicBatchWrapper89_f2817 a h), (publicBatchWrapper89_f2818 a h), kone]⟩


theorem feeAt_75 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 75).isDummy (row a 75).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 71 63) = (row a 75).isDummy := by rw [(publicBatchWrapper89_f620 a h), (publicBatchWrapper89_f618 a h), (publicBatchWrapper89_f619 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2819 a h), by rw [← hd, ← (publicBatchWrapper89_f2820 a h), (publicBatchWrapper89_f2821 a h), kone]⟩


theorem blockAt_75 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 75).isDummy (row a 75).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 71 63) = (row a 75).isDummy := by rw [(publicBatchWrapper89_f620 a h), (publicBatchWrapper89_f618 a h), (publicBatchWrapper89_f619 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606378), a (.virt 606380), a (.virt 606382), a (.virt 606384)],
    ![a (.virt 606379), a (.virt 606381), a (.virt 606383), a (.virt 606385)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2822 a h)
    · exact (publicBatchWrapper89_f2823 a h)
    · exact (publicBatchWrapper89_f2824 a h)
    · exact (publicBatchWrapper89_f2825 a h)
  · show bor (row a 75).isDummy (band (band (a (.virt 606378)) (a (.virt 606380)))
      (band (a (.virt 606382)) (a (.virt 606384)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2826 a h), ← (publicBatchWrapper89_f2827 a h), ← (publicBatchWrapper89_f2828 a h), ← (publicBatchWrapper89_f2829 a h), (publicBatchWrapper89_f2830 a h), kone]


theorem countAt_75 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 75).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 516220)).val = 32
  rw [(publicBatchWrapper89_f75 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_76 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 76) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f621 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f622 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f623 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f624 a h); rw [kzero] at e; exact e


theorem assetAt_76 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 76).isDummy (row a 76).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 73 27) = (row a 76).isDummy := by rw [(publicBatchWrapper89_f627 a h), (publicBatchWrapper89_f625 a h), (publicBatchWrapper89_f626 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2831 a h), by rw [← hd, ← (publicBatchWrapper89_f2832 a h), (publicBatchWrapper89_f2833 a h), kone]⟩


theorem feeAt_76 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 76).isDummy (row a 76).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 73 27) = (row a 76).isDummy := by rw [(publicBatchWrapper89_f627 a h), (publicBatchWrapper89_f625 a h), (publicBatchWrapper89_f626 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2834 a h), by rw [← hd, ← (publicBatchWrapper89_f2835 a h), (publicBatchWrapper89_f2836 a h), kone]⟩


theorem blockAt_76 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 76).isDummy (row a 76).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 73 27) = (row a 76).isDummy := by rw [(publicBatchWrapper89_f627 a h), (publicBatchWrapper89_f625 a h), (publicBatchWrapper89_f626 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606390), a (.virt 606392), a (.virt 606394), a (.virt 606396)],
    ![a (.virt 606391), a (.virt 606393), a (.virt 606395), a (.virt 606397)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2837 a h)
    · exact (publicBatchWrapper89_f2838 a h)
    · exact (publicBatchWrapper89_f2839 a h)
    · exact (publicBatchWrapper89_f2840 a h)
  · show bor (row a 76).isDummy (band (band (a (.virt 606390)) (a (.virt 606392)))
      (band (a (.virt 606394)) (a (.virt 606396)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2841 a h), ← (publicBatchWrapper89_f2842 a h), ← (publicBatchWrapper89_f2843 a h), ← (publicBatchWrapper89_f2844 a h), (publicBatchWrapper89_f2845 a h), kone]


theorem countAt_76 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 76).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 523015)).val = 32
  rw [(publicBatchWrapper89_f76 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_77 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 77) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f628 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f629 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f630 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f631 a h); rw [kzero] at e; exact e


theorem assetAt_77 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 77).isDummy (row a 77).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 73 71) = (row a 77).isDummy := by rw [(publicBatchWrapper89_f634 a h), (publicBatchWrapper89_f632 a h), (publicBatchWrapper89_f633 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2846 a h), by rw [← hd, ← (publicBatchWrapper89_f2847 a h), (publicBatchWrapper89_f2848 a h), kone]⟩


theorem feeAt_77 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 77).isDummy (row a 77).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 73 71) = (row a 77).isDummy := by rw [(publicBatchWrapper89_f634 a h), (publicBatchWrapper89_f632 a h), (publicBatchWrapper89_f633 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2849 a h), by rw [← hd, ← (publicBatchWrapper89_f2850 a h), (publicBatchWrapper89_f2851 a h), kone]⟩


theorem blockAt_77 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 77).isDummy (row a 77).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 73 71) = (row a 77).isDummy := by rw [(publicBatchWrapper89_f634 a h), (publicBatchWrapper89_f632 a h), (publicBatchWrapper89_f633 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606402), a (.virt 606404), a (.virt 606406), a (.virt 606408)],
    ![a (.virt 606403), a (.virt 606405), a (.virt 606407), a (.virt 606409)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2852 a h)
    · exact (publicBatchWrapper89_f2853 a h)
    · exact (publicBatchWrapper89_f2854 a h)
    · exact (publicBatchWrapper89_f2855 a h)
  · show bor (row a 77).isDummy (band (band (a (.virt 606402)) (a (.virt 606404)))
      (band (a (.virt 606406)) (a (.virt 606408)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2856 a h), ← (publicBatchWrapper89_f2857 a h), ← (publicBatchWrapper89_f2858 a h), ← (publicBatchWrapper89_f2859 a h), (publicBatchWrapper89_f2860 a h), kone]


theorem countAt_77 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 77).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 529810)).val = 32
  rw [(publicBatchWrapper89_f77 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_78 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 78) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f635 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f636 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f637 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f638 a h); rw [kzero] at e; exact e


theorem assetAt_78 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 78).isDummy (row a 78).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 75 35) = (row a 78).isDummy := by rw [(publicBatchWrapper89_f641 a h), (publicBatchWrapper89_f639 a h), (publicBatchWrapper89_f640 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2861 a h), by rw [← hd, ← (publicBatchWrapper89_f2862 a h), (publicBatchWrapper89_f2863 a h), kone]⟩


theorem feeAt_78 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 78).isDummy (row a 78).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 75 35) = (row a 78).isDummy := by rw [(publicBatchWrapper89_f641 a h), (publicBatchWrapper89_f639 a h), (publicBatchWrapper89_f640 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2864 a h), by rw [← hd, ← (publicBatchWrapper89_f2865 a h), (publicBatchWrapper89_f2866 a h), kone]⟩


theorem blockAt_78 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 78).isDummy (row a 78).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 75 35) = (row a 78).isDummy := by rw [(publicBatchWrapper89_f641 a h), (publicBatchWrapper89_f639 a h), (publicBatchWrapper89_f640 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606414), a (.virt 606416), a (.virt 606418), a (.virt 606420)],
    ![a (.virt 606415), a (.virt 606417), a (.virt 606419), a (.virt 606421)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2867 a h)
    · exact (publicBatchWrapper89_f2868 a h)
    · exact (publicBatchWrapper89_f2869 a h)
    · exact (publicBatchWrapper89_f2870 a h)
  · show bor (row a 78).isDummy (band (band (a (.virt 606414)) (a (.virt 606416)))
      (band (a (.virt 606418)) (a (.virt 606420)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2871 a h), ← (publicBatchWrapper89_f2872 a h), ← (publicBatchWrapper89_f2873 a h), ← (publicBatchWrapper89_f2874 a h), (publicBatchWrapper89_f2875 a h), kone]


theorem countAt_78 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 78).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 536605)).val = 32
  rw [(publicBatchWrapper89_f78 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_79 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 79) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f642 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f643 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f644 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f645 a h); rw [kzero] at e; exact e


theorem assetAt_79 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 79).isDummy (row a 79).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 75 79) = (row a 79).isDummy := by rw [(publicBatchWrapper89_f648 a h), (publicBatchWrapper89_f646 a h), (publicBatchWrapper89_f647 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2876 a h), by rw [← hd, ← (publicBatchWrapper89_f2877 a h), (publicBatchWrapper89_f2878 a h), kone]⟩


theorem feeAt_79 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 79).isDummy (row a 79).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 75 79) = (row a 79).isDummy := by rw [(publicBatchWrapper89_f648 a h), (publicBatchWrapper89_f646 a h), (publicBatchWrapper89_f647 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2879 a h), by rw [← hd, ← (publicBatchWrapper89_f2880 a h), (publicBatchWrapper89_f2881 a h), kone]⟩


theorem blockAt_79 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 79).isDummy (row a 79).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 75 79) = (row a 79).isDummy := by rw [(publicBatchWrapper89_f648 a h), (publicBatchWrapper89_f646 a h), (publicBatchWrapper89_f647 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606426), a (.virt 606428), a (.virt 606430), a (.virt 606432)],
    ![a (.virt 606427), a (.virt 606429), a (.virt 606431), a (.virt 606433)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2882 a h)
    · exact (publicBatchWrapper89_f2883 a h)
    · exact (publicBatchWrapper89_f2884 a h)
    · exact (publicBatchWrapper89_f2885 a h)
  · show bor (row a 79).isDummy (band (band (a (.virt 606426)) (a (.virt 606428)))
      (band (a (.virt 606430)) (a (.virt 606432)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2886 a h), ← (publicBatchWrapper89_f2887 a h), ← (publicBatchWrapper89_f2888 a h), ← (publicBatchWrapper89_f2889 a h), (publicBatchWrapper89_f2890 a h), kone]


theorem countAt_79 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 79).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 543400)).val = 32
  rw [(publicBatchWrapper89_f79 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_80 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 80) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f649 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f650 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f651 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f652 a h); rw [kzero] at e; exact e


theorem assetAt_80 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 80).isDummy (row a 80).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 77 43) = (row a 80).isDummy := by rw [(publicBatchWrapper89_f655 a h), (publicBatchWrapper89_f653 a h), (publicBatchWrapper89_f654 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2891 a h), by rw [← hd, ← (publicBatchWrapper89_f2892 a h), (publicBatchWrapper89_f2893 a h), kone]⟩


theorem feeAt_80 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 80).isDummy (row a 80).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 77 43) = (row a 80).isDummy := by rw [(publicBatchWrapper89_f655 a h), (publicBatchWrapper89_f653 a h), (publicBatchWrapper89_f654 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2894 a h), by rw [← hd, ← (publicBatchWrapper89_f2895 a h), (publicBatchWrapper89_f2896 a h), kone]⟩


theorem blockAt_80 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 80).isDummy (row a 80).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 77 43) = (row a 80).isDummy := by rw [(publicBatchWrapper89_f655 a h), (publicBatchWrapper89_f653 a h), (publicBatchWrapper89_f654 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606438), a (.virt 606440), a (.virt 606442), a (.virt 606444)],
    ![a (.virt 606439), a (.virt 606441), a (.virt 606443), a (.virt 606445)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2897 a h)
    · exact (publicBatchWrapper89_f2898 a h)
    · exact (publicBatchWrapper89_f2899 a h)
    · exact (publicBatchWrapper89_f2900 a h)
  · show bor (row a 80).isDummy (band (band (a (.virt 606438)) (a (.virt 606440)))
      (band (a (.virt 606442)) (a (.virt 606444)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2901 a h), ← (publicBatchWrapper89_f2902 a h), ← (publicBatchWrapper89_f2903 a h), ← (publicBatchWrapper89_f2904 a h), (publicBatchWrapper89_f2905 a h), kone]


theorem countAt_80 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 80).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 550195)).val = 32
  rw [(publicBatchWrapper89_f80 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_81 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 81) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f656 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f657 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f658 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f659 a h); rw [kzero] at e; exact e


theorem assetAt_81 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 81).isDummy (row a 81).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 78 7) = (row a 81).isDummy := by rw [(publicBatchWrapper89_f662 a h), (publicBatchWrapper89_f660 a h), (publicBatchWrapper89_f661 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2906 a h), by rw [← hd, ← (publicBatchWrapper89_f2907 a h), (publicBatchWrapper89_f2908 a h), kone]⟩


theorem feeAt_81 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 81).isDummy (row a 81).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 78 7) = (row a 81).isDummy := by rw [(publicBatchWrapper89_f662 a h), (publicBatchWrapper89_f660 a h), (publicBatchWrapper89_f661 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2909 a h), by rw [← hd, ← (publicBatchWrapper89_f2910 a h), (publicBatchWrapper89_f2911 a h), kone]⟩


theorem blockAt_81 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 81).isDummy (row a 81).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 78 7) = (row a 81).isDummy := by rw [(publicBatchWrapper89_f662 a h), (publicBatchWrapper89_f660 a h), (publicBatchWrapper89_f661 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606450), a (.virt 606452), a (.virt 606454), a (.virt 606456)],
    ![a (.virt 606451), a (.virt 606453), a (.virt 606455), a (.virt 606457)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2912 a h)
    · exact (publicBatchWrapper89_f2913 a h)
    · exact (publicBatchWrapper89_f2914 a h)
    · exact (publicBatchWrapper89_f2915 a h)
  · show bor (row a 81).isDummy (band (band (a (.virt 606450)) (a (.virt 606452)))
      (band (a (.virt 606454)) (a (.virt 606456)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2916 a h), ← (publicBatchWrapper89_f2917 a h), ← (publicBatchWrapper89_f2918 a h), ← (publicBatchWrapper89_f2919 a h), (publicBatchWrapper89_f2920 a h), kone]


theorem countAt_81 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 81).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 556990)).val = 32
  rw [(publicBatchWrapper89_f81 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_82 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 82) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f663 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f664 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f665 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f666 a h); rw [kzero] at e; exact e


theorem assetAt_82 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 82).isDummy (row a 82).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 78 51) = (row a 82).isDummy := by rw [(publicBatchWrapper89_f669 a h), (publicBatchWrapper89_f667 a h), (publicBatchWrapper89_f668 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2921 a h), by rw [← hd, ← (publicBatchWrapper89_f2922 a h), (publicBatchWrapper89_f2923 a h), kone]⟩


theorem feeAt_82 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 82).isDummy (row a 82).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 78 51) = (row a 82).isDummy := by rw [(publicBatchWrapper89_f669 a h), (publicBatchWrapper89_f667 a h), (publicBatchWrapper89_f668 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2924 a h), by rw [← hd, ← (publicBatchWrapper89_f2925 a h), (publicBatchWrapper89_f2926 a h), kone]⟩


theorem blockAt_82 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 82).isDummy (row a 82).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 78 51) = (row a 82).isDummy := by rw [(publicBatchWrapper89_f669 a h), (publicBatchWrapper89_f667 a h), (publicBatchWrapper89_f668 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606462), a (.virt 606464), a (.virt 606466), a (.virt 606468)],
    ![a (.virt 606463), a (.virt 606465), a (.virt 606467), a (.virt 606469)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2927 a h)
    · exact (publicBatchWrapper89_f2928 a h)
    · exact (publicBatchWrapper89_f2929 a h)
    · exact (publicBatchWrapper89_f2930 a h)
  · show bor (row a 82).isDummy (band (band (a (.virt 606462)) (a (.virt 606464)))
      (band (a (.virt 606466)) (a (.virt 606468)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2931 a h), ← (publicBatchWrapper89_f2932 a h), ← (publicBatchWrapper89_f2933 a h), ← (publicBatchWrapper89_f2934 a h), (publicBatchWrapper89_f2935 a h), kone]


theorem countAt_82 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 82).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 563785)).val = 32
  rw [(publicBatchWrapper89_f82 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_83 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 83) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f670 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f671 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f672 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f673 a h); rw [kzero] at e; exact e


theorem assetAt_83 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 83).isDummy (row a 83).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 80 15) = (row a 83).isDummy := by rw [(publicBatchWrapper89_f676 a h), (publicBatchWrapper89_f674 a h), (publicBatchWrapper89_f675 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2936 a h), by rw [← hd, ← (publicBatchWrapper89_f2937 a h), (publicBatchWrapper89_f2938 a h), kone]⟩


theorem feeAt_83 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 83).isDummy (row a 83).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 80 15) = (row a 83).isDummy := by rw [(publicBatchWrapper89_f676 a h), (publicBatchWrapper89_f674 a h), (publicBatchWrapper89_f675 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2939 a h), by rw [← hd, ← (publicBatchWrapper89_f2940 a h), (publicBatchWrapper89_f2941 a h), kone]⟩


theorem blockAt_83 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 83).isDummy (row a 83).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 80 15) = (row a 83).isDummy := by rw [(publicBatchWrapper89_f676 a h), (publicBatchWrapper89_f674 a h), (publicBatchWrapper89_f675 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606474), a (.virt 606476), a (.virt 606478), a (.virt 606480)],
    ![a (.virt 606475), a (.virt 606477), a (.virt 606479), a (.virt 606481)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2942 a h)
    · exact (publicBatchWrapper89_f2943 a h)
    · exact (publicBatchWrapper89_f2944 a h)
    · exact (publicBatchWrapper89_f2945 a h)
  · show bor (row a 83).isDummy (band (band (a (.virt 606474)) (a (.virt 606476)))
      (band (a (.virt 606478)) (a (.virt 606480)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2946 a h), ← (publicBatchWrapper89_f2947 a h), ← (publicBatchWrapper89_f2948 a h), ← (publicBatchWrapper89_f2949 a h), (publicBatchWrapper89_f2950 a h), kone]


theorem countAt_83 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 83).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 570580)).val = 32
  rw [(publicBatchWrapper89_f83 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_84 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 84) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f677 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f678 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f679 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f680 a h); rw [kzero] at e; exact e


theorem assetAt_84 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 84).isDummy (row a 84).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 80 59) = (row a 84).isDummy := by rw [(publicBatchWrapper89_f683 a h), (publicBatchWrapper89_f681 a h), (publicBatchWrapper89_f682 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2951 a h), by rw [← hd, ← (publicBatchWrapper89_f2952 a h), (publicBatchWrapper89_f2953 a h), kone]⟩


theorem feeAt_84 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 84).isDummy (row a 84).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 80 59) = (row a 84).isDummy := by rw [(publicBatchWrapper89_f683 a h), (publicBatchWrapper89_f681 a h), (publicBatchWrapper89_f682 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2954 a h), by rw [← hd, ← (publicBatchWrapper89_f2955 a h), (publicBatchWrapper89_f2956 a h), kone]⟩


theorem blockAt_84 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 84).isDummy (row a 84).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 80 59) = (row a 84).isDummy := by rw [(publicBatchWrapper89_f683 a h), (publicBatchWrapper89_f681 a h), (publicBatchWrapper89_f682 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606486), a (.virt 606488), a (.virt 606490), a (.virt 606492)],
    ![a (.virt 606487), a (.virt 606489), a (.virt 606491), a (.virt 606493)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2957 a h)
    · exact (publicBatchWrapper89_f2958 a h)
    · exact (publicBatchWrapper89_f2959 a h)
    · exact (publicBatchWrapper89_f2960 a h)
  · show bor (row a 84).isDummy (band (band (a (.virt 606486)) (a (.virt 606488)))
      (band (a (.virt 606490)) (a (.virt 606492)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2961 a h), ← (publicBatchWrapper89_f2962 a h), ← (publicBatchWrapper89_f2963 a h), ← (publicBatchWrapper89_f2964 a h), (publicBatchWrapper89_f2965 a h), kone]


theorem countAt_84 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 84).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 577375)).val = 32
  rw [(publicBatchWrapper89_f84 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_85 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 85) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f684 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f685 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f686 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f687 a h); rw [kzero] at e; exact e


theorem assetAt_85 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 85).isDummy (row a 85).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 82 23) = (row a 85).isDummy := by rw [(publicBatchWrapper89_f690 a h), (publicBatchWrapper89_f688 a h), (publicBatchWrapper89_f689 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2966 a h), by rw [← hd, ← (publicBatchWrapper89_f2967 a h), (publicBatchWrapper89_f2968 a h), kone]⟩


theorem feeAt_85 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 85).isDummy (row a 85).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 82 23) = (row a 85).isDummy := by rw [(publicBatchWrapper89_f690 a h), (publicBatchWrapper89_f688 a h), (publicBatchWrapper89_f689 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2969 a h), by rw [← hd, ← (publicBatchWrapper89_f2970 a h), (publicBatchWrapper89_f2971 a h), kone]⟩


theorem blockAt_85 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 85).isDummy (row a 85).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 82 23) = (row a 85).isDummy := by rw [(publicBatchWrapper89_f690 a h), (publicBatchWrapper89_f688 a h), (publicBatchWrapper89_f689 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606498), a (.virt 606500), a (.virt 606502), a (.virt 606504)],
    ![a (.virt 606499), a (.virt 606501), a (.virt 606503), a (.virt 606505)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2972 a h)
    · exact (publicBatchWrapper89_f2973 a h)
    · exact (publicBatchWrapper89_f2974 a h)
    · exact (publicBatchWrapper89_f2975 a h)
  · show bor (row a 85).isDummy (band (band (a (.virt 606498)) (a (.virt 606500)))
      (band (a (.virt 606502)) (a (.virt 606504)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2976 a h), ← (publicBatchWrapper89_f2977 a h), ← (publicBatchWrapper89_f2978 a h), ← (publicBatchWrapper89_f2979 a h), (publicBatchWrapper89_f2980 a h), kone]


theorem countAt_85 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 85).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 584170)).val = 32
  rw [(publicBatchWrapper89_f85 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_86 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 86) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f691 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f692 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f693 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f694 a h); rw [kzero] at e; exact e


theorem assetAt_86 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 86).isDummy (row a 86).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 82 67) = (row a 86).isDummy := by rw [(publicBatchWrapper89_f697 a h), (publicBatchWrapper89_f695 a h), (publicBatchWrapper89_f696 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2981 a h), by rw [← hd, ← (publicBatchWrapper89_f2982 a h), (publicBatchWrapper89_f2983 a h), kone]⟩


theorem feeAt_86 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 86).isDummy (row a 86).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 82 67) = (row a 86).isDummy := by rw [(publicBatchWrapper89_f697 a h), (publicBatchWrapper89_f695 a h), (publicBatchWrapper89_f696 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2984 a h), by rw [← hd, ← (publicBatchWrapper89_f2985 a h), (publicBatchWrapper89_f2986 a h), kone]⟩


theorem blockAt_86 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 86).isDummy (row a 86).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 82 67) = (row a 86).isDummy := by rw [(publicBatchWrapper89_f697 a h), (publicBatchWrapper89_f695 a h), (publicBatchWrapper89_f696 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606510), a (.virt 606512), a (.virt 606514), a (.virt 606516)],
    ![a (.virt 606511), a (.virt 606513), a (.virt 606515), a (.virt 606517)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2987 a h)
    · exact (publicBatchWrapper89_f2988 a h)
    · exact (publicBatchWrapper89_f2989 a h)
    · exact (publicBatchWrapper89_f2990 a h)
  · show bor (row a 86).isDummy (band (band (a (.virt 606510)) (a (.virt 606512)))
      (band (a (.virt 606514)) (a (.virt 606516)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2991 a h), ← (publicBatchWrapper89_f2992 a h), ← (publicBatchWrapper89_f2993 a h), ← (publicBatchWrapper89_f2994 a h), (publicBatchWrapper89_f2995 a h), kone]


theorem countAt_86 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 86).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 590965)).val = 32
  rw [(publicBatchWrapper89_f86 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_87 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 87) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f698 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f699 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f700 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f701 a h); rw [kzero] at e; exact e


theorem assetAt_87 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 87).isDummy (row a 87).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 83 31) = (row a 87).isDummy := by rw [(publicBatchWrapper89_f704 a h), (publicBatchWrapper89_f702 a h), (publicBatchWrapper89_f703 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2996 a h), by rw [← hd, ← (publicBatchWrapper89_f2997 a h), (publicBatchWrapper89_f2998 a h), kone]⟩


theorem feeAt_87 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 87).isDummy (row a 87).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 83 31) = (row a 87).isDummy := by rw [(publicBatchWrapper89_f704 a h), (publicBatchWrapper89_f702 a h), (publicBatchWrapper89_f703 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2999 a h), by rw [← hd, ← (publicBatchWrapper89_f3000 a h), (publicBatchWrapper89_f3001 a h), kone]⟩


theorem blockAt_87 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 87).isDummy (row a 87).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 83 31) = (row a 87).isDummy := by rw [(publicBatchWrapper89_f704 a h), (publicBatchWrapper89_f702 a h), (publicBatchWrapper89_f703 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606522), a (.virt 606524), a (.virt 606526), a (.virt 606528)],
    ![a (.virt 606523), a (.virt 606525), a (.virt 606527), a (.virt 606529)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f3002 a h)
    · exact (publicBatchWrapper89_f3003 a h)
    · exact (publicBatchWrapper89_f3004 a h)
    · exact (publicBatchWrapper89_f3005 a h)
  · show bor (row a 87).isDummy (band (band (a (.virt 606522)) (a (.virt 606524)))
      (band (a (.virt 606526)) (a (.virt 606528)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f3006 a h), ← (publicBatchWrapper89_f3007 a h), ← (publicBatchWrapper89_f3008 a h), ← (publicBatchWrapper89_f3009 a h), (publicBatchWrapper89_f3010 a h), kone]


theorem countAt_87 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 87).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 597760)).val = 32
  rw [(publicBatchWrapper89_f87 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_88 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 88) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f705 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f706 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f707 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f708 a h); rw [kzero] at e; exact e


theorem assetAt_88 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 88).isDummy (row a 88).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 83 75) = (row a 88).isDummy := by rw [(publicBatchWrapper89_f711 a h), (publicBatchWrapper89_f709 a h), (publicBatchWrapper89_f710 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f3011 a h), by rw [← hd, ← (publicBatchWrapper89_f3012 a h), (publicBatchWrapper89_f3013 a h), kone]⟩


theorem feeAt_88 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 88).isDummy (row a 88).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 83 75) = (row a 88).isDummy := by rw [(publicBatchWrapper89_f711 a h), (publicBatchWrapper89_f709 a h), (publicBatchWrapper89_f710 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f3014 a h), by rw [← hd, ← (publicBatchWrapper89_f3015 a h), (publicBatchWrapper89_f3016 a h), kone]⟩


theorem blockAt_88 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 88).isDummy (row a 88).blockHash (blockRef (rows a)) := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hd : a (.wire 83 75) = (row a 88).isDummy := by rw [(publicBatchWrapper89_f711 a h), (publicBatchWrapper89_f709 a h), (publicBatchWrapper89_f710 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 606534), a (.virt 606536), a (.virt 606538), a (.virt 606540)],
    ![a (.virt 606535), a (.virt 606537), a (.virt 606539), a (.virt 606541)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f3017 a h)
    · exact (publicBatchWrapper89_f3018 a h)
    · exact (publicBatchWrapper89_f3019 a h)
    · exact (publicBatchWrapper89_f3020 a h)
  · show bor (row a 88).isDummy (band (band (a (.virt 606534)) (a (.virt 606536)))
      (band (a (.virt 606538)) (a (.virt 606540)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f3021 a h), ← (publicBatchWrapper89_f3022 a h), ← (publicBatchWrapper89_f3023 a h), ← (publicBatchWrapper89_f3024 a h), (publicBatchWrapper89_f3025 a h), kone]


theorem countAt_88 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 88).numExitSlots = 32 := by
  obtain ⟨kone, kzero, kslots, ktot⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 604555)).val = 32
  rw [(publicBatchWrapper89_f88 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]

end Plonky2Bridge.PublicWrapper89
