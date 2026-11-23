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
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f89 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f90 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f91 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f92 a h); rw [kzero] at e; exact e


theorem assetAt_0 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 0).isDummy (row a 0).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 1 59) = (row a 0).isDummy := by rw [(publicBatchWrapper89_f95 a h), (publicBatchWrapper89_f93 a h), (publicBatchWrapper89_f94 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1691 a h), by rw [← hd, ← (publicBatchWrapper89_f1692 a h), (publicBatchWrapper89_f1693 a h), kone]⟩


theorem feeAt_0 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 0).isDummy (row a 0).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 1 59) = (row a 0).isDummy := by rw [(publicBatchWrapper89_f95 a h), (publicBatchWrapper89_f93 a h), (publicBatchWrapper89_f94 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1694 a h), by rw [← hd, ← (publicBatchWrapper89_f1695 a h), (publicBatchWrapper89_f1696 a h), kone]⟩


theorem blockAt_0 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 0).isDummy (row a 0).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 1 59) = (row a 0).isDummy := by rw [(publicBatchWrapper89_f95 a h), (publicBatchWrapper89_f93 a h), (publicBatchWrapper89_f94 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18523), a (.virt 18525), a (.virt 18527), a (.virt 18529)],
    ![a (.virt 18524), a (.virt 18526), a (.virt 18528), a (.virt 18530)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1697 a h)
    · exact (publicBatchWrapper89_f1698 a h)
    · exact (publicBatchWrapper89_f1699 a h)
    · exact (publicBatchWrapper89_f1700 a h)
  · show bor (row a 0).isDummy (band (band (a (.virt 18523)) (a (.virt 18525)))
      (band (a (.virt 18527)) (a (.virt 18529)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1701 a h), ← (publicBatchWrapper89_f1702 a h), ← (publicBatchWrapper89_f1703 a h), ← (publicBatchWrapper89_f1704 a h), (publicBatchWrapper89_f1705 a h), kone]


theorem countAt_0 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 0).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 0)).val = 32
  rw [(publicBatchWrapper89_f0 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_1 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 1) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f96 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f97 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f98 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f99 a h); rw [kzero] at e; exact e


theorem assetAt_1 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 1).isDummy (row a 1).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 3 23) = (row a 1).isDummy := by rw [(publicBatchWrapper89_f102 a h), (publicBatchWrapper89_f100 a h), (publicBatchWrapper89_f101 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1706 a h), by rw [← hd, ← (publicBatchWrapper89_f1707 a h), (publicBatchWrapper89_f1708 a h), kone]⟩


theorem feeAt_1 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 1).isDummy (row a 1).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 3 23) = (row a 1).isDummy := by rw [(publicBatchWrapper89_f102 a h), (publicBatchWrapper89_f100 a h), (publicBatchWrapper89_f101 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1709 a h), by rw [← hd, ← (publicBatchWrapper89_f1710 a h), (publicBatchWrapper89_f1711 a h), kone]⟩


theorem blockAt_1 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 1).isDummy (row a 1).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 3 23) = (row a 1).isDummy := by rw [(publicBatchWrapper89_f102 a h), (publicBatchWrapper89_f100 a h), (publicBatchWrapper89_f101 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18535), a (.virt 18537), a (.virt 18539), a (.virt 18541)],
    ![a (.virt 18536), a (.virt 18538), a (.virt 18540), a (.virt 18542)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1712 a h)
    · exact (publicBatchWrapper89_f1713 a h)
    · exact (publicBatchWrapper89_f1714 a h)
    · exact (publicBatchWrapper89_f1715 a h)
  · show bor (row a 1).isDummy (band (band (a (.virt 18535)) (a (.virt 18537)))
      (band (a (.virt 18539)) (a (.virt 18541)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1716 a h), ← (publicBatchWrapper89_f1717 a h), ← (publicBatchWrapper89_f1718 a h), ← (publicBatchWrapper89_f1719 a h), (publicBatchWrapper89_f1720 a h), kone]


theorem countAt_1 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 1).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 202)).val = 32
  rw [(publicBatchWrapper89_f1 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_2 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 2) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f103 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f104 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f105 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f106 a h); rw [kzero] at e; exact e


theorem assetAt_2 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 2).isDummy (row a 2).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 3 67) = (row a 2).isDummy := by rw [(publicBatchWrapper89_f109 a h), (publicBatchWrapper89_f107 a h), (publicBatchWrapper89_f108 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1721 a h), by rw [← hd, ← (publicBatchWrapper89_f1722 a h), (publicBatchWrapper89_f1723 a h), kone]⟩


theorem feeAt_2 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 2).isDummy (row a 2).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 3 67) = (row a 2).isDummy := by rw [(publicBatchWrapper89_f109 a h), (publicBatchWrapper89_f107 a h), (publicBatchWrapper89_f108 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1724 a h), by rw [← hd, ← (publicBatchWrapper89_f1725 a h), (publicBatchWrapper89_f1726 a h), kone]⟩


theorem blockAt_2 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 2).isDummy (row a 2).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 3 67) = (row a 2).isDummy := by rw [(publicBatchWrapper89_f109 a h), (publicBatchWrapper89_f107 a h), (publicBatchWrapper89_f108 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18547), a (.virt 18549), a (.virt 18551), a (.virt 18553)],
    ![a (.virt 18548), a (.virt 18550), a (.virt 18552), a (.virt 18554)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1727 a h)
    · exact (publicBatchWrapper89_f1728 a h)
    · exact (publicBatchWrapper89_f1729 a h)
    · exact (publicBatchWrapper89_f1730 a h)
  · show bor (row a 2).isDummy (band (band (a (.virt 18547)) (a (.virt 18549)))
      (band (a (.virt 18551)) (a (.virt 18553)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1731 a h), ← (publicBatchWrapper89_f1732 a h), ← (publicBatchWrapper89_f1733 a h), ← (publicBatchWrapper89_f1734 a h), (publicBatchWrapper89_f1735 a h), kone]


theorem countAt_2 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 2).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 402)).val = 32
  rw [(publicBatchWrapper89_f2 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_3 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 3) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f110 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f111 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f112 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f113 a h); rw [kzero] at e; exact e


theorem assetAt_3 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 3).isDummy (row a 3).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 5 31) = (row a 3).isDummy := by rw [(publicBatchWrapper89_f116 a h), (publicBatchWrapper89_f114 a h), (publicBatchWrapper89_f115 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1736 a h), by rw [← hd, ← (publicBatchWrapper89_f1737 a h), (publicBatchWrapper89_f1738 a h), kone]⟩


theorem feeAt_3 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 3).isDummy (row a 3).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 5 31) = (row a 3).isDummy := by rw [(publicBatchWrapper89_f116 a h), (publicBatchWrapper89_f114 a h), (publicBatchWrapper89_f115 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1739 a h), by rw [← hd, ← (publicBatchWrapper89_f1740 a h), (publicBatchWrapper89_f1741 a h), kone]⟩


theorem blockAt_3 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 3).isDummy (row a 3).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 5 31) = (row a 3).isDummy := by rw [(publicBatchWrapper89_f116 a h), (publicBatchWrapper89_f114 a h), (publicBatchWrapper89_f115 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18559), a (.virt 18561), a (.virt 18563), a (.virt 18565)],
    ![a (.virt 18560), a (.virt 18562), a (.virt 18564), a (.virt 18566)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1742 a h)
    · exact (publicBatchWrapper89_f1743 a h)
    · exact (publicBatchWrapper89_f1744 a h)
    · exact (publicBatchWrapper89_f1745 a h)
  · show bor (row a 3).isDummy (band (band (a (.virt 18559)) (a (.virt 18561)))
      (band (a (.virt 18563)) (a (.virt 18565)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1746 a h), ← (publicBatchWrapper89_f1747 a h), ← (publicBatchWrapper89_f1748 a h), ← (publicBatchWrapper89_f1749 a h), (publicBatchWrapper89_f1750 a h), kone]


theorem countAt_3 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 3).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 602)).val = 32
  rw [(publicBatchWrapper89_f3 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_4 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 4) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f117 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f118 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f119 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f120 a h); rw [kzero] at e; exact e


theorem assetAt_4 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 4).isDummy (row a 4).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 5 75) = (row a 4).isDummy := by rw [(publicBatchWrapper89_f123 a h), (publicBatchWrapper89_f121 a h), (publicBatchWrapper89_f122 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1751 a h), by rw [← hd, ← (publicBatchWrapper89_f1752 a h), (publicBatchWrapper89_f1753 a h), kone]⟩


theorem feeAt_4 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 4).isDummy (row a 4).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 5 75) = (row a 4).isDummy := by rw [(publicBatchWrapper89_f123 a h), (publicBatchWrapper89_f121 a h), (publicBatchWrapper89_f122 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1754 a h), by rw [← hd, ← (publicBatchWrapper89_f1755 a h), (publicBatchWrapper89_f1756 a h), kone]⟩


theorem blockAt_4 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 4).isDummy (row a 4).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 5 75) = (row a 4).isDummy := by rw [(publicBatchWrapper89_f123 a h), (publicBatchWrapper89_f121 a h), (publicBatchWrapper89_f122 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18571), a (.virt 18573), a (.virt 18575), a (.virt 18577)],
    ![a (.virt 18572), a (.virt 18574), a (.virt 18576), a (.virt 18578)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1757 a h)
    · exact (publicBatchWrapper89_f1758 a h)
    · exact (publicBatchWrapper89_f1759 a h)
    · exact (publicBatchWrapper89_f1760 a h)
  · show bor (row a 4).isDummy (band (band (a (.virt 18571)) (a (.virt 18573)))
      (band (a (.virt 18575)) (a (.virt 18577)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1761 a h), ← (publicBatchWrapper89_f1762 a h), ← (publicBatchWrapper89_f1763 a h), ← (publicBatchWrapper89_f1764 a h), (publicBatchWrapper89_f1765 a h), kone]


theorem countAt_4 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 4).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 802)).val = 32
  rw [(publicBatchWrapper89_f4 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_5 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 5) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f124 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f125 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f126 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f127 a h); rw [kzero] at e; exact e


theorem assetAt_5 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 5).isDummy (row a 5).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 7 39) = (row a 5).isDummy := by rw [(publicBatchWrapper89_f130 a h), (publicBatchWrapper89_f128 a h), (publicBatchWrapper89_f129 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1766 a h), by rw [← hd, ← (publicBatchWrapper89_f1767 a h), (publicBatchWrapper89_f1768 a h), kone]⟩


theorem feeAt_5 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 5).isDummy (row a 5).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 7 39) = (row a 5).isDummy := by rw [(publicBatchWrapper89_f130 a h), (publicBatchWrapper89_f128 a h), (publicBatchWrapper89_f129 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1769 a h), by rw [← hd, ← (publicBatchWrapper89_f1770 a h), (publicBatchWrapper89_f1771 a h), kone]⟩


theorem blockAt_5 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 5).isDummy (row a 5).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 7 39) = (row a 5).isDummy := by rw [(publicBatchWrapper89_f130 a h), (publicBatchWrapper89_f128 a h), (publicBatchWrapper89_f129 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18583), a (.virt 18585), a (.virt 18587), a (.virt 18589)],
    ![a (.virt 18584), a (.virt 18586), a (.virt 18588), a (.virt 18590)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1772 a h)
    · exact (publicBatchWrapper89_f1773 a h)
    · exact (publicBatchWrapper89_f1774 a h)
    · exact (publicBatchWrapper89_f1775 a h)
  · show bor (row a 5).isDummy (band (band (a (.virt 18583)) (a (.virt 18585)))
      (band (a (.virt 18587)) (a (.virt 18589)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1776 a h), ← (publicBatchWrapper89_f1777 a h), ← (publicBatchWrapper89_f1778 a h), ← (publicBatchWrapper89_f1779 a h), (publicBatchWrapper89_f1780 a h), kone]


theorem countAt_5 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 5).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 1002)).val = 32
  rw [(publicBatchWrapper89_f5 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_6 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 6) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f131 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f132 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f133 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f134 a h); rw [kzero] at e; exact e


theorem assetAt_6 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 6).isDummy (row a 6).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 8 3) = (row a 6).isDummy := by rw [(publicBatchWrapper89_f137 a h), (publicBatchWrapper89_f135 a h), (publicBatchWrapper89_f136 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1781 a h), by rw [← hd, ← (publicBatchWrapper89_f1782 a h), (publicBatchWrapper89_f1783 a h), kone]⟩


theorem feeAt_6 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 6).isDummy (row a 6).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 8 3) = (row a 6).isDummy := by rw [(publicBatchWrapper89_f137 a h), (publicBatchWrapper89_f135 a h), (publicBatchWrapper89_f136 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1784 a h), by rw [← hd, ← (publicBatchWrapper89_f1785 a h), (publicBatchWrapper89_f1786 a h), kone]⟩


theorem blockAt_6 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 6).isDummy (row a 6).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 8 3) = (row a 6).isDummy := by rw [(publicBatchWrapper89_f137 a h), (publicBatchWrapper89_f135 a h), (publicBatchWrapper89_f136 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18595), a (.virt 18597), a (.virt 18599), a (.virt 18601)],
    ![a (.virt 18596), a (.virt 18598), a (.virt 18600), a (.virt 18602)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1787 a h)
    · exact (publicBatchWrapper89_f1788 a h)
    · exact (publicBatchWrapper89_f1789 a h)
    · exact (publicBatchWrapper89_f1790 a h)
  · show bor (row a 6).isDummy (band (band (a (.virt 18595)) (a (.virt 18597)))
      (band (a (.virt 18599)) (a (.virt 18601)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1791 a h), ← (publicBatchWrapper89_f1792 a h), ← (publicBatchWrapper89_f1793 a h), ← (publicBatchWrapper89_f1794 a h), (publicBatchWrapper89_f1795 a h), kone]


theorem countAt_6 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 6).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 1202)).val = 32
  rw [(publicBatchWrapper89_f6 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_7 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 7) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f138 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f139 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f140 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f141 a h); rw [kzero] at e; exact e


theorem assetAt_7 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 7).isDummy (row a 7).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 8 47) = (row a 7).isDummy := by rw [(publicBatchWrapper89_f144 a h), (publicBatchWrapper89_f142 a h), (publicBatchWrapper89_f143 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1796 a h), by rw [← hd, ← (publicBatchWrapper89_f1797 a h), (publicBatchWrapper89_f1798 a h), kone]⟩


theorem feeAt_7 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 7).isDummy (row a 7).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 8 47) = (row a 7).isDummy := by rw [(publicBatchWrapper89_f144 a h), (publicBatchWrapper89_f142 a h), (publicBatchWrapper89_f143 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1799 a h), by rw [← hd, ← (publicBatchWrapper89_f1800 a h), (publicBatchWrapper89_f1801 a h), kone]⟩


theorem blockAt_7 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 7).isDummy (row a 7).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 8 47) = (row a 7).isDummy := by rw [(publicBatchWrapper89_f144 a h), (publicBatchWrapper89_f142 a h), (publicBatchWrapper89_f143 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18607), a (.virt 18609), a (.virt 18611), a (.virt 18613)],
    ![a (.virt 18608), a (.virt 18610), a (.virt 18612), a (.virt 18614)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1802 a h)
    · exact (publicBatchWrapper89_f1803 a h)
    · exact (publicBatchWrapper89_f1804 a h)
    · exact (publicBatchWrapper89_f1805 a h)
  · show bor (row a 7).isDummy (band (band (a (.virt 18607)) (a (.virt 18609)))
      (band (a (.virt 18611)) (a (.virt 18613)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1806 a h), ← (publicBatchWrapper89_f1807 a h), ← (publicBatchWrapper89_f1808 a h), ← (publicBatchWrapper89_f1809 a h), (publicBatchWrapper89_f1810 a h), kone]


theorem countAt_7 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 7).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 1402)).val = 32
  rw [(publicBatchWrapper89_f7 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_8 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 8) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f145 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f146 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f147 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f148 a h); rw [kzero] at e; exact e


theorem assetAt_8 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 8).isDummy (row a 8).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 10 11) = (row a 8).isDummy := by rw [(publicBatchWrapper89_f151 a h), (publicBatchWrapper89_f149 a h), (publicBatchWrapper89_f150 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1811 a h), by rw [← hd, ← (publicBatchWrapper89_f1812 a h), (publicBatchWrapper89_f1813 a h), kone]⟩


theorem feeAt_8 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 8).isDummy (row a 8).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 10 11) = (row a 8).isDummy := by rw [(publicBatchWrapper89_f151 a h), (publicBatchWrapper89_f149 a h), (publicBatchWrapper89_f150 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1814 a h), by rw [← hd, ← (publicBatchWrapper89_f1815 a h), (publicBatchWrapper89_f1816 a h), kone]⟩


theorem blockAt_8 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 8).isDummy (row a 8).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 10 11) = (row a 8).isDummy := by rw [(publicBatchWrapper89_f151 a h), (publicBatchWrapper89_f149 a h), (publicBatchWrapper89_f150 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18619), a (.virt 18621), a (.virt 18623), a (.virt 18625)],
    ![a (.virt 18620), a (.virt 18622), a (.virt 18624), a (.virt 18626)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1817 a h)
    · exact (publicBatchWrapper89_f1818 a h)
    · exact (publicBatchWrapper89_f1819 a h)
    · exact (publicBatchWrapper89_f1820 a h)
  · show bor (row a 8).isDummy (band (band (a (.virt 18619)) (a (.virt 18621)))
      (band (a (.virt 18623)) (a (.virt 18625)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1821 a h), ← (publicBatchWrapper89_f1822 a h), ← (publicBatchWrapper89_f1823 a h), ← (publicBatchWrapper89_f1824 a h), (publicBatchWrapper89_f1825 a h), kone]


theorem countAt_8 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 8).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 1602)).val = 32
  rw [(publicBatchWrapper89_f8 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_9 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 9) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f152 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f153 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f154 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f155 a h); rw [kzero] at e; exact e


theorem assetAt_9 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 9).isDummy (row a 9).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 10 55) = (row a 9).isDummy := by rw [(publicBatchWrapper89_f158 a h), (publicBatchWrapper89_f156 a h), (publicBatchWrapper89_f157 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1826 a h), by rw [← hd, ← (publicBatchWrapper89_f1827 a h), (publicBatchWrapper89_f1828 a h), kone]⟩


theorem feeAt_9 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 9).isDummy (row a 9).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 10 55) = (row a 9).isDummy := by rw [(publicBatchWrapper89_f158 a h), (publicBatchWrapper89_f156 a h), (publicBatchWrapper89_f157 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1829 a h), by rw [← hd, ← (publicBatchWrapper89_f1830 a h), (publicBatchWrapper89_f1831 a h), kone]⟩


theorem blockAt_9 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 9).isDummy (row a 9).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 10 55) = (row a 9).isDummy := by rw [(publicBatchWrapper89_f158 a h), (publicBatchWrapper89_f156 a h), (publicBatchWrapper89_f157 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18631), a (.virt 18633), a (.virt 18635), a (.virt 18637)],
    ![a (.virt 18632), a (.virt 18634), a (.virt 18636), a (.virt 18638)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1832 a h)
    · exact (publicBatchWrapper89_f1833 a h)
    · exact (publicBatchWrapper89_f1834 a h)
    · exact (publicBatchWrapper89_f1835 a h)
  · show bor (row a 9).isDummy (band (band (a (.virt 18631)) (a (.virt 18633)))
      (band (a (.virt 18635)) (a (.virt 18637)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1836 a h), ← (publicBatchWrapper89_f1837 a h), ← (publicBatchWrapper89_f1838 a h), ← (publicBatchWrapper89_f1839 a h), (publicBatchWrapper89_f1840 a h), kone]


theorem countAt_9 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 9).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 1802)).val = 32
  rw [(publicBatchWrapper89_f9 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_10 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 10) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f159 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f160 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f161 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f162 a h); rw [kzero] at e; exact e


theorem assetAt_10 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 10).isDummy (row a 10).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 12 19) = (row a 10).isDummy := by rw [(publicBatchWrapper89_f165 a h), (publicBatchWrapper89_f163 a h), (publicBatchWrapper89_f164 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1841 a h), by rw [← hd, ← (publicBatchWrapper89_f1842 a h), (publicBatchWrapper89_f1843 a h), kone]⟩


theorem feeAt_10 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 10).isDummy (row a 10).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 12 19) = (row a 10).isDummy := by rw [(publicBatchWrapper89_f165 a h), (publicBatchWrapper89_f163 a h), (publicBatchWrapper89_f164 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1844 a h), by rw [← hd, ← (publicBatchWrapper89_f1845 a h), (publicBatchWrapper89_f1846 a h), kone]⟩


theorem blockAt_10 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 10).isDummy (row a 10).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 12 19) = (row a 10).isDummy := by rw [(publicBatchWrapper89_f165 a h), (publicBatchWrapper89_f163 a h), (publicBatchWrapper89_f164 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18643), a (.virt 18645), a (.virt 18647), a (.virt 18649)],
    ![a (.virt 18644), a (.virt 18646), a (.virt 18648), a (.virt 18650)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1847 a h)
    · exact (publicBatchWrapper89_f1848 a h)
    · exact (publicBatchWrapper89_f1849 a h)
    · exact (publicBatchWrapper89_f1850 a h)
  · show bor (row a 10).isDummy (band (band (a (.virt 18643)) (a (.virt 18645)))
      (band (a (.virt 18647)) (a (.virt 18649)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1851 a h), ← (publicBatchWrapper89_f1852 a h), ← (publicBatchWrapper89_f1853 a h), ← (publicBatchWrapper89_f1854 a h), (publicBatchWrapper89_f1855 a h), kone]


theorem countAt_10 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 10).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 2002)).val = 32
  rw [(publicBatchWrapper89_f10 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_11 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 11) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f166 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f167 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f168 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f169 a h); rw [kzero] at e; exact e


theorem assetAt_11 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 11).isDummy (row a 11).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 12 63) = (row a 11).isDummy := by rw [(publicBatchWrapper89_f172 a h), (publicBatchWrapper89_f170 a h), (publicBatchWrapper89_f171 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1856 a h), by rw [← hd, ← (publicBatchWrapper89_f1857 a h), (publicBatchWrapper89_f1858 a h), kone]⟩


theorem feeAt_11 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 11).isDummy (row a 11).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 12 63) = (row a 11).isDummy := by rw [(publicBatchWrapper89_f172 a h), (publicBatchWrapper89_f170 a h), (publicBatchWrapper89_f171 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1859 a h), by rw [← hd, ← (publicBatchWrapper89_f1860 a h), (publicBatchWrapper89_f1861 a h), kone]⟩


theorem blockAt_11 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 11).isDummy (row a 11).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 12 63) = (row a 11).isDummy := by rw [(publicBatchWrapper89_f172 a h), (publicBatchWrapper89_f170 a h), (publicBatchWrapper89_f171 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18655), a (.virt 18657), a (.virt 18659), a (.virt 18661)],
    ![a (.virt 18656), a (.virt 18658), a (.virt 18660), a (.virt 18662)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1862 a h)
    · exact (publicBatchWrapper89_f1863 a h)
    · exact (publicBatchWrapper89_f1864 a h)
    · exact (publicBatchWrapper89_f1865 a h)
  · show bor (row a 11).isDummy (band (band (a (.virt 18655)) (a (.virt 18657)))
      (band (a (.virt 18659)) (a (.virt 18661)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1866 a h), ← (publicBatchWrapper89_f1867 a h), ← (publicBatchWrapper89_f1868 a h), ← (publicBatchWrapper89_f1869 a h), (publicBatchWrapper89_f1870 a h), kone]


theorem countAt_11 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 11).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 2202)).val = 32
  rw [(publicBatchWrapper89_f11 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_12 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 12) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f173 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f174 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f175 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f176 a h); rw [kzero] at e; exact e


theorem assetAt_12 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 12).isDummy (row a 12).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 14 27) = (row a 12).isDummy := by rw [(publicBatchWrapper89_f179 a h), (publicBatchWrapper89_f177 a h), (publicBatchWrapper89_f178 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1871 a h), by rw [← hd, ← (publicBatchWrapper89_f1872 a h), (publicBatchWrapper89_f1873 a h), kone]⟩


theorem feeAt_12 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 12).isDummy (row a 12).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 14 27) = (row a 12).isDummy := by rw [(publicBatchWrapper89_f179 a h), (publicBatchWrapper89_f177 a h), (publicBatchWrapper89_f178 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1874 a h), by rw [← hd, ← (publicBatchWrapper89_f1875 a h), (publicBatchWrapper89_f1876 a h), kone]⟩


theorem blockAt_12 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 12).isDummy (row a 12).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 14 27) = (row a 12).isDummy := by rw [(publicBatchWrapper89_f179 a h), (publicBatchWrapper89_f177 a h), (publicBatchWrapper89_f178 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18667), a (.virt 18669), a (.virt 18671), a (.virt 18673)],
    ![a (.virt 18668), a (.virt 18670), a (.virt 18672), a (.virt 18674)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1877 a h)
    · exact (publicBatchWrapper89_f1878 a h)
    · exact (publicBatchWrapper89_f1879 a h)
    · exact (publicBatchWrapper89_f1880 a h)
  · show bor (row a 12).isDummy (band (band (a (.virt 18667)) (a (.virt 18669)))
      (band (a (.virt 18671)) (a (.virt 18673)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1881 a h), ← (publicBatchWrapper89_f1882 a h), ← (publicBatchWrapper89_f1883 a h), ← (publicBatchWrapper89_f1884 a h), (publicBatchWrapper89_f1885 a h), kone]


theorem countAt_12 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 12).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 2402)).val = 32
  rw [(publicBatchWrapper89_f12 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_13 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 13) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f180 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f181 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f182 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f183 a h); rw [kzero] at e; exact e


theorem assetAt_13 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 13).isDummy (row a 13).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 14 71) = (row a 13).isDummy := by rw [(publicBatchWrapper89_f186 a h), (publicBatchWrapper89_f184 a h), (publicBatchWrapper89_f185 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1886 a h), by rw [← hd, ← (publicBatchWrapper89_f1887 a h), (publicBatchWrapper89_f1888 a h), kone]⟩


theorem feeAt_13 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 13).isDummy (row a 13).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 14 71) = (row a 13).isDummy := by rw [(publicBatchWrapper89_f186 a h), (publicBatchWrapper89_f184 a h), (publicBatchWrapper89_f185 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1889 a h), by rw [← hd, ← (publicBatchWrapper89_f1890 a h), (publicBatchWrapper89_f1891 a h), kone]⟩


theorem blockAt_13 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 13).isDummy (row a 13).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 14 71) = (row a 13).isDummy := by rw [(publicBatchWrapper89_f186 a h), (publicBatchWrapper89_f184 a h), (publicBatchWrapper89_f185 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18679), a (.virt 18681), a (.virt 18683), a (.virt 18685)],
    ![a (.virt 18680), a (.virt 18682), a (.virt 18684), a (.virt 18686)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1892 a h)
    · exact (publicBatchWrapper89_f1893 a h)
    · exact (publicBatchWrapper89_f1894 a h)
    · exact (publicBatchWrapper89_f1895 a h)
  · show bor (row a 13).isDummy (band (band (a (.virt 18679)) (a (.virt 18681)))
      (band (a (.virt 18683)) (a (.virt 18685)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1896 a h), ← (publicBatchWrapper89_f1897 a h), ← (publicBatchWrapper89_f1898 a h), ← (publicBatchWrapper89_f1899 a h), (publicBatchWrapper89_f1900 a h), kone]


theorem countAt_13 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 13).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 2602)).val = 32
  rw [(publicBatchWrapper89_f13 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_14 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 14) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f187 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f188 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f189 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f190 a h); rw [kzero] at e; exact e


theorem assetAt_14 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 14).isDummy (row a 14).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 15 35) = (row a 14).isDummy := by rw [(publicBatchWrapper89_f193 a h), (publicBatchWrapper89_f191 a h), (publicBatchWrapper89_f192 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1901 a h), by rw [← hd, ← (publicBatchWrapper89_f1902 a h), (publicBatchWrapper89_f1903 a h), kone]⟩


theorem feeAt_14 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 14).isDummy (row a 14).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 15 35) = (row a 14).isDummy := by rw [(publicBatchWrapper89_f193 a h), (publicBatchWrapper89_f191 a h), (publicBatchWrapper89_f192 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1904 a h), by rw [← hd, ← (publicBatchWrapper89_f1905 a h), (publicBatchWrapper89_f1906 a h), kone]⟩


theorem blockAt_14 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 14).isDummy (row a 14).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 15 35) = (row a 14).isDummy := by rw [(publicBatchWrapper89_f193 a h), (publicBatchWrapper89_f191 a h), (publicBatchWrapper89_f192 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18691), a (.virt 18693), a (.virt 18695), a (.virt 18697)],
    ![a (.virt 18692), a (.virt 18694), a (.virt 18696), a (.virt 18698)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1907 a h)
    · exact (publicBatchWrapper89_f1908 a h)
    · exact (publicBatchWrapper89_f1909 a h)
    · exact (publicBatchWrapper89_f1910 a h)
  · show bor (row a 14).isDummy (band (band (a (.virt 18691)) (a (.virt 18693)))
      (band (a (.virt 18695)) (a (.virt 18697)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1911 a h), ← (publicBatchWrapper89_f1912 a h), ← (publicBatchWrapper89_f1913 a h), ← (publicBatchWrapper89_f1914 a h), (publicBatchWrapper89_f1915 a h), kone]


theorem countAt_14 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 14).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 2802)).val = 32
  rw [(publicBatchWrapper89_f14 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_15 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 15) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f194 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f195 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f196 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f197 a h); rw [kzero] at e; exact e


theorem assetAt_15 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 15).isDummy (row a 15).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 15 79) = (row a 15).isDummy := by rw [(publicBatchWrapper89_f200 a h), (publicBatchWrapper89_f198 a h), (publicBatchWrapper89_f199 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1916 a h), by rw [← hd, ← (publicBatchWrapper89_f1917 a h), (publicBatchWrapper89_f1918 a h), kone]⟩


theorem feeAt_15 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 15).isDummy (row a 15).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 15 79) = (row a 15).isDummy := by rw [(publicBatchWrapper89_f200 a h), (publicBatchWrapper89_f198 a h), (publicBatchWrapper89_f199 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1919 a h), by rw [← hd, ← (publicBatchWrapper89_f1920 a h), (publicBatchWrapper89_f1921 a h), kone]⟩


theorem blockAt_15 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 15).isDummy (row a 15).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 15 79) = (row a 15).isDummy := by rw [(publicBatchWrapper89_f200 a h), (publicBatchWrapper89_f198 a h), (publicBatchWrapper89_f199 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18703), a (.virt 18705), a (.virt 18707), a (.virt 18709)],
    ![a (.virt 18704), a (.virt 18706), a (.virt 18708), a (.virt 18710)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1922 a h)
    · exact (publicBatchWrapper89_f1923 a h)
    · exact (publicBatchWrapper89_f1924 a h)
    · exact (publicBatchWrapper89_f1925 a h)
  · show bor (row a 15).isDummy (band (band (a (.virt 18703)) (a (.virt 18705)))
      (band (a (.virt 18707)) (a (.virt 18709)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1926 a h), ← (publicBatchWrapper89_f1927 a h), ← (publicBatchWrapper89_f1928 a h), ← (publicBatchWrapper89_f1929 a h), (publicBatchWrapper89_f1930 a h), kone]


theorem countAt_15 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 15).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 3002)).val = 32
  rw [(publicBatchWrapper89_f15 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_16 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 16) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f201 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f202 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f203 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f204 a h); rw [kzero] at e; exact e


theorem assetAt_16 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 16).isDummy (row a 16).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 17 43) = (row a 16).isDummy := by rw [(publicBatchWrapper89_f207 a h), (publicBatchWrapper89_f205 a h), (publicBatchWrapper89_f206 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1931 a h), by rw [← hd, ← (publicBatchWrapper89_f1932 a h), (publicBatchWrapper89_f1933 a h), kone]⟩


theorem feeAt_16 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 16).isDummy (row a 16).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 17 43) = (row a 16).isDummy := by rw [(publicBatchWrapper89_f207 a h), (publicBatchWrapper89_f205 a h), (publicBatchWrapper89_f206 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1934 a h), by rw [← hd, ← (publicBatchWrapper89_f1935 a h), (publicBatchWrapper89_f1936 a h), kone]⟩


theorem blockAt_16 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 16).isDummy (row a 16).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 17 43) = (row a 16).isDummy := by rw [(publicBatchWrapper89_f207 a h), (publicBatchWrapper89_f205 a h), (publicBatchWrapper89_f206 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18715), a (.virt 18717), a (.virt 18719), a (.virt 18721)],
    ![a (.virt 18716), a (.virt 18718), a (.virt 18720), a (.virt 18722)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1937 a h)
    · exact (publicBatchWrapper89_f1938 a h)
    · exact (publicBatchWrapper89_f1939 a h)
    · exact (publicBatchWrapper89_f1940 a h)
  · show bor (row a 16).isDummy (band (band (a (.virt 18715)) (a (.virt 18717)))
      (band (a (.virt 18719)) (a (.virt 18721)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1941 a h), ← (publicBatchWrapper89_f1942 a h), ← (publicBatchWrapper89_f1943 a h), ← (publicBatchWrapper89_f1944 a h), (publicBatchWrapper89_f1945 a h), kone]


theorem countAt_16 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 16).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 3202)).val = 32
  rw [(publicBatchWrapper89_f16 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_17 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 17) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f208 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f209 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f210 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f211 a h); rw [kzero] at e; exact e


theorem assetAt_17 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 17).isDummy (row a 17).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 19 7) = (row a 17).isDummy := by rw [(publicBatchWrapper89_f214 a h), (publicBatchWrapper89_f212 a h), (publicBatchWrapper89_f213 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1946 a h), by rw [← hd, ← (publicBatchWrapper89_f1947 a h), (publicBatchWrapper89_f1948 a h), kone]⟩


theorem feeAt_17 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 17).isDummy (row a 17).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 19 7) = (row a 17).isDummy := by rw [(publicBatchWrapper89_f214 a h), (publicBatchWrapper89_f212 a h), (publicBatchWrapper89_f213 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1949 a h), by rw [← hd, ← (publicBatchWrapper89_f1950 a h), (publicBatchWrapper89_f1951 a h), kone]⟩


theorem blockAt_17 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 17).isDummy (row a 17).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 19 7) = (row a 17).isDummy := by rw [(publicBatchWrapper89_f214 a h), (publicBatchWrapper89_f212 a h), (publicBatchWrapper89_f213 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18727), a (.virt 18729), a (.virt 18731), a (.virt 18733)],
    ![a (.virt 18728), a (.virt 18730), a (.virt 18732), a (.virt 18734)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1952 a h)
    · exact (publicBatchWrapper89_f1953 a h)
    · exact (publicBatchWrapper89_f1954 a h)
    · exact (publicBatchWrapper89_f1955 a h)
  · show bor (row a 17).isDummy (band (band (a (.virt 18727)) (a (.virt 18729)))
      (band (a (.virt 18731)) (a (.virt 18733)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1956 a h), ← (publicBatchWrapper89_f1957 a h), ← (publicBatchWrapper89_f1958 a h), ← (publicBatchWrapper89_f1959 a h), (publicBatchWrapper89_f1960 a h), kone]


theorem countAt_17 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 17).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 3402)).val = 32
  rw [(publicBatchWrapper89_f17 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_18 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 18) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f215 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f216 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f217 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f218 a h); rw [kzero] at e; exact e


theorem assetAt_18 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 18).isDummy (row a 18).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 19 51) = (row a 18).isDummy := by rw [(publicBatchWrapper89_f221 a h), (publicBatchWrapper89_f219 a h), (publicBatchWrapper89_f220 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1961 a h), by rw [← hd, ← (publicBatchWrapper89_f1962 a h), (publicBatchWrapper89_f1963 a h), kone]⟩


theorem feeAt_18 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 18).isDummy (row a 18).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 19 51) = (row a 18).isDummy := by rw [(publicBatchWrapper89_f221 a h), (publicBatchWrapper89_f219 a h), (publicBatchWrapper89_f220 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1964 a h), by rw [← hd, ← (publicBatchWrapper89_f1965 a h), (publicBatchWrapper89_f1966 a h), kone]⟩


theorem blockAt_18 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 18).isDummy (row a 18).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 19 51) = (row a 18).isDummy := by rw [(publicBatchWrapper89_f221 a h), (publicBatchWrapper89_f219 a h), (publicBatchWrapper89_f220 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18739), a (.virt 18741), a (.virt 18743), a (.virt 18745)],
    ![a (.virt 18740), a (.virt 18742), a (.virt 18744), a (.virt 18746)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1967 a h)
    · exact (publicBatchWrapper89_f1968 a h)
    · exact (publicBatchWrapper89_f1969 a h)
    · exact (publicBatchWrapper89_f1970 a h)
  · show bor (row a 18).isDummy (band (band (a (.virt 18739)) (a (.virt 18741)))
      (band (a (.virt 18743)) (a (.virt 18745)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1971 a h), ← (publicBatchWrapper89_f1972 a h), ← (publicBatchWrapper89_f1973 a h), ← (publicBatchWrapper89_f1974 a h), (publicBatchWrapper89_f1975 a h), kone]


theorem countAt_18 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 18).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 3602)).val = 32
  rw [(publicBatchWrapper89_f18 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_19 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 19) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f222 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f223 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f224 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f225 a h); rw [kzero] at e; exact e


theorem assetAt_19 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 19).isDummy (row a 19).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 20 15) = (row a 19).isDummy := by rw [(publicBatchWrapper89_f228 a h), (publicBatchWrapper89_f226 a h), (publicBatchWrapper89_f227 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1976 a h), by rw [← hd, ← (publicBatchWrapper89_f1977 a h), (publicBatchWrapper89_f1978 a h), kone]⟩


theorem feeAt_19 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 19).isDummy (row a 19).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 20 15) = (row a 19).isDummy := by rw [(publicBatchWrapper89_f228 a h), (publicBatchWrapper89_f226 a h), (publicBatchWrapper89_f227 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1979 a h), by rw [← hd, ← (publicBatchWrapper89_f1980 a h), (publicBatchWrapper89_f1981 a h), kone]⟩


theorem blockAt_19 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 19).isDummy (row a 19).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 20 15) = (row a 19).isDummy := by rw [(publicBatchWrapper89_f228 a h), (publicBatchWrapper89_f226 a h), (publicBatchWrapper89_f227 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18751), a (.virt 18753), a (.virt 18755), a (.virt 18757)],
    ![a (.virt 18752), a (.virt 18754), a (.virt 18756), a (.virt 18758)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1982 a h)
    · exact (publicBatchWrapper89_f1983 a h)
    · exact (publicBatchWrapper89_f1984 a h)
    · exact (publicBatchWrapper89_f1985 a h)
  · show bor (row a 19).isDummy (band (band (a (.virt 18751)) (a (.virt 18753)))
      (band (a (.virt 18755)) (a (.virt 18757)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f1986 a h), ← (publicBatchWrapper89_f1987 a h), ← (publicBatchWrapper89_f1988 a h), ← (publicBatchWrapper89_f1989 a h), (publicBatchWrapper89_f1990 a h), kone]


theorem countAt_19 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 19).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 3802)).val = 32
  rw [(publicBatchWrapper89_f19 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_20 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 20) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f229 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f230 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f231 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f232 a h); rw [kzero] at e; exact e


theorem assetAt_20 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 20).isDummy (row a 20).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 20 59) = (row a 20).isDummy := by rw [(publicBatchWrapper89_f235 a h), (publicBatchWrapper89_f233 a h), (publicBatchWrapper89_f234 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f1991 a h), by rw [← hd, ← (publicBatchWrapper89_f1992 a h), (publicBatchWrapper89_f1993 a h), kone]⟩


theorem feeAt_20 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 20).isDummy (row a 20).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 20 59) = (row a 20).isDummy := by rw [(publicBatchWrapper89_f235 a h), (publicBatchWrapper89_f233 a h), (publicBatchWrapper89_f234 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f1994 a h), by rw [← hd, ← (publicBatchWrapper89_f1995 a h), (publicBatchWrapper89_f1996 a h), kone]⟩


theorem blockAt_20 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 20).isDummy (row a 20).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 20 59) = (row a 20).isDummy := by rw [(publicBatchWrapper89_f235 a h), (publicBatchWrapper89_f233 a h), (publicBatchWrapper89_f234 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18763), a (.virt 18765), a (.virt 18767), a (.virt 18769)],
    ![a (.virt 18764), a (.virt 18766), a (.virt 18768), a (.virt 18770)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f1997 a h)
    · exact (publicBatchWrapper89_f1998 a h)
    · exact (publicBatchWrapper89_f1999 a h)
    · exact (publicBatchWrapper89_f2000 a h)
  · show bor (row a 20).isDummy (band (band (a (.virt 18763)) (a (.virt 18765)))
      (band (a (.virt 18767)) (a (.virt 18769)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2001 a h), ← (publicBatchWrapper89_f2002 a h), ← (publicBatchWrapper89_f2003 a h), ← (publicBatchWrapper89_f2004 a h), (publicBatchWrapper89_f2005 a h), kone]


theorem countAt_20 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 20).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 4002)).val = 32
  rw [(publicBatchWrapper89_f20 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_21 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 21) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f236 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f237 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f238 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f239 a h); rw [kzero] at e; exact e


theorem assetAt_21 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 21).isDummy (row a 21).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 22 23) = (row a 21).isDummy := by rw [(publicBatchWrapper89_f242 a h), (publicBatchWrapper89_f240 a h), (publicBatchWrapper89_f241 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2006 a h), by rw [← hd, ← (publicBatchWrapper89_f2007 a h), (publicBatchWrapper89_f2008 a h), kone]⟩


theorem feeAt_21 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 21).isDummy (row a 21).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 22 23) = (row a 21).isDummy := by rw [(publicBatchWrapper89_f242 a h), (publicBatchWrapper89_f240 a h), (publicBatchWrapper89_f241 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2009 a h), by rw [← hd, ← (publicBatchWrapper89_f2010 a h), (publicBatchWrapper89_f2011 a h), kone]⟩


theorem blockAt_21 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 21).isDummy (row a 21).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 22 23) = (row a 21).isDummy := by rw [(publicBatchWrapper89_f242 a h), (publicBatchWrapper89_f240 a h), (publicBatchWrapper89_f241 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18775), a (.virt 18777), a (.virt 18779), a (.virt 18781)],
    ![a (.virt 18776), a (.virt 18778), a (.virt 18780), a (.virt 18782)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2012 a h)
    · exact (publicBatchWrapper89_f2013 a h)
    · exact (publicBatchWrapper89_f2014 a h)
    · exact (publicBatchWrapper89_f2015 a h)
  · show bor (row a 21).isDummy (band (band (a (.virt 18775)) (a (.virt 18777)))
      (band (a (.virt 18779)) (a (.virt 18781)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2016 a h), ← (publicBatchWrapper89_f2017 a h), ← (publicBatchWrapper89_f2018 a h), ← (publicBatchWrapper89_f2019 a h), (publicBatchWrapper89_f2020 a h), kone]


theorem countAt_21 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 21).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 4202)).val = 32
  rw [(publicBatchWrapper89_f21 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_22 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 22) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f243 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f244 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f245 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f246 a h); rw [kzero] at e; exact e


theorem assetAt_22 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 22).isDummy (row a 22).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 22 67) = (row a 22).isDummy := by rw [(publicBatchWrapper89_f249 a h), (publicBatchWrapper89_f247 a h), (publicBatchWrapper89_f248 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2021 a h), by rw [← hd, ← (publicBatchWrapper89_f2022 a h), (publicBatchWrapper89_f2023 a h), kone]⟩


theorem feeAt_22 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 22).isDummy (row a 22).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 22 67) = (row a 22).isDummy := by rw [(publicBatchWrapper89_f249 a h), (publicBatchWrapper89_f247 a h), (publicBatchWrapper89_f248 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2024 a h), by rw [← hd, ← (publicBatchWrapper89_f2025 a h), (publicBatchWrapper89_f2026 a h), kone]⟩


theorem blockAt_22 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 22).isDummy (row a 22).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 22 67) = (row a 22).isDummy := by rw [(publicBatchWrapper89_f249 a h), (publicBatchWrapper89_f247 a h), (publicBatchWrapper89_f248 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18787), a (.virt 18789), a (.virt 18791), a (.virt 18793)],
    ![a (.virt 18788), a (.virt 18790), a (.virt 18792), a (.virt 18794)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2027 a h)
    · exact (publicBatchWrapper89_f2028 a h)
    · exact (publicBatchWrapper89_f2029 a h)
    · exact (publicBatchWrapper89_f2030 a h)
  · show bor (row a 22).isDummy (band (band (a (.virt 18787)) (a (.virt 18789)))
      (band (a (.virt 18791)) (a (.virt 18793)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2031 a h), ← (publicBatchWrapper89_f2032 a h), ← (publicBatchWrapper89_f2033 a h), ← (publicBatchWrapper89_f2034 a h), (publicBatchWrapper89_f2035 a h), kone]


theorem countAt_22 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 22).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 4402)).val = 32
  rw [(publicBatchWrapper89_f22 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_23 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 23) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f250 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f251 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f252 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f253 a h); rw [kzero] at e; exact e


theorem assetAt_23 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 23).isDummy (row a 23).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 24 31) = (row a 23).isDummy := by rw [(publicBatchWrapper89_f256 a h), (publicBatchWrapper89_f254 a h), (publicBatchWrapper89_f255 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2036 a h), by rw [← hd, ← (publicBatchWrapper89_f2037 a h), (publicBatchWrapper89_f2038 a h), kone]⟩


theorem feeAt_23 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 23).isDummy (row a 23).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 24 31) = (row a 23).isDummy := by rw [(publicBatchWrapper89_f256 a h), (publicBatchWrapper89_f254 a h), (publicBatchWrapper89_f255 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2039 a h), by rw [← hd, ← (publicBatchWrapper89_f2040 a h), (publicBatchWrapper89_f2041 a h), kone]⟩


theorem blockAt_23 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 23).isDummy (row a 23).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 24 31) = (row a 23).isDummy := by rw [(publicBatchWrapper89_f256 a h), (publicBatchWrapper89_f254 a h), (publicBatchWrapper89_f255 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18799), a (.virt 18801), a (.virt 18803), a (.virt 18805)],
    ![a (.virt 18800), a (.virt 18802), a (.virt 18804), a (.virt 18806)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2042 a h)
    · exact (publicBatchWrapper89_f2043 a h)
    · exact (publicBatchWrapper89_f2044 a h)
    · exact (publicBatchWrapper89_f2045 a h)
  · show bor (row a 23).isDummy (band (band (a (.virt 18799)) (a (.virt 18801)))
      (band (a (.virt 18803)) (a (.virt 18805)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2046 a h), ← (publicBatchWrapper89_f2047 a h), ← (publicBatchWrapper89_f2048 a h), ← (publicBatchWrapper89_f2049 a h), (publicBatchWrapper89_f2050 a h), kone]


theorem countAt_23 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 23).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 4602)).val = 32
  rw [(publicBatchWrapper89_f23 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_24 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 24) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f257 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f258 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f259 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f260 a h); rw [kzero] at e; exact e


theorem assetAt_24 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 24).isDummy (row a 24).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 24 75) = (row a 24).isDummy := by rw [(publicBatchWrapper89_f263 a h), (publicBatchWrapper89_f261 a h), (publicBatchWrapper89_f262 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2051 a h), by rw [← hd, ← (publicBatchWrapper89_f2052 a h), (publicBatchWrapper89_f2053 a h), kone]⟩


theorem feeAt_24 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 24).isDummy (row a 24).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 24 75) = (row a 24).isDummy := by rw [(publicBatchWrapper89_f263 a h), (publicBatchWrapper89_f261 a h), (publicBatchWrapper89_f262 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2054 a h), by rw [← hd, ← (publicBatchWrapper89_f2055 a h), (publicBatchWrapper89_f2056 a h), kone]⟩


theorem blockAt_24 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 24).isDummy (row a 24).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 24 75) = (row a 24).isDummy := by rw [(publicBatchWrapper89_f263 a h), (publicBatchWrapper89_f261 a h), (publicBatchWrapper89_f262 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18811), a (.virt 18813), a (.virt 18815), a (.virt 18817)],
    ![a (.virt 18812), a (.virt 18814), a (.virt 18816), a (.virt 18818)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2057 a h)
    · exact (publicBatchWrapper89_f2058 a h)
    · exact (publicBatchWrapper89_f2059 a h)
    · exact (publicBatchWrapper89_f2060 a h)
  · show bor (row a 24).isDummy (band (band (a (.virt 18811)) (a (.virt 18813)))
      (band (a (.virt 18815)) (a (.virt 18817)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2061 a h), ← (publicBatchWrapper89_f2062 a h), ← (publicBatchWrapper89_f2063 a h), ← (publicBatchWrapper89_f2064 a h), (publicBatchWrapper89_f2065 a h), kone]


theorem countAt_24 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 24).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 4802)).val = 32
  rw [(publicBatchWrapper89_f24 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_25 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 25) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f264 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f265 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f266 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f267 a h); rw [kzero] at e; exact e


theorem assetAt_25 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 25).isDummy (row a 25).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 26 39) = (row a 25).isDummy := by rw [(publicBatchWrapper89_f270 a h), (publicBatchWrapper89_f268 a h), (publicBatchWrapper89_f269 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2066 a h), by rw [← hd, ← (publicBatchWrapper89_f2067 a h), (publicBatchWrapper89_f2068 a h), kone]⟩


theorem feeAt_25 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 25).isDummy (row a 25).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 26 39) = (row a 25).isDummy := by rw [(publicBatchWrapper89_f270 a h), (publicBatchWrapper89_f268 a h), (publicBatchWrapper89_f269 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2069 a h), by rw [← hd, ← (publicBatchWrapper89_f2070 a h), (publicBatchWrapper89_f2071 a h), kone]⟩


theorem blockAt_25 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 25).isDummy (row a 25).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 26 39) = (row a 25).isDummy := by rw [(publicBatchWrapper89_f270 a h), (publicBatchWrapper89_f268 a h), (publicBatchWrapper89_f269 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18823), a (.virt 18825), a (.virt 18827), a (.virt 18829)],
    ![a (.virt 18824), a (.virt 18826), a (.virt 18828), a (.virt 18830)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2072 a h)
    · exact (publicBatchWrapper89_f2073 a h)
    · exact (publicBatchWrapper89_f2074 a h)
    · exact (publicBatchWrapper89_f2075 a h)
  · show bor (row a 25).isDummy (band (band (a (.virt 18823)) (a (.virt 18825)))
      (band (a (.virt 18827)) (a (.virt 18829)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2076 a h), ← (publicBatchWrapper89_f2077 a h), ← (publicBatchWrapper89_f2078 a h), ← (publicBatchWrapper89_f2079 a h), (publicBatchWrapper89_f2080 a h), kone]


theorem countAt_25 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 25).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 5002)).val = 32
  rw [(publicBatchWrapper89_f25 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_26 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 26) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f271 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f272 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f273 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f274 a h); rw [kzero] at e; exact e


theorem assetAt_26 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 26).isDummy (row a 26).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 27 3) = (row a 26).isDummy := by rw [(publicBatchWrapper89_f277 a h), (publicBatchWrapper89_f275 a h), (publicBatchWrapper89_f276 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2081 a h), by rw [← hd, ← (publicBatchWrapper89_f2082 a h), (publicBatchWrapper89_f2083 a h), kone]⟩


theorem feeAt_26 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 26).isDummy (row a 26).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 27 3) = (row a 26).isDummy := by rw [(publicBatchWrapper89_f277 a h), (publicBatchWrapper89_f275 a h), (publicBatchWrapper89_f276 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2084 a h), by rw [← hd, ← (publicBatchWrapper89_f2085 a h), (publicBatchWrapper89_f2086 a h), kone]⟩


theorem blockAt_26 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 26).isDummy (row a 26).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 27 3) = (row a 26).isDummy := by rw [(publicBatchWrapper89_f277 a h), (publicBatchWrapper89_f275 a h), (publicBatchWrapper89_f276 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18835), a (.virt 18837), a (.virt 18839), a (.virt 18841)],
    ![a (.virt 18836), a (.virt 18838), a (.virt 18840), a (.virt 18842)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2087 a h)
    · exact (publicBatchWrapper89_f2088 a h)
    · exact (publicBatchWrapper89_f2089 a h)
    · exact (publicBatchWrapper89_f2090 a h)
  · show bor (row a 26).isDummy (band (band (a (.virt 18835)) (a (.virt 18837)))
      (band (a (.virt 18839)) (a (.virt 18841)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2091 a h), ← (publicBatchWrapper89_f2092 a h), ← (publicBatchWrapper89_f2093 a h), ← (publicBatchWrapper89_f2094 a h), (publicBatchWrapper89_f2095 a h), kone]


theorem countAt_26 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 26).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 5202)).val = 32
  rw [(publicBatchWrapper89_f26 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_27 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 27) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f278 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f279 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f280 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f281 a h); rw [kzero] at e; exact e


theorem assetAt_27 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 27).isDummy (row a 27).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 27 47) = (row a 27).isDummy := by rw [(publicBatchWrapper89_f284 a h), (publicBatchWrapper89_f282 a h), (publicBatchWrapper89_f283 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2096 a h), by rw [← hd, ← (publicBatchWrapper89_f2097 a h), (publicBatchWrapper89_f2098 a h), kone]⟩


theorem feeAt_27 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 27).isDummy (row a 27).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 27 47) = (row a 27).isDummy := by rw [(publicBatchWrapper89_f284 a h), (publicBatchWrapper89_f282 a h), (publicBatchWrapper89_f283 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2099 a h), by rw [← hd, ← (publicBatchWrapper89_f2100 a h), (publicBatchWrapper89_f2101 a h), kone]⟩


theorem blockAt_27 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 27).isDummy (row a 27).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 27 47) = (row a 27).isDummy := by rw [(publicBatchWrapper89_f284 a h), (publicBatchWrapper89_f282 a h), (publicBatchWrapper89_f283 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18847), a (.virt 18849), a (.virt 18851), a (.virt 18853)],
    ![a (.virt 18848), a (.virt 18850), a (.virt 18852), a (.virt 18854)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2102 a h)
    · exact (publicBatchWrapper89_f2103 a h)
    · exact (publicBatchWrapper89_f2104 a h)
    · exact (publicBatchWrapper89_f2105 a h)
  · show bor (row a 27).isDummy (band (band (a (.virt 18847)) (a (.virt 18849)))
      (band (a (.virt 18851)) (a (.virt 18853)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2106 a h), ← (publicBatchWrapper89_f2107 a h), ← (publicBatchWrapper89_f2108 a h), ← (publicBatchWrapper89_f2109 a h), (publicBatchWrapper89_f2110 a h), kone]


theorem countAt_27 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 27).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 5402)).val = 32
  rw [(publicBatchWrapper89_f27 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_28 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 28) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f285 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f286 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f287 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f288 a h); rw [kzero] at e; exact e


theorem assetAt_28 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 28).isDummy (row a 28).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 29 11) = (row a 28).isDummy := by rw [(publicBatchWrapper89_f291 a h), (publicBatchWrapper89_f289 a h), (publicBatchWrapper89_f290 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2111 a h), by rw [← hd, ← (publicBatchWrapper89_f2112 a h), (publicBatchWrapper89_f2113 a h), kone]⟩


theorem feeAt_28 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 28).isDummy (row a 28).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 29 11) = (row a 28).isDummy := by rw [(publicBatchWrapper89_f291 a h), (publicBatchWrapper89_f289 a h), (publicBatchWrapper89_f290 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2114 a h), by rw [← hd, ← (publicBatchWrapper89_f2115 a h), (publicBatchWrapper89_f2116 a h), kone]⟩


theorem blockAt_28 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 28).isDummy (row a 28).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 29 11) = (row a 28).isDummy := by rw [(publicBatchWrapper89_f291 a h), (publicBatchWrapper89_f289 a h), (publicBatchWrapper89_f290 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18859), a (.virt 18861), a (.virt 18863), a (.virt 18865)],
    ![a (.virt 18860), a (.virt 18862), a (.virt 18864), a (.virt 18866)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2117 a h)
    · exact (publicBatchWrapper89_f2118 a h)
    · exact (publicBatchWrapper89_f2119 a h)
    · exact (publicBatchWrapper89_f2120 a h)
  · show bor (row a 28).isDummy (band (band (a (.virt 18859)) (a (.virt 18861)))
      (band (a (.virt 18863)) (a (.virt 18865)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2121 a h), ← (publicBatchWrapper89_f2122 a h), ← (publicBatchWrapper89_f2123 a h), ← (publicBatchWrapper89_f2124 a h), (publicBatchWrapper89_f2125 a h), kone]


theorem countAt_28 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 28).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 5602)).val = 32
  rw [(publicBatchWrapper89_f28 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_29 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 29) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f292 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f293 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f294 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f295 a h); rw [kzero] at e; exact e


theorem assetAt_29 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 29).isDummy (row a 29).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 29 55) = (row a 29).isDummy := by rw [(publicBatchWrapper89_f298 a h), (publicBatchWrapper89_f296 a h), (publicBatchWrapper89_f297 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2126 a h), by rw [← hd, ← (publicBatchWrapper89_f2127 a h), (publicBatchWrapper89_f2128 a h), kone]⟩


theorem feeAt_29 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 29).isDummy (row a 29).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 29 55) = (row a 29).isDummy := by rw [(publicBatchWrapper89_f298 a h), (publicBatchWrapper89_f296 a h), (publicBatchWrapper89_f297 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2129 a h), by rw [← hd, ← (publicBatchWrapper89_f2130 a h), (publicBatchWrapper89_f2131 a h), kone]⟩


theorem blockAt_29 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 29).isDummy (row a 29).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 29 55) = (row a 29).isDummy := by rw [(publicBatchWrapper89_f298 a h), (publicBatchWrapper89_f296 a h), (publicBatchWrapper89_f297 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18871), a (.virt 18873), a (.virt 18875), a (.virt 18877)],
    ![a (.virt 18872), a (.virt 18874), a (.virt 18876), a (.virt 18878)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2132 a h)
    · exact (publicBatchWrapper89_f2133 a h)
    · exact (publicBatchWrapper89_f2134 a h)
    · exact (publicBatchWrapper89_f2135 a h)
  · show bor (row a 29).isDummy (band (band (a (.virt 18871)) (a (.virt 18873)))
      (band (a (.virt 18875)) (a (.virt 18877)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2136 a h), ← (publicBatchWrapper89_f2137 a h), ← (publicBatchWrapper89_f2138 a h), ← (publicBatchWrapper89_f2139 a h), (publicBatchWrapper89_f2140 a h), kone]


theorem countAt_29 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 29).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 5802)).val = 32
  rw [(publicBatchWrapper89_f29 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_30 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 30) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f299 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f300 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f301 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f302 a h); rw [kzero] at e; exact e


theorem assetAt_30 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 30).isDummy (row a 30).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 31 19) = (row a 30).isDummy := by rw [(publicBatchWrapper89_f305 a h), (publicBatchWrapper89_f303 a h), (publicBatchWrapper89_f304 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2141 a h), by rw [← hd, ← (publicBatchWrapper89_f2142 a h), (publicBatchWrapper89_f2143 a h), kone]⟩


theorem feeAt_30 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 30).isDummy (row a 30).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 31 19) = (row a 30).isDummy := by rw [(publicBatchWrapper89_f305 a h), (publicBatchWrapper89_f303 a h), (publicBatchWrapper89_f304 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2144 a h), by rw [← hd, ← (publicBatchWrapper89_f2145 a h), (publicBatchWrapper89_f2146 a h), kone]⟩


theorem blockAt_30 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 30).isDummy (row a 30).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 31 19) = (row a 30).isDummy := by rw [(publicBatchWrapper89_f305 a h), (publicBatchWrapper89_f303 a h), (publicBatchWrapper89_f304 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18883), a (.virt 18885), a (.virt 18887), a (.virt 18889)],
    ![a (.virt 18884), a (.virt 18886), a (.virt 18888), a (.virt 18890)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2147 a h)
    · exact (publicBatchWrapper89_f2148 a h)
    · exact (publicBatchWrapper89_f2149 a h)
    · exact (publicBatchWrapper89_f2150 a h)
  · show bor (row a 30).isDummy (band (band (a (.virt 18883)) (a (.virt 18885)))
      (band (a (.virt 18887)) (a (.virt 18889)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2151 a h), ← (publicBatchWrapper89_f2152 a h), ← (publicBatchWrapper89_f2153 a h), ← (publicBatchWrapper89_f2154 a h), (publicBatchWrapper89_f2155 a h), kone]


theorem countAt_30 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 30).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 6002)).val = 32
  rw [(publicBatchWrapper89_f30 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_31 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 31) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f306 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f307 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f308 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f309 a h); rw [kzero] at e; exact e


theorem assetAt_31 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 31).isDummy (row a 31).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 31 63) = (row a 31).isDummy := by rw [(publicBatchWrapper89_f312 a h), (publicBatchWrapper89_f310 a h), (publicBatchWrapper89_f311 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2156 a h), by rw [← hd, ← (publicBatchWrapper89_f2157 a h), (publicBatchWrapper89_f2158 a h), kone]⟩


theorem feeAt_31 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 31).isDummy (row a 31).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 31 63) = (row a 31).isDummy := by rw [(publicBatchWrapper89_f312 a h), (publicBatchWrapper89_f310 a h), (publicBatchWrapper89_f311 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2159 a h), by rw [← hd, ← (publicBatchWrapper89_f2160 a h), (publicBatchWrapper89_f2161 a h), kone]⟩


theorem blockAt_31 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 31).isDummy (row a 31).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 31 63) = (row a 31).isDummy := by rw [(publicBatchWrapper89_f312 a h), (publicBatchWrapper89_f310 a h), (publicBatchWrapper89_f311 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18895), a (.virt 18897), a (.virt 18899), a (.virt 18901)],
    ![a (.virt 18896), a (.virt 18898), a (.virt 18900), a (.virt 18902)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2162 a h)
    · exact (publicBatchWrapper89_f2163 a h)
    · exact (publicBatchWrapper89_f2164 a h)
    · exact (publicBatchWrapper89_f2165 a h)
  · show bor (row a 31).isDummy (band (band (a (.virt 18895)) (a (.virt 18897)))
      (band (a (.virt 18899)) (a (.virt 18901)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2166 a h), ← (publicBatchWrapper89_f2167 a h), ← (publicBatchWrapper89_f2168 a h), ← (publicBatchWrapper89_f2169 a h), (publicBatchWrapper89_f2170 a h), kone]


theorem countAt_31 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 31).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 6202)).val = 32
  rw [(publicBatchWrapper89_f31 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_32 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 32) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f313 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f314 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f315 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f316 a h); rw [kzero] at e; exact e


theorem assetAt_32 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 32).isDummy (row a 32).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 33 27) = (row a 32).isDummy := by rw [(publicBatchWrapper89_f319 a h), (publicBatchWrapper89_f317 a h), (publicBatchWrapper89_f318 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2171 a h), by rw [← hd, ← (publicBatchWrapper89_f2172 a h), (publicBatchWrapper89_f2173 a h), kone]⟩


theorem feeAt_32 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 32).isDummy (row a 32).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 33 27) = (row a 32).isDummy := by rw [(publicBatchWrapper89_f319 a h), (publicBatchWrapper89_f317 a h), (publicBatchWrapper89_f318 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2174 a h), by rw [← hd, ← (publicBatchWrapper89_f2175 a h), (publicBatchWrapper89_f2176 a h), kone]⟩


theorem blockAt_32 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 32).isDummy (row a 32).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 33 27) = (row a 32).isDummy := by rw [(publicBatchWrapper89_f319 a h), (publicBatchWrapper89_f317 a h), (publicBatchWrapper89_f318 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18907), a (.virt 18909), a (.virt 18911), a (.virt 18913)],
    ![a (.virt 18908), a (.virt 18910), a (.virt 18912), a (.virt 18914)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2177 a h)
    · exact (publicBatchWrapper89_f2178 a h)
    · exact (publicBatchWrapper89_f2179 a h)
    · exact (publicBatchWrapper89_f2180 a h)
  · show bor (row a 32).isDummy (band (band (a (.virt 18907)) (a (.virt 18909)))
      (band (a (.virt 18911)) (a (.virt 18913)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2181 a h), ← (publicBatchWrapper89_f2182 a h), ← (publicBatchWrapper89_f2183 a h), ← (publicBatchWrapper89_f2184 a h), (publicBatchWrapper89_f2185 a h), kone]


theorem countAt_32 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 32).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 6402)).val = 32
  rw [(publicBatchWrapper89_f32 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_33 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 33) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f320 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f321 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f322 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f323 a h); rw [kzero] at e; exact e


theorem assetAt_33 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 33).isDummy (row a 33).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 33 71) = (row a 33).isDummy := by rw [(publicBatchWrapper89_f326 a h), (publicBatchWrapper89_f324 a h), (publicBatchWrapper89_f325 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2186 a h), by rw [← hd, ← (publicBatchWrapper89_f2187 a h), (publicBatchWrapper89_f2188 a h), kone]⟩


theorem feeAt_33 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 33).isDummy (row a 33).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 33 71) = (row a 33).isDummy := by rw [(publicBatchWrapper89_f326 a h), (publicBatchWrapper89_f324 a h), (publicBatchWrapper89_f325 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2189 a h), by rw [← hd, ← (publicBatchWrapper89_f2190 a h), (publicBatchWrapper89_f2191 a h), kone]⟩


theorem blockAt_33 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 33).isDummy (row a 33).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 33 71) = (row a 33).isDummy := by rw [(publicBatchWrapper89_f326 a h), (publicBatchWrapper89_f324 a h), (publicBatchWrapper89_f325 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18919), a (.virt 18921), a (.virt 18923), a (.virt 18925)],
    ![a (.virt 18920), a (.virt 18922), a (.virt 18924), a (.virt 18926)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2192 a h)
    · exact (publicBatchWrapper89_f2193 a h)
    · exact (publicBatchWrapper89_f2194 a h)
    · exact (publicBatchWrapper89_f2195 a h)
  · show bor (row a 33).isDummy (band (band (a (.virt 18919)) (a (.virt 18921)))
      (band (a (.virt 18923)) (a (.virt 18925)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2196 a h), ← (publicBatchWrapper89_f2197 a h), ← (publicBatchWrapper89_f2198 a h), ← (publicBatchWrapper89_f2199 a h), (publicBatchWrapper89_f2200 a h), kone]


theorem countAt_33 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 33).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 6602)).val = 32
  rw [(publicBatchWrapper89_f33 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_34 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 34) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f327 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f328 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f329 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f330 a h); rw [kzero] at e; exact e


theorem assetAt_34 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 34).isDummy (row a 34).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 34 35) = (row a 34).isDummy := by rw [(publicBatchWrapper89_f333 a h), (publicBatchWrapper89_f331 a h), (publicBatchWrapper89_f332 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2201 a h), by rw [← hd, ← (publicBatchWrapper89_f2202 a h), (publicBatchWrapper89_f2203 a h), kone]⟩


theorem feeAt_34 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 34).isDummy (row a 34).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 34 35) = (row a 34).isDummy := by rw [(publicBatchWrapper89_f333 a h), (publicBatchWrapper89_f331 a h), (publicBatchWrapper89_f332 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2204 a h), by rw [← hd, ← (publicBatchWrapper89_f2205 a h), (publicBatchWrapper89_f2206 a h), kone]⟩


theorem blockAt_34 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 34).isDummy (row a 34).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 34 35) = (row a 34).isDummy := by rw [(publicBatchWrapper89_f333 a h), (publicBatchWrapper89_f331 a h), (publicBatchWrapper89_f332 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18931), a (.virt 18933), a (.virt 18935), a (.virt 18937)],
    ![a (.virt 18932), a (.virt 18934), a (.virt 18936), a (.virt 18938)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2207 a h)
    · exact (publicBatchWrapper89_f2208 a h)
    · exact (publicBatchWrapper89_f2209 a h)
    · exact (publicBatchWrapper89_f2210 a h)
  · show bor (row a 34).isDummy (band (band (a (.virt 18931)) (a (.virt 18933)))
      (band (a (.virt 18935)) (a (.virt 18937)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2211 a h), ← (publicBatchWrapper89_f2212 a h), ← (publicBatchWrapper89_f2213 a h), ← (publicBatchWrapper89_f2214 a h), (publicBatchWrapper89_f2215 a h), kone]


theorem countAt_34 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 34).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 6802)).val = 32
  rw [(publicBatchWrapper89_f34 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_35 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 35) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f334 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f335 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f336 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f337 a h); rw [kzero] at e; exact e


theorem assetAt_35 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 35).isDummy (row a 35).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 34 79) = (row a 35).isDummy := by rw [(publicBatchWrapper89_f340 a h), (publicBatchWrapper89_f338 a h), (publicBatchWrapper89_f339 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2216 a h), by rw [← hd, ← (publicBatchWrapper89_f2217 a h), (publicBatchWrapper89_f2218 a h), kone]⟩


theorem feeAt_35 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 35).isDummy (row a 35).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 34 79) = (row a 35).isDummy := by rw [(publicBatchWrapper89_f340 a h), (publicBatchWrapper89_f338 a h), (publicBatchWrapper89_f339 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2219 a h), by rw [← hd, ← (publicBatchWrapper89_f2220 a h), (publicBatchWrapper89_f2221 a h), kone]⟩


theorem blockAt_35 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 35).isDummy (row a 35).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 34 79) = (row a 35).isDummy := by rw [(publicBatchWrapper89_f340 a h), (publicBatchWrapper89_f338 a h), (publicBatchWrapper89_f339 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18943), a (.virt 18945), a (.virt 18947), a (.virt 18949)],
    ![a (.virt 18944), a (.virt 18946), a (.virt 18948), a (.virt 18950)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2222 a h)
    · exact (publicBatchWrapper89_f2223 a h)
    · exact (publicBatchWrapper89_f2224 a h)
    · exact (publicBatchWrapper89_f2225 a h)
  · show bor (row a 35).isDummy (band (band (a (.virt 18943)) (a (.virt 18945)))
      (band (a (.virt 18947)) (a (.virt 18949)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2226 a h), ← (publicBatchWrapper89_f2227 a h), ← (publicBatchWrapper89_f2228 a h), ← (publicBatchWrapper89_f2229 a h), (publicBatchWrapper89_f2230 a h), kone]


theorem countAt_35 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 35).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 7002)).val = 32
  rw [(publicBatchWrapper89_f35 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_36 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 36) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f341 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f342 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f343 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f344 a h); rw [kzero] at e; exact e


theorem assetAt_36 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 36).isDummy (row a 36).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 36 43) = (row a 36).isDummy := by rw [(publicBatchWrapper89_f347 a h), (publicBatchWrapper89_f345 a h), (publicBatchWrapper89_f346 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2231 a h), by rw [← hd, ← (publicBatchWrapper89_f2232 a h), (publicBatchWrapper89_f2233 a h), kone]⟩


theorem feeAt_36 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 36).isDummy (row a 36).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 36 43) = (row a 36).isDummy := by rw [(publicBatchWrapper89_f347 a h), (publicBatchWrapper89_f345 a h), (publicBatchWrapper89_f346 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2234 a h), by rw [← hd, ← (publicBatchWrapper89_f2235 a h), (publicBatchWrapper89_f2236 a h), kone]⟩


theorem blockAt_36 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 36).isDummy (row a 36).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 36 43) = (row a 36).isDummy := by rw [(publicBatchWrapper89_f347 a h), (publicBatchWrapper89_f345 a h), (publicBatchWrapper89_f346 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18955), a (.virt 18957), a (.virt 18959), a (.virt 18961)],
    ![a (.virt 18956), a (.virt 18958), a (.virt 18960), a (.virt 18962)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2237 a h)
    · exact (publicBatchWrapper89_f2238 a h)
    · exact (publicBatchWrapper89_f2239 a h)
    · exact (publicBatchWrapper89_f2240 a h)
  · show bor (row a 36).isDummy (band (band (a (.virt 18955)) (a (.virt 18957)))
      (band (a (.virt 18959)) (a (.virt 18961)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2241 a h), ← (publicBatchWrapper89_f2242 a h), ← (publicBatchWrapper89_f2243 a h), ← (publicBatchWrapper89_f2244 a h), (publicBatchWrapper89_f2245 a h), kone]


theorem countAt_36 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 36).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 7202)).val = 32
  rw [(publicBatchWrapper89_f36 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_37 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 37) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f348 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f349 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f350 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f351 a h); rw [kzero] at e; exact e


theorem assetAt_37 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 37).isDummy (row a 37).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 38 7) = (row a 37).isDummy := by rw [(publicBatchWrapper89_f354 a h), (publicBatchWrapper89_f352 a h), (publicBatchWrapper89_f353 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2246 a h), by rw [← hd, ← (publicBatchWrapper89_f2247 a h), (publicBatchWrapper89_f2248 a h), kone]⟩


theorem feeAt_37 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 37).isDummy (row a 37).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 38 7) = (row a 37).isDummy := by rw [(publicBatchWrapper89_f354 a h), (publicBatchWrapper89_f352 a h), (publicBatchWrapper89_f353 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2249 a h), by rw [← hd, ← (publicBatchWrapper89_f2250 a h), (publicBatchWrapper89_f2251 a h), kone]⟩


theorem blockAt_37 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 37).isDummy (row a 37).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 38 7) = (row a 37).isDummy := by rw [(publicBatchWrapper89_f354 a h), (publicBatchWrapper89_f352 a h), (publicBatchWrapper89_f353 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18967), a (.virt 18969), a (.virt 18971), a (.virt 18973)],
    ![a (.virt 18968), a (.virt 18970), a (.virt 18972), a (.virt 18974)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2252 a h)
    · exact (publicBatchWrapper89_f2253 a h)
    · exact (publicBatchWrapper89_f2254 a h)
    · exact (publicBatchWrapper89_f2255 a h)
  · show bor (row a 37).isDummy (band (band (a (.virt 18967)) (a (.virt 18969)))
      (band (a (.virt 18971)) (a (.virt 18973)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2256 a h), ← (publicBatchWrapper89_f2257 a h), ← (publicBatchWrapper89_f2258 a h), ← (publicBatchWrapper89_f2259 a h), (publicBatchWrapper89_f2260 a h), kone]


theorem countAt_37 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 37).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 7402)).val = 32
  rw [(publicBatchWrapper89_f37 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_38 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 38) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f355 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f356 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f357 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f358 a h); rw [kzero] at e; exact e


theorem assetAt_38 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 38).isDummy (row a 38).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 38 51) = (row a 38).isDummy := by rw [(publicBatchWrapper89_f361 a h), (publicBatchWrapper89_f359 a h), (publicBatchWrapper89_f360 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2261 a h), by rw [← hd, ← (publicBatchWrapper89_f2262 a h), (publicBatchWrapper89_f2263 a h), kone]⟩


theorem feeAt_38 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 38).isDummy (row a 38).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 38 51) = (row a 38).isDummy := by rw [(publicBatchWrapper89_f361 a h), (publicBatchWrapper89_f359 a h), (publicBatchWrapper89_f360 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2264 a h), by rw [← hd, ← (publicBatchWrapper89_f2265 a h), (publicBatchWrapper89_f2266 a h), kone]⟩


theorem blockAt_38 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 38).isDummy (row a 38).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 38 51) = (row a 38).isDummy := by rw [(publicBatchWrapper89_f361 a h), (publicBatchWrapper89_f359 a h), (publicBatchWrapper89_f360 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18979), a (.virt 18981), a (.virt 18983), a (.virt 18985)],
    ![a (.virt 18980), a (.virt 18982), a (.virt 18984), a (.virt 18986)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2267 a h)
    · exact (publicBatchWrapper89_f2268 a h)
    · exact (publicBatchWrapper89_f2269 a h)
    · exact (publicBatchWrapper89_f2270 a h)
  · show bor (row a 38).isDummy (band (band (a (.virt 18979)) (a (.virt 18981)))
      (band (a (.virt 18983)) (a (.virt 18985)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2271 a h), ← (publicBatchWrapper89_f2272 a h), ← (publicBatchWrapper89_f2273 a h), ← (publicBatchWrapper89_f2274 a h), (publicBatchWrapper89_f2275 a h), kone]


theorem countAt_38 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 38).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 7602)).val = 32
  rw [(publicBatchWrapper89_f38 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_39 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 39) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f362 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f363 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f364 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f365 a h); rw [kzero] at e; exact e


theorem assetAt_39 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 39).isDummy (row a 39).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 39 15) = (row a 39).isDummy := by rw [(publicBatchWrapper89_f368 a h), (publicBatchWrapper89_f366 a h), (publicBatchWrapper89_f367 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2276 a h), by rw [← hd, ← (publicBatchWrapper89_f2277 a h), (publicBatchWrapper89_f2278 a h), kone]⟩


theorem feeAt_39 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 39).isDummy (row a 39).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 39 15) = (row a 39).isDummy := by rw [(publicBatchWrapper89_f368 a h), (publicBatchWrapper89_f366 a h), (publicBatchWrapper89_f367 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2279 a h), by rw [← hd, ← (publicBatchWrapper89_f2280 a h), (publicBatchWrapper89_f2281 a h), kone]⟩


theorem blockAt_39 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 39).isDummy (row a 39).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 39 15) = (row a 39).isDummy := by rw [(publicBatchWrapper89_f368 a h), (publicBatchWrapper89_f366 a h), (publicBatchWrapper89_f367 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18991), a (.virt 18993), a (.virt 18995), a (.virt 18997)],
    ![a (.virt 18992), a (.virt 18994), a (.virt 18996), a (.virt 18998)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2282 a h)
    · exact (publicBatchWrapper89_f2283 a h)
    · exact (publicBatchWrapper89_f2284 a h)
    · exact (publicBatchWrapper89_f2285 a h)
  · show bor (row a 39).isDummy (band (band (a (.virt 18991)) (a (.virt 18993)))
      (band (a (.virt 18995)) (a (.virt 18997)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2286 a h), ← (publicBatchWrapper89_f2287 a h), ← (publicBatchWrapper89_f2288 a h), ← (publicBatchWrapper89_f2289 a h), (publicBatchWrapper89_f2290 a h), kone]


theorem countAt_39 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 39).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 7802)).val = 32
  rw [(publicBatchWrapper89_f39 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_40 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 40) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f369 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f370 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f371 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f372 a h); rw [kzero] at e; exact e


theorem assetAt_40 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 40).isDummy (row a 40).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 39 59) = (row a 40).isDummy := by rw [(publicBatchWrapper89_f375 a h), (publicBatchWrapper89_f373 a h), (publicBatchWrapper89_f374 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2291 a h), by rw [← hd, ← (publicBatchWrapper89_f2292 a h), (publicBatchWrapper89_f2293 a h), kone]⟩


theorem feeAt_40 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 40).isDummy (row a 40).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 39 59) = (row a 40).isDummy := by rw [(publicBatchWrapper89_f375 a h), (publicBatchWrapper89_f373 a h), (publicBatchWrapper89_f374 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2294 a h), by rw [← hd, ← (publicBatchWrapper89_f2295 a h), (publicBatchWrapper89_f2296 a h), kone]⟩


theorem blockAt_40 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 40).isDummy (row a 40).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 39 59) = (row a 40).isDummy := by rw [(publicBatchWrapper89_f375 a h), (publicBatchWrapper89_f373 a h), (publicBatchWrapper89_f374 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19003), a (.virt 19005), a (.virt 19007), a (.virt 19009)],
    ![a (.virt 19004), a (.virt 19006), a (.virt 19008), a (.virt 19010)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2297 a h)
    · exact (publicBatchWrapper89_f2298 a h)
    · exact (publicBatchWrapper89_f2299 a h)
    · exact (publicBatchWrapper89_f2300 a h)
  · show bor (row a 40).isDummy (band (band (a (.virt 19003)) (a (.virt 19005)))
      (band (a (.virt 19007)) (a (.virt 19009)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2301 a h), ← (publicBatchWrapper89_f2302 a h), ← (publicBatchWrapper89_f2303 a h), ← (publicBatchWrapper89_f2304 a h), (publicBatchWrapper89_f2305 a h), kone]


theorem countAt_40 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 40).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 8002)).val = 32
  rw [(publicBatchWrapper89_f40 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_41 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 41) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f376 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f377 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f378 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f379 a h); rw [kzero] at e; exact e


theorem assetAt_41 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 41).isDummy (row a 41).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 41 23) = (row a 41).isDummy := by rw [(publicBatchWrapper89_f382 a h), (publicBatchWrapper89_f380 a h), (publicBatchWrapper89_f381 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2306 a h), by rw [← hd, ← (publicBatchWrapper89_f2307 a h), (publicBatchWrapper89_f2308 a h), kone]⟩


theorem feeAt_41 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 41).isDummy (row a 41).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 41 23) = (row a 41).isDummy := by rw [(publicBatchWrapper89_f382 a h), (publicBatchWrapper89_f380 a h), (publicBatchWrapper89_f381 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2309 a h), by rw [← hd, ← (publicBatchWrapper89_f2310 a h), (publicBatchWrapper89_f2311 a h), kone]⟩


theorem blockAt_41 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 41).isDummy (row a 41).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 41 23) = (row a 41).isDummy := by rw [(publicBatchWrapper89_f382 a h), (publicBatchWrapper89_f380 a h), (publicBatchWrapper89_f381 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19015), a (.virt 19017), a (.virt 19019), a (.virt 19021)],
    ![a (.virt 19016), a (.virt 19018), a (.virt 19020), a (.virt 19022)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2312 a h)
    · exact (publicBatchWrapper89_f2313 a h)
    · exact (publicBatchWrapper89_f2314 a h)
    · exact (publicBatchWrapper89_f2315 a h)
  · show bor (row a 41).isDummy (band (band (a (.virt 19015)) (a (.virt 19017)))
      (band (a (.virt 19019)) (a (.virt 19021)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2316 a h), ← (publicBatchWrapper89_f2317 a h), ← (publicBatchWrapper89_f2318 a h), ← (publicBatchWrapper89_f2319 a h), (publicBatchWrapper89_f2320 a h), kone]


theorem countAt_41 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 41).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 8202)).val = 32
  rw [(publicBatchWrapper89_f41 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_42 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 42) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f383 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f384 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f385 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f386 a h); rw [kzero] at e; exact e


theorem assetAt_42 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 42).isDummy (row a 42).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 41 67) = (row a 42).isDummy := by rw [(publicBatchWrapper89_f389 a h), (publicBatchWrapper89_f387 a h), (publicBatchWrapper89_f388 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2321 a h), by rw [← hd, ← (publicBatchWrapper89_f2322 a h), (publicBatchWrapper89_f2323 a h), kone]⟩


theorem feeAt_42 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 42).isDummy (row a 42).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 41 67) = (row a 42).isDummy := by rw [(publicBatchWrapper89_f389 a h), (publicBatchWrapper89_f387 a h), (publicBatchWrapper89_f388 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2324 a h), by rw [← hd, ← (publicBatchWrapper89_f2325 a h), (publicBatchWrapper89_f2326 a h), kone]⟩


theorem blockAt_42 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 42).isDummy (row a 42).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 41 67) = (row a 42).isDummy := by rw [(publicBatchWrapper89_f389 a h), (publicBatchWrapper89_f387 a h), (publicBatchWrapper89_f388 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19027), a (.virt 19029), a (.virt 19031), a (.virt 19033)],
    ![a (.virt 19028), a (.virt 19030), a (.virt 19032), a (.virt 19034)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2327 a h)
    · exact (publicBatchWrapper89_f2328 a h)
    · exact (publicBatchWrapper89_f2329 a h)
    · exact (publicBatchWrapper89_f2330 a h)
  · show bor (row a 42).isDummy (band (band (a (.virt 19027)) (a (.virt 19029)))
      (band (a (.virt 19031)) (a (.virt 19033)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2331 a h), ← (publicBatchWrapper89_f2332 a h), ← (publicBatchWrapper89_f2333 a h), ← (publicBatchWrapper89_f2334 a h), (publicBatchWrapper89_f2335 a h), kone]


theorem countAt_42 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 42).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 8402)).val = 32
  rw [(publicBatchWrapper89_f42 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_43 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 43) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f390 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f391 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f392 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f393 a h); rw [kzero] at e; exact e


theorem assetAt_43 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 43).isDummy (row a 43).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 43 31) = (row a 43).isDummy := by rw [(publicBatchWrapper89_f396 a h), (publicBatchWrapper89_f394 a h), (publicBatchWrapper89_f395 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2336 a h), by rw [← hd, ← (publicBatchWrapper89_f2337 a h), (publicBatchWrapper89_f2338 a h), kone]⟩


theorem feeAt_43 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 43).isDummy (row a 43).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 43 31) = (row a 43).isDummy := by rw [(publicBatchWrapper89_f396 a h), (publicBatchWrapper89_f394 a h), (publicBatchWrapper89_f395 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2339 a h), by rw [← hd, ← (publicBatchWrapper89_f2340 a h), (publicBatchWrapper89_f2341 a h), kone]⟩


theorem blockAt_43 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 43).isDummy (row a 43).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 43 31) = (row a 43).isDummy := by rw [(publicBatchWrapper89_f396 a h), (publicBatchWrapper89_f394 a h), (publicBatchWrapper89_f395 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19039), a (.virt 19041), a (.virt 19043), a (.virt 19045)],
    ![a (.virt 19040), a (.virt 19042), a (.virt 19044), a (.virt 19046)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2342 a h)
    · exact (publicBatchWrapper89_f2343 a h)
    · exact (publicBatchWrapper89_f2344 a h)
    · exact (publicBatchWrapper89_f2345 a h)
  · show bor (row a 43).isDummy (band (band (a (.virt 19039)) (a (.virt 19041)))
      (band (a (.virt 19043)) (a (.virt 19045)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2346 a h), ← (publicBatchWrapper89_f2347 a h), ← (publicBatchWrapper89_f2348 a h), ← (publicBatchWrapper89_f2349 a h), (publicBatchWrapper89_f2350 a h), kone]


theorem countAt_43 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 43).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 8602)).val = 32
  rw [(publicBatchWrapper89_f43 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_44 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 44) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f397 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f398 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f399 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f400 a h); rw [kzero] at e; exact e


theorem assetAt_44 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 44).isDummy (row a 44).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 43 75) = (row a 44).isDummy := by rw [(publicBatchWrapper89_f403 a h), (publicBatchWrapper89_f401 a h), (publicBatchWrapper89_f402 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2351 a h), by rw [← hd, ← (publicBatchWrapper89_f2352 a h), (publicBatchWrapper89_f2353 a h), kone]⟩


theorem feeAt_44 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 44).isDummy (row a 44).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 43 75) = (row a 44).isDummy := by rw [(publicBatchWrapper89_f403 a h), (publicBatchWrapper89_f401 a h), (publicBatchWrapper89_f402 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2354 a h), by rw [← hd, ← (publicBatchWrapper89_f2355 a h), (publicBatchWrapper89_f2356 a h), kone]⟩


theorem blockAt_44 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 44).isDummy (row a 44).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 43 75) = (row a 44).isDummy := by rw [(publicBatchWrapper89_f403 a h), (publicBatchWrapper89_f401 a h), (publicBatchWrapper89_f402 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19051), a (.virt 19053), a (.virt 19055), a (.virt 19057)],
    ![a (.virt 19052), a (.virt 19054), a (.virt 19056), a (.virt 19058)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2357 a h)
    · exact (publicBatchWrapper89_f2358 a h)
    · exact (publicBatchWrapper89_f2359 a h)
    · exact (publicBatchWrapper89_f2360 a h)
  · show bor (row a 44).isDummy (band (band (a (.virt 19051)) (a (.virt 19053)))
      (band (a (.virt 19055)) (a (.virt 19057)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2361 a h), ← (publicBatchWrapper89_f2362 a h), ← (publicBatchWrapper89_f2363 a h), ← (publicBatchWrapper89_f2364 a h), (publicBatchWrapper89_f2365 a h), kone]


theorem countAt_44 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 44).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 8802)).val = 32
  rw [(publicBatchWrapper89_f44 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_45 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 45) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f404 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f405 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f406 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f407 a h); rw [kzero] at e; exact e


theorem assetAt_45 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 45).isDummy (row a 45).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 45 39) = (row a 45).isDummy := by rw [(publicBatchWrapper89_f410 a h), (publicBatchWrapper89_f408 a h), (publicBatchWrapper89_f409 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2366 a h), by rw [← hd, ← (publicBatchWrapper89_f2367 a h), (publicBatchWrapper89_f2368 a h), kone]⟩


theorem feeAt_45 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 45).isDummy (row a 45).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 45 39) = (row a 45).isDummy := by rw [(publicBatchWrapper89_f410 a h), (publicBatchWrapper89_f408 a h), (publicBatchWrapper89_f409 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2369 a h), by rw [← hd, ← (publicBatchWrapper89_f2370 a h), (publicBatchWrapper89_f2371 a h), kone]⟩


theorem blockAt_45 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 45).isDummy (row a 45).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 45 39) = (row a 45).isDummy := by rw [(publicBatchWrapper89_f410 a h), (publicBatchWrapper89_f408 a h), (publicBatchWrapper89_f409 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19063), a (.virt 19065), a (.virt 19067), a (.virt 19069)],
    ![a (.virt 19064), a (.virt 19066), a (.virt 19068), a (.virt 19070)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2372 a h)
    · exact (publicBatchWrapper89_f2373 a h)
    · exact (publicBatchWrapper89_f2374 a h)
    · exact (publicBatchWrapper89_f2375 a h)
  · show bor (row a 45).isDummy (band (band (a (.virt 19063)) (a (.virt 19065)))
      (band (a (.virt 19067)) (a (.virt 19069)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2376 a h), ← (publicBatchWrapper89_f2377 a h), ← (publicBatchWrapper89_f2378 a h), ← (publicBatchWrapper89_f2379 a h), (publicBatchWrapper89_f2380 a h), kone]


theorem countAt_45 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 45).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 9002)).val = 32
  rw [(publicBatchWrapper89_f45 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_46 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 46) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f411 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f412 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f413 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f414 a h); rw [kzero] at e; exact e


theorem assetAt_46 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 46).isDummy (row a 46).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 46 3) = (row a 46).isDummy := by rw [(publicBatchWrapper89_f417 a h), (publicBatchWrapper89_f415 a h), (publicBatchWrapper89_f416 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2381 a h), by rw [← hd, ← (publicBatchWrapper89_f2382 a h), (publicBatchWrapper89_f2383 a h), kone]⟩


theorem feeAt_46 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 46).isDummy (row a 46).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 46 3) = (row a 46).isDummy := by rw [(publicBatchWrapper89_f417 a h), (publicBatchWrapper89_f415 a h), (publicBatchWrapper89_f416 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2384 a h), by rw [← hd, ← (publicBatchWrapper89_f2385 a h), (publicBatchWrapper89_f2386 a h), kone]⟩


theorem blockAt_46 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 46).isDummy (row a 46).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 46 3) = (row a 46).isDummy := by rw [(publicBatchWrapper89_f417 a h), (publicBatchWrapper89_f415 a h), (publicBatchWrapper89_f416 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19075), a (.virt 19077), a (.virt 19079), a (.virt 19081)],
    ![a (.virt 19076), a (.virt 19078), a (.virt 19080), a (.virt 19082)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2387 a h)
    · exact (publicBatchWrapper89_f2388 a h)
    · exact (publicBatchWrapper89_f2389 a h)
    · exact (publicBatchWrapper89_f2390 a h)
  · show bor (row a 46).isDummy (band (band (a (.virt 19075)) (a (.virt 19077)))
      (band (a (.virt 19079)) (a (.virt 19081)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2391 a h), ← (publicBatchWrapper89_f2392 a h), ← (publicBatchWrapper89_f2393 a h), ← (publicBatchWrapper89_f2394 a h), (publicBatchWrapper89_f2395 a h), kone]


theorem countAt_46 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 46).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 9202)).val = 32
  rw [(publicBatchWrapper89_f46 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_47 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 47) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f418 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f419 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f420 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f421 a h); rw [kzero] at e; exact e


theorem assetAt_47 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 47).isDummy (row a 47).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 46 47) = (row a 47).isDummy := by rw [(publicBatchWrapper89_f424 a h), (publicBatchWrapper89_f422 a h), (publicBatchWrapper89_f423 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2396 a h), by rw [← hd, ← (publicBatchWrapper89_f2397 a h), (publicBatchWrapper89_f2398 a h), kone]⟩


theorem feeAt_47 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 47).isDummy (row a 47).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 46 47) = (row a 47).isDummy := by rw [(publicBatchWrapper89_f424 a h), (publicBatchWrapper89_f422 a h), (publicBatchWrapper89_f423 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2399 a h), by rw [← hd, ← (publicBatchWrapper89_f2400 a h), (publicBatchWrapper89_f2401 a h), kone]⟩


theorem blockAt_47 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 47).isDummy (row a 47).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 46 47) = (row a 47).isDummy := by rw [(publicBatchWrapper89_f424 a h), (publicBatchWrapper89_f422 a h), (publicBatchWrapper89_f423 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19087), a (.virt 19089), a (.virt 19091), a (.virt 19093)],
    ![a (.virt 19088), a (.virt 19090), a (.virt 19092), a (.virt 19094)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2402 a h)
    · exact (publicBatchWrapper89_f2403 a h)
    · exact (publicBatchWrapper89_f2404 a h)
    · exact (publicBatchWrapper89_f2405 a h)
  · show bor (row a 47).isDummy (band (band (a (.virt 19087)) (a (.virt 19089)))
      (band (a (.virt 19091)) (a (.virt 19093)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2406 a h), ← (publicBatchWrapper89_f2407 a h), ← (publicBatchWrapper89_f2408 a h), ← (publicBatchWrapper89_f2409 a h), (publicBatchWrapper89_f2410 a h), kone]


theorem countAt_47 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 47).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 9402)).val = 32
  rw [(publicBatchWrapper89_f47 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_48 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 48) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f425 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f426 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f427 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f428 a h); rw [kzero] at e; exact e


theorem assetAt_48 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 48).isDummy (row a 48).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 48 11) = (row a 48).isDummy := by rw [(publicBatchWrapper89_f431 a h), (publicBatchWrapper89_f429 a h), (publicBatchWrapper89_f430 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2411 a h), by rw [← hd, ← (publicBatchWrapper89_f2412 a h), (publicBatchWrapper89_f2413 a h), kone]⟩


theorem feeAt_48 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 48).isDummy (row a 48).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 48 11) = (row a 48).isDummy := by rw [(publicBatchWrapper89_f431 a h), (publicBatchWrapper89_f429 a h), (publicBatchWrapper89_f430 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2414 a h), by rw [← hd, ← (publicBatchWrapper89_f2415 a h), (publicBatchWrapper89_f2416 a h), kone]⟩


theorem blockAt_48 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 48).isDummy (row a 48).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 48 11) = (row a 48).isDummy := by rw [(publicBatchWrapper89_f431 a h), (publicBatchWrapper89_f429 a h), (publicBatchWrapper89_f430 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19099), a (.virt 19101), a (.virt 19103), a (.virt 19105)],
    ![a (.virt 19100), a (.virt 19102), a (.virt 19104), a (.virt 19106)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2417 a h)
    · exact (publicBatchWrapper89_f2418 a h)
    · exact (publicBatchWrapper89_f2419 a h)
    · exact (publicBatchWrapper89_f2420 a h)
  · show bor (row a 48).isDummy (band (band (a (.virt 19099)) (a (.virt 19101)))
      (band (a (.virt 19103)) (a (.virt 19105)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2421 a h), ← (publicBatchWrapper89_f2422 a h), ← (publicBatchWrapper89_f2423 a h), ← (publicBatchWrapper89_f2424 a h), (publicBatchWrapper89_f2425 a h), kone]


theorem countAt_48 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 48).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 9602)).val = 32
  rw [(publicBatchWrapper89_f48 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_49 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 49) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f432 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f433 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f434 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f435 a h); rw [kzero] at e; exact e


theorem assetAt_49 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 49).isDummy (row a 49).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 48 55) = (row a 49).isDummy := by rw [(publicBatchWrapper89_f438 a h), (publicBatchWrapper89_f436 a h), (publicBatchWrapper89_f437 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2426 a h), by rw [← hd, ← (publicBatchWrapper89_f2427 a h), (publicBatchWrapper89_f2428 a h), kone]⟩


theorem feeAt_49 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 49).isDummy (row a 49).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 48 55) = (row a 49).isDummy := by rw [(publicBatchWrapper89_f438 a h), (publicBatchWrapper89_f436 a h), (publicBatchWrapper89_f437 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2429 a h), by rw [← hd, ← (publicBatchWrapper89_f2430 a h), (publicBatchWrapper89_f2431 a h), kone]⟩


theorem blockAt_49 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 49).isDummy (row a 49).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 48 55) = (row a 49).isDummy := by rw [(publicBatchWrapper89_f438 a h), (publicBatchWrapper89_f436 a h), (publicBatchWrapper89_f437 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19111), a (.virt 19113), a (.virt 19115), a (.virt 19117)],
    ![a (.virt 19112), a (.virt 19114), a (.virt 19116), a (.virt 19118)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2432 a h)
    · exact (publicBatchWrapper89_f2433 a h)
    · exact (publicBatchWrapper89_f2434 a h)
    · exact (publicBatchWrapper89_f2435 a h)
  · show bor (row a 49).isDummy (band (band (a (.virt 19111)) (a (.virt 19113)))
      (band (a (.virt 19115)) (a (.virt 19117)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2436 a h), ← (publicBatchWrapper89_f2437 a h), ← (publicBatchWrapper89_f2438 a h), ← (publicBatchWrapper89_f2439 a h), (publicBatchWrapper89_f2440 a h), kone]


theorem countAt_49 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 49).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 9802)).val = 32
  rw [(publicBatchWrapper89_f49 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_50 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 50) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f439 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f440 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f441 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f442 a h); rw [kzero] at e; exact e


theorem assetAt_50 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 50).isDummy (row a 50).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 50 19) = (row a 50).isDummy := by rw [(publicBatchWrapper89_f445 a h), (publicBatchWrapper89_f443 a h), (publicBatchWrapper89_f444 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2441 a h), by rw [← hd, ← (publicBatchWrapper89_f2442 a h), (publicBatchWrapper89_f2443 a h), kone]⟩


theorem feeAt_50 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 50).isDummy (row a 50).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 50 19) = (row a 50).isDummy := by rw [(publicBatchWrapper89_f445 a h), (publicBatchWrapper89_f443 a h), (publicBatchWrapper89_f444 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2444 a h), by rw [← hd, ← (publicBatchWrapper89_f2445 a h), (publicBatchWrapper89_f2446 a h), kone]⟩


theorem blockAt_50 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 50).isDummy (row a 50).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 50 19) = (row a 50).isDummy := by rw [(publicBatchWrapper89_f445 a h), (publicBatchWrapper89_f443 a h), (publicBatchWrapper89_f444 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19123), a (.virt 19125), a (.virt 19127), a (.virt 19129)],
    ![a (.virt 19124), a (.virt 19126), a (.virt 19128), a (.virt 19130)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2447 a h)
    · exact (publicBatchWrapper89_f2448 a h)
    · exact (publicBatchWrapper89_f2449 a h)
    · exact (publicBatchWrapper89_f2450 a h)
  · show bor (row a 50).isDummy (band (band (a (.virt 19123)) (a (.virt 19125)))
      (band (a (.virt 19127)) (a (.virt 19129)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2451 a h), ← (publicBatchWrapper89_f2452 a h), ← (publicBatchWrapper89_f2453 a h), ← (publicBatchWrapper89_f2454 a h), (publicBatchWrapper89_f2455 a h), kone]


theorem countAt_50 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 50).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 10002)).val = 32
  rw [(publicBatchWrapper89_f50 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_51 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 51) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f446 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f447 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f448 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f449 a h); rw [kzero] at e; exact e


theorem assetAt_51 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 51).isDummy (row a 51).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 50 63) = (row a 51).isDummy := by rw [(publicBatchWrapper89_f452 a h), (publicBatchWrapper89_f450 a h), (publicBatchWrapper89_f451 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2456 a h), by rw [← hd, ← (publicBatchWrapper89_f2457 a h), (publicBatchWrapper89_f2458 a h), kone]⟩


theorem feeAt_51 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 51).isDummy (row a 51).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 50 63) = (row a 51).isDummy := by rw [(publicBatchWrapper89_f452 a h), (publicBatchWrapper89_f450 a h), (publicBatchWrapper89_f451 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2459 a h), by rw [← hd, ← (publicBatchWrapper89_f2460 a h), (publicBatchWrapper89_f2461 a h), kone]⟩


theorem blockAt_51 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 51).isDummy (row a 51).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 50 63) = (row a 51).isDummy := by rw [(publicBatchWrapper89_f452 a h), (publicBatchWrapper89_f450 a h), (publicBatchWrapper89_f451 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19135), a (.virt 19137), a (.virt 19139), a (.virt 19141)],
    ![a (.virt 19136), a (.virt 19138), a (.virt 19140), a (.virt 19142)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2462 a h)
    · exact (publicBatchWrapper89_f2463 a h)
    · exact (publicBatchWrapper89_f2464 a h)
    · exact (publicBatchWrapper89_f2465 a h)
  · show bor (row a 51).isDummy (band (band (a (.virt 19135)) (a (.virt 19137)))
      (band (a (.virt 19139)) (a (.virt 19141)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2466 a h), ← (publicBatchWrapper89_f2467 a h), ← (publicBatchWrapper89_f2468 a h), ← (publicBatchWrapper89_f2469 a h), (publicBatchWrapper89_f2470 a h), kone]


theorem countAt_51 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 51).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 10202)).val = 32
  rw [(publicBatchWrapper89_f51 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_52 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 52) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f453 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f454 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f455 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f456 a h); rw [kzero] at e; exact e


theorem assetAt_52 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 52).isDummy (row a 52).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 52 27) = (row a 52).isDummy := by rw [(publicBatchWrapper89_f459 a h), (publicBatchWrapper89_f457 a h), (publicBatchWrapper89_f458 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2471 a h), by rw [← hd, ← (publicBatchWrapper89_f2472 a h), (publicBatchWrapper89_f2473 a h), kone]⟩


theorem feeAt_52 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 52).isDummy (row a 52).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 52 27) = (row a 52).isDummy := by rw [(publicBatchWrapper89_f459 a h), (publicBatchWrapper89_f457 a h), (publicBatchWrapper89_f458 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2474 a h), by rw [← hd, ← (publicBatchWrapper89_f2475 a h), (publicBatchWrapper89_f2476 a h), kone]⟩


theorem blockAt_52 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 52).isDummy (row a 52).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 52 27) = (row a 52).isDummy := by rw [(publicBatchWrapper89_f459 a h), (publicBatchWrapper89_f457 a h), (publicBatchWrapper89_f458 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19147), a (.virt 19149), a (.virt 19151), a (.virt 19153)],
    ![a (.virt 19148), a (.virt 19150), a (.virt 19152), a (.virt 19154)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2477 a h)
    · exact (publicBatchWrapper89_f2478 a h)
    · exact (publicBatchWrapper89_f2479 a h)
    · exact (publicBatchWrapper89_f2480 a h)
  · show bor (row a 52).isDummy (band (band (a (.virt 19147)) (a (.virt 19149)))
      (band (a (.virt 19151)) (a (.virt 19153)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2481 a h), ← (publicBatchWrapper89_f2482 a h), ← (publicBatchWrapper89_f2483 a h), ← (publicBatchWrapper89_f2484 a h), (publicBatchWrapper89_f2485 a h), kone]


theorem countAt_52 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 52).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 10402)).val = 32
  rw [(publicBatchWrapper89_f52 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_53 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 53) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f460 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f461 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f462 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f463 a h); rw [kzero] at e; exact e


theorem assetAt_53 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 53).isDummy (row a 53).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 52 71) = (row a 53).isDummy := by rw [(publicBatchWrapper89_f466 a h), (publicBatchWrapper89_f464 a h), (publicBatchWrapper89_f465 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2486 a h), by rw [← hd, ← (publicBatchWrapper89_f2487 a h), (publicBatchWrapper89_f2488 a h), kone]⟩


theorem feeAt_53 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 53).isDummy (row a 53).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 52 71) = (row a 53).isDummy := by rw [(publicBatchWrapper89_f466 a h), (publicBatchWrapper89_f464 a h), (publicBatchWrapper89_f465 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2489 a h), by rw [← hd, ← (publicBatchWrapper89_f2490 a h), (publicBatchWrapper89_f2491 a h), kone]⟩


theorem blockAt_53 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 53).isDummy (row a 53).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 52 71) = (row a 53).isDummy := by rw [(publicBatchWrapper89_f466 a h), (publicBatchWrapper89_f464 a h), (publicBatchWrapper89_f465 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19159), a (.virt 19161), a (.virt 19163), a (.virt 19165)],
    ![a (.virt 19160), a (.virt 19162), a (.virt 19164), a (.virt 19166)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2492 a h)
    · exact (publicBatchWrapper89_f2493 a h)
    · exact (publicBatchWrapper89_f2494 a h)
    · exact (publicBatchWrapper89_f2495 a h)
  · show bor (row a 53).isDummy (band (band (a (.virt 19159)) (a (.virt 19161)))
      (band (a (.virt 19163)) (a (.virt 19165)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2496 a h), ← (publicBatchWrapper89_f2497 a h), ← (publicBatchWrapper89_f2498 a h), ← (publicBatchWrapper89_f2499 a h), (publicBatchWrapper89_f2500 a h), kone]


theorem countAt_53 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 53).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 10602)).val = 32
  rw [(publicBatchWrapper89_f53 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_54 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 54) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f467 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f468 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f469 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f470 a h); rw [kzero] at e; exact e


theorem assetAt_54 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 54).isDummy (row a 54).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 53 35) = (row a 54).isDummy := by rw [(publicBatchWrapper89_f473 a h), (publicBatchWrapper89_f471 a h), (publicBatchWrapper89_f472 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2501 a h), by rw [← hd, ← (publicBatchWrapper89_f2502 a h), (publicBatchWrapper89_f2503 a h), kone]⟩


theorem feeAt_54 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 54).isDummy (row a 54).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 53 35) = (row a 54).isDummy := by rw [(publicBatchWrapper89_f473 a h), (publicBatchWrapper89_f471 a h), (publicBatchWrapper89_f472 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2504 a h), by rw [← hd, ← (publicBatchWrapper89_f2505 a h), (publicBatchWrapper89_f2506 a h), kone]⟩


theorem blockAt_54 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 54).isDummy (row a 54).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 53 35) = (row a 54).isDummy := by rw [(publicBatchWrapper89_f473 a h), (publicBatchWrapper89_f471 a h), (publicBatchWrapper89_f472 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19171), a (.virt 19173), a (.virt 19175), a (.virt 19177)],
    ![a (.virt 19172), a (.virt 19174), a (.virt 19176), a (.virt 19178)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2507 a h)
    · exact (publicBatchWrapper89_f2508 a h)
    · exact (publicBatchWrapper89_f2509 a h)
    · exact (publicBatchWrapper89_f2510 a h)
  · show bor (row a 54).isDummy (band (band (a (.virt 19171)) (a (.virt 19173)))
      (band (a (.virt 19175)) (a (.virt 19177)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2511 a h), ← (publicBatchWrapper89_f2512 a h), ← (publicBatchWrapper89_f2513 a h), ← (publicBatchWrapper89_f2514 a h), (publicBatchWrapper89_f2515 a h), kone]


theorem countAt_54 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 54).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 10802)).val = 32
  rw [(publicBatchWrapper89_f54 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_55 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 55) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f474 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f475 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f476 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f477 a h); rw [kzero] at e; exact e


theorem assetAt_55 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 55).isDummy (row a 55).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 53 79) = (row a 55).isDummy := by rw [(publicBatchWrapper89_f480 a h), (publicBatchWrapper89_f478 a h), (publicBatchWrapper89_f479 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2516 a h), by rw [← hd, ← (publicBatchWrapper89_f2517 a h), (publicBatchWrapper89_f2518 a h), kone]⟩


theorem feeAt_55 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 55).isDummy (row a 55).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 53 79) = (row a 55).isDummy := by rw [(publicBatchWrapper89_f480 a h), (publicBatchWrapper89_f478 a h), (publicBatchWrapper89_f479 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2519 a h), by rw [← hd, ← (publicBatchWrapper89_f2520 a h), (publicBatchWrapper89_f2521 a h), kone]⟩


theorem blockAt_55 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 55).isDummy (row a 55).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 53 79) = (row a 55).isDummy := by rw [(publicBatchWrapper89_f480 a h), (publicBatchWrapper89_f478 a h), (publicBatchWrapper89_f479 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19183), a (.virt 19185), a (.virt 19187), a (.virt 19189)],
    ![a (.virt 19184), a (.virt 19186), a (.virt 19188), a (.virt 19190)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2522 a h)
    · exact (publicBatchWrapper89_f2523 a h)
    · exact (publicBatchWrapper89_f2524 a h)
    · exact (publicBatchWrapper89_f2525 a h)
  · show bor (row a 55).isDummy (band (band (a (.virt 19183)) (a (.virt 19185)))
      (band (a (.virt 19187)) (a (.virt 19189)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2526 a h), ← (publicBatchWrapper89_f2527 a h), ← (publicBatchWrapper89_f2528 a h), ← (publicBatchWrapper89_f2529 a h), (publicBatchWrapper89_f2530 a h), kone]


theorem countAt_55 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 55).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 11002)).val = 32
  rw [(publicBatchWrapper89_f55 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_56 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 56) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f481 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f482 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f483 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f484 a h); rw [kzero] at e; exact e


theorem assetAt_56 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 56).isDummy (row a 56).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 55 43) = (row a 56).isDummy := by rw [(publicBatchWrapper89_f487 a h), (publicBatchWrapper89_f485 a h), (publicBatchWrapper89_f486 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2531 a h), by rw [← hd, ← (publicBatchWrapper89_f2532 a h), (publicBatchWrapper89_f2533 a h), kone]⟩


theorem feeAt_56 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 56).isDummy (row a 56).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 55 43) = (row a 56).isDummy := by rw [(publicBatchWrapper89_f487 a h), (publicBatchWrapper89_f485 a h), (publicBatchWrapper89_f486 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2534 a h), by rw [← hd, ← (publicBatchWrapper89_f2535 a h), (publicBatchWrapper89_f2536 a h), kone]⟩


theorem blockAt_56 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 56).isDummy (row a 56).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 55 43) = (row a 56).isDummy := by rw [(publicBatchWrapper89_f487 a h), (publicBatchWrapper89_f485 a h), (publicBatchWrapper89_f486 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19195), a (.virt 19197), a (.virt 19199), a (.virt 19201)],
    ![a (.virt 19196), a (.virt 19198), a (.virt 19200), a (.virt 19202)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2537 a h)
    · exact (publicBatchWrapper89_f2538 a h)
    · exact (publicBatchWrapper89_f2539 a h)
    · exact (publicBatchWrapper89_f2540 a h)
  · show bor (row a 56).isDummy (band (band (a (.virt 19195)) (a (.virt 19197)))
      (band (a (.virt 19199)) (a (.virt 19201)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2541 a h), ← (publicBatchWrapper89_f2542 a h), ← (publicBatchWrapper89_f2543 a h), ← (publicBatchWrapper89_f2544 a h), (publicBatchWrapper89_f2545 a h), kone]


theorem countAt_56 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 56).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 11202)).val = 32
  rw [(publicBatchWrapper89_f56 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_57 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 57) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f488 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f489 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f490 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f491 a h); rw [kzero] at e; exact e


theorem assetAt_57 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 57).isDummy (row a 57).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 57 7) = (row a 57).isDummy := by rw [(publicBatchWrapper89_f494 a h), (publicBatchWrapper89_f492 a h), (publicBatchWrapper89_f493 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2546 a h), by rw [← hd, ← (publicBatchWrapper89_f2547 a h), (publicBatchWrapper89_f2548 a h), kone]⟩


theorem feeAt_57 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 57).isDummy (row a 57).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 57 7) = (row a 57).isDummy := by rw [(publicBatchWrapper89_f494 a h), (publicBatchWrapper89_f492 a h), (publicBatchWrapper89_f493 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2549 a h), by rw [← hd, ← (publicBatchWrapper89_f2550 a h), (publicBatchWrapper89_f2551 a h), kone]⟩


theorem blockAt_57 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 57).isDummy (row a 57).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 57 7) = (row a 57).isDummy := by rw [(publicBatchWrapper89_f494 a h), (publicBatchWrapper89_f492 a h), (publicBatchWrapper89_f493 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19207), a (.virt 19209), a (.virt 19211), a (.virt 19213)],
    ![a (.virt 19208), a (.virt 19210), a (.virt 19212), a (.virt 19214)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2552 a h)
    · exact (publicBatchWrapper89_f2553 a h)
    · exact (publicBatchWrapper89_f2554 a h)
    · exact (publicBatchWrapper89_f2555 a h)
  · show bor (row a 57).isDummy (band (band (a (.virt 19207)) (a (.virt 19209)))
      (band (a (.virt 19211)) (a (.virt 19213)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2556 a h), ← (publicBatchWrapper89_f2557 a h), ← (publicBatchWrapper89_f2558 a h), ← (publicBatchWrapper89_f2559 a h), (publicBatchWrapper89_f2560 a h), kone]


theorem countAt_57 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 57).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 11402)).val = 32
  rw [(publicBatchWrapper89_f57 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_58 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 58) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f495 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f496 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f497 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f498 a h); rw [kzero] at e; exact e


theorem assetAt_58 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 58).isDummy (row a 58).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 57 51) = (row a 58).isDummy := by rw [(publicBatchWrapper89_f501 a h), (publicBatchWrapper89_f499 a h), (publicBatchWrapper89_f500 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2561 a h), by rw [← hd, ← (publicBatchWrapper89_f2562 a h), (publicBatchWrapper89_f2563 a h), kone]⟩


theorem feeAt_58 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 58).isDummy (row a 58).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 57 51) = (row a 58).isDummy := by rw [(publicBatchWrapper89_f501 a h), (publicBatchWrapper89_f499 a h), (publicBatchWrapper89_f500 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2564 a h), by rw [← hd, ← (publicBatchWrapper89_f2565 a h), (publicBatchWrapper89_f2566 a h), kone]⟩


theorem blockAt_58 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 58).isDummy (row a 58).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 57 51) = (row a 58).isDummy := by rw [(publicBatchWrapper89_f501 a h), (publicBatchWrapper89_f499 a h), (publicBatchWrapper89_f500 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19219), a (.virt 19221), a (.virt 19223), a (.virt 19225)],
    ![a (.virt 19220), a (.virt 19222), a (.virt 19224), a (.virt 19226)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2567 a h)
    · exact (publicBatchWrapper89_f2568 a h)
    · exact (publicBatchWrapper89_f2569 a h)
    · exact (publicBatchWrapper89_f2570 a h)
  · show bor (row a 58).isDummy (band (band (a (.virt 19219)) (a (.virt 19221)))
      (band (a (.virt 19223)) (a (.virt 19225)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2571 a h), ← (publicBatchWrapper89_f2572 a h), ← (publicBatchWrapper89_f2573 a h), ← (publicBatchWrapper89_f2574 a h), (publicBatchWrapper89_f2575 a h), kone]


theorem countAt_58 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 58).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 11602)).val = 32
  rw [(publicBatchWrapper89_f58 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_59 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 59) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f502 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f503 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f504 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f505 a h); rw [kzero] at e; exact e


theorem assetAt_59 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 59).isDummy (row a 59).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 58 15) = (row a 59).isDummy := by rw [(publicBatchWrapper89_f508 a h), (publicBatchWrapper89_f506 a h), (publicBatchWrapper89_f507 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2576 a h), by rw [← hd, ← (publicBatchWrapper89_f2577 a h), (publicBatchWrapper89_f2578 a h), kone]⟩


theorem feeAt_59 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 59).isDummy (row a 59).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 58 15) = (row a 59).isDummy := by rw [(publicBatchWrapper89_f508 a h), (publicBatchWrapper89_f506 a h), (publicBatchWrapper89_f507 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2579 a h), by rw [← hd, ← (publicBatchWrapper89_f2580 a h), (publicBatchWrapper89_f2581 a h), kone]⟩


theorem blockAt_59 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 59).isDummy (row a 59).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 58 15) = (row a 59).isDummy := by rw [(publicBatchWrapper89_f508 a h), (publicBatchWrapper89_f506 a h), (publicBatchWrapper89_f507 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19231), a (.virt 19233), a (.virt 19235), a (.virt 19237)],
    ![a (.virt 19232), a (.virt 19234), a (.virt 19236), a (.virt 19238)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2582 a h)
    · exact (publicBatchWrapper89_f2583 a h)
    · exact (publicBatchWrapper89_f2584 a h)
    · exact (publicBatchWrapper89_f2585 a h)
  · show bor (row a 59).isDummy (band (band (a (.virt 19231)) (a (.virt 19233)))
      (band (a (.virt 19235)) (a (.virt 19237)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2586 a h), ← (publicBatchWrapper89_f2587 a h), ← (publicBatchWrapper89_f2588 a h), ← (publicBatchWrapper89_f2589 a h), (publicBatchWrapper89_f2590 a h), kone]


theorem countAt_59 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 59).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 11802)).val = 32
  rw [(publicBatchWrapper89_f59 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_60 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 60) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f509 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f510 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f511 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f512 a h); rw [kzero] at e; exact e


theorem assetAt_60 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 60).isDummy (row a 60).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 58 59) = (row a 60).isDummy := by rw [(publicBatchWrapper89_f515 a h), (publicBatchWrapper89_f513 a h), (publicBatchWrapper89_f514 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2591 a h), by rw [← hd, ← (publicBatchWrapper89_f2592 a h), (publicBatchWrapper89_f2593 a h), kone]⟩


theorem feeAt_60 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 60).isDummy (row a 60).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 58 59) = (row a 60).isDummy := by rw [(publicBatchWrapper89_f515 a h), (publicBatchWrapper89_f513 a h), (publicBatchWrapper89_f514 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2594 a h), by rw [← hd, ← (publicBatchWrapper89_f2595 a h), (publicBatchWrapper89_f2596 a h), kone]⟩


theorem blockAt_60 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 60).isDummy (row a 60).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 58 59) = (row a 60).isDummy := by rw [(publicBatchWrapper89_f515 a h), (publicBatchWrapper89_f513 a h), (publicBatchWrapper89_f514 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19243), a (.virt 19245), a (.virt 19247), a (.virt 19249)],
    ![a (.virt 19244), a (.virt 19246), a (.virt 19248), a (.virt 19250)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2597 a h)
    · exact (publicBatchWrapper89_f2598 a h)
    · exact (publicBatchWrapper89_f2599 a h)
    · exact (publicBatchWrapper89_f2600 a h)
  · show bor (row a 60).isDummy (band (band (a (.virt 19243)) (a (.virt 19245)))
      (band (a (.virt 19247)) (a (.virt 19249)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2601 a h), ← (publicBatchWrapper89_f2602 a h), ← (publicBatchWrapper89_f2603 a h), ← (publicBatchWrapper89_f2604 a h), (publicBatchWrapper89_f2605 a h), kone]


theorem countAt_60 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 60).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 12002)).val = 32
  rw [(publicBatchWrapper89_f60 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_61 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 61) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f516 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f517 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f518 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f519 a h); rw [kzero] at e; exact e


theorem assetAt_61 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 61).isDummy (row a 61).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 60 23) = (row a 61).isDummy := by rw [(publicBatchWrapper89_f522 a h), (publicBatchWrapper89_f520 a h), (publicBatchWrapper89_f521 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2606 a h), by rw [← hd, ← (publicBatchWrapper89_f2607 a h), (publicBatchWrapper89_f2608 a h), kone]⟩


theorem feeAt_61 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 61).isDummy (row a 61).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 60 23) = (row a 61).isDummy := by rw [(publicBatchWrapper89_f522 a h), (publicBatchWrapper89_f520 a h), (publicBatchWrapper89_f521 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2609 a h), by rw [← hd, ← (publicBatchWrapper89_f2610 a h), (publicBatchWrapper89_f2611 a h), kone]⟩


theorem blockAt_61 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 61).isDummy (row a 61).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 60 23) = (row a 61).isDummy := by rw [(publicBatchWrapper89_f522 a h), (publicBatchWrapper89_f520 a h), (publicBatchWrapper89_f521 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19255), a (.virt 19257), a (.virt 19259), a (.virt 19261)],
    ![a (.virt 19256), a (.virt 19258), a (.virt 19260), a (.virt 19262)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2612 a h)
    · exact (publicBatchWrapper89_f2613 a h)
    · exact (publicBatchWrapper89_f2614 a h)
    · exact (publicBatchWrapper89_f2615 a h)
  · show bor (row a 61).isDummy (band (band (a (.virt 19255)) (a (.virt 19257)))
      (band (a (.virt 19259)) (a (.virt 19261)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2616 a h), ← (publicBatchWrapper89_f2617 a h), ← (publicBatchWrapper89_f2618 a h), ← (publicBatchWrapper89_f2619 a h), (publicBatchWrapper89_f2620 a h), kone]


theorem countAt_61 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 61).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 12202)).val = 32
  rw [(publicBatchWrapper89_f61 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_62 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 62) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f523 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f524 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f525 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f526 a h); rw [kzero] at e; exact e


theorem assetAt_62 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 62).isDummy (row a 62).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 60 67) = (row a 62).isDummy := by rw [(publicBatchWrapper89_f529 a h), (publicBatchWrapper89_f527 a h), (publicBatchWrapper89_f528 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2621 a h), by rw [← hd, ← (publicBatchWrapper89_f2622 a h), (publicBatchWrapper89_f2623 a h), kone]⟩


theorem feeAt_62 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 62).isDummy (row a 62).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 60 67) = (row a 62).isDummy := by rw [(publicBatchWrapper89_f529 a h), (publicBatchWrapper89_f527 a h), (publicBatchWrapper89_f528 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2624 a h), by rw [← hd, ← (publicBatchWrapper89_f2625 a h), (publicBatchWrapper89_f2626 a h), kone]⟩


theorem blockAt_62 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 62).isDummy (row a 62).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 60 67) = (row a 62).isDummy := by rw [(publicBatchWrapper89_f529 a h), (publicBatchWrapper89_f527 a h), (publicBatchWrapper89_f528 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19267), a (.virt 19269), a (.virt 19271), a (.virt 19273)],
    ![a (.virt 19268), a (.virt 19270), a (.virt 19272), a (.virt 19274)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2627 a h)
    · exact (publicBatchWrapper89_f2628 a h)
    · exact (publicBatchWrapper89_f2629 a h)
    · exact (publicBatchWrapper89_f2630 a h)
  · show bor (row a 62).isDummy (band (band (a (.virt 19267)) (a (.virt 19269)))
      (band (a (.virt 19271)) (a (.virt 19273)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2631 a h), ← (publicBatchWrapper89_f2632 a h), ← (publicBatchWrapper89_f2633 a h), ← (publicBatchWrapper89_f2634 a h), (publicBatchWrapper89_f2635 a h), kone]


theorem countAt_62 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 62).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 12402)).val = 32
  rw [(publicBatchWrapper89_f62 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_63 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 63) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f530 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f531 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f532 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f533 a h); rw [kzero] at e; exact e


theorem assetAt_63 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 63).isDummy (row a 63).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 62 31) = (row a 63).isDummy := by rw [(publicBatchWrapper89_f536 a h), (publicBatchWrapper89_f534 a h), (publicBatchWrapper89_f535 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2636 a h), by rw [← hd, ← (publicBatchWrapper89_f2637 a h), (publicBatchWrapper89_f2638 a h), kone]⟩


theorem feeAt_63 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 63).isDummy (row a 63).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 62 31) = (row a 63).isDummy := by rw [(publicBatchWrapper89_f536 a h), (publicBatchWrapper89_f534 a h), (publicBatchWrapper89_f535 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2639 a h), by rw [← hd, ← (publicBatchWrapper89_f2640 a h), (publicBatchWrapper89_f2641 a h), kone]⟩


theorem blockAt_63 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 63).isDummy (row a 63).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 62 31) = (row a 63).isDummy := by rw [(publicBatchWrapper89_f536 a h), (publicBatchWrapper89_f534 a h), (publicBatchWrapper89_f535 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19279), a (.virt 19281), a (.virt 19283), a (.virt 19285)],
    ![a (.virt 19280), a (.virt 19282), a (.virt 19284), a (.virt 19286)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2642 a h)
    · exact (publicBatchWrapper89_f2643 a h)
    · exact (publicBatchWrapper89_f2644 a h)
    · exact (publicBatchWrapper89_f2645 a h)
  · show bor (row a 63).isDummy (band (band (a (.virt 19279)) (a (.virt 19281)))
      (band (a (.virt 19283)) (a (.virt 19285)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2646 a h), ← (publicBatchWrapper89_f2647 a h), ← (publicBatchWrapper89_f2648 a h), ← (publicBatchWrapper89_f2649 a h), (publicBatchWrapper89_f2650 a h), kone]


theorem countAt_63 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 63).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 12602)).val = 32
  rw [(publicBatchWrapper89_f63 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_64 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 64) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f537 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f538 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f539 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f540 a h); rw [kzero] at e; exact e


theorem assetAt_64 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 64).isDummy (row a 64).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 62 75) = (row a 64).isDummy := by rw [(publicBatchWrapper89_f543 a h), (publicBatchWrapper89_f541 a h), (publicBatchWrapper89_f542 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2651 a h), by rw [← hd, ← (publicBatchWrapper89_f2652 a h), (publicBatchWrapper89_f2653 a h), kone]⟩


theorem feeAt_64 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 64).isDummy (row a 64).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 62 75) = (row a 64).isDummy := by rw [(publicBatchWrapper89_f543 a h), (publicBatchWrapper89_f541 a h), (publicBatchWrapper89_f542 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2654 a h), by rw [← hd, ← (publicBatchWrapper89_f2655 a h), (publicBatchWrapper89_f2656 a h), kone]⟩


theorem blockAt_64 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 64).isDummy (row a 64).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 62 75) = (row a 64).isDummy := by rw [(publicBatchWrapper89_f543 a h), (publicBatchWrapper89_f541 a h), (publicBatchWrapper89_f542 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19291), a (.virt 19293), a (.virt 19295), a (.virt 19297)],
    ![a (.virt 19292), a (.virt 19294), a (.virt 19296), a (.virt 19298)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2657 a h)
    · exact (publicBatchWrapper89_f2658 a h)
    · exact (publicBatchWrapper89_f2659 a h)
    · exact (publicBatchWrapper89_f2660 a h)
  · show bor (row a 64).isDummy (band (band (a (.virt 19291)) (a (.virt 19293)))
      (band (a (.virt 19295)) (a (.virt 19297)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2661 a h), ← (publicBatchWrapper89_f2662 a h), ← (publicBatchWrapper89_f2663 a h), ← (publicBatchWrapper89_f2664 a h), (publicBatchWrapper89_f2665 a h), kone]


theorem countAt_64 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 64).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 12802)).val = 32
  rw [(publicBatchWrapper89_f64 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_65 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 65) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f544 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f545 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f546 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f547 a h); rw [kzero] at e; exact e


theorem assetAt_65 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 65).isDummy (row a 65).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 64 39) = (row a 65).isDummy := by rw [(publicBatchWrapper89_f550 a h), (publicBatchWrapper89_f548 a h), (publicBatchWrapper89_f549 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2666 a h), by rw [← hd, ← (publicBatchWrapper89_f2667 a h), (publicBatchWrapper89_f2668 a h), kone]⟩


theorem feeAt_65 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 65).isDummy (row a 65).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 64 39) = (row a 65).isDummy := by rw [(publicBatchWrapper89_f550 a h), (publicBatchWrapper89_f548 a h), (publicBatchWrapper89_f549 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2669 a h), by rw [← hd, ← (publicBatchWrapper89_f2670 a h), (publicBatchWrapper89_f2671 a h), kone]⟩


theorem blockAt_65 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 65).isDummy (row a 65).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 64 39) = (row a 65).isDummy := by rw [(publicBatchWrapper89_f550 a h), (publicBatchWrapper89_f548 a h), (publicBatchWrapper89_f549 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19303), a (.virt 19305), a (.virt 19307), a (.virt 19309)],
    ![a (.virt 19304), a (.virt 19306), a (.virt 19308), a (.virt 19310)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2672 a h)
    · exact (publicBatchWrapper89_f2673 a h)
    · exact (publicBatchWrapper89_f2674 a h)
    · exact (publicBatchWrapper89_f2675 a h)
  · show bor (row a 65).isDummy (band (band (a (.virt 19303)) (a (.virt 19305)))
      (band (a (.virt 19307)) (a (.virt 19309)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2676 a h), ← (publicBatchWrapper89_f2677 a h), ← (publicBatchWrapper89_f2678 a h), ← (publicBatchWrapper89_f2679 a h), (publicBatchWrapper89_f2680 a h), kone]


theorem countAt_65 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 65).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 13002)).val = 32
  rw [(publicBatchWrapper89_f65 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_66 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 66) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f551 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f552 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f553 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f554 a h); rw [kzero] at e; exact e


theorem assetAt_66 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 66).isDummy (row a 66).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 65 3) = (row a 66).isDummy := by rw [(publicBatchWrapper89_f557 a h), (publicBatchWrapper89_f555 a h), (publicBatchWrapper89_f556 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2681 a h), by rw [← hd, ← (publicBatchWrapper89_f2682 a h), (publicBatchWrapper89_f2683 a h), kone]⟩


theorem feeAt_66 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 66).isDummy (row a 66).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 65 3) = (row a 66).isDummy := by rw [(publicBatchWrapper89_f557 a h), (publicBatchWrapper89_f555 a h), (publicBatchWrapper89_f556 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2684 a h), by rw [← hd, ← (publicBatchWrapper89_f2685 a h), (publicBatchWrapper89_f2686 a h), kone]⟩


theorem blockAt_66 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 66).isDummy (row a 66).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 65 3) = (row a 66).isDummy := by rw [(publicBatchWrapper89_f557 a h), (publicBatchWrapper89_f555 a h), (publicBatchWrapper89_f556 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19315), a (.virt 19317), a (.virt 19319), a (.virt 19321)],
    ![a (.virt 19316), a (.virt 19318), a (.virt 19320), a (.virt 19322)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2687 a h)
    · exact (publicBatchWrapper89_f2688 a h)
    · exact (publicBatchWrapper89_f2689 a h)
    · exact (publicBatchWrapper89_f2690 a h)
  · show bor (row a 66).isDummy (band (band (a (.virt 19315)) (a (.virt 19317)))
      (band (a (.virt 19319)) (a (.virt 19321)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2691 a h), ← (publicBatchWrapper89_f2692 a h), ← (publicBatchWrapper89_f2693 a h), ← (publicBatchWrapper89_f2694 a h), (publicBatchWrapper89_f2695 a h), kone]


theorem countAt_66 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 66).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 13202)).val = 32
  rw [(publicBatchWrapper89_f66 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_67 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 67) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f558 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f559 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f560 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f561 a h); rw [kzero] at e; exact e


theorem assetAt_67 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 67).isDummy (row a 67).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 65 47) = (row a 67).isDummy := by rw [(publicBatchWrapper89_f564 a h), (publicBatchWrapper89_f562 a h), (publicBatchWrapper89_f563 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2696 a h), by rw [← hd, ← (publicBatchWrapper89_f2697 a h), (publicBatchWrapper89_f2698 a h), kone]⟩


theorem feeAt_67 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 67).isDummy (row a 67).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 65 47) = (row a 67).isDummy := by rw [(publicBatchWrapper89_f564 a h), (publicBatchWrapper89_f562 a h), (publicBatchWrapper89_f563 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2699 a h), by rw [← hd, ← (publicBatchWrapper89_f2700 a h), (publicBatchWrapper89_f2701 a h), kone]⟩


theorem blockAt_67 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 67).isDummy (row a 67).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 65 47) = (row a 67).isDummy := by rw [(publicBatchWrapper89_f564 a h), (publicBatchWrapper89_f562 a h), (publicBatchWrapper89_f563 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19327), a (.virt 19329), a (.virt 19331), a (.virt 19333)],
    ![a (.virt 19328), a (.virt 19330), a (.virt 19332), a (.virt 19334)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2702 a h)
    · exact (publicBatchWrapper89_f2703 a h)
    · exact (publicBatchWrapper89_f2704 a h)
    · exact (publicBatchWrapper89_f2705 a h)
  · show bor (row a 67).isDummy (band (band (a (.virt 19327)) (a (.virt 19329)))
      (band (a (.virt 19331)) (a (.virt 19333)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2706 a h), ← (publicBatchWrapper89_f2707 a h), ← (publicBatchWrapper89_f2708 a h), ← (publicBatchWrapper89_f2709 a h), (publicBatchWrapper89_f2710 a h), kone]


theorem countAt_67 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 67).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 13402)).val = 32
  rw [(publicBatchWrapper89_f67 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_68 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 68) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f565 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f566 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f567 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f568 a h); rw [kzero] at e; exact e


theorem assetAt_68 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 68).isDummy (row a 68).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 67 11) = (row a 68).isDummy := by rw [(publicBatchWrapper89_f571 a h), (publicBatchWrapper89_f569 a h), (publicBatchWrapper89_f570 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2711 a h), by rw [← hd, ← (publicBatchWrapper89_f2712 a h), (publicBatchWrapper89_f2713 a h), kone]⟩


theorem feeAt_68 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 68).isDummy (row a 68).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 67 11) = (row a 68).isDummy := by rw [(publicBatchWrapper89_f571 a h), (publicBatchWrapper89_f569 a h), (publicBatchWrapper89_f570 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2714 a h), by rw [← hd, ← (publicBatchWrapper89_f2715 a h), (publicBatchWrapper89_f2716 a h), kone]⟩


theorem blockAt_68 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 68).isDummy (row a 68).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 67 11) = (row a 68).isDummy := by rw [(publicBatchWrapper89_f571 a h), (publicBatchWrapper89_f569 a h), (publicBatchWrapper89_f570 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19339), a (.virt 19341), a (.virt 19343), a (.virt 19345)],
    ![a (.virt 19340), a (.virt 19342), a (.virt 19344), a (.virt 19346)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2717 a h)
    · exact (publicBatchWrapper89_f2718 a h)
    · exact (publicBatchWrapper89_f2719 a h)
    · exact (publicBatchWrapper89_f2720 a h)
  · show bor (row a 68).isDummy (band (band (a (.virt 19339)) (a (.virt 19341)))
      (band (a (.virt 19343)) (a (.virt 19345)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2721 a h), ← (publicBatchWrapper89_f2722 a h), ← (publicBatchWrapper89_f2723 a h), ← (publicBatchWrapper89_f2724 a h), (publicBatchWrapper89_f2725 a h), kone]


theorem countAt_68 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 68).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 13602)).val = 32
  rw [(publicBatchWrapper89_f68 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_69 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 69) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f572 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f573 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f574 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f575 a h); rw [kzero] at e; exact e


theorem assetAt_69 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 69).isDummy (row a 69).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 67 55) = (row a 69).isDummy := by rw [(publicBatchWrapper89_f578 a h), (publicBatchWrapper89_f576 a h), (publicBatchWrapper89_f577 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2726 a h), by rw [← hd, ← (publicBatchWrapper89_f2727 a h), (publicBatchWrapper89_f2728 a h), kone]⟩


theorem feeAt_69 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 69).isDummy (row a 69).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 67 55) = (row a 69).isDummy := by rw [(publicBatchWrapper89_f578 a h), (publicBatchWrapper89_f576 a h), (publicBatchWrapper89_f577 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2729 a h), by rw [← hd, ← (publicBatchWrapper89_f2730 a h), (publicBatchWrapper89_f2731 a h), kone]⟩


theorem blockAt_69 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 69).isDummy (row a 69).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 67 55) = (row a 69).isDummy := by rw [(publicBatchWrapper89_f578 a h), (publicBatchWrapper89_f576 a h), (publicBatchWrapper89_f577 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19351), a (.virt 19353), a (.virt 19355), a (.virt 19357)],
    ![a (.virt 19352), a (.virt 19354), a (.virt 19356), a (.virt 19358)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2732 a h)
    · exact (publicBatchWrapper89_f2733 a h)
    · exact (publicBatchWrapper89_f2734 a h)
    · exact (publicBatchWrapper89_f2735 a h)
  · show bor (row a 69).isDummy (band (band (a (.virt 19351)) (a (.virt 19353)))
      (band (a (.virt 19355)) (a (.virt 19357)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2736 a h), ← (publicBatchWrapper89_f2737 a h), ← (publicBatchWrapper89_f2738 a h), ← (publicBatchWrapper89_f2739 a h), (publicBatchWrapper89_f2740 a h), kone]


theorem countAt_69 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 69).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 13802)).val = 32
  rw [(publicBatchWrapper89_f69 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_70 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 70) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f579 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f580 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f581 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f582 a h); rw [kzero] at e; exact e


theorem assetAt_70 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 70).isDummy (row a 70).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 69 19) = (row a 70).isDummy := by rw [(publicBatchWrapper89_f585 a h), (publicBatchWrapper89_f583 a h), (publicBatchWrapper89_f584 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2741 a h), by rw [← hd, ← (publicBatchWrapper89_f2742 a h), (publicBatchWrapper89_f2743 a h), kone]⟩


theorem feeAt_70 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 70).isDummy (row a 70).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 69 19) = (row a 70).isDummy := by rw [(publicBatchWrapper89_f585 a h), (publicBatchWrapper89_f583 a h), (publicBatchWrapper89_f584 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2744 a h), by rw [← hd, ← (publicBatchWrapper89_f2745 a h), (publicBatchWrapper89_f2746 a h), kone]⟩


theorem blockAt_70 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 70).isDummy (row a 70).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 69 19) = (row a 70).isDummy := by rw [(publicBatchWrapper89_f585 a h), (publicBatchWrapper89_f583 a h), (publicBatchWrapper89_f584 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19363), a (.virt 19365), a (.virt 19367), a (.virt 19369)],
    ![a (.virt 19364), a (.virt 19366), a (.virt 19368), a (.virt 19370)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2747 a h)
    · exact (publicBatchWrapper89_f2748 a h)
    · exact (publicBatchWrapper89_f2749 a h)
    · exact (publicBatchWrapper89_f2750 a h)
  · show bor (row a 70).isDummy (band (band (a (.virt 19363)) (a (.virt 19365)))
      (band (a (.virt 19367)) (a (.virt 19369)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2751 a h), ← (publicBatchWrapper89_f2752 a h), ← (publicBatchWrapper89_f2753 a h), ← (publicBatchWrapper89_f2754 a h), (publicBatchWrapper89_f2755 a h), kone]


theorem countAt_70 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 70).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 14002)).val = 32
  rw [(publicBatchWrapper89_f70 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_71 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 71) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f586 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f587 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f588 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f589 a h); rw [kzero] at e; exact e


theorem assetAt_71 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 71).isDummy (row a 71).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 69 63) = (row a 71).isDummy := by rw [(publicBatchWrapper89_f592 a h), (publicBatchWrapper89_f590 a h), (publicBatchWrapper89_f591 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2756 a h), by rw [← hd, ← (publicBatchWrapper89_f2757 a h), (publicBatchWrapper89_f2758 a h), kone]⟩


theorem feeAt_71 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 71).isDummy (row a 71).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 69 63) = (row a 71).isDummy := by rw [(publicBatchWrapper89_f592 a h), (publicBatchWrapper89_f590 a h), (publicBatchWrapper89_f591 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2759 a h), by rw [← hd, ← (publicBatchWrapper89_f2760 a h), (publicBatchWrapper89_f2761 a h), kone]⟩


theorem blockAt_71 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 71).isDummy (row a 71).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 69 63) = (row a 71).isDummy := by rw [(publicBatchWrapper89_f592 a h), (publicBatchWrapper89_f590 a h), (publicBatchWrapper89_f591 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19375), a (.virt 19377), a (.virt 19379), a (.virt 19381)],
    ![a (.virt 19376), a (.virt 19378), a (.virt 19380), a (.virt 19382)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2762 a h)
    · exact (publicBatchWrapper89_f2763 a h)
    · exact (publicBatchWrapper89_f2764 a h)
    · exact (publicBatchWrapper89_f2765 a h)
  · show bor (row a 71).isDummy (band (band (a (.virt 19375)) (a (.virt 19377)))
      (band (a (.virt 19379)) (a (.virt 19381)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2766 a h), ← (publicBatchWrapper89_f2767 a h), ← (publicBatchWrapper89_f2768 a h), ← (publicBatchWrapper89_f2769 a h), (publicBatchWrapper89_f2770 a h), kone]


theorem countAt_71 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 71).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 14202)).val = 32
  rw [(publicBatchWrapper89_f71 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_72 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 72) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f593 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f594 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f595 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f596 a h); rw [kzero] at e; exact e


theorem assetAt_72 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 72).isDummy (row a 72).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 71 27) = (row a 72).isDummy := by rw [(publicBatchWrapper89_f599 a h), (publicBatchWrapper89_f597 a h), (publicBatchWrapper89_f598 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2771 a h), by rw [← hd, ← (publicBatchWrapper89_f2772 a h), (publicBatchWrapper89_f2773 a h), kone]⟩


theorem feeAt_72 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 72).isDummy (row a 72).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 71 27) = (row a 72).isDummy := by rw [(publicBatchWrapper89_f599 a h), (publicBatchWrapper89_f597 a h), (publicBatchWrapper89_f598 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2774 a h), by rw [← hd, ← (publicBatchWrapper89_f2775 a h), (publicBatchWrapper89_f2776 a h), kone]⟩


theorem blockAt_72 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 72).isDummy (row a 72).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 71 27) = (row a 72).isDummy := by rw [(publicBatchWrapper89_f599 a h), (publicBatchWrapper89_f597 a h), (publicBatchWrapper89_f598 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19387), a (.virt 19389), a (.virt 19391), a (.virt 19393)],
    ![a (.virt 19388), a (.virt 19390), a (.virt 19392), a (.virt 19394)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2777 a h)
    · exact (publicBatchWrapper89_f2778 a h)
    · exact (publicBatchWrapper89_f2779 a h)
    · exact (publicBatchWrapper89_f2780 a h)
  · show bor (row a 72).isDummy (band (band (a (.virt 19387)) (a (.virt 19389)))
      (band (a (.virt 19391)) (a (.virt 19393)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2781 a h), ← (publicBatchWrapper89_f2782 a h), ← (publicBatchWrapper89_f2783 a h), ← (publicBatchWrapper89_f2784 a h), (publicBatchWrapper89_f2785 a h), kone]


theorem countAt_72 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 72).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 14402)).val = 32
  rw [(publicBatchWrapper89_f72 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_73 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 73) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f600 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f601 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f602 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f603 a h); rw [kzero] at e; exact e


theorem assetAt_73 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 73).isDummy (row a 73).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 71 71) = (row a 73).isDummy := by rw [(publicBatchWrapper89_f606 a h), (publicBatchWrapper89_f604 a h), (publicBatchWrapper89_f605 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2786 a h), by rw [← hd, ← (publicBatchWrapper89_f2787 a h), (publicBatchWrapper89_f2788 a h), kone]⟩


theorem feeAt_73 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 73).isDummy (row a 73).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 71 71) = (row a 73).isDummy := by rw [(publicBatchWrapper89_f606 a h), (publicBatchWrapper89_f604 a h), (publicBatchWrapper89_f605 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2789 a h), by rw [← hd, ← (publicBatchWrapper89_f2790 a h), (publicBatchWrapper89_f2791 a h), kone]⟩


theorem blockAt_73 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 73).isDummy (row a 73).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 71 71) = (row a 73).isDummy := by rw [(publicBatchWrapper89_f606 a h), (publicBatchWrapper89_f604 a h), (publicBatchWrapper89_f605 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19399), a (.virt 19401), a (.virt 19403), a (.virt 19405)],
    ![a (.virt 19400), a (.virt 19402), a (.virt 19404), a (.virt 19406)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2792 a h)
    · exact (publicBatchWrapper89_f2793 a h)
    · exact (publicBatchWrapper89_f2794 a h)
    · exact (publicBatchWrapper89_f2795 a h)
  · show bor (row a 73).isDummy (band (band (a (.virt 19399)) (a (.virt 19401)))
      (band (a (.virt 19403)) (a (.virt 19405)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2796 a h), ← (publicBatchWrapper89_f2797 a h), ← (publicBatchWrapper89_f2798 a h), ← (publicBatchWrapper89_f2799 a h), (publicBatchWrapper89_f2800 a h), kone]


theorem countAt_73 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 73).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 14602)).val = 32
  rw [(publicBatchWrapper89_f73 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_74 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 74) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f607 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f608 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f609 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f610 a h); rw [kzero] at e; exact e


theorem assetAt_74 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 74).isDummy (row a 74).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 72 35) = (row a 74).isDummy := by rw [(publicBatchWrapper89_f613 a h), (publicBatchWrapper89_f611 a h), (publicBatchWrapper89_f612 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2801 a h), by rw [← hd, ← (publicBatchWrapper89_f2802 a h), (publicBatchWrapper89_f2803 a h), kone]⟩


theorem feeAt_74 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 74).isDummy (row a 74).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 72 35) = (row a 74).isDummy := by rw [(publicBatchWrapper89_f613 a h), (publicBatchWrapper89_f611 a h), (publicBatchWrapper89_f612 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2804 a h), by rw [← hd, ← (publicBatchWrapper89_f2805 a h), (publicBatchWrapper89_f2806 a h), kone]⟩


theorem blockAt_74 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 74).isDummy (row a 74).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 72 35) = (row a 74).isDummy := by rw [(publicBatchWrapper89_f613 a h), (publicBatchWrapper89_f611 a h), (publicBatchWrapper89_f612 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19411), a (.virt 19413), a (.virt 19415), a (.virt 19417)],
    ![a (.virt 19412), a (.virt 19414), a (.virt 19416), a (.virt 19418)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2807 a h)
    · exact (publicBatchWrapper89_f2808 a h)
    · exact (publicBatchWrapper89_f2809 a h)
    · exact (publicBatchWrapper89_f2810 a h)
  · show bor (row a 74).isDummy (band (band (a (.virt 19411)) (a (.virt 19413)))
      (band (a (.virt 19415)) (a (.virt 19417)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2811 a h), ← (publicBatchWrapper89_f2812 a h), ← (publicBatchWrapper89_f2813 a h), ← (publicBatchWrapper89_f2814 a h), (publicBatchWrapper89_f2815 a h), kone]


theorem countAt_74 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 74).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 14802)).val = 32
  rw [(publicBatchWrapper89_f74 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_75 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 75) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f614 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f615 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f616 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f617 a h); rw [kzero] at e; exact e


theorem assetAt_75 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 75).isDummy (row a 75).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 72 79) = (row a 75).isDummy := by rw [(publicBatchWrapper89_f620 a h), (publicBatchWrapper89_f618 a h), (publicBatchWrapper89_f619 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2816 a h), by rw [← hd, ← (publicBatchWrapper89_f2817 a h), (publicBatchWrapper89_f2818 a h), kone]⟩


theorem feeAt_75 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 75).isDummy (row a 75).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 72 79) = (row a 75).isDummy := by rw [(publicBatchWrapper89_f620 a h), (publicBatchWrapper89_f618 a h), (publicBatchWrapper89_f619 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2819 a h), by rw [← hd, ← (publicBatchWrapper89_f2820 a h), (publicBatchWrapper89_f2821 a h), kone]⟩


theorem blockAt_75 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 75).isDummy (row a 75).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 72 79) = (row a 75).isDummy := by rw [(publicBatchWrapper89_f620 a h), (publicBatchWrapper89_f618 a h), (publicBatchWrapper89_f619 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19423), a (.virt 19425), a (.virt 19427), a (.virt 19429)],
    ![a (.virt 19424), a (.virt 19426), a (.virt 19428), a (.virt 19430)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2822 a h)
    · exact (publicBatchWrapper89_f2823 a h)
    · exact (publicBatchWrapper89_f2824 a h)
    · exact (publicBatchWrapper89_f2825 a h)
  · show bor (row a 75).isDummy (band (band (a (.virt 19423)) (a (.virt 19425)))
      (band (a (.virt 19427)) (a (.virt 19429)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2826 a h), ← (publicBatchWrapper89_f2827 a h), ← (publicBatchWrapper89_f2828 a h), ← (publicBatchWrapper89_f2829 a h), (publicBatchWrapper89_f2830 a h), kone]


theorem countAt_75 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 75).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 15002)).val = 32
  rw [(publicBatchWrapper89_f75 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_76 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 76) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f621 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f622 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f623 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f624 a h); rw [kzero] at e; exact e


theorem assetAt_76 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 76).isDummy (row a 76).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 74 43) = (row a 76).isDummy := by rw [(publicBatchWrapper89_f627 a h), (publicBatchWrapper89_f625 a h), (publicBatchWrapper89_f626 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2831 a h), by rw [← hd, ← (publicBatchWrapper89_f2832 a h), (publicBatchWrapper89_f2833 a h), kone]⟩


theorem feeAt_76 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 76).isDummy (row a 76).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 74 43) = (row a 76).isDummy := by rw [(publicBatchWrapper89_f627 a h), (publicBatchWrapper89_f625 a h), (publicBatchWrapper89_f626 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2834 a h), by rw [← hd, ← (publicBatchWrapper89_f2835 a h), (publicBatchWrapper89_f2836 a h), kone]⟩


theorem blockAt_76 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 76).isDummy (row a 76).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 74 43) = (row a 76).isDummy := by rw [(publicBatchWrapper89_f627 a h), (publicBatchWrapper89_f625 a h), (publicBatchWrapper89_f626 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19435), a (.virt 19437), a (.virt 19439), a (.virt 19441)],
    ![a (.virt 19436), a (.virt 19438), a (.virt 19440), a (.virt 19442)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2837 a h)
    · exact (publicBatchWrapper89_f2838 a h)
    · exact (publicBatchWrapper89_f2839 a h)
    · exact (publicBatchWrapper89_f2840 a h)
  · show bor (row a 76).isDummy (band (band (a (.virt 19435)) (a (.virt 19437)))
      (band (a (.virt 19439)) (a (.virt 19441)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2841 a h), ← (publicBatchWrapper89_f2842 a h), ← (publicBatchWrapper89_f2843 a h), ← (publicBatchWrapper89_f2844 a h), (publicBatchWrapper89_f2845 a h), kone]


theorem countAt_76 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 76).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 15202)).val = 32
  rw [(publicBatchWrapper89_f76 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_77 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 77) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f628 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f629 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f630 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f631 a h); rw [kzero] at e; exact e


theorem assetAt_77 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 77).isDummy (row a 77).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 76 7) = (row a 77).isDummy := by rw [(publicBatchWrapper89_f634 a h), (publicBatchWrapper89_f632 a h), (publicBatchWrapper89_f633 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2846 a h), by rw [← hd, ← (publicBatchWrapper89_f2847 a h), (publicBatchWrapper89_f2848 a h), kone]⟩


theorem feeAt_77 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 77).isDummy (row a 77).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 76 7) = (row a 77).isDummy := by rw [(publicBatchWrapper89_f634 a h), (publicBatchWrapper89_f632 a h), (publicBatchWrapper89_f633 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2849 a h), by rw [← hd, ← (publicBatchWrapper89_f2850 a h), (publicBatchWrapper89_f2851 a h), kone]⟩


theorem blockAt_77 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 77).isDummy (row a 77).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 76 7) = (row a 77).isDummy := by rw [(publicBatchWrapper89_f634 a h), (publicBatchWrapper89_f632 a h), (publicBatchWrapper89_f633 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19447), a (.virt 19449), a (.virt 19451), a (.virt 19453)],
    ![a (.virt 19448), a (.virt 19450), a (.virt 19452), a (.virt 19454)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2852 a h)
    · exact (publicBatchWrapper89_f2853 a h)
    · exact (publicBatchWrapper89_f2854 a h)
    · exact (publicBatchWrapper89_f2855 a h)
  · show bor (row a 77).isDummy (band (band (a (.virt 19447)) (a (.virt 19449)))
      (band (a (.virt 19451)) (a (.virt 19453)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2856 a h), ← (publicBatchWrapper89_f2857 a h), ← (publicBatchWrapper89_f2858 a h), ← (publicBatchWrapper89_f2859 a h), (publicBatchWrapper89_f2860 a h), kone]


theorem countAt_77 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 77).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 15402)).val = 32
  rw [(publicBatchWrapper89_f77 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_78 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 78) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f635 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f636 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f637 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f638 a h); rw [kzero] at e; exact e


theorem assetAt_78 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 78).isDummy (row a 78).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 76 51) = (row a 78).isDummy := by rw [(publicBatchWrapper89_f641 a h), (publicBatchWrapper89_f639 a h), (publicBatchWrapper89_f640 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2861 a h), by rw [← hd, ← (publicBatchWrapper89_f2862 a h), (publicBatchWrapper89_f2863 a h), kone]⟩


theorem feeAt_78 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 78).isDummy (row a 78).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 76 51) = (row a 78).isDummy := by rw [(publicBatchWrapper89_f641 a h), (publicBatchWrapper89_f639 a h), (publicBatchWrapper89_f640 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2864 a h), by rw [← hd, ← (publicBatchWrapper89_f2865 a h), (publicBatchWrapper89_f2866 a h), kone]⟩


theorem blockAt_78 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 78).isDummy (row a 78).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 76 51) = (row a 78).isDummy := by rw [(publicBatchWrapper89_f641 a h), (publicBatchWrapper89_f639 a h), (publicBatchWrapper89_f640 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19459), a (.virt 19461), a (.virt 19463), a (.virt 19465)],
    ![a (.virt 19460), a (.virt 19462), a (.virt 19464), a (.virt 19466)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2867 a h)
    · exact (publicBatchWrapper89_f2868 a h)
    · exact (publicBatchWrapper89_f2869 a h)
    · exact (publicBatchWrapper89_f2870 a h)
  · show bor (row a 78).isDummy (band (band (a (.virt 19459)) (a (.virt 19461)))
      (band (a (.virt 19463)) (a (.virt 19465)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2871 a h), ← (publicBatchWrapper89_f2872 a h), ← (publicBatchWrapper89_f2873 a h), ← (publicBatchWrapper89_f2874 a h), (publicBatchWrapper89_f2875 a h), kone]


theorem countAt_78 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 78).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 15602)).val = 32
  rw [(publicBatchWrapper89_f78 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_79 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 79) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f642 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f643 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f644 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f645 a h); rw [kzero] at e; exact e


theorem assetAt_79 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 79).isDummy (row a 79).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 77 15) = (row a 79).isDummy := by rw [(publicBatchWrapper89_f648 a h), (publicBatchWrapper89_f646 a h), (publicBatchWrapper89_f647 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2876 a h), by rw [← hd, ← (publicBatchWrapper89_f2877 a h), (publicBatchWrapper89_f2878 a h), kone]⟩


theorem feeAt_79 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 79).isDummy (row a 79).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 77 15) = (row a 79).isDummy := by rw [(publicBatchWrapper89_f648 a h), (publicBatchWrapper89_f646 a h), (publicBatchWrapper89_f647 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2879 a h), by rw [← hd, ← (publicBatchWrapper89_f2880 a h), (publicBatchWrapper89_f2881 a h), kone]⟩


theorem blockAt_79 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 79).isDummy (row a 79).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 77 15) = (row a 79).isDummy := by rw [(publicBatchWrapper89_f648 a h), (publicBatchWrapper89_f646 a h), (publicBatchWrapper89_f647 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19471), a (.virt 19473), a (.virt 19475), a (.virt 19477)],
    ![a (.virt 19472), a (.virt 19474), a (.virt 19476), a (.virt 19478)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2882 a h)
    · exact (publicBatchWrapper89_f2883 a h)
    · exact (publicBatchWrapper89_f2884 a h)
    · exact (publicBatchWrapper89_f2885 a h)
  · show bor (row a 79).isDummy (band (band (a (.virt 19471)) (a (.virt 19473)))
      (band (a (.virt 19475)) (a (.virt 19477)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2886 a h), ← (publicBatchWrapper89_f2887 a h), ← (publicBatchWrapper89_f2888 a h), ← (publicBatchWrapper89_f2889 a h), (publicBatchWrapper89_f2890 a h), kone]


theorem countAt_79 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 79).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 15802)).val = 32
  rw [(publicBatchWrapper89_f79 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_80 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 80) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f649 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f650 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f651 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f652 a h); rw [kzero] at e; exact e


theorem assetAt_80 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 80).isDummy (row a 80).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 77 59) = (row a 80).isDummy := by rw [(publicBatchWrapper89_f655 a h), (publicBatchWrapper89_f653 a h), (publicBatchWrapper89_f654 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2891 a h), by rw [← hd, ← (publicBatchWrapper89_f2892 a h), (publicBatchWrapper89_f2893 a h), kone]⟩


theorem feeAt_80 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 80).isDummy (row a 80).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 77 59) = (row a 80).isDummy := by rw [(publicBatchWrapper89_f655 a h), (publicBatchWrapper89_f653 a h), (publicBatchWrapper89_f654 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2894 a h), by rw [← hd, ← (publicBatchWrapper89_f2895 a h), (publicBatchWrapper89_f2896 a h), kone]⟩


theorem blockAt_80 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 80).isDummy (row a 80).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 77 59) = (row a 80).isDummy := by rw [(publicBatchWrapper89_f655 a h), (publicBatchWrapper89_f653 a h), (publicBatchWrapper89_f654 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19483), a (.virt 19485), a (.virt 19487), a (.virt 19489)],
    ![a (.virt 19484), a (.virt 19486), a (.virt 19488), a (.virt 19490)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2897 a h)
    · exact (publicBatchWrapper89_f2898 a h)
    · exact (publicBatchWrapper89_f2899 a h)
    · exact (publicBatchWrapper89_f2900 a h)
  · show bor (row a 80).isDummy (band (band (a (.virt 19483)) (a (.virt 19485)))
      (band (a (.virt 19487)) (a (.virt 19489)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2901 a h), ← (publicBatchWrapper89_f2902 a h), ← (publicBatchWrapper89_f2903 a h), ← (publicBatchWrapper89_f2904 a h), (publicBatchWrapper89_f2905 a h), kone]


theorem countAt_80 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 80).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 16002)).val = 32
  rw [(publicBatchWrapper89_f80 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_81 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 81) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f656 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f657 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f658 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f659 a h); rw [kzero] at e; exact e


theorem assetAt_81 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 81).isDummy (row a 81).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 79 23) = (row a 81).isDummy := by rw [(publicBatchWrapper89_f662 a h), (publicBatchWrapper89_f660 a h), (publicBatchWrapper89_f661 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2906 a h), by rw [← hd, ← (publicBatchWrapper89_f2907 a h), (publicBatchWrapper89_f2908 a h), kone]⟩


theorem feeAt_81 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 81).isDummy (row a 81).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 79 23) = (row a 81).isDummy := by rw [(publicBatchWrapper89_f662 a h), (publicBatchWrapper89_f660 a h), (publicBatchWrapper89_f661 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2909 a h), by rw [← hd, ← (publicBatchWrapper89_f2910 a h), (publicBatchWrapper89_f2911 a h), kone]⟩


theorem blockAt_81 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 81).isDummy (row a 81).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 79 23) = (row a 81).isDummy := by rw [(publicBatchWrapper89_f662 a h), (publicBatchWrapper89_f660 a h), (publicBatchWrapper89_f661 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19495), a (.virt 19497), a (.virt 19499), a (.virt 19501)],
    ![a (.virt 19496), a (.virt 19498), a (.virt 19500), a (.virt 19502)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2912 a h)
    · exact (publicBatchWrapper89_f2913 a h)
    · exact (publicBatchWrapper89_f2914 a h)
    · exact (publicBatchWrapper89_f2915 a h)
  · show bor (row a 81).isDummy (band (band (a (.virt 19495)) (a (.virt 19497)))
      (band (a (.virt 19499)) (a (.virt 19501)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2916 a h), ← (publicBatchWrapper89_f2917 a h), ← (publicBatchWrapper89_f2918 a h), ← (publicBatchWrapper89_f2919 a h), (publicBatchWrapper89_f2920 a h), kone]


theorem countAt_81 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 81).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 16202)).val = 32
  rw [(publicBatchWrapper89_f81 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_82 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 82) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f663 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f664 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f665 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f666 a h); rw [kzero] at e; exact e


theorem assetAt_82 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 82).isDummy (row a 82).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 79 67) = (row a 82).isDummy := by rw [(publicBatchWrapper89_f669 a h), (publicBatchWrapper89_f667 a h), (publicBatchWrapper89_f668 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2921 a h), by rw [← hd, ← (publicBatchWrapper89_f2922 a h), (publicBatchWrapper89_f2923 a h), kone]⟩


theorem feeAt_82 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 82).isDummy (row a 82).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 79 67) = (row a 82).isDummy := by rw [(publicBatchWrapper89_f669 a h), (publicBatchWrapper89_f667 a h), (publicBatchWrapper89_f668 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2924 a h), by rw [← hd, ← (publicBatchWrapper89_f2925 a h), (publicBatchWrapper89_f2926 a h), kone]⟩


theorem blockAt_82 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 82).isDummy (row a 82).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 79 67) = (row a 82).isDummy := by rw [(publicBatchWrapper89_f669 a h), (publicBatchWrapper89_f667 a h), (publicBatchWrapper89_f668 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19507), a (.virt 19509), a (.virt 19511), a (.virt 19513)],
    ![a (.virt 19508), a (.virt 19510), a (.virt 19512), a (.virt 19514)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2927 a h)
    · exact (publicBatchWrapper89_f2928 a h)
    · exact (publicBatchWrapper89_f2929 a h)
    · exact (publicBatchWrapper89_f2930 a h)
  · show bor (row a 82).isDummy (band (band (a (.virt 19507)) (a (.virt 19509)))
      (band (a (.virt 19511)) (a (.virt 19513)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2931 a h), ← (publicBatchWrapper89_f2932 a h), ← (publicBatchWrapper89_f2933 a h), ← (publicBatchWrapper89_f2934 a h), (publicBatchWrapper89_f2935 a h), kone]


theorem countAt_82 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 82).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 16402)).val = 32
  rw [(publicBatchWrapper89_f82 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_83 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 83) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f670 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f671 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f672 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f673 a h); rw [kzero] at e; exact e


theorem assetAt_83 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 83).isDummy (row a 83).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 81 31) = (row a 83).isDummy := by rw [(publicBatchWrapper89_f676 a h), (publicBatchWrapper89_f674 a h), (publicBatchWrapper89_f675 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2936 a h), by rw [← hd, ← (publicBatchWrapper89_f2937 a h), (publicBatchWrapper89_f2938 a h), kone]⟩


theorem feeAt_83 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 83).isDummy (row a 83).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 81 31) = (row a 83).isDummy := by rw [(publicBatchWrapper89_f676 a h), (publicBatchWrapper89_f674 a h), (publicBatchWrapper89_f675 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2939 a h), by rw [← hd, ← (publicBatchWrapper89_f2940 a h), (publicBatchWrapper89_f2941 a h), kone]⟩


theorem blockAt_83 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 83).isDummy (row a 83).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 81 31) = (row a 83).isDummy := by rw [(publicBatchWrapper89_f676 a h), (publicBatchWrapper89_f674 a h), (publicBatchWrapper89_f675 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19519), a (.virt 19521), a (.virt 19523), a (.virt 19525)],
    ![a (.virt 19520), a (.virt 19522), a (.virt 19524), a (.virt 19526)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2942 a h)
    · exact (publicBatchWrapper89_f2943 a h)
    · exact (publicBatchWrapper89_f2944 a h)
    · exact (publicBatchWrapper89_f2945 a h)
  · show bor (row a 83).isDummy (band (band (a (.virt 19519)) (a (.virt 19521)))
      (band (a (.virt 19523)) (a (.virt 19525)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2946 a h), ← (publicBatchWrapper89_f2947 a h), ← (publicBatchWrapper89_f2948 a h), ← (publicBatchWrapper89_f2949 a h), (publicBatchWrapper89_f2950 a h), kone]


theorem countAt_83 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 83).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 16602)).val = 32
  rw [(publicBatchWrapper89_f83 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_84 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 84) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f677 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f678 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f679 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f680 a h); rw [kzero] at e; exact e


theorem assetAt_84 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 84).isDummy (row a 84).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 81 75) = (row a 84).isDummy := by rw [(publicBatchWrapper89_f683 a h), (publicBatchWrapper89_f681 a h), (publicBatchWrapper89_f682 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2951 a h), by rw [← hd, ← (publicBatchWrapper89_f2952 a h), (publicBatchWrapper89_f2953 a h), kone]⟩


theorem feeAt_84 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 84).isDummy (row a 84).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 81 75) = (row a 84).isDummy := by rw [(publicBatchWrapper89_f683 a h), (publicBatchWrapper89_f681 a h), (publicBatchWrapper89_f682 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2954 a h), by rw [← hd, ← (publicBatchWrapper89_f2955 a h), (publicBatchWrapper89_f2956 a h), kone]⟩


theorem blockAt_84 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 84).isDummy (row a 84).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 81 75) = (row a 84).isDummy := by rw [(publicBatchWrapper89_f683 a h), (publicBatchWrapper89_f681 a h), (publicBatchWrapper89_f682 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19531), a (.virt 19533), a (.virt 19535), a (.virt 19537)],
    ![a (.virt 19532), a (.virt 19534), a (.virt 19536), a (.virt 19538)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2957 a h)
    · exact (publicBatchWrapper89_f2958 a h)
    · exact (publicBatchWrapper89_f2959 a h)
    · exact (publicBatchWrapper89_f2960 a h)
  · show bor (row a 84).isDummy (band (band (a (.virt 19531)) (a (.virt 19533)))
      (band (a (.virt 19535)) (a (.virt 19537)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2961 a h), ← (publicBatchWrapper89_f2962 a h), ← (publicBatchWrapper89_f2963 a h), ← (publicBatchWrapper89_f2964 a h), (publicBatchWrapper89_f2965 a h), kone]


theorem countAt_84 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 84).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 16802)).val = 32
  rw [(publicBatchWrapper89_f84 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_85 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 85) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f684 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f685 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f686 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f687 a h); rw [kzero] at e; exact e


theorem assetAt_85 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 85).isDummy (row a 85).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 83 39) = (row a 85).isDummy := by rw [(publicBatchWrapper89_f690 a h), (publicBatchWrapper89_f688 a h), (publicBatchWrapper89_f689 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2966 a h), by rw [← hd, ← (publicBatchWrapper89_f2967 a h), (publicBatchWrapper89_f2968 a h), kone]⟩


theorem feeAt_85 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 85).isDummy (row a 85).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 83 39) = (row a 85).isDummy := by rw [(publicBatchWrapper89_f690 a h), (publicBatchWrapper89_f688 a h), (publicBatchWrapper89_f689 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2969 a h), by rw [← hd, ← (publicBatchWrapper89_f2970 a h), (publicBatchWrapper89_f2971 a h), kone]⟩


theorem blockAt_85 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 85).isDummy (row a 85).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 83 39) = (row a 85).isDummy := by rw [(publicBatchWrapper89_f690 a h), (publicBatchWrapper89_f688 a h), (publicBatchWrapper89_f689 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19543), a (.virt 19545), a (.virt 19547), a (.virt 19549)],
    ![a (.virt 19544), a (.virt 19546), a (.virt 19548), a (.virt 19550)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2972 a h)
    · exact (publicBatchWrapper89_f2973 a h)
    · exact (publicBatchWrapper89_f2974 a h)
    · exact (publicBatchWrapper89_f2975 a h)
  · show bor (row a 85).isDummy (band (band (a (.virt 19543)) (a (.virt 19545)))
      (band (a (.virt 19547)) (a (.virt 19549)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2976 a h), ← (publicBatchWrapper89_f2977 a h), ← (publicBatchWrapper89_f2978 a h), ← (publicBatchWrapper89_f2979 a h), (publicBatchWrapper89_f2980 a h), kone]


theorem countAt_85 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 85).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 17002)).val = 32
  rw [(publicBatchWrapper89_f85 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_86 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 86) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f691 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f692 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f693 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f694 a h); rw [kzero] at e; exact e


theorem assetAt_86 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 86).isDummy (row a 86).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 84 3) = (row a 86).isDummy := by rw [(publicBatchWrapper89_f697 a h), (publicBatchWrapper89_f695 a h), (publicBatchWrapper89_f696 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2981 a h), by rw [← hd, ← (publicBatchWrapper89_f2982 a h), (publicBatchWrapper89_f2983 a h), kone]⟩


theorem feeAt_86 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 86).isDummy (row a 86).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 84 3) = (row a 86).isDummy := by rw [(publicBatchWrapper89_f697 a h), (publicBatchWrapper89_f695 a h), (publicBatchWrapper89_f696 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2984 a h), by rw [← hd, ← (publicBatchWrapper89_f2985 a h), (publicBatchWrapper89_f2986 a h), kone]⟩


theorem blockAt_86 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 86).isDummy (row a 86).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 84 3) = (row a 86).isDummy := by rw [(publicBatchWrapper89_f697 a h), (publicBatchWrapper89_f695 a h), (publicBatchWrapper89_f696 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19555), a (.virt 19557), a (.virt 19559), a (.virt 19561)],
    ![a (.virt 19556), a (.virt 19558), a (.virt 19560), a (.virt 19562)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f2987 a h)
    · exact (publicBatchWrapper89_f2988 a h)
    · exact (publicBatchWrapper89_f2989 a h)
    · exact (publicBatchWrapper89_f2990 a h)
  · show bor (row a 86).isDummy (band (band (a (.virt 19555)) (a (.virt 19557)))
      (band (a (.virt 19559)) (a (.virt 19561)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f2991 a h), ← (publicBatchWrapper89_f2992 a h), ← (publicBatchWrapper89_f2993 a h), ← (publicBatchWrapper89_f2994 a h), (publicBatchWrapper89_f2995 a h), kone]


theorem countAt_86 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 86).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 17202)).val = 32
  rw [(publicBatchWrapper89_f86 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_87 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 87) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f698 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f699 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f700 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f701 a h); rw [kzero] at e; exact e


theorem assetAt_87 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 87).isDummy (row a 87).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 84 47) = (row a 87).isDummy := by rw [(publicBatchWrapper89_f704 a h), (publicBatchWrapper89_f702 a h), (publicBatchWrapper89_f703 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f2996 a h), by rw [← hd, ← (publicBatchWrapper89_f2997 a h), (publicBatchWrapper89_f2998 a h), kone]⟩


theorem feeAt_87 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 87).isDummy (row a 87).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 84 47) = (row a 87).isDummy := by rw [(publicBatchWrapper89_f704 a h), (publicBatchWrapper89_f702 a h), (publicBatchWrapper89_f703 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f2999 a h), by rw [← hd, ← (publicBatchWrapper89_f3000 a h), (publicBatchWrapper89_f3001 a h), kone]⟩


theorem blockAt_87 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 87).isDummy (row a 87).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 84 47) = (row a 87).isDummy := by rw [(publicBatchWrapper89_f704 a h), (publicBatchWrapper89_f702 a h), (publicBatchWrapper89_f703 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19567), a (.virt 19569), a (.virt 19571), a (.virt 19573)],
    ![a (.virt 19568), a (.virt 19570), a (.virt 19572), a (.virt 19574)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f3002 a h)
    · exact (publicBatchWrapper89_f3003 a h)
    · exact (publicBatchWrapper89_f3004 a h)
    · exact (publicBatchWrapper89_f3005 a h)
  · show bor (row a 87).isDummy (band (band (a (.virt 19567)) (a (.virt 19569)))
      (band (a (.virt 19571)) (a (.virt 19573)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f3006 a h), ← (publicBatchWrapper89_f3007 a h), ← (publicBatchWrapper89_f3008 a h), ← (publicBatchWrapper89_f3009 a h), (publicBatchWrapper89_f3010 a h), kone]


theorem countAt_87 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 87).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 17402)).val = 32
  rw [(publicBatchWrapper89_f87 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_88 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DummyCheck (row a 88) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper89_f705 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f706 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f707 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper89_f708 a h); rw [kzero] at e; exact e


theorem assetAt_88 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 88).isDummy (row a 88).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 86 11) = (row a 88).isDummy := by rw [(publicBatchWrapper89_f711 a h), (publicBatchWrapper89_f709 a h), (publicBatchWrapper89_f710 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper89_f3011 a h), by rw [← hd, ← (publicBatchWrapper89_f3012 a h), (publicBatchWrapper89_f3013 a h), kone]⟩


theorem feeAt_88 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    ConsistencyCheck (row a 88).isDummy (row a 88).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 86 11) = (row a 88).isDummy := by rw [(publicBatchWrapper89_f711 a h), (publicBatchWrapper89_f709 a h), (publicBatchWrapper89_f710 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper89_f3014 a h), by rw [← hd, ← (publicBatchWrapper89_f3015 a h), (publicBatchWrapper89_f3016 a h), kone]⟩


theorem blockAt_88 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    DigestConsistencyCheck (row a 88).isDummy (row a 88).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 86 11) = (row a 88).isDummy := by rw [(publicBatchWrapper89_f711 a h), (publicBatchWrapper89_f709 a h), (publicBatchWrapper89_f710 a h)]; rfl
  have hblk : ∀ j, a (![.wire 169 63, .wire 169 71, .wire 169 79, .wire 170 7] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 19579), a (.virt 19581), a (.virt 19583), a (.virt 19585)],
    ![a (.virt 19580), a (.virt 19582), a (.virt 19584), a (.virt 19586)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper89_f3017 a h)
    · exact (publicBatchWrapper89_f3018 a h)
    · exact (publicBatchWrapper89_f3019 a h)
    · exact (publicBatchWrapper89_f3020 a h)
  · show bor (row a 88).isDummy (band (band (a (.virt 19579)) (a (.virt 19581)))
      (band (a (.virt 19583)) (a (.virt 19585)))) = 1
    rw [← hd, ← (publicBatchWrapper89_f3021 a h), ← (publicBatchWrapper89_f3022 a h), ← (publicBatchWrapper89_f3023 a h), ← (publicBatchWrapper89_f3024 a h), (publicBatchWrapper89_f3025 a h), kone]


theorem countAt_88 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 88).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 17602)).val = 32
  rw [(publicBatchWrapper89_f88 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]

end Plonky2Bridge.PublicWrapper89
