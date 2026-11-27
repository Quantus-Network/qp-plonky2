import Plonky2Bridge.Generated.PublicWrapper86.Limb0
import Plonky2Bridge.Generated.PublicWrapper86.Limb1
import Plonky2Bridge.Generated.PublicWrapper86.Limb2
import Plonky2Bridge.Generated.PublicWrapper86.Limb3
import Plonky2Bridge.Generated.PublicWrapper86.Scalar

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

theorem dummyAt_0 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 0) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f86 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f87 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f88 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f89 a h); rw [kzero] at e; exact e


theorem assetAt_0 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 0).isDummy (row a 0).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 3 27) = (row a 0).isDummy := by rw [(publicBatchWrapper86_f92 a h), (publicBatchWrapper86_f90 a h), (publicBatchWrapper86_f91 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1634 a h), by rw [← hd, ← (publicBatchWrapper86_f1635 a h), (publicBatchWrapper86_f1636 a h), kone]⟩


theorem feeAt_0 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 0).isDummy (row a 0).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 3 27) = (row a 0).isDummy := by rw [(publicBatchWrapper86_f92 a h), (publicBatchWrapper86_f90 a h), (publicBatchWrapper86_f91 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1637 a h), by rw [← hd, ← (publicBatchWrapper86_f1638 a h), (publicBatchWrapper86_f1639 a h), kone]⟩


theorem blockAt_0 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 0).isDummy (row a 0).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 3 27) = (row a 0).isDummy := by rw [(publicBatchWrapper86_f92 a h), (publicBatchWrapper86_f90 a h), (publicBatchWrapper86_f91 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 17899), a (.virt 17901), a (.virt 17903), a (.virt 17905)],
    ![a (.virt 17900), a (.virt 17902), a (.virt 17904), a (.virt 17906)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1640 a h)
    · exact (publicBatchWrapper86_f1641 a h)
    · exact (publicBatchWrapper86_f1642 a h)
    · exact (publicBatchWrapper86_f1643 a h)
  · show bor (row a 0).isDummy (band (band (a (.virt 17899)) (a (.virt 17901)))
      (band (a (.virt 17903)) (a (.virt 17905)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1644 a h), ← (publicBatchWrapper86_f1645 a h), ← (publicBatchWrapper86_f1646 a h), ← (publicBatchWrapper86_f1647 a h), (publicBatchWrapper86_f1648 a h), kone]


theorem countAt_0 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 0).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 0)).val = 32
  rw [(publicBatchWrapper86_f0 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_1 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 1) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f93 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f94 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f95 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f96 a h); rw [kzero] at e; exact e


theorem assetAt_1 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 1).isDummy (row a 1).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 3 71) = (row a 1).isDummy := by rw [(publicBatchWrapper86_f99 a h), (publicBatchWrapper86_f97 a h), (publicBatchWrapper86_f98 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1649 a h), by rw [← hd, ← (publicBatchWrapper86_f1650 a h), (publicBatchWrapper86_f1651 a h), kone]⟩


theorem feeAt_1 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 1).isDummy (row a 1).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 3 71) = (row a 1).isDummy := by rw [(publicBatchWrapper86_f99 a h), (publicBatchWrapper86_f97 a h), (publicBatchWrapper86_f98 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1652 a h), by rw [← hd, ← (publicBatchWrapper86_f1653 a h), (publicBatchWrapper86_f1654 a h), kone]⟩


theorem blockAt_1 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 1).isDummy (row a 1).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 3 71) = (row a 1).isDummy := by rw [(publicBatchWrapper86_f99 a h), (publicBatchWrapper86_f97 a h), (publicBatchWrapper86_f98 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 17911), a (.virt 17913), a (.virt 17915), a (.virt 17917)],
    ![a (.virt 17912), a (.virt 17914), a (.virt 17916), a (.virt 17918)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1655 a h)
    · exact (publicBatchWrapper86_f1656 a h)
    · exact (publicBatchWrapper86_f1657 a h)
    · exact (publicBatchWrapper86_f1658 a h)
  · show bor (row a 1).isDummy (band (band (a (.virt 17911)) (a (.virt 17913)))
      (band (a (.virt 17915)) (a (.virt 17917)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1659 a h), ← (publicBatchWrapper86_f1660 a h), ← (publicBatchWrapper86_f1661 a h), ← (publicBatchWrapper86_f1662 a h), (publicBatchWrapper86_f1663 a h), kone]


theorem countAt_1 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 1).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 202)).val = 32
  rw [(publicBatchWrapper86_f1 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_2 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 2) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f100 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f101 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f102 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f103 a h); rw [kzero] at e; exact e


theorem assetAt_2 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 2).isDummy (row a 2).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 4 35) = (row a 2).isDummy := by rw [(publicBatchWrapper86_f106 a h), (publicBatchWrapper86_f104 a h), (publicBatchWrapper86_f105 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1664 a h), by rw [← hd, ← (publicBatchWrapper86_f1665 a h), (publicBatchWrapper86_f1666 a h), kone]⟩


theorem feeAt_2 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 2).isDummy (row a 2).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 4 35) = (row a 2).isDummy := by rw [(publicBatchWrapper86_f106 a h), (publicBatchWrapper86_f104 a h), (publicBatchWrapper86_f105 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1667 a h), by rw [← hd, ← (publicBatchWrapper86_f1668 a h), (publicBatchWrapper86_f1669 a h), kone]⟩


theorem blockAt_2 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 2).isDummy (row a 2).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 4 35) = (row a 2).isDummy := by rw [(publicBatchWrapper86_f106 a h), (publicBatchWrapper86_f104 a h), (publicBatchWrapper86_f105 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 17923), a (.virt 17925), a (.virt 17927), a (.virt 17929)],
    ![a (.virt 17924), a (.virt 17926), a (.virt 17928), a (.virt 17930)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1670 a h)
    · exact (publicBatchWrapper86_f1671 a h)
    · exact (publicBatchWrapper86_f1672 a h)
    · exact (publicBatchWrapper86_f1673 a h)
  · show bor (row a 2).isDummy (band (band (a (.virt 17923)) (a (.virt 17925)))
      (band (a (.virt 17927)) (a (.virt 17929)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1674 a h), ← (publicBatchWrapper86_f1675 a h), ← (publicBatchWrapper86_f1676 a h), ← (publicBatchWrapper86_f1677 a h), (publicBatchWrapper86_f1678 a h), kone]


theorem countAt_2 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 2).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 402)).val = 32
  rw [(publicBatchWrapper86_f2 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_3 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 3) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f107 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f108 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f109 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f110 a h); rw [kzero] at e; exact e


theorem assetAt_3 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 3).isDummy (row a 3).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 4 79) = (row a 3).isDummy := by rw [(publicBatchWrapper86_f113 a h), (publicBatchWrapper86_f111 a h), (publicBatchWrapper86_f112 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1679 a h), by rw [← hd, ← (publicBatchWrapper86_f1680 a h), (publicBatchWrapper86_f1681 a h), kone]⟩


theorem feeAt_3 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 3).isDummy (row a 3).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 4 79) = (row a 3).isDummy := by rw [(publicBatchWrapper86_f113 a h), (publicBatchWrapper86_f111 a h), (publicBatchWrapper86_f112 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1682 a h), by rw [← hd, ← (publicBatchWrapper86_f1683 a h), (publicBatchWrapper86_f1684 a h), kone]⟩


theorem blockAt_3 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 3).isDummy (row a 3).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 4 79) = (row a 3).isDummy := by rw [(publicBatchWrapper86_f113 a h), (publicBatchWrapper86_f111 a h), (publicBatchWrapper86_f112 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 17935), a (.virt 17937), a (.virt 17939), a (.virt 17941)],
    ![a (.virt 17936), a (.virt 17938), a (.virt 17940), a (.virt 17942)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1685 a h)
    · exact (publicBatchWrapper86_f1686 a h)
    · exact (publicBatchWrapper86_f1687 a h)
    · exact (publicBatchWrapper86_f1688 a h)
  · show bor (row a 3).isDummy (band (band (a (.virt 17935)) (a (.virt 17937)))
      (band (a (.virt 17939)) (a (.virt 17941)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1689 a h), ← (publicBatchWrapper86_f1690 a h), ← (publicBatchWrapper86_f1691 a h), ← (publicBatchWrapper86_f1692 a h), (publicBatchWrapper86_f1693 a h), kone]


theorem countAt_3 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 3).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 602)).val = 32
  rw [(publicBatchWrapper86_f3 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_4 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 4) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f114 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f115 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f116 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f117 a h); rw [kzero] at e; exact e


theorem assetAt_4 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 4).isDummy (row a 4).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 6 43) = (row a 4).isDummy := by rw [(publicBatchWrapper86_f120 a h), (publicBatchWrapper86_f118 a h), (publicBatchWrapper86_f119 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1694 a h), by rw [← hd, ← (publicBatchWrapper86_f1695 a h), (publicBatchWrapper86_f1696 a h), kone]⟩


theorem feeAt_4 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 4).isDummy (row a 4).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 6 43) = (row a 4).isDummy := by rw [(publicBatchWrapper86_f120 a h), (publicBatchWrapper86_f118 a h), (publicBatchWrapper86_f119 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1697 a h), by rw [← hd, ← (publicBatchWrapper86_f1698 a h), (publicBatchWrapper86_f1699 a h), kone]⟩


theorem blockAt_4 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 4).isDummy (row a 4).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 6 43) = (row a 4).isDummy := by rw [(publicBatchWrapper86_f120 a h), (publicBatchWrapper86_f118 a h), (publicBatchWrapper86_f119 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 17947), a (.virt 17949), a (.virt 17951), a (.virt 17953)],
    ![a (.virt 17948), a (.virt 17950), a (.virt 17952), a (.virt 17954)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1700 a h)
    · exact (publicBatchWrapper86_f1701 a h)
    · exact (publicBatchWrapper86_f1702 a h)
    · exact (publicBatchWrapper86_f1703 a h)
  · show bor (row a 4).isDummy (band (band (a (.virt 17947)) (a (.virt 17949)))
      (band (a (.virt 17951)) (a (.virt 17953)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1704 a h), ← (publicBatchWrapper86_f1705 a h), ← (publicBatchWrapper86_f1706 a h), ← (publicBatchWrapper86_f1707 a h), (publicBatchWrapper86_f1708 a h), kone]


theorem countAt_4 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 4).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 802)).val = 32
  rw [(publicBatchWrapper86_f4 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_5 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 5) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f121 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f122 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f123 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f124 a h); rw [kzero] at e; exact e


theorem assetAt_5 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 5).isDummy (row a 5).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 8 7) = (row a 5).isDummy := by rw [(publicBatchWrapper86_f127 a h), (publicBatchWrapper86_f125 a h), (publicBatchWrapper86_f126 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1709 a h), by rw [← hd, ← (publicBatchWrapper86_f1710 a h), (publicBatchWrapper86_f1711 a h), kone]⟩


theorem feeAt_5 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 5).isDummy (row a 5).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 8 7) = (row a 5).isDummy := by rw [(publicBatchWrapper86_f127 a h), (publicBatchWrapper86_f125 a h), (publicBatchWrapper86_f126 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1712 a h), by rw [← hd, ← (publicBatchWrapper86_f1713 a h), (publicBatchWrapper86_f1714 a h), kone]⟩


theorem blockAt_5 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 5).isDummy (row a 5).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 8 7) = (row a 5).isDummy := by rw [(publicBatchWrapper86_f127 a h), (publicBatchWrapper86_f125 a h), (publicBatchWrapper86_f126 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 17959), a (.virt 17961), a (.virt 17963), a (.virt 17965)],
    ![a (.virt 17960), a (.virt 17962), a (.virt 17964), a (.virt 17966)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1715 a h)
    · exact (publicBatchWrapper86_f1716 a h)
    · exact (publicBatchWrapper86_f1717 a h)
    · exact (publicBatchWrapper86_f1718 a h)
  · show bor (row a 5).isDummy (band (band (a (.virt 17959)) (a (.virt 17961)))
      (band (a (.virt 17963)) (a (.virt 17965)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1719 a h), ← (publicBatchWrapper86_f1720 a h), ← (publicBatchWrapper86_f1721 a h), ← (publicBatchWrapper86_f1722 a h), (publicBatchWrapper86_f1723 a h), kone]


theorem countAt_5 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 5).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 1002)).val = 32
  rw [(publicBatchWrapper86_f5 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_6 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 6) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f128 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f129 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f130 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f131 a h); rw [kzero] at e; exact e


theorem assetAt_6 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 6).isDummy (row a 6).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 8 51) = (row a 6).isDummy := by rw [(publicBatchWrapper86_f134 a h), (publicBatchWrapper86_f132 a h), (publicBatchWrapper86_f133 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1724 a h), by rw [← hd, ← (publicBatchWrapper86_f1725 a h), (publicBatchWrapper86_f1726 a h), kone]⟩


theorem feeAt_6 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 6).isDummy (row a 6).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 8 51) = (row a 6).isDummy := by rw [(publicBatchWrapper86_f134 a h), (publicBatchWrapper86_f132 a h), (publicBatchWrapper86_f133 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1727 a h), by rw [← hd, ← (publicBatchWrapper86_f1728 a h), (publicBatchWrapper86_f1729 a h), kone]⟩


theorem blockAt_6 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 6).isDummy (row a 6).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 8 51) = (row a 6).isDummy := by rw [(publicBatchWrapper86_f134 a h), (publicBatchWrapper86_f132 a h), (publicBatchWrapper86_f133 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 17971), a (.virt 17973), a (.virt 17975), a (.virt 17977)],
    ![a (.virt 17972), a (.virt 17974), a (.virt 17976), a (.virt 17978)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1730 a h)
    · exact (publicBatchWrapper86_f1731 a h)
    · exact (publicBatchWrapper86_f1732 a h)
    · exact (publicBatchWrapper86_f1733 a h)
  · show bor (row a 6).isDummy (band (band (a (.virt 17971)) (a (.virt 17973)))
      (band (a (.virt 17975)) (a (.virt 17977)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1734 a h), ← (publicBatchWrapper86_f1735 a h), ← (publicBatchWrapper86_f1736 a h), ← (publicBatchWrapper86_f1737 a h), (publicBatchWrapper86_f1738 a h), kone]


theorem countAt_6 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 6).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 1202)).val = 32
  rw [(publicBatchWrapper86_f6 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_7 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 7) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f135 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f136 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f137 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f138 a h); rw [kzero] at e; exact e


theorem assetAt_7 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 7).isDummy (row a 7).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 10 15) = (row a 7).isDummy := by rw [(publicBatchWrapper86_f141 a h), (publicBatchWrapper86_f139 a h), (publicBatchWrapper86_f140 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1739 a h), by rw [← hd, ← (publicBatchWrapper86_f1740 a h), (publicBatchWrapper86_f1741 a h), kone]⟩


theorem feeAt_7 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 7).isDummy (row a 7).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 10 15) = (row a 7).isDummy := by rw [(publicBatchWrapper86_f141 a h), (publicBatchWrapper86_f139 a h), (publicBatchWrapper86_f140 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1742 a h), by rw [← hd, ← (publicBatchWrapper86_f1743 a h), (publicBatchWrapper86_f1744 a h), kone]⟩


theorem blockAt_7 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 7).isDummy (row a 7).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 10 15) = (row a 7).isDummy := by rw [(publicBatchWrapper86_f141 a h), (publicBatchWrapper86_f139 a h), (publicBatchWrapper86_f140 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 17983), a (.virt 17985), a (.virt 17987), a (.virt 17989)],
    ![a (.virt 17984), a (.virt 17986), a (.virt 17988), a (.virt 17990)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1745 a h)
    · exact (publicBatchWrapper86_f1746 a h)
    · exact (publicBatchWrapper86_f1747 a h)
    · exact (publicBatchWrapper86_f1748 a h)
  · show bor (row a 7).isDummy (band (band (a (.virt 17983)) (a (.virt 17985)))
      (band (a (.virt 17987)) (a (.virt 17989)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1749 a h), ← (publicBatchWrapper86_f1750 a h), ← (publicBatchWrapper86_f1751 a h), ← (publicBatchWrapper86_f1752 a h), (publicBatchWrapper86_f1753 a h), kone]


theorem countAt_7 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 7).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 1402)).val = 32
  rw [(publicBatchWrapper86_f7 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_8 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 8) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f142 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f143 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f144 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f145 a h); rw [kzero] at e; exact e


theorem assetAt_8 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 8).isDummy (row a 8).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 10 59) = (row a 8).isDummy := by rw [(publicBatchWrapper86_f148 a h), (publicBatchWrapper86_f146 a h), (publicBatchWrapper86_f147 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1754 a h), by rw [← hd, ← (publicBatchWrapper86_f1755 a h), (publicBatchWrapper86_f1756 a h), kone]⟩


theorem feeAt_8 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 8).isDummy (row a 8).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 10 59) = (row a 8).isDummy := by rw [(publicBatchWrapper86_f148 a h), (publicBatchWrapper86_f146 a h), (publicBatchWrapper86_f147 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1757 a h), by rw [← hd, ← (publicBatchWrapper86_f1758 a h), (publicBatchWrapper86_f1759 a h), kone]⟩


theorem blockAt_8 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 8).isDummy (row a 8).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 10 59) = (row a 8).isDummy := by rw [(publicBatchWrapper86_f148 a h), (publicBatchWrapper86_f146 a h), (publicBatchWrapper86_f147 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 17995), a (.virt 17997), a (.virt 17999), a (.virt 18001)],
    ![a (.virt 17996), a (.virt 17998), a (.virt 18000), a (.virt 18002)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1760 a h)
    · exact (publicBatchWrapper86_f1761 a h)
    · exact (publicBatchWrapper86_f1762 a h)
    · exact (publicBatchWrapper86_f1763 a h)
  · show bor (row a 8).isDummy (band (band (a (.virt 17995)) (a (.virt 17997)))
      (band (a (.virt 17999)) (a (.virt 18001)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1764 a h), ← (publicBatchWrapper86_f1765 a h), ← (publicBatchWrapper86_f1766 a h), ← (publicBatchWrapper86_f1767 a h), (publicBatchWrapper86_f1768 a h), kone]


theorem countAt_8 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 8).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 1602)).val = 32
  rw [(publicBatchWrapper86_f8 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_9 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 9) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f149 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f150 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f151 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f152 a h); rw [kzero] at e; exact e


theorem assetAt_9 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 9).isDummy (row a 9).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 11 23) = (row a 9).isDummy := by rw [(publicBatchWrapper86_f155 a h), (publicBatchWrapper86_f153 a h), (publicBatchWrapper86_f154 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1769 a h), by rw [← hd, ← (publicBatchWrapper86_f1770 a h), (publicBatchWrapper86_f1771 a h), kone]⟩


theorem feeAt_9 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 9).isDummy (row a 9).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 11 23) = (row a 9).isDummy := by rw [(publicBatchWrapper86_f155 a h), (publicBatchWrapper86_f153 a h), (publicBatchWrapper86_f154 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1772 a h), by rw [← hd, ← (publicBatchWrapper86_f1773 a h), (publicBatchWrapper86_f1774 a h), kone]⟩


theorem blockAt_9 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 9).isDummy (row a 9).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 11 23) = (row a 9).isDummy := by rw [(publicBatchWrapper86_f155 a h), (publicBatchWrapper86_f153 a h), (publicBatchWrapper86_f154 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18007), a (.virt 18009), a (.virt 18011), a (.virt 18013)],
    ![a (.virt 18008), a (.virt 18010), a (.virt 18012), a (.virt 18014)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1775 a h)
    · exact (publicBatchWrapper86_f1776 a h)
    · exact (publicBatchWrapper86_f1777 a h)
    · exact (publicBatchWrapper86_f1778 a h)
  · show bor (row a 9).isDummy (band (band (a (.virt 18007)) (a (.virt 18009)))
      (band (a (.virt 18011)) (a (.virt 18013)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1779 a h), ← (publicBatchWrapper86_f1780 a h), ← (publicBatchWrapper86_f1781 a h), ← (publicBatchWrapper86_f1782 a h), (publicBatchWrapper86_f1783 a h), kone]


theorem countAt_9 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 9).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 1802)).val = 32
  rw [(publicBatchWrapper86_f9 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_10 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 10) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f156 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f157 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f158 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f159 a h); rw [kzero] at e; exact e


theorem assetAt_10 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 10).isDummy (row a 10).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 11 67) = (row a 10).isDummy := by rw [(publicBatchWrapper86_f162 a h), (publicBatchWrapper86_f160 a h), (publicBatchWrapper86_f161 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1784 a h), by rw [← hd, ← (publicBatchWrapper86_f1785 a h), (publicBatchWrapper86_f1786 a h), kone]⟩


theorem feeAt_10 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 10).isDummy (row a 10).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 11 67) = (row a 10).isDummy := by rw [(publicBatchWrapper86_f162 a h), (publicBatchWrapper86_f160 a h), (publicBatchWrapper86_f161 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1787 a h), by rw [← hd, ← (publicBatchWrapper86_f1788 a h), (publicBatchWrapper86_f1789 a h), kone]⟩


theorem blockAt_10 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 10).isDummy (row a 10).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 11 67) = (row a 10).isDummy := by rw [(publicBatchWrapper86_f162 a h), (publicBatchWrapper86_f160 a h), (publicBatchWrapper86_f161 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18019), a (.virt 18021), a (.virt 18023), a (.virt 18025)],
    ![a (.virt 18020), a (.virt 18022), a (.virt 18024), a (.virt 18026)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1790 a h)
    · exact (publicBatchWrapper86_f1791 a h)
    · exact (publicBatchWrapper86_f1792 a h)
    · exact (publicBatchWrapper86_f1793 a h)
  · show bor (row a 10).isDummy (band (band (a (.virt 18019)) (a (.virt 18021)))
      (band (a (.virt 18023)) (a (.virt 18025)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1794 a h), ← (publicBatchWrapper86_f1795 a h), ← (publicBatchWrapper86_f1796 a h), ← (publicBatchWrapper86_f1797 a h), (publicBatchWrapper86_f1798 a h), kone]


theorem countAt_10 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 10).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 2002)).val = 32
  rw [(publicBatchWrapper86_f10 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_11 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 11) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f163 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f164 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f165 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f166 a h); rw [kzero] at e; exact e


theorem assetAt_11 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 11).isDummy (row a 11).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 13 31) = (row a 11).isDummy := by rw [(publicBatchWrapper86_f169 a h), (publicBatchWrapper86_f167 a h), (publicBatchWrapper86_f168 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1799 a h), by rw [← hd, ← (publicBatchWrapper86_f1800 a h), (publicBatchWrapper86_f1801 a h), kone]⟩


theorem feeAt_11 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 11).isDummy (row a 11).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 13 31) = (row a 11).isDummy := by rw [(publicBatchWrapper86_f169 a h), (publicBatchWrapper86_f167 a h), (publicBatchWrapper86_f168 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1802 a h), by rw [← hd, ← (publicBatchWrapper86_f1803 a h), (publicBatchWrapper86_f1804 a h), kone]⟩


theorem blockAt_11 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 11).isDummy (row a 11).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 13 31) = (row a 11).isDummy := by rw [(publicBatchWrapper86_f169 a h), (publicBatchWrapper86_f167 a h), (publicBatchWrapper86_f168 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18031), a (.virt 18033), a (.virt 18035), a (.virt 18037)],
    ![a (.virt 18032), a (.virt 18034), a (.virt 18036), a (.virt 18038)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1805 a h)
    · exact (publicBatchWrapper86_f1806 a h)
    · exact (publicBatchWrapper86_f1807 a h)
    · exact (publicBatchWrapper86_f1808 a h)
  · show bor (row a 11).isDummy (band (band (a (.virt 18031)) (a (.virt 18033)))
      (band (a (.virt 18035)) (a (.virt 18037)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1809 a h), ← (publicBatchWrapper86_f1810 a h), ← (publicBatchWrapper86_f1811 a h), ← (publicBatchWrapper86_f1812 a h), (publicBatchWrapper86_f1813 a h), kone]


theorem countAt_11 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 11).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 2202)).val = 32
  rw [(publicBatchWrapper86_f11 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_12 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 12) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f170 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f171 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f172 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f173 a h); rw [kzero] at e; exact e


theorem assetAt_12 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 12).isDummy (row a 12).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 13 75) = (row a 12).isDummy := by rw [(publicBatchWrapper86_f176 a h), (publicBatchWrapper86_f174 a h), (publicBatchWrapper86_f175 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1814 a h), by rw [← hd, ← (publicBatchWrapper86_f1815 a h), (publicBatchWrapper86_f1816 a h), kone]⟩


theorem feeAt_12 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 12).isDummy (row a 12).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 13 75) = (row a 12).isDummy := by rw [(publicBatchWrapper86_f176 a h), (publicBatchWrapper86_f174 a h), (publicBatchWrapper86_f175 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1817 a h), by rw [← hd, ← (publicBatchWrapper86_f1818 a h), (publicBatchWrapper86_f1819 a h), kone]⟩


theorem blockAt_12 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 12).isDummy (row a 12).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 13 75) = (row a 12).isDummy := by rw [(publicBatchWrapper86_f176 a h), (publicBatchWrapper86_f174 a h), (publicBatchWrapper86_f175 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18043), a (.virt 18045), a (.virt 18047), a (.virt 18049)],
    ![a (.virt 18044), a (.virt 18046), a (.virt 18048), a (.virt 18050)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1820 a h)
    · exact (publicBatchWrapper86_f1821 a h)
    · exact (publicBatchWrapper86_f1822 a h)
    · exact (publicBatchWrapper86_f1823 a h)
  · show bor (row a 12).isDummy (band (band (a (.virt 18043)) (a (.virt 18045)))
      (band (a (.virt 18047)) (a (.virt 18049)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1824 a h), ← (publicBatchWrapper86_f1825 a h), ← (publicBatchWrapper86_f1826 a h), ← (publicBatchWrapper86_f1827 a h), (publicBatchWrapper86_f1828 a h), kone]


theorem countAt_12 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 12).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 2402)).val = 32
  rw [(publicBatchWrapper86_f12 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_13 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 13) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f177 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f178 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f179 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f180 a h); rw [kzero] at e; exact e


theorem assetAt_13 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 13).isDummy (row a 13).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 15 39) = (row a 13).isDummy := by rw [(publicBatchWrapper86_f183 a h), (publicBatchWrapper86_f181 a h), (publicBatchWrapper86_f182 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1829 a h), by rw [← hd, ← (publicBatchWrapper86_f1830 a h), (publicBatchWrapper86_f1831 a h), kone]⟩


theorem feeAt_13 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 13).isDummy (row a 13).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 15 39) = (row a 13).isDummy := by rw [(publicBatchWrapper86_f183 a h), (publicBatchWrapper86_f181 a h), (publicBatchWrapper86_f182 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1832 a h), by rw [← hd, ← (publicBatchWrapper86_f1833 a h), (publicBatchWrapper86_f1834 a h), kone]⟩


theorem blockAt_13 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 13).isDummy (row a 13).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 15 39) = (row a 13).isDummy := by rw [(publicBatchWrapper86_f183 a h), (publicBatchWrapper86_f181 a h), (publicBatchWrapper86_f182 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18055), a (.virt 18057), a (.virt 18059), a (.virt 18061)],
    ![a (.virt 18056), a (.virt 18058), a (.virt 18060), a (.virt 18062)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1835 a h)
    · exact (publicBatchWrapper86_f1836 a h)
    · exact (publicBatchWrapper86_f1837 a h)
    · exact (publicBatchWrapper86_f1838 a h)
  · show bor (row a 13).isDummy (band (band (a (.virt 18055)) (a (.virt 18057)))
      (band (a (.virt 18059)) (a (.virt 18061)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1839 a h), ← (publicBatchWrapper86_f1840 a h), ← (publicBatchWrapper86_f1841 a h), ← (publicBatchWrapper86_f1842 a h), (publicBatchWrapper86_f1843 a h), kone]


theorem countAt_13 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 13).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 2602)).val = 32
  rw [(publicBatchWrapper86_f13 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_14 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 14) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f184 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f185 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f186 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f187 a h); rw [kzero] at e; exact e


theorem assetAt_14 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 14).isDummy (row a 14).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 16 3) = (row a 14).isDummy := by rw [(publicBatchWrapper86_f190 a h), (publicBatchWrapper86_f188 a h), (publicBatchWrapper86_f189 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1844 a h), by rw [← hd, ← (publicBatchWrapper86_f1845 a h), (publicBatchWrapper86_f1846 a h), kone]⟩


theorem feeAt_14 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 14).isDummy (row a 14).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 16 3) = (row a 14).isDummy := by rw [(publicBatchWrapper86_f190 a h), (publicBatchWrapper86_f188 a h), (publicBatchWrapper86_f189 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1847 a h), by rw [← hd, ← (publicBatchWrapper86_f1848 a h), (publicBatchWrapper86_f1849 a h), kone]⟩


theorem blockAt_14 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 14).isDummy (row a 14).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 16 3) = (row a 14).isDummy := by rw [(publicBatchWrapper86_f190 a h), (publicBatchWrapper86_f188 a h), (publicBatchWrapper86_f189 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18067), a (.virt 18069), a (.virt 18071), a (.virt 18073)],
    ![a (.virt 18068), a (.virt 18070), a (.virt 18072), a (.virt 18074)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1850 a h)
    · exact (publicBatchWrapper86_f1851 a h)
    · exact (publicBatchWrapper86_f1852 a h)
    · exact (publicBatchWrapper86_f1853 a h)
  · show bor (row a 14).isDummy (band (band (a (.virt 18067)) (a (.virt 18069)))
      (band (a (.virt 18071)) (a (.virt 18073)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1854 a h), ← (publicBatchWrapper86_f1855 a h), ← (publicBatchWrapper86_f1856 a h), ← (publicBatchWrapper86_f1857 a h), (publicBatchWrapper86_f1858 a h), kone]


theorem countAt_14 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 14).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 2802)).val = 32
  rw [(publicBatchWrapper86_f14 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_15 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 15) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f191 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f192 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f193 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f194 a h); rw [kzero] at e; exact e


theorem assetAt_15 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 15).isDummy (row a 15).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 16 47) = (row a 15).isDummy := by rw [(publicBatchWrapper86_f197 a h), (publicBatchWrapper86_f195 a h), (publicBatchWrapper86_f196 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1859 a h), by rw [← hd, ← (publicBatchWrapper86_f1860 a h), (publicBatchWrapper86_f1861 a h), kone]⟩


theorem feeAt_15 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 15).isDummy (row a 15).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 16 47) = (row a 15).isDummy := by rw [(publicBatchWrapper86_f197 a h), (publicBatchWrapper86_f195 a h), (publicBatchWrapper86_f196 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1862 a h), by rw [← hd, ← (publicBatchWrapper86_f1863 a h), (publicBatchWrapper86_f1864 a h), kone]⟩


theorem blockAt_15 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 15).isDummy (row a 15).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 16 47) = (row a 15).isDummy := by rw [(publicBatchWrapper86_f197 a h), (publicBatchWrapper86_f195 a h), (publicBatchWrapper86_f196 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18079), a (.virt 18081), a (.virt 18083), a (.virt 18085)],
    ![a (.virt 18080), a (.virt 18082), a (.virt 18084), a (.virt 18086)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1865 a h)
    · exact (publicBatchWrapper86_f1866 a h)
    · exact (publicBatchWrapper86_f1867 a h)
    · exact (publicBatchWrapper86_f1868 a h)
  · show bor (row a 15).isDummy (band (band (a (.virt 18079)) (a (.virt 18081)))
      (band (a (.virt 18083)) (a (.virt 18085)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1869 a h), ← (publicBatchWrapper86_f1870 a h), ← (publicBatchWrapper86_f1871 a h), ← (publicBatchWrapper86_f1872 a h), (publicBatchWrapper86_f1873 a h), kone]


theorem countAt_15 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 15).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 3002)).val = 32
  rw [(publicBatchWrapper86_f15 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_16 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 16) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f198 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f199 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f200 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f201 a h); rw [kzero] at e; exact e


theorem assetAt_16 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 16).isDummy (row a 16).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 18 11) = (row a 16).isDummy := by rw [(publicBatchWrapper86_f204 a h), (publicBatchWrapper86_f202 a h), (publicBatchWrapper86_f203 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1874 a h), by rw [← hd, ← (publicBatchWrapper86_f1875 a h), (publicBatchWrapper86_f1876 a h), kone]⟩


theorem feeAt_16 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 16).isDummy (row a 16).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 18 11) = (row a 16).isDummy := by rw [(publicBatchWrapper86_f204 a h), (publicBatchWrapper86_f202 a h), (publicBatchWrapper86_f203 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1877 a h), by rw [← hd, ← (publicBatchWrapper86_f1878 a h), (publicBatchWrapper86_f1879 a h), kone]⟩


theorem blockAt_16 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 16).isDummy (row a 16).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 18 11) = (row a 16).isDummy := by rw [(publicBatchWrapper86_f204 a h), (publicBatchWrapper86_f202 a h), (publicBatchWrapper86_f203 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18091), a (.virt 18093), a (.virt 18095), a (.virt 18097)],
    ![a (.virt 18092), a (.virt 18094), a (.virt 18096), a (.virt 18098)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1880 a h)
    · exact (publicBatchWrapper86_f1881 a h)
    · exact (publicBatchWrapper86_f1882 a h)
    · exact (publicBatchWrapper86_f1883 a h)
  · show bor (row a 16).isDummy (band (band (a (.virt 18091)) (a (.virt 18093)))
      (band (a (.virt 18095)) (a (.virt 18097)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1884 a h), ← (publicBatchWrapper86_f1885 a h), ← (publicBatchWrapper86_f1886 a h), ← (publicBatchWrapper86_f1887 a h), (publicBatchWrapper86_f1888 a h), kone]


theorem countAt_16 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 16).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 3202)).val = 32
  rw [(publicBatchWrapper86_f16 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_17 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 17) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f205 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f206 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f207 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f208 a h); rw [kzero] at e; exact e


theorem assetAt_17 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 17).isDummy (row a 17).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 18 55) = (row a 17).isDummy := by rw [(publicBatchWrapper86_f211 a h), (publicBatchWrapper86_f209 a h), (publicBatchWrapper86_f210 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1889 a h), by rw [← hd, ← (publicBatchWrapper86_f1890 a h), (publicBatchWrapper86_f1891 a h), kone]⟩


theorem feeAt_17 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 17).isDummy (row a 17).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 18 55) = (row a 17).isDummy := by rw [(publicBatchWrapper86_f211 a h), (publicBatchWrapper86_f209 a h), (publicBatchWrapper86_f210 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1892 a h), by rw [← hd, ← (publicBatchWrapper86_f1893 a h), (publicBatchWrapper86_f1894 a h), kone]⟩


theorem blockAt_17 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 17).isDummy (row a 17).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 18 55) = (row a 17).isDummy := by rw [(publicBatchWrapper86_f211 a h), (publicBatchWrapper86_f209 a h), (publicBatchWrapper86_f210 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18103), a (.virt 18105), a (.virt 18107), a (.virt 18109)],
    ![a (.virt 18104), a (.virt 18106), a (.virt 18108), a (.virt 18110)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1895 a h)
    · exact (publicBatchWrapper86_f1896 a h)
    · exact (publicBatchWrapper86_f1897 a h)
    · exact (publicBatchWrapper86_f1898 a h)
  · show bor (row a 17).isDummy (band (band (a (.virt 18103)) (a (.virt 18105)))
      (band (a (.virt 18107)) (a (.virt 18109)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1899 a h), ← (publicBatchWrapper86_f1900 a h), ← (publicBatchWrapper86_f1901 a h), ← (publicBatchWrapper86_f1902 a h), (publicBatchWrapper86_f1903 a h), kone]


theorem countAt_17 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 17).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 3402)).val = 32
  rw [(publicBatchWrapper86_f17 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_18 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 18) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f212 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f213 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f214 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f215 a h); rw [kzero] at e; exact e


theorem assetAt_18 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 18).isDummy (row a 18).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 20 19) = (row a 18).isDummy := by rw [(publicBatchWrapper86_f218 a h), (publicBatchWrapper86_f216 a h), (publicBatchWrapper86_f217 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1904 a h), by rw [← hd, ← (publicBatchWrapper86_f1905 a h), (publicBatchWrapper86_f1906 a h), kone]⟩


theorem feeAt_18 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 18).isDummy (row a 18).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 20 19) = (row a 18).isDummy := by rw [(publicBatchWrapper86_f218 a h), (publicBatchWrapper86_f216 a h), (publicBatchWrapper86_f217 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1907 a h), by rw [← hd, ← (publicBatchWrapper86_f1908 a h), (publicBatchWrapper86_f1909 a h), kone]⟩


theorem blockAt_18 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 18).isDummy (row a 18).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 20 19) = (row a 18).isDummy := by rw [(publicBatchWrapper86_f218 a h), (publicBatchWrapper86_f216 a h), (publicBatchWrapper86_f217 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18115), a (.virt 18117), a (.virt 18119), a (.virt 18121)],
    ![a (.virt 18116), a (.virt 18118), a (.virt 18120), a (.virt 18122)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1910 a h)
    · exact (publicBatchWrapper86_f1911 a h)
    · exact (publicBatchWrapper86_f1912 a h)
    · exact (publicBatchWrapper86_f1913 a h)
  · show bor (row a 18).isDummy (band (band (a (.virt 18115)) (a (.virt 18117)))
      (band (a (.virt 18119)) (a (.virt 18121)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1914 a h), ← (publicBatchWrapper86_f1915 a h), ← (publicBatchWrapper86_f1916 a h), ← (publicBatchWrapper86_f1917 a h), (publicBatchWrapper86_f1918 a h), kone]


theorem countAt_18 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 18).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 3602)).val = 32
  rw [(publicBatchWrapper86_f18 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_19 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 19) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f219 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f220 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f221 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f222 a h); rw [kzero] at e; exact e


theorem assetAt_19 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 19).isDummy (row a 19).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 20 63) = (row a 19).isDummy := by rw [(publicBatchWrapper86_f225 a h), (publicBatchWrapper86_f223 a h), (publicBatchWrapper86_f224 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1919 a h), by rw [← hd, ← (publicBatchWrapper86_f1920 a h), (publicBatchWrapper86_f1921 a h), kone]⟩


theorem feeAt_19 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 19).isDummy (row a 19).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 20 63) = (row a 19).isDummy := by rw [(publicBatchWrapper86_f225 a h), (publicBatchWrapper86_f223 a h), (publicBatchWrapper86_f224 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1922 a h), by rw [← hd, ← (publicBatchWrapper86_f1923 a h), (publicBatchWrapper86_f1924 a h), kone]⟩


theorem blockAt_19 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 19).isDummy (row a 19).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 20 63) = (row a 19).isDummy := by rw [(publicBatchWrapper86_f225 a h), (publicBatchWrapper86_f223 a h), (publicBatchWrapper86_f224 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18127), a (.virt 18129), a (.virt 18131), a (.virt 18133)],
    ![a (.virt 18128), a (.virt 18130), a (.virt 18132), a (.virt 18134)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1925 a h)
    · exact (publicBatchWrapper86_f1926 a h)
    · exact (publicBatchWrapper86_f1927 a h)
    · exact (publicBatchWrapper86_f1928 a h)
  · show bor (row a 19).isDummy (band (band (a (.virt 18127)) (a (.virt 18129)))
      (band (a (.virt 18131)) (a (.virt 18133)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1929 a h), ← (publicBatchWrapper86_f1930 a h), ← (publicBatchWrapper86_f1931 a h), ← (publicBatchWrapper86_f1932 a h), (publicBatchWrapper86_f1933 a h), kone]


theorem countAt_19 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 19).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 3802)).val = 32
  rw [(publicBatchWrapper86_f19 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_20 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 20) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f226 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f227 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f228 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f229 a h); rw [kzero] at e; exact e


theorem assetAt_20 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 20).isDummy (row a 20).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 22 27) = (row a 20).isDummy := by rw [(publicBatchWrapper86_f232 a h), (publicBatchWrapper86_f230 a h), (publicBatchWrapper86_f231 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1934 a h), by rw [← hd, ← (publicBatchWrapper86_f1935 a h), (publicBatchWrapper86_f1936 a h), kone]⟩


theorem feeAt_20 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 20).isDummy (row a 20).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 22 27) = (row a 20).isDummy := by rw [(publicBatchWrapper86_f232 a h), (publicBatchWrapper86_f230 a h), (publicBatchWrapper86_f231 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1937 a h), by rw [← hd, ← (publicBatchWrapper86_f1938 a h), (publicBatchWrapper86_f1939 a h), kone]⟩


theorem blockAt_20 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 20).isDummy (row a 20).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 22 27) = (row a 20).isDummy := by rw [(publicBatchWrapper86_f232 a h), (publicBatchWrapper86_f230 a h), (publicBatchWrapper86_f231 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18139), a (.virt 18141), a (.virt 18143), a (.virt 18145)],
    ![a (.virt 18140), a (.virt 18142), a (.virt 18144), a (.virt 18146)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1940 a h)
    · exact (publicBatchWrapper86_f1941 a h)
    · exact (publicBatchWrapper86_f1942 a h)
    · exact (publicBatchWrapper86_f1943 a h)
  · show bor (row a 20).isDummy (band (band (a (.virt 18139)) (a (.virt 18141)))
      (band (a (.virt 18143)) (a (.virt 18145)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1944 a h), ← (publicBatchWrapper86_f1945 a h), ← (publicBatchWrapper86_f1946 a h), ← (publicBatchWrapper86_f1947 a h), (publicBatchWrapper86_f1948 a h), kone]


theorem countAt_20 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 20).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 4002)).val = 32
  rw [(publicBatchWrapper86_f20 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_21 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 21) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f233 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f234 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f235 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f236 a h); rw [kzero] at e; exact e


theorem assetAt_21 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 21).isDummy (row a 21).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 22 71) = (row a 21).isDummy := by rw [(publicBatchWrapper86_f239 a h), (publicBatchWrapper86_f237 a h), (publicBatchWrapper86_f238 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1949 a h), by rw [← hd, ← (publicBatchWrapper86_f1950 a h), (publicBatchWrapper86_f1951 a h), kone]⟩


theorem feeAt_21 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 21).isDummy (row a 21).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 22 71) = (row a 21).isDummy := by rw [(publicBatchWrapper86_f239 a h), (publicBatchWrapper86_f237 a h), (publicBatchWrapper86_f238 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1952 a h), by rw [← hd, ← (publicBatchWrapper86_f1953 a h), (publicBatchWrapper86_f1954 a h), kone]⟩


theorem blockAt_21 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 21).isDummy (row a 21).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 22 71) = (row a 21).isDummy := by rw [(publicBatchWrapper86_f239 a h), (publicBatchWrapper86_f237 a h), (publicBatchWrapper86_f238 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18151), a (.virt 18153), a (.virt 18155), a (.virt 18157)],
    ![a (.virt 18152), a (.virt 18154), a (.virt 18156), a (.virt 18158)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1955 a h)
    · exact (publicBatchWrapper86_f1956 a h)
    · exact (publicBatchWrapper86_f1957 a h)
    · exact (publicBatchWrapper86_f1958 a h)
  · show bor (row a 21).isDummy (band (band (a (.virt 18151)) (a (.virt 18153)))
      (band (a (.virt 18155)) (a (.virt 18157)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1959 a h), ← (publicBatchWrapper86_f1960 a h), ← (publicBatchWrapper86_f1961 a h), ← (publicBatchWrapper86_f1962 a h), (publicBatchWrapper86_f1963 a h), kone]


theorem countAt_21 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 21).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 4202)).val = 32
  rw [(publicBatchWrapper86_f21 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_22 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 22) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f240 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f241 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f242 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f243 a h); rw [kzero] at e; exact e


theorem assetAt_22 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 22).isDummy (row a 22).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 23 35) = (row a 22).isDummy := by rw [(publicBatchWrapper86_f246 a h), (publicBatchWrapper86_f244 a h), (publicBatchWrapper86_f245 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1964 a h), by rw [← hd, ← (publicBatchWrapper86_f1965 a h), (publicBatchWrapper86_f1966 a h), kone]⟩


theorem feeAt_22 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 22).isDummy (row a 22).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 23 35) = (row a 22).isDummy := by rw [(publicBatchWrapper86_f246 a h), (publicBatchWrapper86_f244 a h), (publicBatchWrapper86_f245 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1967 a h), by rw [← hd, ← (publicBatchWrapper86_f1968 a h), (publicBatchWrapper86_f1969 a h), kone]⟩


theorem blockAt_22 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 22).isDummy (row a 22).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 23 35) = (row a 22).isDummy := by rw [(publicBatchWrapper86_f246 a h), (publicBatchWrapper86_f244 a h), (publicBatchWrapper86_f245 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18163), a (.virt 18165), a (.virt 18167), a (.virt 18169)],
    ![a (.virt 18164), a (.virt 18166), a (.virt 18168), a (.virt 18170)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1970 a h)
    · exact (publicBatchWrapper86_f1971 a h)
    · exact (publicBatchWrapper86_f1972 a h)
    · exact (publicBatchWrapper86_f1973 a h)
  · show bor (row a 22).isDummy (band (band (a (.virt 18163)) (a (.virt 18165)))
      (band (a (.virt 18167)) (a (.virt 18169)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1974 a h), ← (publicBatchWrapper86_f1975 a h), ← (publicBatchWrapper86_f1976 a h), ← (publicBatchWrapper86_f1977 a h), (publicBatchWrapper86_f1978 a h), kone]


theorem countAt_22 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 22).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 4402)).val = 32
  rw [(publicBatchWrapper86_f22 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_23 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 23) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f247 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f248 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f249 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f250 a h); rw [kzero] at e; exact e


theorem assetAt_23 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 23).isDummy (row a 23).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 23 79) = (row a 23).isDummy := by rw [(publicBatchWrapper86_f253 a h), (publicBatchWrapper86_f251 a h), (publicBatchWrapper86_f252 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1979 a h), by rw [← hd, ← (publicBatchWrapper86_f1980 a h), (publicBatchWrapper86_f1981 a h), kone]⟩


theorem feeAt_23 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 23).isDummy (row a 23).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 23 79) = (row a 23).isDummy := by rw [(publicBatchWrapper86_f253 a h), (publicBatchWrapper86_f251 a h), (publicBatchWrapper86_f252 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1982 a h), by rw [← hd, ← (publicBatchWrapper86_f1983 a h), (publicBatchWrapper86_f1984 a h), kone]⟩


theorem blockAt_23 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 23).isDummy (row a 23).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 23 79) = (row a 23).isDummy := by rw [(publicBatchWrapper86_f253 a h), (publicBatchWrapper86_f251 a h), (publicBatchWrapper86_f252 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18175), a (.virt 18177), a (.virt 18179), a (.virt 18181)],
    ![a (.virt 18176), a (.virt 18178), a (.virt 18180), a (.virt 18182)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f1985 a h)
    · exact (publicBatchWrapper86_f1986 a h)
    · exact (publicBatchWrapper86_f1987 a h)
    · exact (publicBatchWrapper86_f1988 a h)
  · show bor (row a 23).isDummy (band (band (a (.virt 18175)) (a (.virt 18177)))
      (band (a (.virt 18179)) (a (.virt 18181)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f1989 a h), ← (publicBatchWrapper86_f1990 a h), ← (publicBatchWrapper86_f1991 a h), ← (publicBatchWrapper86_f1992 a h), (publicBatchWrapper86_f1993 a h), kone]


theorem countAt_23 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 23).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 4602)).val = 32
  rw [(publicBatchWrapper86_f23 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_24 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 24) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f254 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f255 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f256 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f257 a h); rw [kzero] at e; exact e


theorem assetAt_24 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 24).isDummy (row a 24).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 25 43) = (row a 24).isDummy := by rw [(publicBatchWrapper86_f260 a h), (publicBatchWrapper86_f258 a h), (publicBatchWrapper86_f259 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f1994 a h), by rw [← hd, ← (publicBatchWrapper86_f1995 a h), (publicBatchWrapper86_f1996 a h), kone]⟩


theorem feeAt_24 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 24).isDummy (row a 24).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 25 43) = (row a 24).isDummy := by rw [(publicBatchWrapper86_f260 a h), (publicBatchWrapper86_f258 a h), (publicBatchWrapper86_f259 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f1997 a h), by rw [← hd, ← (publicBatchWrapper86_f1998 a h), (publicBatchWrapper86_f1999 a h), kone]⟩


theorem blockAt_24 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 24).isDummy (row a 24).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 25 43) = (row a 24).isDummy := by rw [(publicBatchWrapper86_f260 a h), (publicBatchWrapper86_f258 a h), (publicBatchWrapper86_f259 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18187), a (.virt 18189), a (.virt 18191), a (.virt 18193)],
    ![a (.virt 18188), a (.virt 18190), a (.virt 18192), a (.virt 18194)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2000 a h)
    · exact (publicBatchWrapper86_f2001 a h)
    · exact (publicBatchWrapper86_f2002 a h)
    · exact (publicBatchWrapper86_f2003 a h)
  · show bor (row a 24).isDummy (band (band (a (.virt 18187)) (a (.virt 18189)))
      (band (a (.virt 18191)) (a (.virt 18193)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2004 a h), ← (publicBatchWrapper86_f2005 a h), ← (publicBatchWrapper86_f2006 a h), ← (publicBatchWrapper86_f2007 a h), (publicBatchWrapper86_f2008 a h), kone]


theorem countAt_24 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 24).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 4802)).val = 32
  rw [(publicBatchWrapper86_f24 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_25 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 25) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f261 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f262 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f263 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f264 a h); rw [kzero] at e; exact e


theorem assetAt_25 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 25).isDummy (row a 25).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 27 7) = (row a 25).isDummy := by rw [(publicBatchWrapper86_f267 a h), (publicBatchWrapper86_f265 a h), (publicBatchWrapper86_f266 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2009 a h), by rw [← hd, ← (publicBatchWrapper86_f2010 a h), (publicBatchWrapper86_f2011 a h), kone]⟩


theorem feeAt_25 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 25).isDummy (row a 25).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 27 7) = (row a 25).isDummy := by rw [(publicBatchWrapper86_f267 a h), (publicBatchWrapper86_f265 a h), (publicBatchWrapper86_f266 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2012 a h), by rw [← hd, ← (publicBatchWrapper86_f2013 a h), (publicBatchWrapper86_f2014 a h), kone]⟩


theorem blockAt_25 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 25).isDummy (row a 25).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 27 7) = (row a 25).isDummy := by rw [(publicBatchWrapper86_f267 a h), (publicBatchWrapper86_f265 a h), (publicBatchWrapper86_f266 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18199), a (.virt 18201), a (.virt 18203), a (.virt 18205)],
    ![a (.virt 18200), a (.virt 18202), a (.virt 18204), a (.virt 18206)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2015 a h)
    · exact (publicBatchWrapper86_f2016 a h)
    · exact (publicBatchWrapper86_f2017 a h)
    · exact (publicBatchWrapper86_f2018 a h)
  · show bor (row a 25).isDummy (band (band (a (.virt 18199)) (a (.virt 18201)))
      (band (a (.virt 18203)) (a (.virt 18205)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2019 a h), ← (publicBatchWrapper86_f2020 a h), ← (publicBatchWrapper86_f2021 a h), ← (publicBatchWrapper86_f2022 a h), (publicBatchWrapper86_f2023 a h), kone]


theorem countAt_25 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 25).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 5002)).val = 32
  rw [(publicBatchWrapper86_f25 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_26 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 26) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f268 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f269 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f270 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f271 a h); rw [kzero] at e; exact e


theorem assetAt_26 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 26).isDummy (row a 26).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 27 51) = (row a 26).isDummy := by rw [(publicBatchWrapper86_f274 a h), (publicBatchWrapper86_f272 a h), (publicBatchWrapper86_f273 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2024 a h), by rw [← hd, ← (publicBatchWrapper86_f2025 a h), (publicBatchWrapper86_f2026 a h), kone]⟩


theorem feeAt_26 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 26).isDummy (row a 26).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 27 51) = (row a 26).isDummy := by rw [(publicBatchWrapper86_f274 a h), (publicBatchWrapper86_f272 a h), (publicBatchWrapper86_f273 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2027 a h), by rw [← hd, ← (publicBatchWrapper86_f2028 a h), (publicBatchWrapper86_f2029 a h), kone]⟩


theorem blockAt_26 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 26).isDummy (row a 26).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 27 51) = (row a 26).isDummy := by rw [(publicBatchWrapper86_f274 a h), (publicBatchWrapper86_f272 a h), (publicBatchWrapper86_f273 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18211), a (.virt 18213), a (.virt 18215), a (.virt 18217)],
    ![a (.virt 18212), a (.virt 18214), a (.virt 18216), a (.virt 18218)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2030 a h)
    · exact (publicBatchWrapper86_f2031 a h)
    · exact (publicBatchWrapper86_f2032 a h)
    · exact (publicBatchWrapper86_f2033 a h)
  · show bor (row a 26).isDummy (band (band (a (.virt 18211)) (a (.virt 18213)))
      (band (a (.virt 18215)) (a (.virt 18217)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2034 a h), ← (publicBatchWrapper86_f2035 a h), ← (publicBatchWrapper86_f2036 a h), ← (publicBatchWrapper86_f2037 a h), (publicBatchWrapper86_f2038 a h), kone]


theorem countAt_26 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 26).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 5202)).val = 32
  rw [(publicBatchWrapper86_f26 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_27 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 27) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f275 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f276 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f277 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f278 a h); rw [kzero] at e; exact e


theorem assetAt_27 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 27).isDummy (row a 27).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 29 15) = (row a 27).isDummy := by rw [(publicBatchWrapper86_f281 a h), (publicBatchWrapper86_f279 a h), (publicBatchWrapper86_f280 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2039 a h), by rw [← hd, ← (publicBatchWrapper86_f2040 a h), (publicBatchWrapper86_f2041 a h), kone]⟩


theorem feeAt_27 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 27).isDummy (row a 27).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 29 15) = (row a 27).isDummy := by rw [(publicBatchWrapper86_f281 a h), (publicBatchWrapper86_f279 a h), (publicBatchWrapper86_f280 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2042 a h), by rw [← hd, ← (publicBatchWrapper86_f2043 a h), (publicBatchWrapper86_f2044 a h), kone]⟩


theorem blockAt_27 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 27).isDummy (row a 27).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 29 15) = (row a 27).isDummy := by rw [(publicBatchWrapper86_f281 a h), (publicBatchWrapper86_f279 a h), (publicBatchWrapper86_f280 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18223), a (.virt 18225), a (.virt 18227), a (.virt 18229)],
    ![a (.virt 18224), a (.virt 18226), a (.virt 18228), a (.virt 18230)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2045 a h)
    · exact (publicBatchWrapper86_f2046 a h)
    · exact (publicBatchWrapper86_f2047 a h)
    · exact (publicBatchWrapper86_f2048 a h)
  · show bor (row a 27).isDummy (band (band (a (.virt 18223)) (a (.virt 18225)))
      (band (a (.virt 18227)) (a (.virt 18229)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2049 a h), ← (publicBatchWrapper86_f2050 a h), ← (publicBatchWrapper86_f2051 a h), ← (publicBatchWrapper86_f2052 a h), (publicBatchWrapper86_f2053 a h), kone]


theorem countAt_27 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 27).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 5402)).val = 32
  rw [(publicBatchWrapper86_f27 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_28 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 28) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f282 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f283 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f284 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f285 a h); rw [kzero] at e; exact e


theorem assetAt_28 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 28).isDummy (row a 28).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 29 59) = (row a 28).isDummy := by rw [(publicBatchWrapper86_f288 a h), (publicBatchWrapper86_f286 a h), (publicBatchWrapper86_f287 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2054 a h), by rw [← hd, ← (publicBatchWrapper86_f2055 a h), (publicBatchWrapper86_f2056 a h), kone]⟩


theorem feeAt_28 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 28).isDummy (row a 28).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 29 59) = (row a 28).isDummy := by rw [(publicBatchWrapper86_f288 a h), (publicBatchWrapper86_f286 a h), (publicBatchWrapper86_f287 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2057 a h), by rw [← hd, ← (publicBatchWrapper86_f2058 a h), (publicBatchWrapper86_f2059 a h), kone]⟩


theorem blockAt_28 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 28).isDummy (row a 28).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 29 59) = (row a 28).isDummy := by rw [(publicBatchWrapper86_f288 a h), (publicBatchWrapper86_f286 a h), (publicBatchWrapper86_f287 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18235), a (.virt 18237), a (.virt 18239), a (.virt 18241)],
    ![a (.virt 18236), a (.virt 18238), a (.virt 18240), a (.virt 18242)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2060 a h)
    · exact (publicBatchWrapper86_f2061 a h)
    · exact (publicBatchWrapper86_f2062 a h)
    · exact (publicBatchWrapper86_f2063 a h)
  · show bor (row a 28).isDummy (band (band (a (.virt 18235)) (a (.virt 18237)))
      (band (a (.virt 18239)) (a (.virt 18241)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2064 a h), ← (publicBatchWrapper86_f2065 a h), ← (publicBatchWrapper86_f2066 a h), ← (publicBatchWrapper86_f2067 a h), (publicBatchWrapper86_f2068 a h), kone]


theorem countAt_28 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 28).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 5602)).val = 32
  rw [(publicBatchWrapper86_f28 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_29 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 29) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f289 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f290 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f291 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f292 a h); rw [kzero] at e; exact e


theorem assetAt_29 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 29).isDummy (row a 29).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 30 23) = (row a 29).isDummy := by rw [(publicBatchWrapper86_f295 a h), (publicBatchWrapper86_f293 a h), (publicBatchWrapper86_f294 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2069 a h), by rw [← hd, ← (publicBatchWrapper86_f2070 a h), (publicBatchWrapper86_f2071 a h), kone]⟩


theorem feeAt_29 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 29).isDummy (row a 29).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 30 23) = (row a 29).isDummy := by rw [(publicBatchWrapper86_f295 a h), (publicBatchWrapper86_f293 a h), (publicBatchWrapper86_f294 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2072 a h), by rw [← hd, ← (publicBatchWrapper86_f2073 a h), (publicBatchWrapper86_f2074 a h), kone]⟩


theorem blockAt_29 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 29).isDummy (row a 29).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 30 23) = (row a 29).isDummy := by rw [(publicBatchWrapper86_f295 a h), (publicBatchWrapper86_f293 a h), (publicBatchWrapper86_f294 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18247), a (.virt 18249), a (.virt 18251), a (.virt 18253)],
    ![a (.virt 18248), a (.virt 18250), a (.virt 18252), a (.virt 18254)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2075 a h)
    · exact (publicBatchWrapper86_f2076 a h)
    · exact (publicBatchWrapper86_f2077 a h)
    · exact (publicBatchWrapper86_f2078 a h)
  · show bor (row a 29).isDummy (band (band (a (.virt 18247)) (a (.virt 18249)))
      (band (a (.virt 18251)) (a (.virt 18253)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2079 a h), ← (publicBatchWrapper86_f2080 a h), ← (publicBatchWrapper86_f2081 a h), ← (publicBatchWrapper86_f2082 a h), (publicBatchWrapper86_f2083 a h), kone]


theorem countAt_29 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 29).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 5802)).val = 32
  rw [(publicBatchWrapper86_f29 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_30 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 30) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f296 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f297 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f298 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f299 a h); rw [kzero] at e; exact e


theorem assetAt_30 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 30).isDummy (row a 30).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 30 67) = (row a 30).isDummy := by rw [(publicBatchWrapper86_f302 a h), (publicBatchWrapper86_f300 a h), (publicBatchWrapper86_f301 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2084 a h), by rw [← hd, ← (publicBatchWrapper86_f2085 a h), (publicBatchWrapper86_f2086 a h), kone]⟩


theorem feeAt_30 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 30).isDummy (row a 30).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 30 67) = (row a 30).isDummy := by rw [(publicBatchWrapper86_f302 a h), (publicBatchWrapper86_f300 a h), (publicBatchWrapper86_f301 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2087 a h), by rw [← hd, ← (publicBatchWrapper86_f2088 a h), (publicBatchWrapper86_f2089 a h), kone]⟩


theorem blockAt_30 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 30).isDummy (row a 30).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 30 67) = (row a 30).isDummy := by rw [(publicBatchWrapper86_f302 a h), (publicBatchWrapper86_f300 a h), (publicBatchWrapper86_f301 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18259), a (.virt 18261), a (.virt 18263), a (.virt 18265)],
    ![a (.virt 18260), a (.virt 18262), a (.virt 18264), a (.virt 18266)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2090 a h)
    · exact (publicBatchWrapper86_f2091 a h)
    · exact (publicBatchWrapper86_f2092 a h)
    · exact (publicBatchWrapper86_f2093 a h)
  · show bor (row a 30).isDummy (band (band (a (.virt 18259)) (a (.virt 18261)))
      (band (a (.virt 18263)) (a (.virt 18265)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2094 a h), ← (publicBatchWrapper86_f2095 a h), ← (publicBatchWrapper86_f2096 a h), ← (publicBatchWrapper86_f2097 a h), (publicBatchWrapper86_f2098 a h), kone]


theorem countAt_30 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 30).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 6002)).val = 32
  rw [(publicBatchWrapper86_f30 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_31 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 31) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f303 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f304 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f305 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f306 a h); rw [kzero] at e; exact e


theorem assetAt_31 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 31).isDummy (row a 31).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 32 31) = (row a 31).isDummy := by rw [(publicBatchWrapper86_f309 a h), (publicBatchWrapper86_f307 a h), (publicBatchWrapper86_f308 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2099 a h), by rw [← hd, ← (publicBatchWrapper86_f2100 a h), (publicBatchWrapper86_f2101 a h), kone]⟩


theorem feeAt_31 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 31).isDummy (row a 31).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 32 31) = (row a 31).isDummy := by rw [(publicBatchWrapper86_f309 a h), (publicBatchWrapper86_f307 a h), (publicBatchWrapper86_f308 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2102 a h), by rw [← hd, ← (publicBatchWrapper86_f2103 a h), (publicBatchWrapper86_f2104 a h), kone]⟩


theorem blockAt_31 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 31).isDummy (row a 31).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 32 31) = (row a 31).isDummy := by rw [(publicBatchWrapper86_f309 a h), (publicBatchWrapper86_f307 a h), (publicBatchWrapper86_f308 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18271), a (.virt 18273), a (.virt 18275), a (.virt 18277)],
    ![a (.virt 18272), a (.virt 18274), a (.virt 18276), a (.virt 18278)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2105 a h)
    · exact (publicBatchWrapper86_f2106 a h)
    · exact (publicBatchWrapper86_f2107 a h)
    · exact (publicBatchWrapper86_f2108 a h)
  · show bor (row a 31).isDummy (band (band (a (.virt 18271)) (a (.virt 18273)))
      (band (a (.virt 18275)) (a (.virt 18277)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2109 a h), ← (publicBatchWrapper86_f2110 a h), ← (publicBatchWrapper86_f2111 a h), ← (publicBatchWrapper86_f2112 a h), (publicBatchWrapper86_f2113 a h), kone]


theorem countAt_31 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 31).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 6202)).val = 32
  rw [(publicBatchWrapper86_f31 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_32 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 32) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f310 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f311 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f312 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f313 a h); rw [kzero] at e; exact e


theorem assetAt_32 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 32).isDummy (row a 32).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 32 75) = (row a 32).isDummy := by rw [(publicBatchWrapper86_f316 a h), (publicBatchWrapper86_f314 a h), (publicBatchWrapper86_f315 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2114 a h), by rw [← hd, ← (publicBatchWrapper86_f2115 a h), (publicBatchWrapper86_f2116 a h), kone]⟩


theorem feeAt_32 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 32).isDummy (row a 32).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 32 75) = (row a 32).isDummy := by rw [(publicBatchWrapper86_f316 a h), (publicBatchWrapper86_f314 a h), (publicBatchWrapper86_f315 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2117 a h), by rw [← hd, ← (publicBatchWrapper86_f2118 a h), (publicBatchWrapper86_f2119 a h), kone]⟩


theorem blockAt_32 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 32).isDummy (row a 32).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 32 75) = (row a 32).isDummy := by rw [(publicBatchWrapper86_f316 a h), (publicBatchWrapper86_f314 a h), (publicBatchWrapper86_f315 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18283), a (.virt 18285), a (.virt 18287), a (.virt 18289)],
    ![a (.virt 18284), a (.virt 18286), a (.virt 18288), a (.virt 18290)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2120 a h)
    · exact (publicBatchWrapper86_f2121 a h)
    · exact (publicBatchWrapper86_f2122 a h)
    · exact (publicBatchWrapper86_f2123 a h)
  · show bor (row a 32).isDummy (band (band (a (.virt 18283)) (a (.virt 18285)))
      (band (a (.virt 18287)) (a (.virt 18289)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2124 a h), ← (publicBatchWrapper86_f2125 a h), ← (publicBatchWrapper86_f2126 a h), ← (publicBatchWrapper86_f2127 a h), (publicBatchWrapper86_f2128 a h), kone]


theorem countAt_32 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 32).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 6402)).val = 32
  rw [(publicBatchWrapper86_f32 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_33 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 33) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f317 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f318 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f319 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f320 a h); rw [kzero] at e; exact e


theorem assetAt_33 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 33).isDummy (row a 33).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 34 39) = (row a 33).isDummy := by rw [(publicBatchWrapper86_f323 a h), (publicBatchWrapper86_f321 a h), (publicBatchWrapper86_f322 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2129 a h), by rw [← hd, ← (publicBatchWrapper86_f2130 a h), (publicBatchWrapper86_f2131 a h), kone]⟩


theorem feeAt_33 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 33).isDummy (row a 33).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 34 39) = (row a 33).isDummy := by rw [(publicBatchWrapper86_f323 a h), (publicBatchWrapper86_f321 a h), (publicBatchWrapper86_f322 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2132 a h), by rw [← hd, ← (publicBatchWrapper86_f2133 a h), (publicBatchWrapper86_f2134 a h), kone]⟩


theorem blockAt_33 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 33).isDummy (row a 33).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 34 39) = (row a 33).isDummy := by rw [(publicBatchWrapper86_f323 a h), (publicBatchWrapper86_f321 a h), (publicBatchWrapper86_f322 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18295), a (.virt 18297), a (.virt 18299), a (.virt 18301)],
    ![a (.virt 18296), a (.virt 18298), a (.virt 18300), a (.virt 18302)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2135 a h)
    · exact (publicBatchWrapper86_f2136 a h)
    · exact (publicBatchWrapper86_f2137 a h)
    · exact (publicBatchWrapper86_f2138 a h)
  · show bor (row a 33).isDummy (band (band (a (.virt 18295)) (a (.virt 18297)))
      (band (a (.virt 18299)) (a (.virt 18301)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2139 a h), ← (publicBatchWrapper86_f2140 a h), ← (publicBatchWrapper86_f2141 a h), ← (publicBatchWrapper86_f2142 a h), (publicBatchWrapper86_f2143 a h), kone]


theorem countAt_33 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 33).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 6602)).val = 32
  rw [(publicBatchWrapper86_f33 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_34 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 34) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f324 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f325 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f326 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f327 a h); rw [kzero] at e; exact e


theorem assetAt_34 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 34).isDummy (row a 34).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 35 3) = (row a 34).isDummy := by rw [(publicBatchWrapper86_f330 a h), (publicBatchWrapper86_f328 a h), (publicBatchWrapper86_f329 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2144 a h), by rw [← hd, ← (publicBatchWrapper86_f2145 a h), (publicBatchWrapper86_f2146 a h), kone]⟩


theorem feeAt_34 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 34).isDummy (row a 34).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 35 3) = (row a 34).isDummy := by rw [(publicBatchWrapper86_f330 a h), (publicBatchWrapper86_f328 a h), (publicBatchWrapper86_f329 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2147 a h), by rw [← hd, ← (publicBatchWrapper86_f2148 a h), (publicBatchWrapper86_f2149 a h), kone]⟩


theorem blockAt_34 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 34).isDummy (row a 34).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 35 3) = (row a 34).isDummy := by rw [(publicBatchWrapper86_f330 a h), (publicBatchWrapper86_f328 a h), (publicBatchWrapper86_f329 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18307), a (.virt 18309), a (.virt 18311), a (.virt 18313)],
    ![a (.virt 18308), a (.virt 18310), a (.virt 18312), a (.virt 18314)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2150 a h)
    · exact (publicBatchWrapper86_f2151 a h)
    · exact (publicBatchWrapper86_f2152 a h)
    · exact (publicBatchWrapper86_f2153 a h)
  · show bor (row a 34).isDummy (band (band (a (.virt 18307)) (a (.virt 18309)))
      (band (a (.virt 18311)) (a (.virt 18313)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2154 a h), ← (publicBatchWrapper86_f2155 a h), ← (publicBatchWrapper86_f2156 a h), ← (publicBatchWrapper86_f2157 a h), (publicBatchWrapper86_f2158 a h), kone]


theorem countAt_34 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 34).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 6802)).val = 32
  rw [(publicBatchWrapper86_f34 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_35 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 35) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f331 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f332 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f333 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f334 a h); rw [kzero] at e; exact e


theorem assetAt_35 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 35).isDummy (row a 35).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 35 47) = (row a 35).isDummy := by rw [(publicBatchWrapper86_f337 a h), (publicBatchWrapper86_f335 a h), (publicBatchWrapper86_f336 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2159 a h), by rw [← hd, ← (publicBatchWrapper86_f2160 a h), (publicBatchWrapper86_f2161 a h), kone]⟩


theorem feeAt_35 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 35).isDummy (row a 35).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 35 47) = (row a 35).isDummy := by rw [(publicBatchWrapper86_f337 a h), (publicBatchWrapper86_f335 a h), (publicBatchWrapper86_f336 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2162 a h), by rw [← hd, ← (publicBatchWrapper86_f2163 a h), (publicBatchWrapper86_f2164 a h), kone]⟩


theorem blockAt_35 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 35).isDummy (row a 35).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 35 47) = (row a 35).isDummy := by rw [(publicBatchWrapper86_f337 a h), (publicBatchWrapper86_f335 a h), (publicBatchWrapper86_f336 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18319), a (.virt 18321), a (.virt 18323), a (.virt 18325)],
    ![a (.virt 18320), a (.virt 18322), a (.virt 18324), a (.virt 18326)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2165 a h)
    · exact (publicBatchWrapper86_f2166 a h)
    · exact (publicBatchWrapper86_f2167 a h)
    · exact (publicBatchWrapper86_f2168 a h)
  · show bor (row a 35).isDummy (band (band (a (.virt 18319)) (a (.virt 18321)))
      (band (a (.virt 18323)) (a (.virt 18325)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2169 a h), ← (publicBatchWrapper86_f2170 a h), ← (publicBatchWrapper86_f2171 a h), ← (publicBatchWrapper86_f2172 a h), (publicBatchWrapper86_f2173 a h), kone]


theorem countAt_35 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 35).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 7002)).val = 32
  rw [(publicBatchWrapper86_f35 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_36 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 36) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f338 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f339 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f340 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f341 a h); rw [kzero] at e; exact e


theorem assetAt_36 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 36).isDummy (row a 36).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 37 11) = (row a 36).isDummy := by rw [(publicBatchWrapper86_f344 a h), (publicBatchWrapper86_f342 a h), (publicBatchWrapper86_f343 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2174 a h), by rw [← hd, ← (publicBatchWrapper86_f2175 a h), (publicBatchWrapper86_f2176 a h), kone]⟩


theorem feeAt_36 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 36).isDummy (row a 36).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 37 11) = (row a 36).isDummy := by rw [(publicBatchWrapper86_f344 a h), (publicBatchWrapper86_f342 a h), (publicBatchWrapper86_f343 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2177 a h), by rw [← hd, ← (publicBatchWrapper86_f2178 a h), (publicBatchWrapper86_f2179 a h), kone]⟩


theorem blockAt_36 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 36).isDummy (row a 36).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 37 11) = (row a 36).isDummy := by rw [(publicBatchWrapper86_f344 a h), (publicBatchWrapper86_f342 a h), (publicBatchWrapper86_f343 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18331), a (.virt 18333), a (.virt 18335), a (.virt 18337)],
    ![a (.virt 18332), a (.virt 18334), a (.virt 18336), a (.virt 18338)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2180 a h)
    · exact (publicBatchWrapper86_f2181 a h)
    · exact (publicBatchWrapper86_f2182 a h)
    · exact (publicBatchWrapper86_f2183 a h)
  · show bor (row a 36).isDummy (band (band (a (.virt 18331)) (a (.virt 18333)))
      (band (a (.virt 18335)) (a (.virt 18337)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2184 a h), ← (publicBatchWrapper86_f2185 a h), ← (publicBatchWrapper86_f2186 a h), ← (publicBatchWrapper86_f2187 a h), (publicBatchWrapper86_f2188 a h), kone]


theorem countAt_36 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 36).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 7202)).val = 32
  rw [(publicBatchWrapper86_f36 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_37 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 37) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f345 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f346 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f347 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f348 a h); rw [kzero] at e; exact e


theorem assetAt_37 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 37).isDummy (row a 37).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 37 55) = (row a 37).isDummy := by rw [(publicBatchWrapper86_f351 a h), (publicBatchWrapper86_f349 a h), (publicBatchWrapper86_f350 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2189 a h), by rw [← hd, ← (publicBatchWrapper86_f2190 a h), (publicBatchWrapper86_f2191 a h), kone]⟩


theorem feeAt_37 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 37).isDummy (row a 37).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 37 55) = (row a 37).isDummy := by rw [(publicBatchWrapper86_f351 a h), (publicBatchWrapper86_f349 a h), (publicBatchWrapper86_f350 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2192 a h), by rw [← hd, ← (publicBatchWrapper86_f2193 a h), (publicBatchWrapper86_f2194 a h), kone]⟩


theorem blockAt_37 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 37).isDummy (row a 37).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 37 55) = (row a 37).isDummy := by rw [(publicBatchWrapper86_f351 a h), (publicBatchWrapper86_f349 a h), (publicBatchWrapper86_f350 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18343), a (.virt 18345), a (.virt 18347), a (.virt 18349)],
    ![a (.virt 18344), a (.virt 18346), a (.virt 18348), a (.virt 18350)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2195 a h)
    · exact (publicBatchWrapper86_f2196 a h)
    · exact (publicBatchWrapper86_f2197 a h)
    · exact (publicBatchWrapper86_f2198 a h)
  · show bor (row a 37).isDummy (band (band (a (.virt 18343)) (a (.virt 18345)))
      (band (a (.virt 18347)) (a (.virt 18349)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2199 a h), ← (publicBatchWrapper86_f2200 a h), ← (publicBatchWrapper86_f2201 a h), ← (publicBatchWrapper86_f2202 a h), (publicBatchWrapper86_f2203 a h), kone]


theorem countAt_37 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 37).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 7402)).val = 32
  rw [(publicBatchWrapper86_f37 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_38 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 38) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f352 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f353 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f354 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f355 a h); rw [kzero] at e; exact e


theorem assetAt_38 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 38).isDummy (row a 38).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 39 19) = (row a 38).isDummy := by rw [(publicBatchWrapper86_f358 a h), (publicBatchWrapper86_f356 a h), (publicBatchWrapper86_f357 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2204 a h), by rw [← hd, ← (publicBatchWrapper86_f2205 a h), (publicBatchWrapper86_f2206 a h), kone]⟩


theorem feeAt_38 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 38).isDummy (row a 38).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 39 19) = (row a 38).isDummy := by rw [(publicBatchWrapper86_f358 a h), (publicBatchWrapper86_f356 a h), (publicBatchWrapper86_f357 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2207 a h), by rw [← hd, ← (publicBatchWrapper86_f2208 a h), (publicBatchWrapper86_f2209 a h), kone]⟩


theorem blockAt_38 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 38).isDummy (row a 38).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 39 19) = (row a 38).isDummy := by rw [(publicBatchWrapper86_f358 a h), (publicBatchWrapper86_f356 a h), (publicBatchWrapper86_f357 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18355), a (.virt 18357), a (.virt 18359), a (.virt 18361)],
    ![a (.virt 18356), a (.virt 18358), a (.virt 18360), a (.virt 18362)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2210 a h)
    · exact (publicBatchWrapper86_f2211 a h)
    · exact (publicBatchWrapper86_f2212 a h)
    · exact (publicBatchWrapper86_f2213 a h)
  · show bor (row a 38).isDummy (band (band (a (.virt 18355)) (a (.virt 18357)))
      (band (a (.virt 18359)) (a (.virt 18361)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2214 a h), ← (publicBatchWrapper86_f2215 a h), ← (publicBatchWrapper86_f2216 a h), ← (publicBatchWrapper86_f2217 a h), (publicBatchWrapper86_f2218 a h), kone]


theorem countAt_38 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 38).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 7602)).val = 32
  rw [(publicBatchWrapper86_f38 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_39 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 39) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f359 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f360 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f361 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f362 a h); rw [kzero] at e; exact e


theorem assetAt_39 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 39).isDummy (row a 39).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 39 63) = (row a 39).isDummy := by rw [(publicBatchWrapper86_f365 a h), (publicBatchWrapper86_f363 a h), (publicBatchWrapper86_f364 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2219 a h), by rw [← hd, ← (publicBatchWrapper86_f2220 a h), (publicBatchWrapper86_f2221 a h), kone]⟩


theorem feeAt_39 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 39).isDummy (row a 39).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 39 63) = (row a 39).isDummy := by rw [(publicBatchWrapper86_f365 a h), (publicBatchWrapper86_f363 a h), (publicBatchWrapper86_f364 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2222 a h), by rw [← hd, ← (publicBatchWrapper86_f2223 a h), (publicBatchWrapper86_f2224 a h), kone]⟩


theorem blockAt_39 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 39).isDummy (row a 39).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 39 63) = (row a 39).isDummy := by rw [(publicBatchWrapper86_f365 a h), (publicBatchWrapper86_f363 a h), (publicBatchWrapper86_f364 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18367), a (.virt 18369), a (.virt 18371), a (.virt 18373)],
    ![a (.virt 18368), a (.virt 18370), a (.virt 18372), a (.virt 18374)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2225 a h)
    · exact (publicBatchWrapper86_f2226 a h)
    · exact (publicBatchWrapper86_f2227 a h)
    · exact (publicBatchWrapper86_f2228 a h)
  · show bor (row a 39).isDummy (band (band (a (.virt 18367)) (a (.virt 18369)))
      (band (a (.virt 18371)) (a (.virt 18373)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2229 a h), ← (publicBatchWrapper86_f2230 a h), ← (publicBatchWrapper86_f2231 a h), ← (publicBatchWrapper86_f2232 a h), (publicBatchWrapper86_f2233 a h), kone]


theorem countAt_39 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 39).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 7802)).val = 32
  rw [(publicBatchWrapper86_f39 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_40 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 40) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f366 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f367 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f368 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f369 a h); rw [kzero] at e; exact e


theorem assetAt_40 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 40).isDummy (row a 40).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 41 27) = (row a 40).isDummy := by rw [(publicBatchWrapper86_f372 a h), (publicBatchWrapper86_f370 a h), (publicBatchWrapper86_f371 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2234 a h), by rw [← hd, ← (publicBatchWrapper86_f2235 a h), (publicBatchWrapper86_f2236 a h), kone]⟩


theorem feeAt_40 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 40).isDummy (row a 40).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 41 27) = (row a 40).isDummy := by rw [(publicBatchWrapper86_f372 a h), (publicBatchWrapper86_f370 a h), (publicBatchWrapper86_f371 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2237 a h), by rw [← hd, ← (publicBatchWrapper86_f2238 a h), (publicBatchWrapper86_f2239 a h), kone]⟩


theorem blockAt_40 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 40).isDummy (row a 40).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 41 27) = (row a 40).isDummy := by rw [(publicBatchWrapper86_f372 a h), (publicBatchWrapper86_f370 a h), (publicBatchWrapper86_f371 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18379), a (.virt 18381), a (.virt 18383), a (.virt 18385)],
    ![a (.virt 18380), a (.virt 18382), a (.virt 18384), a (.virt 18386)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2240 a h)
    · exact (publicBatchWrapper86_f2241 a h)
    · exact (publicBatchWrapper86_f2242 a h)
    · exact (publicBatchWrapper86_f2243 a h)
  · show bor (row a 40).isDummy (band (band (a (.virt 18379)) (a (.virt 18381)))
      (band (a (.virt 18383)) (a (.virt 18385)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2244 a h), ← (publicBatchWrapper86_f2245 a h), ← (publicBatchWrapper86_f2246 a h), ← (publicBatchWrapper86_f2247 a h), (publicBatchWrapper86_f2248 a h), kone]


theorem countAt_40 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 40).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 8002)).val = 32
  rw [(publicBatchWrapper86_f40 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_41 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 41) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f373 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f374 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f375 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f376 a h); rw [kzero] at e; exact e


theorem assetAt_41 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 41).isDummy (row a 41).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 41 71) = (row a 41).isDummy := by rw [(publicBatchWrapper86_f379 a h), (publicBatchWrapper86_f377 a h), (publicBatchWrapper86_f378 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2249 a h), by rw [← hd, ← (publicBatchWrapper86_f2250 a h), (publicBatchWrapper86_f2251 a h), kone]⟩


theorem feeAt_41 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 41).isDummy (row a 41).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 41 71) = (row a 41).isDummy := by rw [(publicBatchWrapper86_f379 a h), (publicBatchWrapper86_f377 a h), (publicBatchWrapper86_f378 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2252 a h), by rw [← hd, ← (publicBatchWrapper86_f2253 a h), (publicBatchWrapper86_f2254 a h), kone]⟩


theorem blockAt_41 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 41).isDummy (row a 41).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 41 71) = (row a 41).isDummy := by rw [(publicBatchWrapper86_f379 a h), (publicBatchWrapper86_f377 a h), (publicBatchWrapper86_f378 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18391), a (.virt 18393), a (.virt 18395), a (.virt 18397)],
    ![a (.virt 18392), a (.virt 18394), a (.virt 18396), a (.virt 18398)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2255 a h)
    · exact (publicBatchWrapper86_f2256 a h)
    · exact (publicBatchWrapper86_f2257 a h)
    · exact (publicBatchWrapper86_f2258 a h)
  · show bor (row a 41).isDummy (band (band (a (.virt 18391)) (a (.virt 18393)))
      (band (a (.virt 18395)) (a (.virt 18397)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2259 a h), ← (publicBatchWrapper86_f2260 a h), ← (publicBatchWrapper86_f2261 a h), ← (publicBatchWrapper86_f2262 a h), (publicBatchWrapper86_f2263 a h), kone]


theorem countAt_41 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 41).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 8202)).val = 32
  rw [(publicBatchWrapper86_f41 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_42 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 42) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f380 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f381 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f382 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f383 a h); rw [kzero] at e; exact e


theorem assetAt_42 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 42).isDummy (row a 42).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 42 35) = (row a 42).isDummy := by rw [(publicBatchWrapper86_f386 a h), (publicBatchWrapper86_f384 a h), (publicBatchWrapper86_f385 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2264 a h), by rw [← hd, ← (publicBatchWrapper86_f2265 a h), (publicBatchWrapper86_f2266 a h), kone]⟩


theorem feeAt_42 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 42).isDummy (row a 42).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 42 35) = (row a 42).isDummy := by rw [(publicBatchWrapper86_f386 a h), (publicBatchWrapper86_f384 a h), (publicBatchWrapper86_f385 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2267 a h), by rw [← hd, ← (publicBatchWrapper86_f2268 a h), (publicBatchWrapper86_f2269 a h), kone]⟩


theorem blockAt_42 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 42).isDummy (row a 42).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 42 35) = (row a 42).isDummy := by rw [(publicBatchWrapper86_f386 a h), (publicBatchWrapper86_f384 a h), (publicBatchWrapper86_f385 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18403), a (.virt 18405), a (.virt 18407), a (.virt 18409)],
    ![a (.virt 18404), a (.virt 18406), a (.virt 18408), a (.virt 18410)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2270 a h)
    · exact (publicBatchWrapper86_f2271 a h)
    · exact (publicBatchWrapper86_f2272 a h)
    · exact (publicBatchWrapper86_f2273 a h)
  · show bor (row a 42).isDummy (band (band (a (.virt 18403)) (a (.virt 18405)))
      (band (a (.virt 18407)) (a (.virt 18409)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2274 a h), ← (publicBatchWrapper86_f2275 a h), ← (publicBatchWrapper86_f2276 a h), ← (publicBatchWrapper86_f2277 a h), (publicBatchWrapper86_f2278 a h), kone]


theorem countAt_42 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 42).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 8402)).val = 32
  rw [(publicBatchWrapper86_f42 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_43 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 43) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f387 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f388 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f389 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f390 a h); rw [kzero] at e; exact e


theorem assetAt_43 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 43).isDummy (row a 43).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 42 79) = (row a 43).isDummy := by rw [(publicBatchWrapper86_f393 a h), (publicBatchWrapper86_f391 a h), (publicBatchWrapper86_f392 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2279 a h), by rw [← hd, ← (publicBatchWrapper86_f2280 a h), (publicBatchWrapper86_f2281 a h), kone]⟩


theorem feeAt_43 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 43).isDummy (row a 43).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 42 79) = (row a 43).isDummy := by rw [(publicBatchWrapper86_f393 a h), (publicBatchWrapper86_f391 a h), (publicBatchWrapper86_f392 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2282 a h), by rw [← hd, ← (publicBatchWrapper86_f2283 a h), (publicBatchWrapper86_f2284 a h), kone]⟩


theorem blockAt_43 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 43).isDummy (row a 43).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 42 79) = (row a 43).isDummy := by rw [(publicBatchWrapper86_f393 a h), (publicBatchWrapper86_f391 a h), (publicBatchWrapper86_f392 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18415), a (.virt 18417), a (.virt 18419), a (.virt 18421)],
    ![a (.virt 18416), a (.virt 18418), a (.virt 18420), a (.virt 18422)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2285 a h)
    · exact (publicBatchWrapper86_f2286 a h)
    · exact (publicBatchWrapper86_f2287 a h)
    · exact (publicBatchWrapper86_f2288 a h)
  · show bor (row a 43).isDummy (band (band (a (.virt 18415)) (a (.virt 18417)))
      (band (a (.virt 18419)) (a (.virt 18421)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2289 a h), ← (publicBatchWrapper86_f2290 a h), ← (publicBatchWrapper86_f2291 a h), ← (publicBatchWrapper86_f2292 a h), (publicBatchWrapper86_f2293 a h), kone]


theorem countAt_43 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 43).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 8602)).val = 32
  rw [(publicBatchWrapper86_f43 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_44 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 44) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f394 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f395 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f396 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f397 a h); rw [kzero] at e; exact e


theorem assetAt_44 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 44).isDummy (row a 44).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 44 43) = (row a 44).isDummy := by rw [(publicBatchWrapper86_f400 a h), (publicBatchWrapper86_f398 a h), (publicBatchWrapper86_f399 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2294 a h), by rw [← hd, ← (publicBatchWrapper86_f2295 a h), (publicBatchWrapper86_f2296 a h), kone]⟩


theorem feeAt_44 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 44).isDummy (row a 44).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 44 43) = (row a 44).isDummy := by rw [(publicBatchWrapper86_f400 a h), (publicBatchWrapper86_f398 a h), (publicBatchWrapper86_f399 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2297 a h), by rw [← hd, ← (publicBatchWrapper86_f2298 a h), (publicBatchWrapper86_f2299 a h), kone]⟩


theorem blockAt_44 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 44).isDummy (row a 44).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 44 43) = (row a 44).isDummy := by rw [(publicBatchWrapper86_f400 a h), (publicBatchWrapper86_f398 a h), (publicBatchWrapper86_f399 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18427), a (.virt 18429), a (.virt 18431), a (.virt 18433)],
    ![a (.virt 18428), a (.virt 18430), a (.virt 18432), a (.virt 18434)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2300 a h)
    · exact (publicBatchWrapper86_f2301 a h)
    · exact (publicBatchWrapper86_f2302 a h)
    · exact (publicBatchWrapper86_f2303 a h)
  · show bor (row a 44).isDummy (band (band (a (.virt 18427)) (a (.virt 18429)))
      (band (a (.virt 18431)) (a (.virt 18433)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2304 a h), ← (publicBatchWrapper86_f2305 a h), ← (publicBatchWrapper86_f2306 a h), ← (publicBatchWrapper86_f2307 a h), (publicBatchWrapper86_f2308 a h), kone]


theorem countAt_44 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 44).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 8802)).val = 32
  rw [(publicBatchWrapper86_f44 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_45 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 45) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f401 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f402 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f403 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f404 a h); rw [kzero] at e; exact e


theorem assetAt_45 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 45).isDummy (row a 45).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 46 7) = (row a 45).isDummy := by rw [(publicBatchWrapper86_f407 a h), (publicBatchWrapper86_f405 a h), (publicBatchWrapper86_f406 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2309 a h), by rw [← hd, ← (publicBatchWrapper86_f2310 a h), (publicBatchWrapper86_f2311 a h), kone]⟩


theorem feeAt_45 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 45).isDummy (row a 45).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 46 7) = (row a 45).isDummy := by rw [(publicBatchWrapper86_f407 a h), (publicBatchWrapper86_f405 a h), (publicBatchWrapper86_f406 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2312 a h), by rw [← hd, ← (publicBatchWrapper86_f2313 a h), (publicBatchWrapper86_f2314 a h), kone]⟩


theorem blockAt_45 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 45).isDummy (row a 45).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 46 7) = (row a 45).isDummy := by rw [(publicBatchWrapper86_f407 a h), (publicBatchWrapper86_f405 a h), (publicBatchWrapper86_f406 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18439), a (.virt 18441), a (.virt 18443), a (.virt 18445)],
    ![a (.virt 18440), a (.virt 18442), a (.virt 18444), a (.virt 18446)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2315 a h)
    · exact (publicBatchWrapper86_f2316 a h)
    · exact (publicBatchWrapper86_f2317 a h)
    · exact (publicBatchWrapper86_f2318 a h)
  · show bor (row a 45).isDummy (band (band (a (.virt 18439)) (a (.virt 18441)))
      (band (a (.virt 18443)) (a (.virt 18445)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2319 a h), ← (publicBatchWrapper86_f2320 a h), ← (publicBatchWrapper86_f2321 a h), ← (publicBatchWrapper86_f2322 a h), (publicBatchWrapper86_f2323 a h), kone]


theorem countAt_45 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 45).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 9002)).val = 32
  rw [(publicBatchWrapper86_f45 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_46 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 46) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f408 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f409 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f410 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f411 a h); rw [kzero] at e; exact e


theorem assetAt_46 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 46).isDummy (row a 46).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 46 51) = (row a 46).isDummy := by rw [(publicBatchWrapper86_f414 a h), (publicBatchWrapper86_f412 a h), (publicBatchWrapper86_f413 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2324 a h), by rw [← hd, ← (publicBatchWrapper86_f2325 a h), (publicBatchWrapper86_f2326 a h), kone]⟩


theorem feeAt_46 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 46).isDummy (row a 46).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 46 51) = (row a 46).isDummy := by rw [(publicBatchWrapper86_f414 a h), (publicBatchWrapper86_f412 a h), (publicBatchWrapper86_f413 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2327 a h), by rw [← hd, ← (publicBatchWrapper86_f2328 a h), (publicBatchWrapper86_f2329 a h), kone]⟩


theorem blockAt_46 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 46).isDummy (row a 46).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 46 51) = (row a 46).isDummy := by rw [(publicBatchWrapper86_f414 a h), (publicBatchWrapper86_f412 a h), (publicBatchWrapper86_f413 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18451), a (.virt 18453), a (.virt 18455), a (.virt 18457)],
    ![a (.virt 18452), a (.virt 18454), a (.virt 18456), a (.virt 18458)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2330 a h)
    · exact (publicBatchWrapper86_f2331 a h)
    · exact (publicBatchWrapper86_f2332 a h)
    · exact (publicBatchWrapper86_f2333 a h)
  · show bor (row a 46).isDummy (band (band (a (.virt 18451)) (a (.virt 18453)))
      (band (a (.virt 18455)) (a (.virt 18457)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2334 a h), ← (publicBatchWrapper86_f2335 a h), ← (publicBatchWrapper86_f2336 a h), ← (publicBatchWrapper86_f2337 a h), (publicBatchWrapper86_f2338 a h), kone]


theorem countAt_46 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 46).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 9202)).val = 32
  rw [(publicBatchWrapper86_f46 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_47 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 47) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f415 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f416 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f417 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f418 a h); rw [kzero] at e; exact e


theorem assetAt_47 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 47).isDummy (row a 47).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 48 15) = (row a 47).isDummy := by rw [(publicBatchWrapper86_f421 a h), (publicBatchWrapper86_f419 a h), (publicBatchWrapper86_f420 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2339 a h), by rw [← hd, ← (publicBatchWrapper86_f2340 a h), (publicBatchWrapper86_f2341 a h), kone]⟩


theorem feeAt_47 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 47).isDummy (row a 47).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 48 15) = (row a 47).isDummy := by rw [(publicBatchWrapper86_f421 a h), (publicBatchWrapper86_f419 a h), (publicBatchWrapper86_f420 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2342 a h), by rw [← hd, ← (publicBatchWrapper86_f2343 a h), (publicBatchWrapper86_f2344 a h), kone]⟩


theorem blockAt_47 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 47).isDummy (row a 47).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 48 15) = (row a 47).isDummy := by rw [(publicBatchWrapper86_f421 a h), (publicBatchWrapper86_f419 a h), (publicBatchWrapper86_f420 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18463), a (.virt 18465), a (.virt 18467), a (.virt 18469)],
    ![a (.virt 18464), a (.virt 18466), a (.virt 18468), a (.virt 18470)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2345 a h)
    · exact (publicBatchWrapper86_f2346 a h)
    · exact (publicBatchWrapper86_f2347 a h)
    · exact (publicBatchWrapper86_f2348 a h)
  · show bor (row a 47).isDummy (band (band (a (.virt 18463)) (a (.virt 18465)))
      (band (a (.virt 18467)) (a (.virt 18469)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2349 a h), ← (publicBatchWrapper86_f2350 a h), ← (publicBatchWrapper86_f2351 a h), ← (publicBatchWrapper86_f2352 a h), (publicBatchWrapper86_f2353 a h), kone]


theorem countAt_47 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 47).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 9402)).val = 32
  rw [(publicBatchWrapper86_f47 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_48 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 48) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f422 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f423 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f424 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f425 a h); rw [kzero] at e; exact e


theorem assetAt_48 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 48).isDummy (row a 48).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 48 59) = (row a 48).isDummy := by rw [(publicBatchWrapper86_f428 a h), (publicBatchWrapper86_f426 a h), (publicBatchWrapper86_f427 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2354 a h), by rw [← hd, ← (publicBatchWrapper86_f2355 a h), (publicBatchWrapper86_f2356 a h), kone]⟩


theorem feeAt_48 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 48).isDummy (row a 48).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 48 59) = (row a 48).isDummy := by rw [(publicBatchWrapper86_f428 a h), (publicBatchWrapper86_f426 a h), (publicBatchWrapper86_f427 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2357 a h), by rw [← hd, ← (publicBatchWrapper86_f2358 a h), (publicBatchWrapper86_f2359 a h), kone]⟩


theorem blockAt_48 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 48).isDummy (row a 48).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 48 59) = (row a 48).isDummy := by rw [(publicBatchWrapper86_f428 a h), (publicBatchWrapper86_f426 a h), (publicBatchWrapper86_f427 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18475), a (.virt 18477), a (.virt 18479), a (.virt 18481)],
    ![a (.virt 18476), a (.virt 18478), a (.virt 18480), a (.virt 18482)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2360 a h)
    · exact (publicBatchWrapper86_f2361 a h)
    · exact (publicBatchWrapper86_f2362 a h)
    · exact (publicBatchWrapper86_f2363 a h)
  · show bor (row a 48).isDummy (band (band (a (.virt 18475)) (a (.virt 18477)))
      (band (a (.virt 18479)) (a (.virt 18481)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2364 a h), ← (publicBatchWrapper86_f2365 a h), ← (publicBatchWrapper86_f2366 a h), ← (publicBatchWrapper86_f2367 a h), (publicBatchWrapper86_f2368 a h), kone]


theorem countAt_48 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 48).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 9602)).val = 32
  rw [(publicBatchWrapper86_f48 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_49 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 49) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f429 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f430 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f431 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f432 a h); rw [kzero] at e; exact e


theorem assetAt_49 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 49).isDummy (row a 49).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 49 23) = (row a 49).isDummy := by rw [(publicBatchWrapper86_f435 a h), (publicBatchWrapper86_f433 a h), (publicBatchWrapper86_f434 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2369 a h), by rw [← hd, ← (publicBatchWrapper86_f2370 a h), (publicBatchWrapper86_f2371 a h), kone]⟩


theorem feeAt_49 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 49).isDummy (row a 49).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 49 23) = (row a 49).isDummy := by rw [(publicBatchWrapper86_f435 a h), (publicBatchWrapper86_f433 a h), (publicBatchWrapper86_f434 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2372 a h), by rw [← hd, ← (publicBatchWrapper86_f2373 a h), (publicBatchWrapper86_f2374 a h), kone]⟩


theorem blockAt_49 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 49).isDummy (row a 49).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 49 23) = (row a 49).isDummy := by rw [(publicBatchWrapper86_f435 a h), (publicBatchWrapper86_f433 a h), (publicBatchWrapper86_f434 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18487), a (.virt 18489), a (.virt 18491), a (.virt 18493)],
    ![a (.virt 18488), a (.virt 18490), a (.virt 18492), a (.virt 18494)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2375 a h)
    · exact (publicBatchWrapper86_f2376 a h)
    · exact (publicBatchWrapper86_f2377 a h)
    · exact (publicBatchWrapper86_f2378 a h)
  · show bor (row a 49).isDummy (band (band (a (.virt 18487)) (a (.virt 18489)))
      (band (a (.virt 18491)) (a (.virt 18493)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2379 a h), ← (publicBatchWrapper86_f2380 a h), ← (publicBatchWrapper86_f2381 a h), ← (publicBatchWrapper86_f2382 a h), (publicBatchWrapper86_f2383 a h), kone]


theorem countAt_49 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 49).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 9802)).val = 32
  rw [(publicBatchWrapper86_f49 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_50 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 50) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f436 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f437 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f438 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f439 a h); rw [kzero] at e; exact e


theorem assetAt_50 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 50).isDummy (row a 50).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 49 67) = (row a 50).isDummy := by rw [(publicBatchWrapper86_f442 a h), (publicBatchWrapper86_f440 a h), (publicBatchWrapper86_f441 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2384 a h), by rw [← hd, ← (publicBatchWrapper86_f2385 a h), (publicBatchWrapper86_f2386 a h), kone]⟩


theorem feeAt_50 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 50).isDummy (row a 50).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 49 67) = (row a 50).isDummy := by rw [(publicBatchWrapper86_f442 a h), (publicBatchWrapper86_f440 a h), (publicBatchWrapper86_f441 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2387 a h), by rw [← hd, ← (publicBatchWrapper86_f2388 a h), (publicBatchWrapper86_f2389 a h), kone]⟩


theorem blockAt_50 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 50).isDummy (row a 50).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 49 67) = (row a 50).isDummy := by rw [(publicBatchWrapper86_f442 a h), (publicBatchWrapper86_f440 a h), (publicBatchWrapper86_f441 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18499), a (.virt 18501), a (.virt 18503), a (.virt 18505)],
    ![a (.virt 18500), a (.virt 18502), a (.virt 18504), a (.virt 18506)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2390 a h)
    · exact (publicBatchWrapper86_f2391 a h)
    · exact (publicBatchWrapper86_f2392 a h)
    · exact (publicBatchWrapper86_f2393 a h)
  · show bor (row a 50).isDummy (band (band (a (.virt 18499)) (a (.virt 18501)))
      (band (a (.virt 18503)) (a (.virt 18505)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2394 a h), ← (publicBatchWrapper86_f2395 a h), ← (publicBatchWrapper86_f2396 a h), ← (publicBatchWrapper86_f2397 a h), (publicBatchWrapper86_f2398 a h), kone]


theorem countAt_50 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 50).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 10002)).val = 32
  rw [(publicBatchWrapper86_f50 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_51 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 51) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f443 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f444 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f445 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f446 a h); rw [kzero] at e; exact e


theorem assetAt_51 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 51).isDummy (row a 51).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 51 31) = (row a 51).isDummy := by rw [(publicBatchWrapper86_f449 a h), (publicBatchWrapper86_f447 a h), (publicBatchWrapper86_f448 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2399 a h), by rw [← hd, ← (publicBatchWrapper86_f2400 a h), (publicBatchWrapper86_f2401 a h), kone]⟩


theorem feeAt_51 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 51).isDummy (row a 51).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 51 31) = (row a 51).isDummy := by rw [(publicBatchWrapper86_f449 a h), (publicBatchWrapper86_f447 a h), (publicBatchWrapper86_f448 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2402 a h), by rw [← hd, ← (publicBatchWrapper86_f2403 a h), (publicBatchWrapper86_f2404 a h), kone]⟩


theorem blockAt_51 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 51).isDummy (row a 51).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 51 31) = (row a 51).isDummy := by rw [(publicBatchWrapper86_f449 a h), (publicBatchWrapper86_f447 a h), (publicBatchWrapper86_f448 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18511), a (.virt 18513), a (.virt 18515), a (.virt 18517)],
    ![a (.virt 18512), a (.virt 18514), a (.virt 18516), a (.virt 18518)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2405 a h)
    · exact (publicBatchWrapper86_f2406 a h)
    · exact (publicBatchWrapper86_f2407 a h)
    · exact (publicBatchWrapper86_f2408 a h)
  · show bor (row a 51).isDummy (band (band (a (.virt 18511)) (a (.virt 18513)))
      (band (a (.virt 18515)) (a (.virt 18517)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2409 a h), ← (publicBatchWrapper86_f2410 a h), ← (publicBatchWrapper86_f2411 a h), ← (publicBatchWrapper86_f2412 a h), (publicBatchWrapper86_f2413 a h), kone]


theorem countAt_51 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 51).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 10202)).val = 32
  rw [(publicBatchWrapper86_f51 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_52 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 52) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f450 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f451 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f452 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f453 a h); rw [kzero] at e; exact e


theorem assetAt_52 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 52).isDummy (row a 52).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 51 75) = (row a 52).isDummy := by rw [(publicBatchWrapper86_f456 a h), (publicBatchWrapper86_f454 a h), (publicBatchWrapper86_f455 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2414 a h), by rw [← hd, ← (publicBatchWrapper86_f2415 a h), (publicBatchWrapper86_f2416 a h), kone]⟩


theorem feeAt_52 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 52).isDummy (row a 52).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 51 75) = (row a 52).isDummy := by rw [(publicBatchWrapper86_f456 a h), (publicBatchWrapper86_f454 a h), (publicBatchWrapper86_f455 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2417 a h), by rw [← hd, ← (publicBatchWrapper86_f2418 a h), (publicBatchWrapper86_f2419 a h), kone]⟩


theorem blockAt_52 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 52).isDummy (row a 52).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 51 75) = (row a 52).isDummy := by rw [(publicBatchWrapper86_f456 a h), (publicBatchWrapper86_f454 a h), (publicBatchWrapper86_f455 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18523), a (.virt 18525), a (.virt 18527), a (.virt 18529)],
    ![a (.virt 18524), a (.virt 18526), a (.virt 18528), a (.virt 18530)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2420 a h)
    · exact (publicBatchWrapper86_f2421 a h)
    · exact (publicBatchWrapper86_f2422 a h)
    · exact (publicBatchWrapper86_f2423 a h)
  · show bor (row a 52).isDummy (band (band (a (.virt 18523)) (a (.virt 18525)))
      (band (a (.virt 18527)) (a (.virt 18529)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2424 a h), ← (publicBatchWrapper86_f2425 a h), ← (publicBatchWrapper86_f2426 a h), ← (publicBatchWrapper86_f2427 a h), (publicBatchWrapper86_f2428 a h), kone]


theorem countAt_52 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 52).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 10402)).val = 32
  rw [(publicBatchWrapper86_f52 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_53 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 53) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f457 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f458 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f459 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f460 a h); rw [kzero] at e; exact e


theorem assetAt_53 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 53).isDummy (row a 53).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 53 39) = (row a 53).isDummy := by rw [(publicBatchWrapper86_f463 a h), (publicBatchWrapper86_f461 a h), (publicBatchWrapper86_f462 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2429 a h), by rw [← hd, ← (publicBatchWrapper86_f2430 a h), (publicBatchWrapper86_f2431 a h), kone]⟩


theorem feeAt_53 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 53).isDummy (row a 53).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 53 39) = (row a 53).isDummy := by rw [(publicBatchWrapper86_f463 a h), (publicBatchWrapper86_f461 a h), (publicBatchWrapper86_f462 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2432 a h), by rw [← hd, ← (publicBatchWrapper86_f2433 a h), (publicBatchWrapper86_f2434 a h), kone]⟩


theorem blockAt_53 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 53).isDummy (row a 53).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 53 39) = (row a 53).isDummy := by rw [(publicBatchWrapper86_f463 a h), (publicBatchWrapper86_f461 a h), (publicBatchWrapper86_f462 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18535), a (.virt 18537), a (.virt 18539), a (.virt 18541)],
    ![a (.virt 18536), a (.virt 18538), a (.virt 18540), a (.virt 18542)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2435 a h)
    · exact (publicBatchWrapper86_f2436 a h)
    · exact (publicBatchWrapper86_f2437 a h)
    · exact (publicBatchWrapper86_f2438 a h)
  · show bor (row a 53).isDummy (band (band (a (.virt 18535)) (a (.virt 18537)))
      (band (a (.virt 18539)) (a (.virt 18541)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2439 a h), ← (publicBatchWrapper86_f2440 a h), ← (publicBatchWrapper86_f2441 a h), ← (publicBatchWrapper86_f2442 a h), (publicBatchWrapper86_f2443 a h), kone]


theorem countAt_53 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 53).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 10602)).val = 32
  rw [(publicBatchWrapper86_f53 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_54 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 54) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f464 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f465 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f466 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f467 a h); rw [kzero] at e; exact e


theorem assetAt_54 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 54).isDummy (row a 54).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 54 3) = (row a 54).isDummy := by rw [(publicBatchWrapper86_f470 a h), (publicBatchWrapper86_f468 a h), (publicBatchWrapper86_f469 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2444 a h), by rw [← hd, ← (publicBatchWrapper86_f2445 a h), (publicBatchWrapper86_f2446 a h), kone]⟩


theorem feeAt_54 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 54).isDummy (row a 54).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 54 3) = (row a 54).isDummy := by rw [(publicBatchWrapper86_f470 a h), (publicBatchWrapper86_f468 a h), (publicBatchWrapper86_f469 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2447 a h), by rw [← hd, ← (publicBatchWrapper86_f2448 a h), (publicBatchWrapper86_f2449 a h), kone]⟩


theorem blockAt_54 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 54).isDummy (row a 54).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 54 3) = (row a 54).isDummy := by rw [(publicBatchWrapper86_f470 a h), (publicBatchWrapper86_f468 a h), (publicBatchWrapper86_f469 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18547), a (.virt 18549), a (.virt 18551), a (.virt 18553)],
    ![a (.virt 18548), a (.virt 18550), a (.virt 18552), a (.virt 18554)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2450 a h)
    · exact (publicBatchWrapper86_f2451 a h)
    · exact (publicBatchWrapper86_f2452 a h)
    · exact (publicBatchWrapper86_f2453 a h)
  · show bor (row a 54).isDummy (band (band (a (.virt 18547)) (a (.virt 18549)))
      (band (a (.virt 18551)) (a (.virt 18553)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2454 a h), ← (publicBatchWrapper86_f2455 a h), ← (publicBatchWrapper86_f2456 a h), ← (publicBatchWrapper86_f2457 a h), (publicBatchWrapper86_f2458 a h), kone]


theorem countAt_54 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 54).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 10802)).val = 32
  rw [(publicBatchWrapper86_f54 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_55 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 55) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f471 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f472 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f473 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f474 a h); rw [kzero] at e; exact e


theorem assetAt_55 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 55).isDummy (row a 55).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 54 47) = (row a 55).isDummy := by rw [(publicBatchWrapper86_f477 a h), (publicBatchWrapper86_f475 a h), (publicBatchWrapper86_f476 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2459 a h), by rw [← hd, ← (publicBatchWrapper86_f2460 a h), (publicBatchWrapper86_f2461 a h), kone]⟩


theorem feeAt_55 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 55).isDummy (row a 55).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 54 47) = (row a 55).isDummy := by rw [(publicBatchWrapper86_f477 a h), (publicBatchWrapper86_f475 a h), (publicBatchWrapper86_f476 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2462 a h), by rw [← hd, ← (publicBatchWrapper86_f2463 a h), (publicBatchWrapper86_f2464 a h), kone]⟩


theorem blockAt_55 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 55).isDummy (row a 55).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 54 47) = (row a 55).isDummy := by rw [(publicBatchWrapper86_f477 a h), (publicBatchWrapper86_f475 a h), (publicBatchWrapper86_f476 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18559), a (.virt 18561), a (.virt 18563), a (.virt 18565)],
    ![a (.virt 18560), a (.virt 18562), a (.virt 18564), a (.virt 18566)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2465 a h)
    · exact (publicBatchWrapper86_f2466 a h)
    · exact (publicBatchWrapper86_f2467 a h)
    · exact (publicBatchWrapper86_f2468 a h)
  · show bor (row a 55).isDummy (band (band (a (.virt 18559)) (a (.virt 18561)))
      (band (a (.virt 18563)) (a (.virt 18565)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2469 a h), ← (publicBatchWrapper86_f2470 a h), ← (publicBatchWrapper86_f2471 a h), ← (publicBatchWrapper86_f2472 a h), (publicBatchWrapper86_f2473 a h), kone]


theorem countAt_55 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 55).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 11002)).val = 32
  rw [(publicBatchWrapper86_f55 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_56 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 56) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f478 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f479 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f480 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f481 a h); rw [kzero] at e; exact e


theorem assetAt_56 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 56).isDummy (row a 56).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 56 11) = (row a 56).isDummy := by rw [(publicBatchWrapper86_f484 a h), (publicBatchWrapper86_f482 a h), (publicBatchWrapper86_f483 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2474 a h), by rw [← hd, ← (publicBatchWrapper86_f2475 a h), (publicBatchWrapper86_f2476 a h), kone]⟩


theorem feeAt_56 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 56).isDummy (row a 56).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 56 11) = (row a 56).isDummy := by rw [(publicBatchWrapper86_f484 a h), (publicBatchWrapper86_f482 a h), (publicBatchWrapper86_f483 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2477 a h), by rw [← hd, ← (publicBatchWrapper86_f2478 a h), (publicBatchWrapper86_f2479 a h), kone]⟩


theorem blockAt_56 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 56).isDummy (row a 56).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 56 11) = (row a 56).isDummy := by rw [(publicBatchWrapper86_f484 a h), (publicBatchWrapper86_f482 a h), (publicBatchWrapper86_f483 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18571), a (.virt 18573), a (.virt 18575), a (.virt 18577)],
    ![a (.virt 18572), a (.virt 18574), a (.virt 18576), a (.virt 18578)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2480 a h)
    · exact (publicBatchWrapper86_f2481 a h)
    · exact (publicBatchWrapper86_f2482 a h)
    · exact (publicBatchWrapper86_f2483 a h)
  · show bor (row a 56).isDummy (band (band (a (.virt 18571)) (a (.virt 18573)))
      (band (a (.virt 18575)) (a (.virt 18577)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2484 a h), ← (publicBatchWrapper86_f2485 a h), ← (publicBatchWrapper86_f2486 a h), ← (publicBatchWrapper86_f2487 a h), (publicBatchWrapper86_f2488 a h), kone]


theorem countAt_56 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 56).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 11202)).val = 32
  rw [(publicBatchWrapper86_f56 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_57 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 57) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f485 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f486 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f487 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f488 a h); rw [kzero] at e; exact e


theorem assetAt_57 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 57).isDummy (row a 57).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 56 55) = (row a 57).isDummy := by rw [(publicBatchWrapper86_f491 a h), (publicBatchWrapper86_f489 a h), (publicBatchWrapper86_f490 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2489 a h), by rw [← hd, ← (publicBatchWrapper86_f2490 a h), (publicBatchWrapper86_f2491 a h), kone]⟩


theorem feeAt_57 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 57).isDummy (row a 57).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 56 55) = (row a 57).isDummy := by rw [(publicBatchWrapper86_f491 a h), (publicBatchWrapper86_f489 a h), (publicBatchWrapper86_f490 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2492 a h), by rw [← hd, ← (publicBatchWrapper86_f2493 a h), (publicBatchWrapper86_f2494 a h), kone]⟩


theorem blockAt_57 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 57).isDummy (row a 57).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 56 55) = (row a 57).isDummy := by rw [(publicBatchWrapper86_f491 a h), (publicBatchWrapper86_f489 a h), (publicBatchWrapper86_f490 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18583), a (.virt 18585), a (.virt 18587), a (.virt 18589)],
    ![a (.virt 18584), a (.virt 18586), a (.virt 18588), a (.virt 18590)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2495 a h)
    · exact (publicBatchWrapper86_f2496 a h)
    · exact (publicBatchWrapper86_f2497 a h)
    · exact (publicBatchWrapper86_f2498 a h)
  · show bor (row a 57).isDummy (band (band (a (.virt 18583)) (a (.virt 18585)))
      (band (a (.virt 18587)) (a (.virt 18589)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2499 a h), ← (publicBatchWrapper86_f2500 a h), ← (publicBatchWrapper86_f2501 a h), ← (publicBatchWrapper86_f2502 a h), (publicBatchWrapper86_f2503 a h), kone]


theorem countAt_57 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 57).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 11402)).val = 32
  rw [(publicBatchWrapper86_f57 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_58 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 58) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f492 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f493 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f494 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f495 a h); rw [kzero] at e; exact e


theorem assetAt_58 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 58).isDummy (row a 58).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 58 19) = (row a 58).isDummy := by rw [(publicBatchWrapper86_f498 a h), (publicBatchWrapper86_f496 a h), (publicBatchWrapper86_f497 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2504 a h), by rw [← hd, ← (publicBatchWrapper86_f2505 a h), (publicBatchWrapper86_f2506 a h), kone]⟩


theorem feeAt_58 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 58).isDummy (row a 58).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 58 19) = (row a 58).isDummy := by rw [(publicBatchWrapper86_f498 a h), (publicBatchWrapper86_f496 a h), (publicBatchWrapper86_f497 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2507 a h), by rw [← hd, ← (publicBatchWrapper86_f2508 a h), (publicBatchWrapper86_f2509 a h), kone]⟩


theorem blockAt_58 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 58).isDummy (row a 58).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 58 19) = (row a 58).isDummy := by rw [(publicBatchWrapper86_f498 a h), (publicBatchWrapper86_f496 a h), (publicBatchWrapper86_f497 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18595), a (.virt 18597), a (.virt 18599), a (.virt 18601)],
    ![a (.virt 18596), a (.virt 18598), a (.virt 18600), a (.virt 18602)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2510 a h)
    · exact (publicBatchWrapper86_f2511 a h)
    · exact (publicBatchWrapper86_f2512 a h)
    · exact (publicBatchWrapper86_f2513 a h)
  · show bor (row a 58).isDummy (band (band (a (.virt 18595)) (a (.virt 18597)))
      (band (a (.virt 18599)) (a (.virt 18601)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2514 a h), ← (publicBatchWrapper86_f2515 a h), ← (publicBatchWrapper86_f2516 a h), ← (publicBatchWrapper86_f2517 a h), (publicBatchWrapper86_f2518 a h), kone]


theorem countAt_58 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 58).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 11602)).val = 32
  rw [(publicBatchWrapper86_f58 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_59 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 59) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f499 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f500 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f501 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f502 a h); rw [kzero] at e; exact e


theorem assetAt_59 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 59).isDummy (row a 59).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 58 63) = (row a 59).isDummy := by rw [(publicBatchWrapper86_f505 a h), (publicBatchWrapper86_f503 a h), (publicBatchWrapper86_f504 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2519 a h), by rw [← hd, ← (publicBatchWrapper86_f2520 a h), (publicBatchWrapper86_f2521 a h), kone]⟩


theorem feeAt_59 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 59).isDummy (row a 59).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 58 63) = (row a 59).isDummy := by rw [(publicBatchWrapper86_f505 a h), (publicBatchWrapper86_f503 a h), (publicBatchWrapper86_f504 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2522 a h), by rw [← hd, ← (publicBatchWrapper86_f2523 a h), (publicBatchWrapper86_f2524 a h), kone]⟩


theorem blockAt_59 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 59).isDummy (row a 59).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 58 63) = (row a 59).isDummy := by rw [(publicBatchWrapper86_f505 a h), (publicBatchWrapper86_f503 a h), (publicBatchWrapper86_f504 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18607), a (.virt 18609), a (.virt 18611), a (.virt 18613)],
    ![a (.virt 18608), a (.virt 18610), a (.virt 18612), a (.virt 18614)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2525 a h)
    · exact (publicBatchWrapper86_f2526 a h)
    · exact (publicBatchWrapper86_f2527 a h)
    · exact (publicBatchWrapper86_f2528 a h)
  · show bor (row a 59).isDummy (band (band (a (.virt 18607)) (a (.virt 18609)))
      (band (a (.virt 18611)) (a (.virt 18613)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2529 a h), ← (publicBatchWrapper86_f2530 a h), ← (publicBatchWrapper86_f2531 a h), ← (publicBatchWrapper86_f2532 a h), (publicBatchWrapper86_f2533 a h), kone]


theorem countAt_59 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 59).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 11802)).val = 32
  rw [(publicBatchWrapper86_f59 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_60 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 60) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f506 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f507 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f508 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f509 a h); rw [kzero] at e; exact e


theorem assetAt_60 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 60).isDummy (row a 60).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 60 27) = (row a 60).isDummy := by rw [(publicBatchWrapper86_f512 a h), (publicBatchWrapper86_f510 a h), (publicBatchWrapper86_f511 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2534 a h), by rw [← hd, ← (publicBatchWrapper86_f2535 a h), (publicBatchWrapper86_f2536 a h), kone]⟩


theorem feeAt_60 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 60).isDummy (row a 60).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 60 27) = (row a 60).isDummy := by rw [(publicBatchWrapper86_f512 a h), (publicBatchWrapper86_f510 a h), (publicBatchWrapper86_f511 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2537 a h), by rw [← hd, ← (publicBatchWrapper86_f2538 a h), (publicBatchWrapper86_f2539 a h), kone]⟩


theorem blockAt_60 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 60).isDummy (row a 60).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 60 27) = (row a 60).isDummy := by rw [(publicBatchWrapper86_f512 a h), (publicBatchWrapper86_f510 a h), (publicBatchWrapper86_f511 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18619), a (.virt 18621), a (.virt 18623), a (.virt 18625)],
    ![a (.virt 18620), a (.virt 18622), a (.virt 18624), a (.virt 18626)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2540 a h)
    · exact (publicBatchWrapper86_f2541 a h)
    · exact (publicBatchWrapper86_f2542 a h)
    · exact (publicBatchWrapper86_f2543 a h)
  · show bor (row a 60).isDummy (band (band (a (.virt 18619)) (a (.virt 18621)))
      (band (a (.virt 18623)) (a (.virt 18625)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2544 a h), ← (publicBatchWrapper86_f2545 a h), ← (publicBatchWrapper86_f2546 a h), ← (publicBatchWrapper86_f2547 a h), (publicBatchWrapper86_f2548 a h), kone]


theorem countAt_60 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 60).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 12002)).val = 32
  rw [(publicBatchWrapper86_f60 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_61 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 61) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f513 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f514 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f515 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f516 a h); rw [kzero] at e; exact e


theorem assetAt_61 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 61).isDummy (row a 61).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 60 71) = (row a 61).isDummy := by rw [(publicBatchWrapper86_f519 a h), (publicBatchWrapper86_f517 a h), (publicBatchWrapper86_f518 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2549 a h), by rw [← hd, ← (publicBatchWrapper86_f2550 a h), (publicBatchWrapper86_f2551 a h), kone]⟩


theorem feeAt_61 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 61).isDummy (row a 61).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 60 71) = (row a 61).isDummy := by rw [(publicBatchWrapper86_f519 a h), (publicBatchWrapper86_f517 a h), (publicBatchWrapper86_f518 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2552 a h), by rw [← hd, ← (publicBatchWrapper86_f2553 a h), (publicBatchWrapper86_f2554 a h), kone]⟩


theorem blockAt_61 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 61).isDummy (row a 61).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 60 71) = (row a 61).isDummy := by rw [(publicBatchWrapper86_f519 a h), (publicBatchWrapper86_f517 a h), (publicBatchWrapper86_f518 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18631), a (.virt 18633), a (.virt 18635), a (.virt 18637)],
    ![a (.virt 18632), a (.virt 18634), a (.virt 18636), a (.virt 18638)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2555 a h)
    · exact (publicBatchWrapper86_f2556 a h)
    · exact (publicBatchWrapper86_f2557 a h)
    · exact (publicBatchWrapper86_f2558 a h)
  · show bor (row a 61).isDummy (band (band (a (.virt 18631)) (a (.virt 18633)))
      (band (a (.virt 18635)) (a (.virt 18637)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2559 a h), ← (publicBatchWrapper86_f2560 a h), ← (publicBatchWrapper86_f2561 a h), ← (publicBatchWrapper86_f2562 a h), (publicBatchWrapper86_f2563 a h), kone]


theorem countAt_61 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 61).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 12202)).val = 32
  rw [(publicBatchWrapper86_f61 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_62 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 62) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f520 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f521 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f522 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f523 a h); rw [kzero] at e; exact e


theorem assetAt_62 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 62).isDummy (row a 62).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 61 35) = (row a 62).isDummy := by rw [(publicBatchWrapper86_f526 a h), (publicBatchWrapper86_f524 a h), (publicBatchWrapper86_f525 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2564 a h), by rw [← hd, ← (publicBatchWrapper86_f2565 a h), (publicBatchWrapper86_f2566 a h), kone]⟩


theorem feeAt_62 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 62).isDummy (row a 62).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 61 35) = (row a 62).isDummy := by rw [(publicBatchWrapper86_f526 a h), (publicBatchWrapper86_f524 a h), (publicBatchWrapper86_f525 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2567 a h), by rw [← hd, ← (publicBatchWrapper86_f2568 a h), (publicBatchWrapper86_f2569 a h), kone]⟩


theorem blockAt_62 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 62).isDummy (row a 62).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 61 35) = (row a 62).isDummy := by rw [(publicBatchWrapper86_f526 a h), (publicBatchWrapper86_f524 a h), (publicBatchWrapper86_f525 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18643), a (.virt 18645), a (.virt 18647), a (.virt 18649)],
    ![a (.virt 18644), a (.virt 18646), a (.virt 18648), a (.virt 18650)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2570 a h)
    · exact (publicBatchWrapper86_f2571 a h)
    · exact (publicBatchWrapper86_f2572 a h)
    · exact (publicBatchWrapper86_f2573 a h)
  · show bor (row a 62).isDummy (band (band (a (.virt 18643)) (a (.virt 18645)))
      (band (a (.virt 18647)) (a (.virt 18649)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2574 a h), ← (publicBatchWrapper86_f2575 a h), ← (publicBatchWrapper86_f2576 a h), ← (publicBatchWrapper86_f2577 a h), (publicBatchWrapper86_f2578 a h), kone]


theorem countAt_62 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 62).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 12402)).val = 32
  rw [(publicBatchWrapper86_f62 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_63 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 63) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f527 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f528 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f529 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f530 a h); rw [kzero] at e; exact e


theorem assetAt_63 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 63).isDummy (row a 63).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 61 79) = (row a 63).isDummy := by rw [(publicBatchWrapper86_f533 a h), (publicBatchWrapper86_f531 a h), (publicBatchWrapper86_f532 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2579 a h), by rw [← hd, ← (publicBatchWrapper86_f2580 a h), (publicBatchWrapper86_f2581 a h), kone]⟩


theorem feeAt_63 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 63).isDummy (row a 63).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 61 79) = (row a 63).isDummy := by rw [(publicBatchWrapper86_f533 a h), (publicBatchWrapper86_f531 a h), (publicBatchWrapper86_f532 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2582 a h), by rw [← hd, ← (publicBatchWrapper86_f2583 a h), (publicBatchWrapper86_f2584 a h), kone]⟩


theorem blockAt_63 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 63).isDummy (row a 63).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 61 79) = (row a 63).isDummy := by rw [(publicBatchWrapper86_f533 a h), (publicBatchWrapper86_f531 a h), (publicBatchWrapper86_f532 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18655), a (.virt 18657), a (.virt 18659), a (.virt 18661)],
    ![a (.virt 18656), a (.virt 18658), a (.virt 18660), a (.virt 18662)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2585 a h)
    · exact (publicBatchWrapper86_f2586 a h)
    · exact (publicBatchWrapper86_f2587 a h)
    · exact (publicBatchWrapper86_f2588 a h)
  · show bor (row a 63).isDummy (band (band (a (.virt 18655)) (a (.virt 18657)))
      (band (a (.virt 18659)) (a (.virt 18661)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2589 a h), ← (publicBatchWrapper86_f2590 a h), ← (publicBatchWrapper86_f2591 a h), ← (publicBatchWrapper86_f2592 a h), (publicBatchWrapper86_f2593 a h), kone]


theorem countAt_63 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 63).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 12602)).val = 32
  rw [(publicBatchWrapper86_f63 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_64 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 64) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f534 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f535 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f536 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f537 a h); rw [kzero] at e; exact e


theorem assetAt_64 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 64).isDummy (row a 64).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 63 43) = (row a 64).isDummy := by rw [(publicBatchWrapper86_f540 a h), (publicBatchWrapper86_f538 a h), (publicBatchWrapper86_f539 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2594 a h), by rw [← hd, ← (publicBatchWrapper86_f2595 a h), (publicBatchWrapper86_f2596 a h), kone]⟩


theorem feeAt_64 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 64).isDummy (row a 64).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 63 43) = (row a 64).isDummy := by rw [(publicBatchWrapper86_f540 a h), (publicBatchWrapper86_f538 a h), (publicBatchWrapper86_f539 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2597 a h), by rw [← hd, ← (publicBatchWrapper86_f2598 a h), (publicBatchWrapper86_f2599 a h), kone]⟩


theorem blockAt_64 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 64).isDummy (row a 64).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 63 43) = (row a 64).isDummy := by rw [(publicBatchWrapper86_f540 a h), (publicBatchWrapper86_f538 a h), (publicBatchWrapper86_f539 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18667), a (.virt 18669), a (.virt 18671), a (.virt 18673)],
    ![a (.virt 18668), a (.virt 18670), a (.virt 18672), a (.virt 18674)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2600 a h)
    · exact (publicBatchWrapper86_f2601 a h)
    · exact (publicBatchWrapper86_f2602 a h)
    · exact (publicBatchWrapper86_f2603 a h)
  · show bor (row a 64).isDummy (band (band (a (.virt 18667)) (a (.virt 18669)))
      (band (a (.virt 18671)) (a (.virt 18673)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2604 a h), ← (publicBatchWrapper86_f2605 a h), ← (publicBatchWrapper86_f2606 a h), ← (publicBatchWrapper86_f2607 a h), (publicBatchWrapper86_f2608 a h), kone]


theorem countAt_64 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 64).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 12802)).val = 32
  rw [(publicBatchWrapper86_f64 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_65 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 65) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f541 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f542 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f543 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f544 a h); rw [kzero] at e; exact e


theorem assetAt_65 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 65).isDummy (row a 65).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 65 7) = (row a 65).isDummy := by rw [(publicBatchWrapper86_f547 a h), (publicBatchWrapper86_f545 a h), (publicBatchWrapper86_f546 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2609 a h), by rw [← hd, ← (publicBatchWrapper86_f2610 a h), (publicBatchWrapper86_f2611 a h), kone]⟩


theorem feeAt_65 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 65).isDummy (row a 65).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 65 7) = (row a 65).isDummy := by rw [(publicBatchWrapper86_f547 a h), (publicBatchWrapper86_f545 a h), (publicBatchWrapper86_f546 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2612 a h), by rw [← hd, ← (publicBatchWrapper86_f2613 a h), (publicBatchWrapper86_f2614 a h), kone]⟩


theorem blockAt_65 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 65).isDummy (row a 65).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 65 7) = (row a 65).isDummy := by rw [(publicBatchWrapper86_f547 a h), (publicBatchWrapper86_f545 a h), (publicBatchWrapper86_f546 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18679), a (.virt 18681), a (.virt 18683), a (.virt 18685)],
    ![a (.virt 18680), a (.virt 18682), a (.virt 18684), a (.virt 18686)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2615 a h)
    · exact (publicBatchWrapper86_f2616 a h)
    · exact (publicBatchWrapper86_f2617 a h)
    · exact (publicBatchWrapper86_f2618 a h)
  · show bor (row a 65).isDummy (band (band (a (.virt 18679)) (a (.virt 18681)))
      (band (a (.virt 18683)) (a (.virt 18685)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2619 a h), ← (publicBatchWrapper86_f2620 a h), ← (publicBatchWrapper86_f2621 a h), ← (publicBatchWrapper86_f2622 a h), (publicBatchWrapper86_f2623 a h), kone]


theorem countAt_65 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 65).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 13002)).val = 32
  rw [(publicBatchWrapper86_f65 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_66 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 66) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f548 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f549 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f550 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f551 a h); rw [kzero] at e; exact e


theorem assetAt_66 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 66).isDummy (row a 66).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 65 51) = (row a 66).isDummy := by rw [(publicBatchWrapper86_f554 a h), (publicBatchWrapper86_f552 a h), (publicBatchWrapper86_f553 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2624 a h), by rw [← hd, ← (publicBatchWrapper86_f2625 a h), (publicBatchWrapper86_f2626 a h), kone]⟩


theorem feeAt_66 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 66).isDummy (row a 66).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 65 51) = (row a 66).isDummy := by rw [(publicBatchWrapper86_f554 a h), (publicBatchWrapper86_f552 a h), (publicBatchWrapper86_f553 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2627 a h), by rw [← hd, ← (publicBatchWrapper86_f2628 a h), (publicBatchWrapper86_f2629 a h), kone]⟩


theorem blockAt_66 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 66).isDummy (row a 66).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 65 51) = (row a 66).isDummy := by rw [(publicBatchWrapper86_f554 a h), (publicBatchWrapper86_f552 a h), (publicBatchWrapper86_f553 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18691), a (.virt 18693), a (.virt 18695), a (.virt 18697)],
    ![a (.virt 18692), a (.virt 18694), a (.virt 18696), a (.virt 18698)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2630 a h)
    · exact (publicBatchWrapper86_f2631 a h)
    · exact (publicBatchWrapper86_f2632 a h)
    · exact (publicBatchWrapper86_f2633 a h)
  · show bor (row a 66).isDummy (band (band (a (.virt 18691)) (a (.virt 18693)))
      (band (a (.virt 18695)) (a (.virt 18697)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2634 a h), ← (publicBatchWrapper86_f2635 a h), ← (publicBatchWrapper86_f2636 a h), ← (publicBatchWrapper86_f2637 a h), (publicBatchWrapper86_f2638 a h), kone]


theorem countAt_66 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 66).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 13202)).val = 32
  rw [(publicBatchWrapper86_f66 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_67 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 67) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f555 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f556 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f557 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f558 a h); rw [kzero] at e; exact e


theorem assetAt_67 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 67).isDummy (row a 67).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 67 15) = (row a 67).isDummy := by rw [(publicBatchWrapper86_f561 a h), (publicBatchWrapper86_f559 a h), (publicBatchWrapper86_f560 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2639 a h), by rw [← hd, ← (publicBatchWrapper86_f2640 a h), (publicBatchWrapper86_f2641 a h), kone]⟩


theorem feeAt_67 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 67).isDummy (row a 67).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 67 15) = (row a 67).isDummy := by rw [(publicBatchWrapper86_f561 a h), (publicBatchWrapper86_f559 a h), (publicBatchWrapper86_f560 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2642 a h), by rw [← hd, ← (publicBatchWrapper86_f2643 a h), (publicBatchWrapper86_f2644 a h), kone]⟩


theorem blockAt_67 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 67).isDummy (row a 67).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 67 15) = (row a 67).isDummy := by rw [(publicBatchWrapper86_f561 a h), (publicBatchWrapper86_f559 a h), (publicBatchWrapper86_f560 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18703), a (.virt 18705), a (.virt 18707), a (.virt 18709)],
    ![a (.virt 18704), a (.virt 18706), a (.virt 18708), a (.virt 18710)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2645 a h)
    · exact (publicBatchWrapper86_f2646 a h)
    · exact (publicBatchWrapper86_f2647 a h)
    · exact (publicBatchWrapper86_f2648 a h)
  · show bor (row a 67).isDummy (band (band (a (.virt 18703)) (a (.virt 18705)))
      (band (a (.virt 18707)) (a (.virt 18709)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2649 a h), ← (publicBatchWrapper86_f2650 a h), ← (publicBatchWrapper86_f2651 a h), ← (publicBatchWrapper86_f2652 a h), (publicBatchWrapper86_f2653 a h), kone]


theorem countAt_67 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 67).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 13402)).val = 32
  rw [(publicBatchWrapper86_f67 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_68 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 68) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f562 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f563 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f564 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f565 a h); rw [kzero] at e; exact e


theorem assetAt_68 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 68).isDummy (row a 68).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 67 59) = (row a 68).isDummy := by rw [(publicBatchWrapper86_f568 a h), (publicBatchWrapper86_f566 a h), (publicBatchWrapper86_f567 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2654 a h), by rw [← hd, ← (publicBatchWrapper86_f2655 a h), (publicBatchWrapper86_f2656 a h), kone]⟩


theorem feeAt_68 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 68).isDummy (row a 68).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 67 59) = (row a 68).isDummy := by rw [(publicBatchWrapper86_f568 a h), (publicBatchWrapper86_f566 a h), (publicBatchWrapper86_f567 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2657 a h), by rw [← hd, ← (publicBatchWrapper86_f2658 a h), (publicBatchWrapper86_f2659 a h), kone]⟩


theorem blockAt_68 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 68).isDummy (row a 68).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 67 59) = (row a 68).isDummy := by rw [(publicBatchWrapper86_f568 a h), (publicBatchWrapper86_f566 a h), (publicBatchWrapper86_f567 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18715), a (.virt 18717), a (.virt 18719), a (.virt 18721)],
    ![a (.virt 18716), a (.virt 18718), a (.virt 18720), a (.virt 18722)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2660 a h)
    · exact (publicBatchWrapper86_f2661 a h)
    · exact (publicBatchWrapper86_f2662 a h)
    · exact (publicBatchWrapper86_f2663 a h)
  · show bor (row a 68).isDummy (band (band (a (.virt 18715)) (a (.virt 18717)))
      (band (a (.virt 18719)) (a (.virt 18721)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2664 a h), ← (publicBatchWrapper86_f2665 a h), ← (publicBatchWrapper86_f2666 a h), ← (publicBatchWrapper86_f2667 a h), (publicBatchWrapper86_f2668 a h), kone]


theorem countAt_68 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 68).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 13602)).val = 32
  rw [(publicBatchWrapper86_f68 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_69 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 69) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f569 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f570 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f571 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f572 a h); rw [kzero] at e; exact e


theorem assetAt_69 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 69).isDummy (row a 69).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 68 23) = (row a 69).isDummy := by rw [(publicBatchWrapper86_f575 a h), (publicBatchWrapper86_f573 a h), (publicBatchWrapper86_f574 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2669 a h), by rw [← hd, ← (publicBatchWrapper86_f2670 a h), (publicBatchWrapper86_f2671 a h), kone]⟩


theorem feeAt_69 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 69).isDummy (row a 69).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 68 23) = (row a 69).isDummy := by rw [(publicBatchWrapper86_f575 a h), (publicBatchWrapper86_f573 a h), (publicBatchWrapper86_f574 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2672 a h), by rw [← hd, ← (publicBatchWrapper86_f2673 a h), (publicBatchWrapper86_f2674 a h), kone]⟩


theorem blockAt_69 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 69).isDummy (row a 69).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 68 23) = (row a 69).isDummy := by rw [(publicBatchWrapper86_f575 a h), (publicBatchWrapper86_f573 a h), (publicBatchWrapper86_f574 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18727), a (.virt 18729), a (.virt 18731), a (.virt 18733)],
    ![a (.virt 18728), a (.virt 18730), a (.virt 18732), a (.virt 18734)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2675 a h)
    · exact (publicBatchWrapper86_f2676 a h)
    · exact (publicBatchWrapper86_f2677 a h)
    · exact (publicBatchWrapper86_f2678 a h)
  · show bor (row a 69).isDummy (band (band (a (.virt 18727)) (a (.virt 18729)))
      (band (a (.virt 18731)) (a (.virt 18733)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2679 a h), ← (publicBatchWrapper86_f2680 a h), ← (publicBatchWrapper86_f2681 a h), ← (publicBatchWrapper86_f2682 a h), (publicBatchWrapper86_f2683 a h), kone]


theorem countAt_69 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 69).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 13802)).val = 32
  rw [(publicBatchWrapper86_f69 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_70 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 70) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f576 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f577 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f578 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f579 a h); rw [kzero] at e; exact e


theorem assetAt_70 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 70).isDummy (row a 70).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 68 67) = (row a 70).isDummy := by rw [(publicBatchWrapper86_f582 a h), (publicBatchWrapper86_f580 a h), (publicBatchWrapper86_f581 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2684 a h), by rw [← hd, ← (publicBatchWrapper86_f2685 a h), (publicBatchWrapper86_f2686 a h), kone]⟩


theorem feeAt_70 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 70).isDummy (row a 70).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 68 67) = (row a 70).isDummy := by rw [(publicBatchWrapper86_f582 a h), (publicBatchWrapper86_f580 a h), (publicBatchWrapper86_f581 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2687 a h), by rw [← hd, ← (publicBatchWrapper86_f2688 a h), (publicBatchWrapper86_f2689 a h), kone]⟩


theorem blockAt_70 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 70).isDummy (row a 70).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 68 67) = (row a 70).isDummy := by rw [(publicBatchWrapper86_f582 a h), (publicBatchWrapper86_f580 a h), (publicBatchWrapper86_f581 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18739), a (.virt 18741), a (.virt 18743), a (.virt 18745)],
    ![a (.virt 18740), a (.virt 18742), a (.virt 18744), a (.virt 18746)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2690 a h)
    · exact (publicBatchWrapper86_f2691 a h)
    · exact (publicBatchWrapper86_f2692 a h)
    · exact (publicBatchWrapper86_f2693 a h)
  · show bor (row a 70).isDummy (band (band (a (.virt 18739)) (a (.virt 18741)))
      (band (a (.virt 18743)) (a (.virt 18745)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2694 a h), ← (publicBatchWrapper86_f2695 a h), ← (publicBatchWrapper86_f2696 a h), ← (publicBatchWrapper86_f2697 a h), (publicBatchWrapper86_f2698 a h), kone]


theorem countAt_70 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 70).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 14002)).val = 32
  rw [(publicBatchWrapper86_f70 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_71 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 71) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f583 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f584 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f585 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f586 a h); rw [kzero] at e; exact e


theorem assetAt_71 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 71).isDummy (row a 71).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 70 31) = (row a 71).isDummy := by rw [(publicBatchWrapper86_f589 a h), (publicBatchWrapper86_f587 a h), (publicBatchWrapper86_f588 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2699 a h), by rw [← hd, ← (publicBatchWrapper86_f2700 a h), (publicBatchWrapper86_f2701 a h), kone]⟩


theorem feeAt_71 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 71).isDummy (row a 71).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 70 31) = (row a 71).isDummy := by rw [(publicBatchWrapper86_f589 a h), (publicBatchWrapper86_f587 a h), (publicBatchWrapper86_f588 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2702 a h), by rw [← hd, ← (publicBatchWrapper86_f2703 a h), (publicBatchWrapper86_f2704 a h), kone]⟩


theorem blockAt_71 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 71).isDummy (row a 71).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 70 31) = (row a 71).isDummy := by rw [(publicBatchWrapper86_f589 a h), (publicBatchWrapper86_f587 a h), (publicBatchWrapper86_f588 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18751), a (.virt 18753), a (.virt 18755), a (.virt 18757)],
    ![a (.virt 18752), a (.virt 18754), a (.virt 18756), a (.virt 18758)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2705 a h)
    · exact (publicBatchWrapper86_f2706 a h)
    · exact (publicBatchWrapper86_f2707 a h)
    · exact (publicBatchWrapper86_f2708 a h)
  · show bor (row a 71).isDummy (band (band (a (.virt 18751)) (a (.virt 18753)))
      (band (a (.virt 18755)) (a (.virt 18757)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2709 a h), ← (publicBatchWrapper86_f2710 a h), ← (publicBatchWrapper86_f2711 a h), ← (publicBatchWrapper86_f2712 a h), (publicBatchWrapper86_f2713 a h), kone]


theorem countAt_71 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 71).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 14202)).val = 32
  rw [(publicBatchWrapper86_f71 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_72 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 72) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f590 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f591 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f592 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f593 a h); rw [kzero] at e; exact e


theorem assetAt_72 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 72).isDummy (row a 72).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 70 75) = (row a 72).isDummy := by rw [(publicBatchWrapper86_f596 a h), (publicBatchWrapper86_f594 a h), (publicBatchWrapper86_f595 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2714 a h), by rw [← hd, ← (publicBatchWrapper86_f2715 a h), (publicBatchWrapper86_f2716 a h), kone]⟩


theorem feeAt_72 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 72).isDummy (row a 72).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 70 75) = (row a 72).isDummy := by rw [(publicBatchWrapper86_f596 a h), (publicBatchWrapper86_f594 a h), (publicBatchWrapper86_f595 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2717 a h), by rw [← hd, ← (publicBatchWrapper86_f2718 a h), (publicBatchWrapper86_f2719 a h), kone]⟩


theorem blockAt_72 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 72).isDummy (row a 72).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 70 75) = (row a 72).isDummy := by rw [(publicBatchWrapper86_f596 a h), (publicBatchWrapper86_f594 a h), (publicBatchWrapper86_f595 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18763), a (.virt 18765), a (.virt 18767), a (.virt 18769)],
    ![a (.virt 18764), a (.virt 18766), a (.virt 18768), a (.virt 18770)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2720 a h)
    · exact (publicBatchWrapper86_f2721 a h)
    · exact (publicBatchWrapper86_f2722 a h)
    · exact (publicBatchWrapper86_f2723 a h)
  · show bor (row a 72).isDummy (band (band (a (.virt 18763)) (a (.virt 18765)))
      (band (a (.virt 18767)) (a (.virt 18769)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2724 a h), ← (publicBatchWrapper86_f2725 a h), ← (publicBatchWrapper86_f2726 a h), ← (publicBatchWrapper86_f2727 a h), (publicBatchWrapper86_f2728 a h), kone]


theorem countAt_72 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 72).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 14402)).val = 32
  rw [(publicBatchWrapper86_f72 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_73 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 73) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f597 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f598 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f599 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f600 a h); rw [kzero] at e; exact e


theorem assetAt_73 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 73).isDummy (row a 73).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 72 39) = (row a 73).isDummy := by rw [(publicBatchWrapper86_f603 a h), (publicBatchWrapper86_f601 a h), (publicBatchWrapper86_f602 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2729 a h), by rw [← hd, ← (publicBatchWrapper86_f2730 a h), (publicBatchWrapper86_f2731 a h), kone]⟩


theorem feeAt_73 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 73).isDummy (row a 73).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 72 39) = (row a 73).isDummy := by rw [(publicBatchWrapper86_f603 a h), (publicBatchWrapper86_f601 a h), (publicBatchWrapper86_f602 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2732 a h), by rw [← hd, ← (publicBatchWrapper86_f2733 a h), (publicBatchWrapper86_f2734 a h), kone]⟩


theorem blockAt_73 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 73).isDummy (row a 73).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 72 39) = (row a 73).isDummy := by rw [(publicBatchWrapper86_f603 a h), (publicBatchWrapper86_f601 a h), (publicBatchWrapper86_f602 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18775), a (.virt 18777), a (.virt 18779), a (.virt 18781)],
    ![a (.virt 18776), a (.virt 18778), a (.virt 18780), a (.virt 18782)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2735 a h)
    · exact (publicBatchWrapper86_f2736 a h)
    · exact (publicBatchWrapper86_f2737 a h)
    · exact (publicBatchWrapper86_f2738 a h)
  · show bor (row a 73).isDummy (band (band (a (.virt 18775)) (a (.virt 18777)))
      (band (a (.virt 18779)) (a (.virt 18781)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2739 a h), ← (publicBatchWrapper86_f2740 a h), ← (publicBatchWrapper86_f2741 a h), ← (publicBatchWrapper86_f2742 a h), (publicBatchWrapper86_f2743 a h), kone]


theorem countAt_73 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 73).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 14602)).val = 32
  rw [(publicBatchWrapper86_f73 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_74 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 74) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f604 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f605 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f606 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f607 a h); rw [kzero] at e; exact e


theorem assetAt_74 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 74).isDummy (row a 74).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 73 3) = (row a 74).isDummy := by rw [(publicBatchWrapper86_f610 a h), (publicBatchWrapper86_f608 a h), (publicBatchWrapper86_f609 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2744 a h), by rw [← hd, ← (publicBatchWrapper86_f2745 a h), (publicBatchWrapper86_f2746 a h), kone]⟩


theorem feeAt_74 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 74).isDummy (row a 74).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 73 3) = (row a 74).isDummy := by rw [(publicBatchWrapper86_f610 a h), (publicBatchWrapper86_f608 a h), (publicBatchWrapper86_f609 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2747 a h), by rw [← hd, ← (publicBatchWrapper86_f2748 a h), (publicBatchWrapper86_f2749 a h), kone]⟩


theorem blockAt_74 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 74).isDummy (row a 74).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 73 3) = (row a 74).isDummy := by rw [(publicBatchWrapper86_f610 a h), (publicBatchWrapper86_f608 a h), (publicBatchWrapper86_f609 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18787), a (.virt 18789), a (.virt 18791), a (.virt 18793)],
    ![a (.virt 18788), a (.virt 18790), a (.virt 18792), a (.virt 18794)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2750 a h)
    · exact (publicBatchWrapper86_f2751 a h)
    · exact (publicBatchWrapper86_f2752 a h)
    · exact (publicBatchWrapper86_f2753 a h)
  · show bor (row a 74).isDummy (band (band (a (.virt 18787)) (a (.virt 18789)))
      (band (a (.virt 18791)) (a (.virt 18793)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2754 a h), ← (publicBatchWrapper86_f2755 a h), ← (publicBatchWrapper86_f2756 a h), ← (publicBatchWrapper86_f2757 a h), (publicBatchWrapper86_f2758 a h), kone]


theorem countAt_74 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 74).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 14802)).val = 32
  rw [(publicBatchWrapper86_f74 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_75 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 75) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f611 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f612 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f613 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f614 a h); rw [kzero] at e; exact e


theorem assetAt_75 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 75).isDummy (row a 75).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 73 47) = (row a 75).isDummy := by rw [(publicBatchWrapper86_f617 a h), (publicBatchWrapper86_f615 a h), (publicBatchWrapper86_f616 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2759 a h), by rw [← hd, ← (publicBatchWrapper86_f2760 a h), (publicBatchWrapper86_f2761 a h), kone]⟩


theorem feeAt_75 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 75).isDummy (row a 75).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 73 47) = (row a 75).isDummy := by rw [(publicBatchWrapper86_f617 a h), (publicBatchWrapper86_f615 a h), (publicBatchWrapper86_f616 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2762 a h), by rw [← hd, ← (publicBatchWrapper86_f2763 a h), (publicBatchWrapper86_f2764 a h), kone]⟩


theorem blockAt_75 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 75).isDummy (row a 75).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 73 47) = (row a 75).isDummy := by rw [(publicBatchWrapper86_f617 a h), (publicBatchWrapper86_f615 a h), (publicBatchWrapper86_f616 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18799), a (.virt 18801), a (.virt 18803), a (.virt 18805)],
    ![a (.virt 18800), a (.virt 18802), a (.virt 18804), a (.virt 18806)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2765 a h)
    · exact (publicBatchWrapper86_f2766 a h)
    · exact (publicBatchWrapper86_f2767 a h)
    · exact (publicBatchWrapper86_f2768 a h)
  · show bor (row a 75).isDummy (band (band (a (.virt 18799)) (a (.virt 18801)))
      (band (a (.virt 18803)) (a (.virt 18805)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2769 a h), ← (publicBatchWrapper86_f2770 a h), ← (publicBatchWrapper86_f2771 a h), ← (publicBatchWrapper86_f2772 a h), (publicBatchWrapper86_f2773 a h), kone]


theorem countAt_75 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 75).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 15002)).val = 32
  rw [(publicBatchWrapper86_f75 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_76 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 76) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f618 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f619 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f620 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f621 a h); rw [kzero] at e; exact e


theorem assetAt_76 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 76).isDummy (row a 76).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 75 11) = (row a 76).isDummy := by rw [(publicBatchWrapper86_f624 a h), (publicBatchWrapper86_f622 a h), (publicBatchWrapper86_f623 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2774 a h), by rw [← hd, ← (publicBatchWrapper86_f2775 a h), (publicBatchWrapper86_f2776 a h), kone]⟩


theorem feeAt_76 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 76).isDummy (row a 76).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 75 11) = (row a 76).isDummy := by rw [(publicBatchWrapper86_f624 a h), (publicBatchWrapper86_f622 a h), (publicBatchWrapper86_f623 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2777 a h), by rw [← hd, ← (publicBatchWrapper86_f2778 a h), (publicBatchWrapper86_f2779 a h), kone]⟩


theorem blockAt_76 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 76).isDummy (row a 76).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 75 11) = (row a 76).isDummy := by rw [(publicBatchWrapper86_f624 a h), (publicBatchWrapper86_f622 a h), (publicBatchWrapper86_f623 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18811), a (.virt 18813), a (.virt 18815), a (.virt 18817)],
    ![a (.virt 18812), a (.virt 18814), a (.virt 18816), a (.virt 18818)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2780 a h)
    · exact (publicBatchWrapper86_f2781 a h)
    · exact (publicBatchWrapper86_f2782 a h)
    · exact (publicBatchWrapper86_f2783 a h)
  · show bor (row a 76).isDummy (band (band (a (.virt 18811)) (a (.virt 18813)))
      (band (a (.virt 18815)) (a (.virt 18817)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2784 a h), ← (publicBatchWrapper86_f2785 a h), ← (publicBatchWrapper86_f2786 a h), ← (publicBatchWrapper86_f2787 a h), (publicBatchWrapper86_f2788 a h), kone]


theorem countAt_76 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 76).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 15202)).val = 32
  rw [(publicBatchWrapper86_f76 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_77 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 77) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f625 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f626 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f627 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f628 a h); rw [kzero] at e; exact e


theorem assetAt_77 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 77).isDummy (row a 77).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 75 55) = (row a 77).isDummy := by rw [(publicBatchWrapper86_f631 a h), (publicBatchWrapper86_f629 a h), (publicBatchWrapper86_f630 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2789 a h), by rw [← hd, ← (publicBatchWrapper86_f2790 a h), (publicBatchWrapper86_f2791 a h), kone]⟩


theorem feeAt_77 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 77).isDummy (row a 77).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 75 55) = (row a 77).isDummy := by rw [(publicBatchWrapper86_f631 a h), (publicBatchWrapper86_f629 a h), (publicBatchWrapper86_f630 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2792 a h), by rw [← hd, ← (publicBatchWrapper86_f2793 a h), (publicBatchWrapper86_f2794 a h), kone]⟩


theorem blockAt_77 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 77).isDummy (row a 77).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 75 55) = (row a 77).isDummy := by rw [(publicBatchWrapper86_f631 a h), (publicBatchWrapper86_f629 a h), (publicBatchWrapper86_f630 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18823), a (.virt 18825), a (.virt 18827), a (.virt 18829)],
    ![a (.virt 18824), a (.virt 18826), a (.virt 18828), a (.virt 18830)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2795 a h)
    · exact (publicBatchWrapper86_f2796 a h)
    · exact (publicBatchWrapper86_f2797 a h)
    · exact (publicBatchWrapper86_f2798 a h)
  · show bor (row a 77).isDummy (band (band (a (.virt 18823)) (a (.virt 18825)))
      (band (a (.virt 18827)) (a (.virt 18829)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2799 a h), ← (publicBatchWrapper86_f2800 a h), ← (publicBatchWrapper86_f2801 a h), ← (publicBatchWrapper86_f2802 a h), (publicBatchWrapper86_f2803 a h), kone]


theorem countAt_77 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 77).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 15402)).val = 32
  rw [(publicBatchWrapper86_f77 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_78 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 78) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f632 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f633 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f634 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f635 a h); rw [kzero] at e; exact e


theorem assetAt_78 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 78).isDummy (row a 78).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 77 19) = (row a 78).isDummy := by rw [(publicBatchWrapper86_f638 a h), (publicBatchWrapper86_f636 a h), (publicBatchWrapper86_f637 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2804 a h), by rw [← hd, ← (publicBatchWrapper86_f2805 a h), (publicBatchWrapper86_f2806 a h), kone]⟩


theorem feeAt_78 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 78).isDummy (row a 78).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 77 19) = (row a 78).isDummy := by rw [(publicBatchWrapper86_f638 a h), (publicBatchWrapper86_f636 a h), (publicBatchWrapper86_f637 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2807 a h), by rw [← hd, ← (publicBatchWrapper86_f2808 a h), (publicBatchWrapper86_f2809 a h), kone]⟩


theorem blockAt_78 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 78).isDummy (row a 78).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 77 19) = (row a 78).isDummy := by rw [(publicBatchWrapper86_f638 a h), (publicBatchWrapper86_f636 a h), (publicBatchWrapper86_f637 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18835), a (.virt 18837), a (.virt 18839), a (.virt 18841)],
    ![a (.virt 18836), a (.virt 18838), a (.virt 18840), a (.virt 18842)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2810 a h)
    · exact (publicBatchWrapper86_f2811 a h)
    · exact (publicBatchWrapper86_f2812 a h)
    · exact (publicBatchWrapper86_f2813 a h)
  · show bor (row a 78).isDummy (band (band (a (.virt 18835)) (a (.virt 18837)))
      (band (a (.virt 18839)) (a (.virt 18841)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2814 a h), ← (publicBatchWrapper86_f2815 a h), ← (publicBatchWrapper86_f2816 a h), ← (publicBatchWrapper86_f2817 a h), (publicBatchWrapper86_f2818 a h), kone]


theorem countAt_78 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 78).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 15602)).val = 32
  rw [(publicBatchWrapper86_f78 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_79 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 79) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f639 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f640 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f641 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f642 a h); rw [kzero] at e; exact e


theorem assetAt_79 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 79).isDummy (row a 79).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 77 63) = (row a 79).isDummy := by rw [(publicBatchWrapper86_f645 a h), (publicBatchWrapper86_f643 a h), (publicBatchWrapper86_f644 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2819 a h), by rw [← hd, ← (publicBatchWrapper86_f2820 a h), (publicBatchWrapper86_f2821 a h), kone]⟩


theorem feeAt_79 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 79).isDummy (row a 79).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 77 63) = (row a 79).isDummy := by rw [(publicBatchWrapper86_f645 a h), (publicBatchWrapper86_f643 a h), (publicBatchWrapper86_f644 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2822 a h), by rw [← hd, ← (publicBatchWrapper86_f2823 a h), (publicBatchWrapper86_f2824 a h), kone]⟩


theorem blockAt_79 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 79).isDummy (row a 79).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 77 63) = (row a 79).isDummy := by rw [(publicBatchWrapper86_f645 a h), (publicBatchWrapper86_f643 a h), (publicBatchWrapper86_f644 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18847), a (.virt 18849), a (.virt 18851), a (.virt 18853)],
    ![a (.virt 18848), a (.virt 18850), a (.virt 18852), a (.virt 18854)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2825 a h)
    · exact (publicBatchWrapper86_f2826 a h)
    · exact (publicBatchWrapper86_f2827 a h)
    · exact (publicBatchWrapper86_f2828 a h)
  · show bor (row a 79).isDummy (band (band (a (.virt 18847)) (a (.virt 18849)))
      (band (a (.virt 18851)) (a (.virt 18853)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2829 a h), ← (publicBatchWrapper86_f2830 a h), ← (publicBatchWrapper86_f2831 a h), ← (publicBatchWrapper86_f2832 a h), (publicBatchWrapper86_f2833 a h), kone]


theorem countAt_79 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 79).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 15802)).val = 32
  rw [(publicBatchWrapper86_f79 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_80 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 80) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f646 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f647 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f648 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f649 a h); rw [kzero] at e; exact e


theorem assetAt_80 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 80).isDummy (row a 80).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 79 27) = (row a 80).isDummy := by rw [(publicBatchWrapper86_f652 a h), (publicBatchWrapper86_f650 a h), (publicBatchWrapper86_f651 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2834 a h), by rw [← hd, ← (publicBatchWrapper86_f2835 a h), (publicBatchWrapper86_f2836 a h), kone]⟩


theorem feeAt_80 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 80).isDummy (row a 80).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 79 27) = (row a 80).isDummy := by rw [(publicBatchWrapper86_f652 a h), (publicBatchWrapper86_f650 a h), (publicBatchWrapper86_f651 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2837 a h), by rw [← hd, ← (publicBatchWrapper86_f2838 a h), (publicBatchWrapper86_f2839 a h), kone]⟩


theorem blockAt_80 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 80).isDummy (row a 80).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 79 27) = (row a 80).isDummy := by rw [(publicBatchWrapper86_f652 a h), (publicBatchWrapper86_f650 a h), (publicBatchWrapper86_f651 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18859), a (.virt 18861), a (.virt 18863), a (.virt 18865)],
    ![a (.virt 18860), a (.virt 18862), a (.virt 18864), a (.virt 18866)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2840 a h)
    · exact (publicBatchWrapper86_f2841 a h)
    · exact (publicBatchWrapper86_f2842 a h)
    · exact (publicBatchWrapper86_f2843 a h)
  · show bor (row a 80).isDummy (band (band (a (.virt 18859)) (a (.virt 18861)))
      (band (a (.virt 18863)) (a (.virt 18865)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2844 a h), ← (publicBatchWrapper86_f2845 a h), ← (publicBatchWrapper86_f2846 a h), ← (publicBatchWrapper86_f2847 a h), (publicBatchWrapper86_f2848 a h), kone]


theorem countAt_80 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 80).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 16002)).val = 32
  rw [(publicBatchWrapper86_f80 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_81 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 81) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f653 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f654 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f655 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f656 a h); rw [kzero] at e; exact e


theorem assetAt_81 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 81).isDummy (row a 81).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 79 71) = (row a 81).isDummy := by rw [(publicBatchWrapper86_f659 a h), (publicBatchWrapper86_f657 a h), (publicBatchWrapper86_f658 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2849 a h), by rw [← hd, ← (publicBatchWrapper86_f2850 a h), (publicBatchWrapper86_f2851 a h), kone]⟩


theorem feeAt_81 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 81).isDummy (row a 81).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 79 71) = (row a 81).isDummy := by rw [(publicBatchWrapper86_f659 a h), (publicBatchWrapper86_f657 a h), (publicBatchWrapper86_f658 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2852 a h), by rw [← hd, ← (publicBatchWrapper86_f2853 a h), (publicBatchWrapper86_f2854 a h), kone]⟩


theorem blockAt_81 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 81).isDummy (row a 81).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 79 71) = (row a 81).isDummy := by rw [(publicBatchWrapper86_f659 a h), (publicBatchWrapper86_f657 a h), (publicBatchWrapper86_f658 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18871), a (.virt 18873), a (.virt 18875), a (.virt 18877)],
    ![a (.virt 18872), a (.virt 18874), a (.virt 18876), a (.virt 18878)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2855 a h)
    · exact (publicBatchWrapper86_f2856 a h)
    · exact (publicBatchWrapper86_f2857 a h)
    · exact (publicBatchWrapper86_f2858 a h)
  · show bor (row a 81).isDummy (band (band (a (.virt 18871)) (a (.virt 18873)))
      (band (a (.virt 18875)) (a (.virt 18877)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2859 a h), ← (publicBatchWrapper86_f2860 a h), ← (publicBatchWrapper86_f2861 a h), ← (publicBatchWrapper86_f2862 a h), (publicBatchWrapper86_f2863 a h), kone]


theorem countAt_81 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 81).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 16202)).val = 32
  rw [(publicBatchWrapper86_f81 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_82 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 82) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f660 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f661 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f662 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f663 a h); rw [kzero] at e; exact e


theorem assetAt_82 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 82).isDummy (row a 82).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 80 35) = (row a 82).isDummy := by rw [(publicBatchWrapper86_f666 a h), (publicBatchWrapper86_f664 a h), (publicBatchWrapper86_f665 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2864 a h), by rw [← hd, ← (publicBatchWrapper86_f2865 a h), (publicBatchWrapper86_f2866 a h), kone]⟩


theorem feeAt_82 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 82).isDummy (row a 82).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 80 35) = (row a 82).isDummy := by rw [(publicBatchWrapper86_f666 a h), (publicBatchWrapper86_f664 a h), (publicBatchWrapper86_f665 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2867 a h), by rw [← hd, ← (publicBatchWrapper86_f2868 a h), (publicBatchWrapper86_f2869 a h), kone]⟩


theorem blockAt_82 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 82).isDummy (row a 82).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 80 35) = (row a 82).isDummy := by rw [(publicBatchWrapper86_f666 a h), (publicBatchWrapper86_f664 a h), (publicBatchWrapper86_f665 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18883), a (.virt 18885), a (.virt 18887), a (.virt 18889)],
    ![a (.virt 18884), a (.virt 18886), a (.virt 18888), a (.virt 18890)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2870 a h)
    · exact (publicBatchWrapper86_f2871 a h)
    · exact (publicBatchWrapper86_f2872 a h)
    · exact (publicBatchWrapper86_f2873 a h)
  · show bor (row a 82).isDummy (band (band (a (.virt 18883)) (a (.virt 18885)))
      (band (a (.virt 18887)) (a (.virt 18889)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2874 a h), ← (publicBatchWrapper86_f2875 a h), ← (publicBatchWrapper86_f2876 a h), ← (publicBatchWrapper86_f2877 a h), (publicBatchWrapper86_f2878 a h), kone]


theorem countAt_82 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 82).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 16402)).val = 32
  rw [(publicBatchWrapper86_f82 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_83 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 83) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f667 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f668 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f669 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f670 a h); rw [kzero] at e; exact e


theorem assetAt_83 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 83).isDummy (row a 83).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 80 79) = (row a 83).isDummy := by rw [(publicBatchWrapper86_f673 a h), (publicBatchWrapper86_f671 a h), (publicBatchWrapper86_f672 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2879 a h), by rw [← hd, ← (publicBatchWrapper86_f2880 a h), (publicBatchWrapper86_f2881 a h), kone]⟩


theorem feeAt_83 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 83).isDummy (row a 83).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 80 79) = (row a 83).isDummy := by rw [(publicBatchWrapper86_f673 a h), (publicBatchWrapper86_f671 a h), (publicBatchWrapper86_f672 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2882 a h), by rw [← hd, ← (publicBatchWrapper86_f2883 a h), (publicBatchWrapper86_f2884 a h), kone]⟩


theorem blockAt_83 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 83).isDummy (row a 83).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 80 79) = (row a 83).isDummy := by rw [(publicBatchWrapper86_f673 a h), (publicBatchWrapper86_f671 a h), (publicBatchWrapper86_f672 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18895), a (.virt 18897), a (.virt 18899), a (.virt 18901)],
    ![a (.virt 18896), a (.virt 18898), a (.virt 18900), a (.virt 18902)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2885 a h)
    · exact (publicBatchWrapper86_f2886 a h)
    · exact (publicBatchWrapper86_f2887 a h)
    · exact (publicBatchWrapper86_f2888 a h)
  · show bor (row a 83).isDummy (band (band (a (.virt 18895)) (a (.virt 18897)))
      (band (a (.virt 18899)) (a (.virt 18901)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2889 a h), ← (publicBatchWrapper86_f2890 a h), ← (publicBatchWrapper86_f2891 a h), ← (publicBatchWrapper86_f2892 a h), (publicBatchWrapper86_f2893 a h), kone]


theorem countAt_83 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 83).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 16602)).val = 32
  rw [(publicBatchWrapper86_f83 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_84 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 84) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f674 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f675 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f676 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f677 a h); rw [kzero] at e; exact e


theorem assetAt_84 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 84).isDummy (row a 84).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 82 43) = (row a 84).isDummy := by rw [(publicBatchWrapper86_f680 a h), (publicBatchWrapper86_f678 a h), (publicBatchWrapper86_f679 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2894 a h), by rw [← hd, ← (publicBatchWrapper86_f2895 a h), (publicBatchWrapper86_f2896 a h), kone]⟩


theorem feeAt_84 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 84).isDummy (row a 84).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 82 43) = (row a 84).isDummy := by rw [(publicBatchWrapper86_f680 a h), (publicBatchWrapper86_f678 a h), (publicBatchWrapper86_f679 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2897 a h), by rw [← hd, ← (publicBatchWrapper86_f2898 a h), (publicBatchWrapper86_f2899 a h), kone]⟩


theorem blockAt_84 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 84).isDummy (row a 84).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 82 43) = (row a 84).isDummy := by rw [(publicBatchWrapper86_f680 a h), (publicBatchWrapper86_f678 a h), (publicBatchWrapper86_f679 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18907), a (.virt 18909), a (.virt 18911), a (.virt 18913)],
    ![a (.virt 18908), a (.virt 18910), a (.virt 18912), a (.virt 18914)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2900 a h)
    · exact (publicBatchWrapper86_f2901 a h)
    · exact (publicBatchWrapper86_f2902 a h)
    · exact (publicBatchWrapper86_f2903 a h)
  · show bor (row a 84).isDummy (band (band (a (.virt 18907)) (a (.virt 18909)))
      (band (a (.virt 18911)) (a (.virt 18913)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2904 a h), ← (publicBatchWrapper86_f2905 a h), ← (publicBatchWrapper86_f2906 a h), ← (publicBatchWrapper86_f2907 a h), (publicBatchWrapper86_f2908 a h), kone]


theorem countAt_84 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 84).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 16802)).val = 32
  rw [(publicBatchWrapper86_f84 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]


theorem dummyAt_85 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DummyCheck (row a 85) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  intro j; fin_cases j
  · have e := (publicBatchWrapper86_f681 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f682 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f683 a h); rw [kzero] at e; exact e
  · have e := (publicBatchWrapper86_f684 a h); rw [kzero] at e; exact e


theorem assetAt_85 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 85).isDummy (row a 85).assetId
      (scanRef (rows a) fun t => t.1.assetId) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 84 7) = (row a 85).isDummy := by rw [(publicBatchWrapper86_f687 a h), (publicBatchWrapper86_f685 a h), (publicBatchWrapper86_f686 a h)]; rfl
  exact ⟨_, _, scan_sa a h ▸ (publicBatchWrapper86_f2909 a h), by rw [← hd, ← (publicBatchWrapper86_f2910 a h), (publicBatchWrapper86_f2911 a h), kone]⟩


theorem feeAt_85 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    ConsistencyCheck (row a 85).isDummy (row a 85).fee
      (scanRef (rows a) fun t => t.1.fee) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 84 7) = (row a 85).isDummy := by rw [(publicBatchWrapper86_f687 a h), (publicBatchWrapper86_f685 a h), (publicBatchWrapper86_f686 a h)]; rfl
  exact ⟨_, _, scan_sf a h ▸ (publicBatchWrapper86_f2912 a h), by rw [← hd, ← (publicBatchWrapper86_f2913 a h), (publicBatchWrapper86_f2914 a h), kone]⟩


theorem blockAt_85 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    DigestConsistencyCheck (row a 85).isDummy (row a 85).blockHash (blockRef (rows a)) := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hd : a (.wire 84 7) = (row a 85).isDummy := by rw [(publicBatchWrapper86_f687 a h), (publicBatchWrapper86_f685 a h), (publicBatchWrapper86_f686 a h)]; rfl
  have hblk : ∀ j, a (![.wire 165 15, .wire 165 23, .wire 165 31, .wire 165 39] j) = blockRef (rows a) j := by
    intro j; fin_cases j
    · exact blockLimb_0 a h
    · exact blockLimb_1 a h
    · exact blockLimb_2 a h
    · exact blockLimb_3 a h
  refine ⟨![a (.virt 18919), a (.virt 18921), a (.virt 18923), a (.virt 18925)],
    ![a (.virt 18920), a (.virt 18922), a (.virt 18924), a (.virt 18926)], ?_, ?_⟩
  · intro j; rw [← hblk j]; fin_cases j
    · exact (publicBatchWrapper86_f2915 a h)
    · exact (publicBatchWrapper86_f2916 a h)
    · exact (publicBatchWrapper86_f2917 a h)
    · exact (publicBatchWrapper86_f2918 a h)
  · show bor (row a 85).isDummy (band (band (a (.virt 18919)) (a (.virt 18921)))
      (band (a (.virt 18923)) (a (.virt 18925)))) = 1
    rw [← hd, ← (publicBatchWrapper86_f2919 a h), ← (publicBatchWrapper86_f2920 a h), ← (publicBatchWrapper86_f2921 a h), ← (publicBatchWrapper86_f2922 a h), (publicBatchWrapper86_f2923 a h), kone]


theorem countAt_85 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a)
    (hpg : WormholeSpec.goldilocks ≤ p) :
    (inner a 85).numExitSlots = 32 := by
  obtain ⟨kslots, ktot, kzero, kone⟩ := consts a h
  have hc : (32 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
  show (a (.virt 17002)).val = 32
  rw [(publicBatchWrapper86_f85 a h), kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]

end Plonky2Bridge.PublicWrapper86
