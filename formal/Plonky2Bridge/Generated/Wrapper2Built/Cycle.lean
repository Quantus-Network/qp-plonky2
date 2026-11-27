/-
  AUTO-GENERATED — do not edit by hand.

  Part of `Plonky2Bridge.Generated.Wrapper2Built`, split out of the generated module
  so each Lean module elaborates in bounded memory.
  Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
-/
import Plonky2Bridge.Trusted
import Plonky2Bridge.Generated.LeafBuilt
import Plonky2Spec.Generated.PrivateBatchWrapper2
import Plonky2Bridge.Generated.Wrapper2Built.Data

namespace Plonky2Bridge.Wrapper2

open Plonky2Spec.Wiring
open Plonky2Spec.Generated

set_option maxRecDepth 200000

set_option maxHeartbeats 0

set_option linter.all false

set_option Elab.async false

variable {p : ℕ}

/-- The circuit with its recursion gadgets: `verify_proof` under the `leaf_circuit`
    verifier key on each `leaf_pis_i`. -/
noncomputable def tree : Recursive p :=
  .node "private_batch_wrapper_n2" (privateBatchWrapper2 p)
    [(LeafCircuit.tree, List.ofFn (leafPis 0)),
     (LeafCircuit.tree, List.ofFn (leafPis 1))]

/-- Cycle `0` of `buildCert`, 4162 positions. -/
noncomputable def buildCert.cycle0 : Tbl (ℕ × ℕ) := (Tbl.node 2048 (Tbl.node 1024 (Tbl.node 512 (Tbl.node 256 buildCert.cycle0_0 buildCert.cycle0_1) (Tbl.node 256 buildCert.cycle0_2 buildCert.cycle0_3)) (Tbl.node 512 (Tbl.node 256 buildCert.cycle0_4 buildCert.cycle0_5) (Tbl.node 256 buildCert.cycle0_6 buildCert.cycle0_7))) (Tbl.node 1024 (Tbl.node 512 (Tbl.node 256 buildCert.cycle0_8 buildCert.cycle0_9) (Tbl.node 256 buildCert.cycle0_10 buildCert.cycle0_11)) (Tbl.node 512 (Tbl.node 256 buildCert.cycle0_12 buildCert.cycle0_13) (Tbl.node 256 buildCert.cycle0_14 (Tbl.node 256 buildCert.cycle0_15 buildCert.cycle0_16)))))

/-- Cycle `6` of `buildCert`, 28836 positions. -/
noncomputable def buildCert.cycle6 : Tbl (ℕ × ℕ) := (Tbl.node 14336 (Tbl.node 7168 (Tbl.node 3584 (Tbl.node 1792 (Tbl.node 768 (Tbl.node 256 buildCert.cycle6_0 (Tbl.node 256 buildCert.cycle6_1 buildCert.cycle6_2)) (Tbl.node 512 (Tbl.node 256 buildCert.cycle6_3 buildCert.cycle6_4) (Tbl.node 256 buildCert.cycle6_5 buildCert.cycle6_6))) (Tbl.node 768 (Tbl.node 256 buildCert.cycle6_7 (Tbl.node 256 buildCert.cycle6_8 buildCert.cycle6_9)) (Tbl.node 512 (Tbl.node 256 buildCert.cycle6_10 buildCert.cycle6_11) (Tbl.node 256 buildCert.cycle6_12 buildCert.cycle6_13)))) (Tbl.node 1792 (Tbl.node 768 (Tbl.node 256 buildCert.cycle6_14 (Tbl.node 256 buildCert.cycle6_15 buildCert.cycle6_16)) (Tbl.node 512 (Tbl.node 256 buildCert.cycle6_17 buildCert.cycle6_18) (Tbl.node 256 buildCert.cycle6_19 buildCert.cycle6_20))) (Tbl.node 768 (Tbl.node 256 buildCert.cycle6_21 (Tbl.node 256 buildCert.cycle6_22 buildCert.cycle6_23)) (Tbl.node 512 (Tbl.node 256 buildCert.cycle6_24 buildCert.cycle6_25) (Tbl.node 256 buildCert.cycle6_26 buildCert.cycle6_27))))) (Tbl.node 3584 (Tbl.node 1792 (Tbl.node 768 (Tbl.node 256 buildCert.cycle6_28 (Tbl.node 256 buildCert.cycle6_29 buildCert.cycle6_30)) (Tbl.node 512 (Tbl.node 256 buildCert.cycle6_31 buildCert.cycle6_32) (Tbl.node 256 buildCert.cycle6_33 buildCert.cycle6_34))) (Tbl.node 768 (Tbl.node 256 buildCert.cycle6_35 (Tbl.node 256 buildCert.cycle6_36 buildCert.cycle6_37)) (Tbl.node 512 (Tbl.node 256 buildCert.cycle6_38 buildCert.cycle6_39) (Tbl.node 256 buildCert.cycle6_40 buildCert.cycle6_41)))) (Tbl.node 1792 (Tbl.node 768 (Tbl.node 256 buildCert.cycle6_42 (Tbl.node 256 buildCert.cycle6_43 buildCert.cycle6_44)) (Tbl.node 512 (Tbl.node 256 buildCert.cycle6_45 buildCert.cycle6_46) (Tbl.node 256 buildCert.cycle6_47 buildCert.cycle6_48))) (Tbl.node 768 (Tbl.node 256 buildCert.cycle6_49 (Tbl.node 256 buildCert.cycle6_50 buildCert.cycle6_51)) (Tbl.node 512 (Tbl.node 256 buildCert.cycle6_52 buildCert.cycle6_53) (Tbl.node 256 buildCert.cycle6_54 buildCert.cycle6_55)))))) (Tbl.node 7168 (Tbl.node 3584 (Tbl.node 1792 (Tbl.node 768 (Tbl.node 256 buildCert.cycle6_56 (Tbl.node 256 buildCert.cycle6_57 buildCert.cycle6_58)) (Tbl.node 512 (Tbl.node 256 buildCert.cycle6_59 buildCert.cycle6_60) (Tbl.node 256 buildCert.cycle6_61 buildCert.cycle6_62))) (Tbl.node 768 (Tbl.node 256 buildCert.cycle6_63 (Tbl.node 256 buildCert.cycle6_64 buildCert.cycle6_65)) (Tbl.node 512 (Tbl.node 256 buildCert.cycle6_66 buildCert.cycle6_67) (Tbl.node 256 buildCert.cycle6_68 buildCert.cycle6_69)))) (Tbl.node 1792 (Tbl.node 768 (Tbl.node 256 buildCert.cycle6_70 (Tbl.node 256 buildCert.cycle6_71 buildCert.cycle6_72)) (Tbl.node 512 (Tbl.node 256 buildCert.cycle6_73 buildCert.cycle6_74) (Tbl.node 256 buildCert.cycle6_75 buildCert.cycle6_76))) (Tbl.node 768 (Tbl.node 256 buildCert.cycle6_77 (Tbl.node 256 buildCert.cycle6_78 buildCert.cycle6_79)) (Tbl.node 512 (Tbl.node 256 buildCert.cycle6_80 buildCert.cycle6_81) (Tbl.node 256 buildCert.cycle6_82 buildCert.cycle6_83))))) (Tbl.node 3584 (Tbl.node 1792 (Tbl.node 768 (Tbl.node 256 buildCert.cycle6_84 (Tbl.node 256 buildCert.cycle6_85 buildCert.cycle6_86)) (Tbl.node 512 (Tbl.node 256 buildCert.cycle6_87 buildCert.cycle6_88) (Tbl.node 256 buildCert.cycle6_89 buildCert.cycle6_90))) (Tbl.node 768 (Tbl.node 256 buildCert.cycle6_91 (Tbl.node 256 buildCert.cycle6_92 buildCert.cycle6_93)) (Tbl.node 512 (Tbl.node 256 buildCert.cycle6_94 buildCert.cycle6_95) (Tbl.node 256 buildCert.cycle6_96 buildCert.cycle6_97)))) (Tbl.node 1792 (Tbl.node 768 (Tbl.node 256 buildCert.cycle6_98 (Tbl.node 256 buildCert.cycle6_99 buildCert.cycle6_100)) (Tbl.node 512 (Tbl.node 256 buildCert.cycle6_101 buildCert.cycle6_102) (Tbl.node 256 buildCert.cycle6_103 buildCert.cycle6_104))) (Tbl.node 1024 (Tbl.node 512 (Tbl.node 256 buildCert.cycle6_105 buildCert.cycle6_106) (Tbl.node 256 buildCert.cycle6_107 buildCert.cycle6_108)) (Tbl.node 512 (Tbl.node 256 buildCert.cycle6_109 buildCert.cycle6_110) (Tbl.node 256 buildCert.cycle6_111 buildCert.cycle6_112)))))))

/-- One witness per constant target of the export. -/
noncomputable def constWits : List (ℕ × ℕ × ℕ) := ((constWits0 ++ constWits1) ++ (constWits2 ++ (constWits3 ++ constWits4)))

/-- The export's gate kinds, row by row. -/
noncomputable def rowKinds : List GateKind := (((((rowKinds0 ++ rowKinds1) ++ (rowKinds2 ++ (rowKinds3 ++ rowKinds4))) ++ ((rowKinds5 ++ rowKinds6) ++ (rowKinds7 ++ (rowKinds8 ++ rowKinds9)))) ++ (((rowKinds10 ++ rowKinds11) ++ (rowKinds12 ++ (rowKinds13 ++ rowKinds14))) ++ ((rowKinds15 ++ rowKinds16) ++ (rowKinds17 ++ (rowKinds18 ++ rowKinds19))))) ++ ((((rowKinds20 ++ rowKinds21) ++ (rowKinds22 ++ (rowKinds23 ++ rowKinds24))) ++ ((rowKinds25 ++ rowKinds26) ++ (rowKinds27 ++ (rowKinds28 ++ rowKinds29)))) ++ (((rowKinds30 ++ rowKinds31) ++ (rowKinds32 ++ (rowKinds33 ++ rowKinds34))) ++ ((rowKinds35 ++ (rowKinds36 ++ rowKinds37)) ++ (rowKinds38 ++ (rowKinds39 ++ rowKinds40))))))

theorem constants_fst : (privateBatchWrapper2 p).constants.map Prod.fst = constTargets := rfl

theorem publicInputs_targets : (privateBatchWrapper2 p).publicInputs = piTargets := rfl

end Plonky2Bridge.Wrapper2
