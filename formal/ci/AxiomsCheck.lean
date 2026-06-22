/-
  CI axiom-footprint gate for the capstone `layer0_end_to_end` theorem.

  The shell step in `.github/workflows/ci.yml` runs this file and parses `#print axioms`
  output. It is not part of `defaultTargets` or `Plonky2Bridge`; import-only.
-/
import Plonky2Bridge

#print axioms Plonky2Bridge.layer0_end_to_end
