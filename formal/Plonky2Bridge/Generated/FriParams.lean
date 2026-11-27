/-
  AUTO-GENERATED — do not edit by hand.

  The FRI query parameters of the deployed circuits, read off their recorded traces
  (`constraint-exporter/traces/`), and the query exponents they determine: the capacity
  conjecture's (`friQueryBits`, refuted), the proved Johnson-radius one
  (`friJohnsonQueryBits`), and `security_bits`'s accounting (`friConjecturedBits`). See
  the FRI section of `Plonky2Bridge/Trusted.lean`. Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
-/
import Plonky2Bridge.Trusted

namespace Plonky2Bridge

/-- `wormhole_leaf_circuit_config` (`qp-zk-circuits/common/src/circuit.rs`), as the traces
    `leaf_circuit` record. -/
def leafFri : FriQueryParams where
  rateBits := 7
  numQueries := 16
  proofOfWorkBits := 16

example : friQueryBits leafFri = 112 := rfl
example : friJohnsonQueryBits leafFri = 56 := rfl
example : friConjecturedBits leafFri = 128 := rfl

/-- `wormhole_private_batch_circuit_config` (`qp-zk-circuits/common/src/circuit.rs`), as the traces
    `private_batch_wrapper_n2`, `private_batch_wrapper_n4`, `private_batch_wrapper_n8` record. -/
def privateBatchFri : FriQueryParams where
  rateBits := 3
  numQueries := 36
  proofOfWorkBits := 20

example : friQueryBits privateBatchFri = 108 := rfl
example : friJohnsonQueryBits privateBatchFri = 54 := rfl
example : friConjecturedBits privateBatchFri = 128 := rfl

/-- `wormhole_public_batch_circuit_config` (`qp-zk-circuits/common/src/circuit.rs`), as the traces
    `public_batch_wrapper_n2`, `public_batch_wrapper_n4`, `public_batch_wrapper_n86` record. -/
def publicBatchFri : FriQueryParams where
  rateBits := 3
  numQueries := 36
  proofOfWorkBits := 20

example : friQueryBits publicBatchFri = 108 := rfl
example : friJohnsonQueryBits publicBatchFri = 54 := rfl
example : friConjecturedBits publicBatchFri = 128 := rfl

end Plonky2Bridge
