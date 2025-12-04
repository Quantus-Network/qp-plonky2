mod allocator;

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Sample, Field};
use plonky2::hash::hash_types::{BytesHash, RichField};
use plonky2::hash::hashing::PlonkyPermutation;
use plonky2::hash::keccak::KeccakHash;
use plonky2::hash::poseidon::{Poseidon, PoseidonHash, SPONGE_WIDTH};
use plonky2::hash::poseidon2::{P2Permuter, Poseidon2Hash, Poseidon2Permutation};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig, Hasher, PoseidonGoldilocksConfig};
use tynm::type_name;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub(crate) fn bench_keccak<F: RichField>(c: &mut Criterion) {
    c.bench_function("keccak256", |b| {
        b.iter_batched(
            || (BytesHash::<32>::rand(), BytesHash::<32>::rand()),
            |(left, right)| <KeccakHash<32> as Hasher<F>>::two_to_one(left, right),
            BatchSize::SmallInput,
        )
    });
}

pub(crate) fn bench_poseidon<F: Poseidon>(c: &mut Criterion) {
    c.bench_function(
        &format!("poseidon<{}, {SPONGE_WIDTH}>", type_name::<F>()),
        |b| {
            b.iter_batched(
                || F::rand_array::<SPONGE_WIDTH>(),
                |state| F::poseidon(state),
                BatchSize::SmallInput,
            )
        },
    );
}
pub(crate) fn bench_poseidon2<F: Poseidon + P2Permuter>(c: &mut Criterion) {
    c.bench_function(
        &format!("poseidon2<{}, {SPONGE_WIDTH}>", type_name::<F>()),
        |b| {
            b.iter_batched(
                || F::rand_array::<SPONGE_WIDTH>(),
                |state| Poseidon2Permutation::new(state).permute(),
                BatchSize::SmallInput,
            )
        },
    );
}
const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;

const NUM_PERMS: usize = 100;  // Number of permutations to perform in the circuit

// Helper: Generate fixed random inputs for fairness
fn generate_inputs(rng: &mut StdRng) -> Vec<[F; SPONGE_WIDTH]> {
    (0..NUM_PERMS)
        .map(|_| {
            let mut state = [F::ZERO; SPONGE_WIDTH];
            for i in 0..SPONGE_WIDTH {
                state[i] = F::from_canonical_u64(rng.gen());
            }
            state
        })
        .collect()
}

fn bench_poseidon_air(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0xdeadbeef);
    let inputs = generate_inputs(&mut rng);

    c.bench_function("poseidon_prove", |b| {
        b.iter(|| {
            let config = CircuitConfig::standard_recursion_config();
            let mut builder = CircuitBuilder::<F, D>::new(config);

            // Add inputs as targets
            let input_targets: Vec<[Target; SPONGE_WIDTH]> = (0..NUM_PERMS)
                .map(|_| [(); SPONGE_WIDTH].map(|_| builder.add_virtual_target()))
                .collect();

            // Perform permutations
            for input_t in input_targets.iter() {
                let perm = <PoseidonHash as AlgebraicHasher<F>>::AlgebraicPermutation::new(input_t.iter().cloned());
                let swap = builder.zero();  // No swap
                let out = PoseidonHash::permute_swapped(perm, BoolTarget::new_unsafe(swap), &mut builder);
                // Register output to ensure it's computed (dummy)
                builder.register_public_inputs(&out.squeeze());
            }

            let data = builder.build::<C>();

            // Set witness
            let mut pw = PartialWitness::new();
            for (i, state) in inputs.iter().enumerate() {
                for (j, &val) in state.iter().enumerate() {
                    pw.set_target(input_targets[i][j], val).unwrap();
                }
            }

            // Prove
            let proof = data.prove(pw.clone()).unwrap();
            black_box(proof);
        })
    });
}

fn bench_poseidon2_air(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0xdeadbeef);
    let inputs = generate_inputs(&mut rng);

    c.bench_function("poseidon2_prove", |b| {
        b.iter(|| {
            let config = CircuitConfig::standard_recursion_config();
            let mut builder = CircuitBuilder::<F, D>::new(config);

            // Add inputs as targets
            let input_targets: Vec<[Target; SPONGE_WIDTH]> = (0..NUM_PERMS)
                .map(|_| [(); SPONGE_WIDTH].map(|_| builder.add_virtual_target()))
                .collect();

            // Perform permutations
            for input_t in input_targets.iter() {
                let perm = <Poseidon2Hash as AlgebraicHasher<F>>::AlgebraicPermutation::new(input_t.iter().cloned());
                let swap = builder.zero();  // No swap
                let out = Poseidon2Hash::permute_swapped(perm, BoolTarget::new_unsafe(swap), &mut builder);
                // Register output to ensure it's computed (dummy)
                builder.register_public_inputs(&out.squeeze());
            }

            let data = builder.build::<C>();

            // Set witness
            let mut pw = PartialWitness::new();
            for (i, state) in inputs.iter().enumerate() {
                for (j, &val) in state.iter().enumerate() {
                    pw.set_target(input_targets[i][j], val).unwrap();
                }
            }

            // Prove
            let proof = data.prove(pw.clone()).unwrap();
            black_box(proof);
        })
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_poseidon::<GoldilocksField>(c);
    bench_keccak::<GoldilocksField>(c);
    bench_poseidon2::<GoldilocksField>(c);
}

criterion_group!(benches, criterion_benchmark, bench_poseidon2_air, bench_poseidon_air);
criterion_main!(benches);
