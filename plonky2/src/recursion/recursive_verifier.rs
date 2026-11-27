#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::field::extension::Extendable;
use crate::hash::hash_types::{HashOutTarget, RichField};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::{CommonCircuitData, VerifierCircuitTarget};
use crate::plonk::config::{AlgebraicHasher, GenericConfig};
use crate::plonk::plonk_common::salt_size;
use crate::plonk::proof::{
    OpeningSetTarget, ProofChallengesTarget, ProofTarget, ProofWithPublicInputsTarget,
};
use crate::plonk::vanishing_poly::eval_vanishing_poly_circuit;
use crate::plonk::vars::EvaluationTargets;
use crate::util::reducing::ReducingFactorTarget;
use crate::with_context;

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilder<F, D> {
    /// Recursively verifies an inner proof.
    pub fn verify_proof<C: GenericConfig<D, F = F>>(
        &mut self,
        proof_with_pis: &ProofWithPublicInputsTarget<D>,
        inner_verifier_data: &VerifierCircuitTarget,
        inner_common_data: &CommonCircuitData<F, D>,
    ) where
        C::Hasher: AlgebraicHasher<F>,
        C::InnerHasher: AlgebraicHasher<F>,
    {
        assert_eq!(
            proof_with_pis.public_inputs.len(),
            inner_common_data.num_public_inputs
        );
        let public_inputs_hash =
            self.hash_n_to_hash_no_pad::<C::InnerHasher>(proof_with_pis.public_inputs.clone());
        let challenges = proof_with_pis.get_challenges::<F, C>(
            self,
            public_inputs_hash,
            inner_verifier_data.circuit_digest,
            inner_common_data,
        );

        self.verify_proof_with_challenges::<C>(
            &proof_with_pis.proof,
            public_inputs_hash,
            challenges,
            inner_verifier_data,
            inner_common_data,
        );
    }

    /// Recursively verifies an inner proof.
    fn verify_proof_with_challenges<C: GenericConfig<D, F = F>>(
        &mut self,
        proof: &ProofTarget<D>,
        public_inputs_hash: HashOutTarget,
        challenges: ProofChallengesTarget<D>,
        inner_verifier_data: &VerifierCircuitTarget,
        inner_common_data: &CommonCircuitData<F, D>,
    ) where
        C::Hasher: AlgebraicHasher<F>,
    {
        let one = self.one_extension();

        let local_constants = &proof.openings.constants;
        let local_wires = &proof.openings.wires;
        let vars = EvaluationTargets {
            local_constants,
            local_wires,
            public_inputs_hash: &public_inputs_hash,
        };
        let local_zs = &proof.openings.plonk_zs;
        let next_zs = &proof.openings.plonk_zs_next;
        let s_sigmas = &proof.openings.plonk_sigmas;
        let partial_products = &proof.openings.partial_products;

        let zeta_pow_deg =
            self.exp_power_of_2_extension(challenges.plonk_zeta, inner_common_data.degree_bits());
        let vanishing_polys_zeta = with_context!(
            self,
            "evaluate the vanishing polynomial at our challenge point, zeta.",
            eval_vanishing_poly_circuit::<F, D>(
                self,
                inner_common_data,
                challenges.plonk_zeta,
                zeta_pow_deg,
                vars,
                local_zs,
                next_zs,
                partial_products,
                s_sigmas,
                &challenges.plonk_betas,
                &challenges.plonk_gammas,
                &challenges.plonk_alphas,
            )
        );

        with_context!(self, "check vanishing and quotient polynomials.", {
            let quotient_polys_zeta = &proof.openings.quotient_polys;
            let mut scale = ReducingFactorTarget::new(zeta_pow_deg);
            let z_h_zeta = self.sub_extension(zeta_pow_deg, one);
            for (i, chunk) in quotient_polys_zeta
                .chunks(inner_common_data.quotient_degree_factor())
                .enumerate()
            {
                let recombined_quotient = scale.reduce(chunk, self);
                let computed_vanishing_poly = self.mul_extension(z_h_zeta, recombined_quotient);
                self.connect_extension(vanishing_polys_zeta[i], computed_vanishing_poly);
            }
        });

        let merkle_caps = &[
            inner_verifier_data.constants_sigmas_cap.clone(),
            proof.wires_cap.clone(),
            proof.plonk_zs_partial_products_cap.clone(),
            proof.quotient_polys_cap.clone(),
        ];

        let fri_instance = inner_common_data.get_fri_instance_target(self, challenges.plonk_zeta);
        with_context!(
            self,
            "verify FRI proof",
            self.verify_fri_proof::<C>(
                &fri_instance,
                &proof.openings.to_fri_openings(),
                &challenges.fri_challenges,
                merkle_caps,
                &proof.opening_proof,
                &inner_common_data.fri_params,
            )
        );
    }

    pub fn add_virtual_proof_with_pis(
        &mut self,
        common_data: &CommonCircuitData<F, D>,
    ) -> ProofWithPublicInputsTarget<D> {
        let proof = self.add_virtual_proof(common_data);
        let public_inputs = self.add_virtual_targets(common_data.num_public_inputs);
        ProofWithPublicInputsTarget {
            proof,
            public_inputs,
        }
    }

    fn add_virtual_proof(&mut self, common_data: &CommonCircuitData<F, D>) -> ProofTarget<D> {
        let fri_params = &common_data.fri_params;
        let cap_height = fri_params.config.cap_height;
        let config = &common_data.config;
        let salt = salt_size(common_data.fri_params.leaf_hiding);

        // Compute num_leaves_per_oracle from the circuit data
        let num_leaves_per_oracle = vec![
            // CONSTANTS_SIGMAS: num_constants + num_routed_wires (no salt - not blinded)
            common_data.num_constants + config.num_routed_wires,
            // WIRES: num_wires + salt (blinded)
            config.num_wires + salt,
            // ZS_PARTIAL_PRODUCTS: num_challenges * (1 + num_partial_products) + salt (blinded)
            config.num_challenges * (1 + common_data.num_partial_products) + salt,
            // QUOTIENT: quotient_degree_factor * num_challenges + salt (blinded)
            common_data.quotient_degree_factor() * config.num_challenges + salt,
        ];

        ProofTarget {
            wires_cap: self.add_virtual_cap(cap_height),
            plonk_zs_partial_products_cap: self.add_virtual_cap(cap_height),
            quotient_polys_cap: self.add_virtual_cap(cap_height),
            openings: self.add_opening_set(common_data),
            opening_proof: self.add_virtual_fri_proof(&num_leaves_per_oracle, fri_params),
        }
    }

    fn add_opening_set(&mut self, common_data: &CommonCircuitData<F, D>) -> OpeningSetTarget<D> {
        let config = &common_data.config;
        let num_challenges = config.num_challenges;
        let total_partial_products = num_challenges * common_data.num_partial_products;
        OpeningSetTarget {
            constants: self.add_virtual_extension_targets(common_data.num_constants),
            plonk_sigmas: self.add_virtual_extension_targets(config.num_routed_wires),
            wires: self.add_virtual_extension_targets(config.num_wires),
            plonk_zs: self.add_virtual_extension_targets(num_challenges),
            plonk_zs_next: self.add_virtual_extension_targets(num_challenges),
            partial_products: self.add_virtual_extension_targets(total_partial_products),
            quotient_polys: self.add_virtual_extension_targets(common_data.num_quotient_polys()),
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use anyhow::Result;
    use log::{info, Level};

    use super::*;
    use crate::fri::{FriConfig, FriReductionStrategy};
    use crate::gates::noop::NoopGate;
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_data::{CircuitConfig, VerifierOnlyCircuitData};
    use crate::plonk::config::{KeccakGoldilocksConfig, PoseidonGoldilocksConfig};
    use crate::plonk::proof::ProofWithPublicInputs;
    use crate::plonk::prover::prove;
    use crate::util::timing::TimingTree;

    #[test]
    #[cfg(feature = "rand")]
    fn test_recursive_verifier() -> Result<()> {
        init_logger();
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let config = CircuitConfig::standard_recursion_config();

        let (proof, vd, common_data) = dummy_proof::<F, C, D>(&config, 4_000)?;
        let (proof, _vd, common_data) =
            recursive_proof::<F, C, C, D>(proof, vd, common_data, &config, None, true, true)?;
        test_serialization(&proof, &common_data)?;

        Ok(())
    }

    #[test]
    fn test_recursive_recursive_verifier() -> Result<()> {
        init_logger();
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();

        // Start with a degree 2^14 proof
        let (proof, vd, common_data) = dummy_proof::<F, C, D>(&config, 16_000)?;
        assert_eq!(common_data.degree_bits(), 14);

        // Shrink it to 2^13.
        let (proof, vd, common_data) =
            recursive_proof::<F, C, C, D>(proof, vd, common_data, &config, Some(13), false, false)?;
        assert_eq!(common_data.degree_bits(), 13);

        // Shrink it to 2^12.
        let (proof, _vd, common_data) =
            recursive_proof::<F, C, C, D>(proof, vd, common_data, &config, None, true, true)?;
        assert_eq!(common_data.degree_bits(), 12);

        test_serialization(&proof, &common_data)?;

        Ok(())
    }

    /// Creates a chain of recursive proofs where the last proof is made as small as reasonably
    /// possible, using a high rate, high PoW bits, etc.
    #[test]
    #[ignore]
    fn test_size_optimized_recursion() -> Result<()> {
        init_logger();
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type KC = KeccakGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let standard_config = CircuitConfig::standard_recursion_config();

        // An initial dummy proof.
        let (proof, vd, common_data) = dummy_proof::<F, C, D>(&standard_config, 4_000)?;
        assert_eq!(common_data.degree_bits(), 12);

        // A standard recursive proof.
        let (proof, vd, common_data) = recursive_proof::<F, C, C, D>(
            proof,
            vd,
            common_data,
            &standard_config,
            None,
            false,
            false,
        )?;
        assert_eq!(common_data.degree_bits(), 12);

        // A high-rate recursive proof, designed to be verifiable with fewer routed wires.
        let high_rate_config = CircuitConfig {
            fri_config: FriConfig {
                rate_bits: 7,
                proof_of_work_bits: 16,
                num_query_rounds: 12,
                ..standard_config.fri_config.clone()
            },
            ..standard_config
        };
        let (proof, vd, common_data) = recursive_proof::<F, C, C, D>(
            proof,
            vd,
            common_data,
            &high_rate_config,
            None,
            true,
            true,
        )?;
        assert_eq!(common_data.degree_bits(), 12);

        // A final proof, optimized for size.
        let final_config = CircuitConfig {
            num_routed_wires: 37,
            fri_config: FriConfig {
                rate_bits: 8,
                cap_height: 0,
                proof_of_work_bits: 20,
                reduction_strategy: FriReductionStrategy::MinSize(None),
                num_query_rounds: 10,
            },
            ..high_rate_config
        };
        let (proof, _vd, common_data) = recursive_proof::<F, KC, C, D>(
            proof,
            vd,
            common_data,
            &final_config,
            None,
            true,
            true,
        )?;
        assert_eq!(common_data.degree_bits(), 12, "final proof too large");

        test_serialization(&proof, &common_data)?;

        Ok(())
    }

    #[test]
    fn test_recursive_verifier_multi_hash() -> Result<()> {
        init_logger();
        const D: usize = 2;
        type PC = PoseidonGoldilocksConfig;
        type KC = KeccakGoldilocksConfig;
        type F = <PC as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();
        let (proof, vd, common_data) = dummy_proof::<F, PC, D>(&config, 4_000)?;

        let (proof, vd, common_data) =
            recursive_proof::<F, PC, PC, D>(proof, vd, common_data, &config, None, false, false)?;
        test_serialization(&proof, &common_data)?;

        let (proof, _vd, common_data) =
            recursive_proof::<F, KC, PC, D>(proof, vd, common_data, &config, None, false, false)?;
        test_serialization(&proof, &common_data)?;

        Ok(())
    }

    type Proof<F, C, const D: usize> = (
        ProofWithPublicInputs<F, C, D>,
        VerifierOnlyCircuitData<C, D>,
        CommonCircuitData<F, D>,
    );

    /// Creates a dummy proof which should have roughly `num_dummy_gates` gates.
    fn dummy_proof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
        config: &CircuitConfig,
        num_dummy_gates: u64,
    ) -> Result<Proof<F, C, D>>
    where
        C::InnerHasher: AlgebraicHasher<F>,
    {
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        for _ in 0..num_dummy_gates {
            builder.add_gate(NoopGate, vec![]);
        }

        let data = builder.build::<C>();
        let inputs = PartialWitness::new();
        let proof = data.prove(inputs)?;
        data.verify(proof.clone())?;

        Ok((proof, data.verifier_only, data.common))
    }

    fn recursive_proof<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        InnerC: GenericConfig<D, F = F>,
        const D: usize,
    >(
        inner_proof: ProofWithPublicInputs<F, InnerC, D>,
        inner_vd: VerifierOnlyCircuitData<InnerC, D>,
        inner_cd: CommonCircuitData<F, D>,
        config: &CircuitConfig,
        min_degree_bits: Option<usize>,
        print_gate_counts: bool,
        print_timing: bool,
    ) -> Result<Proof<F, C, D>>
    where
        InnerC::Hasher: AlgebraicHasher<F>,
        InnerC::InnerHasher: AlgebraicHasher<F>,
        C::InnerHasher: AlgebraicHasher<F>,
    {
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        let mut pw = PartialWitness::new();
        let pt = builder.add_virtual_proof_with_pis(&inner_cd);
        pw.set_proof_with_pis_target(&pt, &inner_proof)?;

        let inner_data = builder.add_virtual_verifier_data(inner_cd.config.fri_config.cap_height);
        pw.set_cap_target(
            &inner_data.constants_sigmas_cap,
            &inner_vd.constants_sigmas_cap,
        )?;
        pw.set_hash_target(inner_data.circuit_digest, inner_vd.circuit_digest)?;

        builder.verify_proof::<InnerC>(&pt, &inner_data, &inner_cd);

        if print_gate_counts {
            builder.print_gate_counts(0);
        }

        if let Some(min_degree_bits) = min_degree_bits {
            // We don't want to pad all the way up to 2^min_degree_bits, as the builder will add a
            // few special gates afterward. So just pad to 2^(min_degree_bits - 1) + 1. Then the
            // builder will pad to the next power of two, 2^min_degree_bits.
            let min_gates = (1 << (min_degree_bits - 1)) + 1;
            for _ in builder.num_gates()..min_gates {
                builder.add_gate(NoopGate, vec![]);
            }
        }

        let data = builder.build::<C>();

        let mut timing = TimingTree::new("prove", Level::Debug);
        let proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;
        if print_timing {
            timing.print();
        }

        data.verify(proof.clone())?;

        Ok((proof, data.verifier_only, data.common))
    }

    /// Test serialization and print some size info.
    fn test_serialization<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        proof: &ProofWithPublicInputs<F, C, D>,
        common_data: &CommonCircuitData<F, D>,
    ) -> Result<()> {
        let proof_bytes = proof.to_bytes();
        info!("Proof length: {} bytes", proof_bytes.len());
        let proof_from_bytes = ProofWithPublicInputs::from_bytes(proof_bytes, common_data)?;
        assert_eq!(proof, &proof_from_bytes);
        Ok(())
    }

    fn init_logger() {
        let _ = env_logger::builder().format_timestamp(None).try_init();
    }
}
