//! `Plonky2Bridge/Generated/FriParams.lean` is current and pins the deployed configs.

use constraint_exporter::trace::generate_fri_params_lean;

#[test]
fn fri_params_lean_is_current() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../formal/Plonky2Bridge/Generated/FriParams.lean"
    );
    let on_disk = std::fs::read_to_string(path).unwrap_or_default();
    let generated = generate_fri_params_lean().unwrap();
    assert_eq!(on_disk, generated, "run export-constraints ({path})");
    assert!(generated.contains(
        "def leafFri : FriQueryParams where\n  rateBits := 7\n  numQueries := 16\n  proofOfWorkBits := 16\n"
    ));
    assert!(generated.contains(
        "def privateBatchFri : FriQueryParams where\n  rateBits := 3\n  numQueries := 36\n  proofOfWorkBits := 20\n"
    ));
    assert!(generated.contains("example : friJohnsonQueryBits publicBatchFri = 54 := rfl"));
    assert!(generated.contains("example : friConjecturedBits publicBatchFri = 128 := rfl"));
}
