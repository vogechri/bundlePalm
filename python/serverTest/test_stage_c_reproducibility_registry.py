from validate_stage_c_reproducibility_registry import DEFAULT_REGISTRY, validate


def test_stage_c_reproducibility_registry_covers_modes_tables_and_archive_boundary():
    result = validate(DEFAULT_REGISTRY)

    assert result["modes"] == 8
    assert result["tables"] == 6
    assert result["maintained_tests"] == 223
    assert result["worker_sha256"] == (
        "c408131a6977de3d92281f195c9ca888fe6c11e05182915bc36ba45d10326b3d"
    )
