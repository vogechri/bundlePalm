"""Shared publication-panel coverage contract."""


PUBLICATION_PANELS = {
    "all15_1dsfm": {
        "scenes": 15,
        "labels": (
            "Ceres",
            "DRS K1 BAE-style",
            "DRS K1 Schur-PCG",
            "Base DRS K24",
            "DRS+Schur fast",
            "DRS+Schur balanced",
            "DRS+Schur quality",
            "DRS K4",
            "DRS K16",
            "DRS K4 + terminal correction",
            "DRS K16 + terminal correction",
        ),
    },
    "all29_bal": {
        "scenes": 29,
        "labels": (
            "Ceres",
            "Base DRS K24",
            "DRS K4",
            "DRS K16",
            "DRS K4 + terminal correction",
            "DRS K16 + terminal correction",
        ),
    },
    "bae_six_scene_inset": {
        "scenes": 6,
        "labels": (
            "Ceres",
            "DRS K1 BAE-style",
            "DRS K1 Schur-PCG",
            "Base DRS K24",
            "DRS K4",
            "DRS K16",
            "DRS K4 + terminal correction",
            "DRS K16 + terminal correction",
            "BAE Schur-PCG CG",
            "BAE Schur-PCG Nesterov",
        ),
    },
}
