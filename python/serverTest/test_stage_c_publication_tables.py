from build_stage_c_publication_tables import (
    DEFAULT_CARRYOVER_SUMMARY,
    DEFAULT_PUBLICATION_SUMMARY,
    display_label,
    load_carryover,
    load_publication,
    write_carryover_table,
    write_final_table,
)


def test_base_horizon_label_is_panel_specific():
    assert display_label("all15_1dsfm", "Base DRS K24") == "Base K24/I200"
    assert display_label("all29_bal", "Base DRS K24") == "Base K24/I90"


def test_generated_tables_cover_authoritative_rows(tmp_path):
    panels = load_publication(DEFAULT_PUBLICATION_SUMMARY)
    families = load_carryover(DEFAULT_CARRYOVER_SUMMARY)
    final_path = tmp_path / "final.tex"
    carryover_path = tmp_path / "carryover.tex"

    with final_path.open("w", encoding="utf-8") as output:
        write_final_table(output, panels)
    with carryover_path.open("w", encoding="utf-8") as output:
        write_carryover_table(output, families)

    final = final_path.read_text(encoding="utf-8")
    carryover = carryover_path.read_text(encoding="utf-8")
    assert final.count("1DSfM-15 &") == 11
    assert final.count("BAL-29 &") == 6
    assert "K16 + terminal correction & 1.523013" in final
    assert "1DSfM-15 & 0.992877 & 0.787326 & 0.762902" in carryover
    assert "BAL-29 & 1.000177 & 0.999998 & 1.000131" in carryover
