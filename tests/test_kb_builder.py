import json

import pytest

from src.data.kb_builder import (
    THRESHOLD_METADATA,
    build_thresholds_json,
    load_thresholds,
    validate_thresholds_schema,
)


@pytest.fixture
def sample_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "thresholds:\n"
        "  sugar_high_g_per_100g: 22.5\n"
        "  who_salt_daily_intake_g: 5\n",
        encoding="utf-8",
    )
    return config_path


def test_build_thresholds_json_creates_valid_file(sample_config, tmp_path):
    out_path = tmp_path / "thresholds.json"
    build_thresholds_json(sample_config, out_path)

    assert out_path.exists()
    data = load_thresholds(out_path)
    assert data["schema_version"] == "1.0"
    assert data["thresholds"]["sugar_high_g_per_100g"]["value"] == 22.5
    assert data["thresholds"]["who_salt_daily_intake_g"]["verified"] is True


def test_build_thresholds_json_raises_on_unknown_key(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("thresholds:\n  bilinmeyen_esik: 1\n", encoding="utf-8")
    out_path = tmp_path / "thresholds.json"

    with pytest.raises(ValueError):
        build_thresholds_json(config_path, out_path)


def test_validate_thresholds_schema_accepts_well_formed_data():
    data = {
        "schema_version": "1.0",
        "thresholds": {
            "sugar_high_g_per_100g": {
                "value": 22.5,
                "unit": "g/100g",
                "source": "x",
                "source_note": "y",
                "verified": False,
            }
        },
    }
    assert validate_thresholds_schema(data) == []


def test_validate_thresholds_schema_flags_missing_fields():
    data = {"schema_version": "1.0", "thresholds": {"foo": {"value": 1}}}
    errors = validate_thresholds_schema(data)
    assert len(errors) == 1
    assert "foo" in errors[0]


def test_validate_thresholds_schema_flags_missing_top_level_keys():
    errors = validate_thresholds_schema({})
    assert "'schema_version' alani eksik" in errors
    assert "'thresholds' alani eksik" in errors


def test_all_config_thresholds_have_metadata():
    """Gercek config.yaml'daki her esigin THRESHOLD_METADATA'da karsiligi olmali."""
    from src.common.config import load_config

    config = load_config()
    for key in config["thresholds"]:
        assert key in THRESHOLD_METADATA, f"'{key}' icin metadata eksik"
