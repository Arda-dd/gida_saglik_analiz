from src.rag.compare_providers import parse_model_spec

CONFIG = {"llm": {"model_openai": "gpt-4.1-mini", "model_huggingface": "meta-llama/Llama-3.1-8B-Instruct"}}


def test_parse_model_spec_with_explicit_model():
    assert parse_model_spec("openai:gpt-4o-mini", CONFIG) == ("openai", "gpt-4o-mini")


def test_parse_model_spec_falls_back_to_config_model():
    assert parse_model_spec("huggingface", CONFIG) == ("huggingface", "meta-llama/Llama-3.1-8B-Instruct")


def test_parse_model_spec_keeps_slashes_in_model_name():
    assert parse_model_spec("huggingface:Qwen/Qwen2.5-7B-Instruct", CONFIG) == (
        "huggingface",
        "Qwen/Qwen2.5-7B-Instruct",
    )
