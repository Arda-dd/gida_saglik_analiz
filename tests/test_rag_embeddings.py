from unittest.mock import MagicMock

import numpy as np
import pytest

from src.rag.embeddings import embed_texts, get_embedding_settings


def test_embed_texts_openai_normalizes_vectors():
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[3.0, 4.0]), MagicMock(embedding=[1.0, 0.0])]
    )

    vectors = embed_texts(["a", "b"], provider="openai", model="text-embedding-3-small", client=mock_client)

    assert vectors.shape == (2, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert np.allclose(vectors[0], [0.6, 0.8])


def test_embed_texts_huggingface_normalizes_vectors():
    mock_client = MagicMock()
    mock_client.feature_extraction.side_effect = [
        np.array([3.0, 4.0]),
        np.array([1.0, 0.0]),
    ]

    vectors = embed_texts(["a", "b"], provider="huggingface", model="some/model", client=mock_client)

    assert vectors.shape == (2, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert np.allclose(vectors[0], [0.6, 0.8])


def test_embed_texts_huggingface_mean_pools_token_level_output():
    mock_client = MagicMock()
    # Bazi modeller token-bazinda (seq_len, dim) cikti doner - mean pooling gerekir.
    mock_client.feature_extraction.side_effect = [np.array([[1.0, 0.0], [3.0, 0.0]])]

    vectors = embed_texts(["tek metin"], provider="huggingface", model="some/model", client=mock_client)

    assert vectors.shape == (1, 2)
    assert np.allclose(vectors[0], [1.0, 0.0])


def test_embed_texts_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Bilinmeyen embedding saglayici"):
        embed_texts(["a"], provider="not_real")


def test_get_embedding_settings_reads_config():
    config = {"rag": {"embedding_provider": "openai", "embedding_model": "custom-model"}}
    provider, model = get_embedding_settings(config)
    assert provider == "openai"
    assert model == "custom-model"


def test_get_embedding_settings_defaults_to_huggingface_when_missing():
    provider, _ = get_embedding_settings({})
    assert provider == "huggingface"


def test_get_embedding_settings_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Bilinmeyen embedding saglayici"):
        get_embedding_settings({"rag": {"embedding_provider": "not_real"}})
