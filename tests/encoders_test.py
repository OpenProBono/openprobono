"""Tests for different encoder models."""
import unittest

from langchain_openai import OpenAIEmbeddings

import encoders
from models import EncoderParams


class EncoderTests(unittest.TestCase):
    """Test class for different encoder models."""

    # 1024 dimensions, 512 max tokens
    test_model = EncoderParams("WhereIsAI/UAE-Large-V1", 1024)

    def test_langchain_embeddings_openai(self: "EncoderTests") -> None:
        """Check LangChain embeddings for an OpenAI model.

        Parameters
        ----------
        self: EncoderTests
            The test class instance

        """
        self.assertTrue(isinstance(
            encoders.get_langchain_embedding_model(EncoderParams()), OpenAIEmbeddings,
        ))

    def test_langchain_embeddings_hf(self: "EncoderTests") -> None:
        """Check LangChain embeddings for a HuggingFaceHub model.

        Parameters
        ----------
        self: EncoderTests
            The test class instance

        """
        self.assertTrue(isinstance(
            encoders.get_langchain_embedding_model(self.test_model),
            encoders.HuggingFaceHubEmbeddings,
        ))
