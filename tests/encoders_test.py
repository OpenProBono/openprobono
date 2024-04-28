"""Tests for different encoder models."""
import unittest

import encoders


class EncoderTests(unittest.TestCase):
    """Test class for different encoder models."""

    test_model = encoders.EncoderParams(encoders.UAE_LARGE, 1024)

    def test_huggingface_model(self: "EncoderTests") -> None:
        """Check that a PreTrainedModel loads properly.

        Parameters
        ----------
        self : EncoderTests
            The test class instance

        """
        model = encoders.get_huggingface_model(self.test_model.name)
        self.assertTrue(isinstance(model, encoders.PreTrainedModel))

    def test_huggingface_tokenizer(self: "EncoderTests") -> None:
        """Check that a PreTrainedTokenizerBase loads properly.

        Parameters
        ----------
        self : EncoderTests
            The test class instance

        """
        tokenizer = encoders.get_huggingface_tokenizer(self.test_model.name)
        self.assertTrue(isinstance(tokenizer, encoders.PreTrainedTokenizerBase))

    def test_langchain_embeddings_openai(self: "EncoderTests") -> None:
        """Check LangChain embeddings for an OpenAI model.

        Parameters
        ----------
        self: EncoderTests
            The test class instance

        """
        self.assertTrue(isinstance(
            encoders.get_langchain_embedding_model(), encoders.OpenAIEmbeddings,
        ))

    def test_langchain_embeddings_st(self: "EncoderTests") -> None:
        """Check LangChain embeddings for a sentence-transformers model.

        Parameters
        ----------
        self: EncoderTests
            The test class instance

        """
        self.assertTrue(isinstance(
            encoders.get_langchain_embedding_model(
                encoders.EncoderParams(encoders.MPNET, None),
            ),
            encoders.HuggingFaceEmbeddings,
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
