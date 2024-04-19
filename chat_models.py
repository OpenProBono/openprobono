GPT_3_5 = "gpt-3.5-turbo-0125"
GPT_4 = "gpt-4"
GPT_4_TURBO = "gpt-4-turbo-preview"

class ChatModelParams:
    """Define a chat model for RAG."""

    def __init__(self: "ChatModelParams", name: str) -> None:
        """Define parameters for a chat model.

        Parameters
        ----------
        name : str
            The name of the chat model

        """
        self.name = name
