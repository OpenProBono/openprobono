"""Prompts to use with LLMs."""

# for bots.py

MAX_NUM_TOOLS = 8

# https://docs.anthropic.com/claude/docs/tool-use-examples
# To prompt Sonnet or Haiku to better assess the user query before making tool calls,
# the following prompt can be used:"
ANTHROPIC_SONNET_TOOL_PROMPT = (
    "Answer the user's request using relevant tools (if they are available). "
    "Before calling a tool, do some analysis within <thinking></thinking> tags. "
    "First, think about which of the provided tools is the relevant tool to answer the "
    "user's request. Second, go through each of the required parameters of the "
    " relevant tool and determine if the user has directly provided or given enough "
    "information to infer a value. When deciding if the parameter can be inferred, "
    "carefully consider all the context to see if it supports a specific value. If all "
    "of the required parameters are present or can be reasonably inferred, close the "
    "thinking tag and proceed with the tool call. BUT, if one of the values for a "
    "required parameter is missing, DO NOT invoke the function (not even with fillers "
    "for the missing params) and instead, ask the user to provide the missing "
    "parameters. DO NOT ask for more information on optional parameters if it is not "
    "provided."
)

COMBINE_TOOL_OUTPUTS_TEMPLATE = """You are Lacuna, a legal expert who helps the public research the law, find information, and explore their first steps to address their legal problem. You are equipped with tools that enable you to query the web or databases for specific information. Always use your tools to form your responses. Every tool accepts a search query argument and returns a list of search results. Each result contains a source URL. In your responses, every sentence containing information from a tool result must have a numbered in-text citation for the associated source URL. If the same information came from multiple tool results, or there is information from multiple tool results, include a citation with the number for each source URL used. At the end of your responses, you must include a "Sources" section with each associated number and URL. Your responses should be in plain language."""

NEW_TEST_TEMPLATE = """You are a legal expert, tasked with answering any questions about law. ALWAYS use tools to answer questions.

You can use multiple tools and the same tool multiple times with different inputs with the goal of gettting relevant resulsts.

Considering the results from the tools you used, decide if another tool call could be useful in providing a comprehensive answer.

If not, then provide an answer to the user's question. ALWAYS return a "SOURCES" part in your answer.
"""

# for moderation.py

# based on moderation prompt from Anthropic's API:
# https://docs.anthropic.com/claude/docs/content-moderation
MODERATION_PROMPT = (
    "A human user is in dialogue with an AI. The human is asking the AI questions "
    "about legal information and resources. Here is the most recent request from "
    "the user:<user query>{user_input}</user query>\n\n"
    "If the user's request refers to harmful, pornographic, or illegal activities, "
    "reply with (Y). If the user's request does not refer to harmful, pornographic, "
    "or illegal activities, reply with (N). Reply with nothing else other than (Y) or "
    "(N)."
)

# for summarization.py

SUMMARY_PROMPT = (
    "Write a concise summary of the following:"
    "\n\n{text}\n\n"
    "CONCISE SUMMARY:"
)

SUMMARY_MAP_PROMPT = (
    "Write a concise summary of the following:"
    "\n\n{text}\n\n"
    "CONCISE SUMMARY:"
)

SUMMARY_REFINE_PROMPT = (
    "Taking the following context delimited by triple backquotes into consideration:"
    "\n\n```{context}```\n\n"
    "Write a concise summary of the following text delimited by triple backquotes."
    "\n\n```{text}```\n\n"
    "CONCISE SUMMARY:"
)

OPINION_SUMMARY_BASE_PROMPT = """Write your summary as a list of bullet points in the following format:

- **Parties**: <parties>
- **Introduction**: <introduction>
- **Background**: <background>
- **Procedural History**: <procedural history>
- **Issues Presented**: <issues presented>
- **Analysis**: <analysis>
- **Holding**: <holding>

Each bullet point should be no longer than 3 sentences. Only include the bullets above, and do not change the titles."""

OPINION_SUMMARY_MAP_PROMPT = \
    "Your task is to summarize a judicial opinion. " +\
    OPINION_SUMMARY_BASE_PROMPT +\
    """ If there is not any information related to a title, write "I could not find any information." for that bullet point."""

OPINION_SUMMARY_REDUCE_PROMPT = \
    """Your task is to combine partial summaries of a judicial opinion into a single, comprehensive summary.""" +\
    OPINION_SUMMARY_BASE_PROMPT

# for chat_models.py

HIVE_QA_PROMPT = (
    "You are a legal assistant that provides factual information about "
    "legal code based on the following prompts.  After using information "
    "from a selected prompt, please attempt to provide a reference to the "
    "section that the information was pulled from. If a question does not "
    "make sense or is not factually coherent, explain why instead of "
    "answering something not correct. Please respond clearly and concisely "
    "as if talking to the customer directly."
)

# for evaluations.py

EVALUATION_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback:"""

COMPARISON_PROMPT = """You will be given a question, the true answer to that question, and two answers generated by AI language models in response to the question. Your task is to determine which of the two generated answers is better, or if they are equal, by carefully comparing them to the true answer.

Here is the question:
{question}

Here is the true answer to the question:
{true_answer}

Here is generated answer A:
{answer_a}

And here is generated answer B:
{answer_b}

First, analyze each generated answer and compare it to the true answer. Consider the following:

- How accurate is the generated answer compared to the true answer? Does it contain any incorrect statements or information?
- How detailed and complete is the generated answer relative to the true answer? Does it cover all the key points, or is important information missing? 
- How relevant and on-topic is the generated answer? Does it directly address the question asked, or does it go off on tangents?

Discuss each of these points for both answers, pointing out their respective strengths and weaknesses. Aim to be as objective as possible in your analysis.

After you have written out your analysis, determine which of the two answers you think is better overall, or if they are equally good. Remember, do not simply choose the longer answer - a shorter answer that efficiently covers the key points may be better than a longer one that includes extraneous or incorrect information. Focus on accuracy, relevance and completeness relative to the true answer.

The output format should look as follows: \"Feedback: {{Write out your analysis and reasoning}} [RESULT] {{A or B or Tie}}\"
Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output."""

# for search_tools.py

FILTERED_CASELAW_PROMPT = (
    "Use to find case law and optionally filter by jurisdiction and date range. "
    "You can search by semantic or keyword. For example, to search for "
    "cases related to workers compensation that cite the Jones Act, you "
    'can search semantically for "workers compensation" and search by '
    'keyword for "Jones Act". Be sure to enter dates in YYYY-MM-DD format. '
    "The citation for this tool should include a link in the following format: "
    "'https://www.courtlistener.com/opinion/' + metadata['cluster_id'] + '/' + metadata['slug']"
)

# for vdb_tools.py

VDB_PROMPT = (
    "This tool queries a database named {collection_name} "
    "and returns the top {k} results. "
    "The database description is: {description}."
)

# for list_classifier.py

ISSUE_CLASSIFER_PROMPT = """You are a legal analysis AI trained to categorize non-lawyer descriptions of situations into predefined legal categories. Your task is to analyze a given situation and provide a probability distribution of possible classifications over a set of legal categories.

First, here are the legal categories you will be considering:

{terms}

Now, here is the situation description provided by a non-lawyer:

{message}

To complete this task, follow these steps:

1. Carefully read and analyze the situation description.
2. Consider how the described situation might relate to each of the provided legal categories.
3. Assess the probability that the situation falls under each category. The probabilities should sum to 1 (100%).

Present your probability distribution in this JSON format:

{{
  "categories": [
    {{
      "title": "Category Name",
      "probability": 0.XX,
    }},
    ...
  ]
}}

Here's an example of how your output should look:

{{
  "categories": [
    {{
      "title": "Accidents and Torts",
      "probability": 0.65,
    }},
    {{
      "title": "Money, Debt, and Consumer Issues",
      "probability": 0.25,
    }},
    {{
      "title": "Housing",
      "probability": 0.10,
    }}
  ]
}}

Remember to consider all provided categories in your analysis, even if some have very low or zero probability. Ensure that your probabilities sum to 1 (100%) across all categories.

Do not output anything besides valid JSON."""
