"""Prompts to use with LLMs."""

from pathlib import Path

prompts_path = Path(Path.cwd() / "app/prompts")

# for bots.py

with (prompts_path / "bot.md").open("r") as f:
    BOT_PROMPT = f.read()

with (prompts_path / "title_chat.md").open("r") as f:
    TITLE_CHAT_PROMPT = f.read()

# for moderation.py

# based on moderation prompt from Anthropic's API:
# https://docs.anthropic.com/claude/docs/content-moderation

with (prompts_path / "moderation.md").open("r") as f:
    MODERATION_PROMPT = f.read()

# for summarization.py

with (prompts_path / "summary.md").open("r") as f:
    SUMMARY_PROMPT = f.read()

with (prompts_path / "summary_refine.md").open("r") as f:
    SUMMARY_REFINE_PROMPT = f.read()

with (prompts_path / "summary_opinion_base.md").open("r") as f:
    OPINION_SUMMARY_BASE_PROMPT = f.read()

with (prompts_path / "summary_opinion_map.md").open("r") as f:
    OPINION_SUMMARY_MAP_PROMPT = f.read().format(
        base_prompt=OPINION_SUMMARY_BASE_PROMPT,
    )

with (prompts_path / "summary_opinion_reduce.md").open("r") as f:
    OPINION_SUMMARY_REDUCE_PROMPT = f.read().format(
        base_prompt=OPINION_SUMMARY_BASE_PROMPT,
    )

# for chat_models.py

with (prompts_path / "hive.md").open("r") as f:
    HIVE_QA_PROMPT = f.read()

# for evaluations.py

with (prompts_path / "evaluation.md").open("r") as f:
    EVALUATION_PROMPT = f.read()

with (prompts_path / "comparison.md").open("r") as f:
    COMPARISON_PROMPT = f.read()

# for search_tools.py

with (prompts_path / "opinion_search.md").open("r") as f:
    FILTERED_CASELAW_PROMPT = f.read()

# for vdb_tools.py

with (prompts_path / "vdb_query.md").open("r") as f:
    VDB_QUERY_PROMPT = f.read()

with (prompts_path / "vdb_source.md").open("r") as f:
    VDB_SOURCE_PROMPT = f.read()

# for classifiers.py

with (prompts_path / "jurisdiction_summary.md").open("r") as f:
    JURISDICTION_SUMMARY_PROMPT = f.read()

with (prompts_path / "jurisdiction.md").open("r") as f:
    JURISDICTION_PROMPT = f.read()

with (prompts_path / "issue_classifier.md").open("r") as f:
    ISSUE_CLASSIFER_PROMPT = f.read()
