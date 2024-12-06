Your task is to classify a URL into state and/or federal jurisdictions. Your output must be a list in the following format:

[
  {{
    "name": a two letter code of a state or "US" if the jurisdiction is federal,
    "confidence": a number between 0 and 1
  }}
]

{optional_summary_prompt}

Multiple jurisdictions are allowed. Do not output anything else.