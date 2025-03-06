Your task is to classify a URL into state, federal, and/or UK jurisdictions. Your output must be a list in the following format:

[
  {{
    "name": a two-letter code of a U.S. state, "US" if the jurisdiction is federal, or "UK" if the jurisdiction is the United Kingdom,
    "confidence": a number between 0 and 1
  }}
]

{optional_summary_prompt}

Multiple jurisdictions are allowed. Do not output anything else.