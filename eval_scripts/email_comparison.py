import json
from pathlib import Path

files_to_compare = [
    "eval-of-" + "states-responsesdaaa_401d8762-b0c7-450b-8f41-13d41fae37c9.txt.txt",
    "eval-of-" + "states-responsesdynaj4a_401d8762-b0c7-450b-8f41-13d41fae37c2.txt.txt"
]

responses = []
for file in files_to_compare:
    if Path(file).is_file():
        with Path(file).open() as f:
            responses.append(json.loads(f.read()))

print(responses)

for result1, result2 in zip(responses[0], responses[1]):
    print(result1[3] + "  ----- " + result2[3])
    print("===============")