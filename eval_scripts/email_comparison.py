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

score_card = [0,0]
for result1, result2 in zip(responses[0], responses[1]):
    if(result1[3] != result2[3]):
        print(result1[3] + "  ----- " + result2[3])
        if(result1[3] > result2[3]):
            score_card[0] += 1
        if(result1[3] < result2[3]):
            score_card[1] += 1
        print(result1[1])
        print("===============")

print(score_card)