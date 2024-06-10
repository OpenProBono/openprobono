import json
from pathlib import Path

files_to_compare = [
    "app/eval/eval-of-" + "states-responsesdaaa_401d8762-b0c7-450b-8f41-13d41fae37c9.txt.txt",
    "app/eval/eval-of-" + "states-responsesdynaj4a_401d8762-b0c7-450b-8f41-13d41fae37c2.txt.txt",
    "app/eval/eval-of-states-responses_metadata2_401d8762-b0c7-450b-8f41-13d41fae37c2.txt",
    "app/eval/eval-of-states-responses_j8_q_k10401d8762-b0c7-450b-8f41-13d41fae37c2.txt",
    "app/eval/eval-of-states-responses_j8_q_k3_dyserp_nr10401d8762-b0c7-450b-8f41-13d41fae37c2.txt",
    "app/eval/eval-of-states-responses.txt"
]

responses = []
for file in files_to_compare:
    if Path(file).is_file():
        with Path(file).open() as f:
            responses.append(json.loads(f.read()))

score_card = [0,0,0,0,0]
for result1, result2, result3, result4, result5 in zip(responses[0], responses[1], responses[2], responses[3], responses[4]):
    print(result1[3] + "  ----- " + result2[3] + "  ----- " + result3[3] + "  ----- " + result4[3] + "  ----- " + result5[3])
    score_card[0] += int(result1[3])
    score_card[1] += int(result2[3])
    score_card[2] += int(result3[3])
    score_card[3] += int(result4[3])
    score_card[4] += int(result5[3])

print(score_card)