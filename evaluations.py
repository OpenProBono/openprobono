from pdfs import get_docs
from encoder import OPENAI_3_SMALL
from os import getcwd

documents = get_docs(getcwd() + "/data/US/", "usc04@118-30.pdf")

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# generator with openai models
generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
critic_llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings(model=OPENAI_3_SMALL)

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

# generate testset
testset = generator.generate_with_langchain_docs(documents, test_size=10, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})
testset.to_pandas().to_csv("synthetic-data.csv", sep='\t')

# OPENAI EXAMPLE from 

# import pandas as pd
# import os
# import yaml

# data_path = "/data/MMLU/"

# # Build the prompts using Chat format. We support converting Chat conversations to text for non-Chat models

# choices = ["A", "B", "C", "D"]
# sys_msg = "The following are multiple choice questions (with answers) about {}."
# def create_chat_prompt(sys_msg, question, answers, subject):
#     user_prompt = f"{question}\n" + "\n".join([f"{choice}. {answer}" for choice, answer in zip(choices, answers)]) + "\nAnswer:"
#     return [
#         {"role": "system", "content": sys_msg.format(subject)}, 
#         {"role": "user", "content": user_prompt}
#     ]

# def create_chat_example(question, answers, correct_answer):
#     """
#     Form few-shot prompts in the recommended format: https://github.com/openai/openai-python/blob/main/chatml.md#few-shot-prompting
#     """
#     user_prompt = f"{question}\n" + "\n".join([f"{choice}. {answer}" for choice, answer in zip(choices, answers)]) + "\nAnswer:"
#     return [
#         {"role": "system", "content": user_prompt, "name": "example_user"},
#         {"role": "system", "content": correct_answer, "name": "example_assistant"},
#     ]

# subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(data_path, "test")) if "_test.csv" in f])
# registry_yaml = {}

# for subject in subjects:
#     subject_path = os.path.join(registry_path, "data", "mmlu", subject)
#     os.makedirs(subject_path, exist_ok=True)

#     # Create few-shot prompts
#     dev_df = pd.read_csv(os.path.join(data_path, "dev", subject + "_dev.csv"), names=("Question", "A", "B", "C", "D", "Answer"))
#     dev_df["sample"] = dev_df.apply(lambda x: create_chat_example(x["Question"], x[["A", "B", "C", "D"]], x["Answer"]), axis=1)
#     few_shot_path = os.path.join(subject_path, "few_shot.jsonl")     
#     dev_df[["sample"]].to_json(few_shot_path, lines=True, orient="records")

#     # Create test prompts and ideal completions
#     test_df = pd.read_csv(os.path.join(data_path, "test", subject + "_test.csv"), names=("Question", "A", "B", "C", "D", "Answer"))
#     test_df["input"] = test_df.apply(lambda x: create_chat_prompt(sys_msg, x["Question"], x[["A", "B", "C", "D"]], subject), axis=1)
#     test_df["ideal"] = test_df.Answer
#     samples_path = os.path.join(subject_path, "samples.jsonl")     
#     test_df[["input", "ideal"]].to_json(samples_path, lines=True, orient="records")

#     eval_id = f"match_mmlu_{subject}"

#     registry_yaml[eval_id] = {
#         "id": f"{eval_id}.test.v1",
#         "metrics": ["accuracy"]
#     }
#     registry_yaml[f"{eval_id}.test.v1"] = {
#         "class": "evals.elsuite.basic.match:Match",
#         "args": {
#             "samples_jsonl": samples_path,
#             "few_shot_jsonl": few_shot_path,
#             "num_few_shot": 4,
#         }
#     }

# with open(os.path.join(registry_path, "evals", "mmlu.yaml"), "w") as f:
#     yaml.dump(registry_yaml, f)