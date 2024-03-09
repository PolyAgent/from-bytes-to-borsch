import jsonlines
import requests
import json
import os
from huggingface_hub import InferenceClient


def compose_zno_question_vertex(quest_dict: dict):
    question = quest_dict["question"]
    quest_string = question
    for opt in quest_dict["answers"]:
        var = opt["marker"]
        text = opt["text"]
        quest_string += "Ти повинен вибрати єдиний правильний варіант. Відповідь повинна містити тільки букву."
        quest_string += "\n"
        quest_string += f"{var}: {text}"
    data = {
        "instances": [
            {
                "prompt": f"<start_of_turn>user {quest_string}<end_of_turn>\n<start_of_turn>model\nВідповідь: ",
                "max_tokens": 400,
                "temperature": 1,
                "top_p": 1.0,
                "top_k": 1,
            }
        ]
    }
    return data


def compose_open_question_vertex(quest_dict: dict):
    question = quest_dict["instruction"]
    quest_string = "Ти повинен правильно відповісти на питання. \n"
    quest_string += question
    data = {
        "instances": [
            {
                "prompt": f"<start_of_turn>user {quest_string}<end_of_turn>\n<start_of_turn>model\nВідповідь: ",
                "max_tokens": 120,
                "temperature": 1,
                "top_p": 1.0,
                "top_k": 1,
            }
        ]
    }
    return data


def run_vertex(
    token: str,
    url: str,
    output_file: str,
    input_file="../unlp-2024-shared-task/data/zno.train.jsonl",
    open_questions: bool = False,
    verbose=False,
):
    """Vertex AI running pipeline. If output file exists the run will skip the first n=len(output) questions."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    all_answers = []
    assert os.path.isfile(input_file) == True, "Input file not found."
    if os.path.isfile(output_file):
        with open(output_file, "r") as f:
            all_answers = json.loads(f.read())

    with jsonlines.open(input_file) as f:
        for n, line in enumerate(f.iter()):
            if n % 10 == 0:
                print("question", n)
            if n <= len(all_answers):
                continue
            if open_questions:
                question = compose_open_question_vertex(line)
            else:
                question = compose_zno_question_vertex(line)

            response = requests.post(url, headers=headers, json=question)
            if "predictions" in json.loads(response.text):
                output = str(json.loads(response.text)["predictions"]).split(
                    "Output:\\n"
                )[1]
                if verbose:
                    print(output)
            else:
                output = None
                if verbose:
                    print("Nope")
            all_answers[n] = output
            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(
                    all_answers,
                    file,
                    ensure_ascii=False,
                )


def compose_zno_question_hf(quest_dict):
    question = quest_dict["question"]
    quest_string = "Ти повинен вибрати єдиний правильний варіант. \n"
    quest_string += question
    for opt in quest_dict["answers"]:
        var = opt["marker"]
        text = opt["text"]

        quest_string += "\n "
        quest_string += f"{var}: {text}"

    return f"<start_of_turn>user\n{quest_string}. Твоя відповідь повинна містити тільки букву. <end_of_turn> <start_of_turn>model \n Відповідь:"


def get_response(stream, gen_kwargs):
    phrase = []
    # yield each generated token
    for r in stream:
        # skip special tokens
        if r.token.special:
            continue
        # stop if we encounter a stop sequence
        if r.token.text in gen_kwargs["stop_sequences"]:
            break
        # yield the generated token
        phrase.append(r.token.text)
        # yield r.token.text
    return "".join(phrase)


def compose_open_question_hf(quest_dict):
    question = quest_dict
    quest_string = "Ти повинен відповісти на питання. \n"
    quest_string += question
    return f"<start_of_turn>user\n{quest_string}. <end_of_turn> <start_of_turn>model \n Відповідь:"


def run_hf(
    token: str,
    url: str,
    output_file: str,
    input_file="../unlp-2024-shared-task/data/zno.train.jsonl",
    open_questions: bool = False,
    verbose=False,
):
    """Hugging face running pipeline. If output file exists the run will skip the existing answers."""

    client = InferenceClient(url, token=token)
    gen_kwargs = dict(
        max_new_tokens=488,
        top_k=30,
        top_p=0.9,
        temperature=1,
        repetition_penalty=1.02,
        stop_sequences=["\nUser:", "<|endoftext|>", "</s>"],
    )
    all_answers = {}
    assert os.path.isfile(input_file) == True, "Input file not found."
    if os.path.isfile(output_file):
        with open(output_file, "r") as f:
            all_answers = json.loads(f.read())

    with jsonlines.open(input_file) as f:
        for n, line in enumerate(f.iter()):
            if n % 10 == 0:
                print("question", n)
            if n in all_answers:
                continue

            if open_questions:
                question = compose_open_question_hf(line)
            else:
                question = compose_zno_question_hf(line)

            main_stream = client.text_generation(
                question, stream=True, details=True, **gen_kwargs
            )
            resp = get_response(main_stream, gen_kwargs)
            if verbose:
                print((question))
                print(resp)
            all_answers[n] = resp
            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(
                    all_answers,
                    file,
                    ensure_ascii=False,
                )


replacement_dict = {
    "A": "А",
    "B": "В",
    "D": "Д",
    "Ґ": "Г",
    "1": "А",
    "2": "Б",
    "3": "В",
    "4": "Г",
    "5": "Д",
}


def answer_pp(raw_answers):
    text = raw_answers
    if not text:
        return "X"
    for orig, replacement in replacement_dict.items():
        text = text.replace(orig, replacement)
    for l in text:
        if l not in ["А", "Б", "В", "Г", "Д"]:
            continue
        else:
            return l
    return "Х"


def measure_metrics(gt_file, predictions_file):
    accuracy = 0
    with open(predictions_file, "r") as f:
        predictions = json.loads(f.read())
    with jsonlines.open(gt_file) as f:
        for n, line in enumerate(f.iter()):
            if str(n) in predictions and line["correct_answers"][0] == answer_pp(
                predictions[str(n)]
            ):
                accuracy += 1

    print("accuracy:", accuracy / len(predictions))
