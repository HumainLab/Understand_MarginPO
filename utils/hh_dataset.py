import re
from typing import List, Dict


def common_start(str1: str, str2: str) -> str:
    # Zip the two strings and iterate over them together
    common_chars = []
    for c1, c2 in zip(str1, str2):
        if c1 == c2:
            common_chars.append(c1)
        else:
            break
    # Join the common characters and return as a string
    return "".join(common_chars)


def extract_dialogue(example: str) -> List[Dict[str, str]]:
    # Extract the prompt, which corresponds to the common start of the chosen and rejected dialogues
    prompt_text = common_start(example["chosen"], example["rejected"])

    # The chosen and rejected may share a common start, so we need to remove the common part
    if not prompt_text.endswith("\n\nAssistant: "):
        prompt_text = prompt_text[: prompt_text.rfind("\n\nAssistant: ")] + "\n\nAssistant: "

    # Extract the chosen and rejected lines
    chosen_line = example["chosen"][len(prompt_text) :]
    rejected_line = example["rejected"][len(prompt_text) :]

    # Remove the generation prompt ("\n\nAssistant: ") from the prompt
    prompt_text = prompt_text[: -len("\n\nAssistant: ")]

    # Split the string at every occurrence of "Human: " or "Assistant: "
    prompt_lines = re.split(r"(\n\nAssistant: |\n\nHuman: )", prompt_text)

    # Remove the first element as it's empty
    prompt_lines = prompt_lines[1:]

    prompt = []
    for idx in range(0, len(prompt_lines), 2):
        role = "user" if prompt_lines[idx] == "\n\nHuman: " else "assistant"
        content = prompt_lines[idx + 1]
        prompt.append({"role": role, "content": content})

    # Remove the prompt from the chosen and rejected dialogues
    chosen = [{"role": "assitant", "content": chosen_line}]
    rejected = [{"role": "assistant", "content": rejected_line}]

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}