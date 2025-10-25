# Copyright 2025 SNU MLLAB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass
from pprint import pprint
from typing import Any

import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer

from recipe.code_repair.rewards.code_reward import rllm_reward_fn_code

DUMMY_SOLUTION = """
```python
# REFERENCE CODE HERE
```
""".strip()

USER_PROMPT_NO_CODE = """
No code found. Please output your final code.
""".strip()

USER_PROMPT_NO_TESTS = """
No test cases found. Please review your solution once more for correctness and efficiency, then output your final code if you're confident it's optimal.
""".strip()

USER_PROMPT_PASSED = """
Congratulations! You've successfully passed all test cases. Please carefully review your solution one more time to ensure it handles all edge cases properly. If you're confident your code is optimal, you can proceed with outputting your final solution.
""".strip()

USER_PROMPT_FAILED = """
Here are the results on the public test cases:

{formatted_tests}

Some test cases are still failing. Please carefully analyze the error patterns, revise your code to address these issues, and ensure your solution handles all the test cases correctly. Then, output your final code.
""".strip()

USER_PROMPT_SOLUTION = """
{user_prompt}

Here is the reference code to assist you:

{solution}
""".strip()


@dataclass
class Batch:
    data_sources: list[str]
    prompts: list[list[dict[str, str]]]
    questions: list[str]
    ground_truths: list[str]
    solutions: list[str]
    responses: list[list[dict[str, str]]]
    observations: list[dict[str, Any]]
    scores: list[float]
    dones: list[bool]

    def __len__(self):
        return len(self.data_sources)


def normalize_string(s: str) -> str:
    return "".join(s.split())


def truncate_string(s: str, length: int = 300) -> str:
    if len(s) <= length:
        return s
    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


def get_public_tests(tests: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    normalized_question = normalize_string(question)
    public_tests = []
    for test in tests:
        if isinstance(test["input"], list):
            strings_to_match = [normalize_string(str(s)) for s in test["input"]]
        elif isinstance(test["input"], str):
            strings_to_match = [normalize_string(s) for s in test["input"].split("\n")]
        else:
            raise TypeError(f"input must be a list or a string, but got {type(test['input'])}")
        if all(s in normalized_question for s in strings_to_match):
            public_tests.append(test)
    if not public_tests:
        public_tests = tests[:2]
    return public_tests


def is_done(observation: dict[str, Any], question: str) -> bool:
    tests = observation.get("test_results")
    if tests is None:
        return False
    public_tests = get_public_tests(tests, question)
    if not public_tests:
        return False
    done = all(test["passed"] for test in public_tests)
    return done


def format_tests(tests: list[dict[str, Any]]) -> str:
    formatted_tests = ""
    for i, test in enumerate(tests):
        if not test["passed"]:
            formatted_tests += f"### Test {i + 1} failed\n"
            formatted_tests += f"Input: {truncate_string(str(test['input']))}\n"
            formatted_tests += f"Expected: {truncate_string(str(test['expected']))}\n"
            formatted_tests += f"Output: {truncate_string(str(test.get('output')))}\n"
            formatted_tests += f"Error message: {truncate_string(str(test.get('error_message')))}\n"
    formatted_tests = formatted_tests.strip()
    return formatted_tests


def get_user_prompt(observation: dict[str, Any], question: str) -> str:
    tests = observation.get("test_results")
    if tests is None:
        return USER_PROMPT_NO_CODE
    public_tests = get_public_tests(tests, question)
    if not public_tests:
        return USER_PROMPT_NO_TESTS
    formatted_tests = format_tests(public_tests)
    if not formatted_tests:
        return USER_PROMPT_PASSED
    else:
        return USER_PROMPT_FAILED.format(formatted_tests=formatted_tests)


def append_solution(user_prompt: str, solution: str) -> str:
    return USER_PROMPT_SOLUTION.format(user_prompt=user_prompt, solution=solution)


def get_sequence_length(message: list[dict[str, str]], tokenizer: AnyTokenizer) -> int:
    tokens = tokenizer.apply_chat_template(message, add_generation_prompt=message[-1]["role"] == "user")
    sequence_length = len(tokens)
    return sequence_length


def get_response_length(prompt: list[dict[str, str]], response: list[dict[str, str]], tokenizer: AnyTokenizer) -> int:
    message = prompt + response
    prompt_length = get_sequence_length(prompt, tokenizer)
    message_length = get_sequence_length(message, tokenizer)
    response_length = message_length - prompt_length
    return response_length


def chat(batch: Batch, llm: LLM, tokenizer: AnyTokenizer, config: DictConfig, turn: int, use_solution: bool = False):
    # get messages
    indices = []
    messages = []
    sampling_params = []
    for i in range(len(batch)):
        if batch.dones[i]:
            continue
        response = batch.responses[i][: 2 * turn - 1]
        if turn > 0:
            if not response:
                continue
            user_prompt = get_user_prompt(batch.observations[i], batch.questions[i])
            if use_solution:
                user_prompt = append_solution(user_prompt, batch.solutions[i])
            response += [{"role": "user", "content": user_prompt}]
        response_length = get_response_length(batch.prompts[i], response, tokenizer)
        if response_length >= config.max_response_length:
            continue
        message = batch.prompts[i] + response
        max_tokens = config.max_response_length - response_length
        sampling_param = SamplingParams(**config.sampling_params, max_tokens=max_tokens)
        indices.append(i)
        messages.append(message)
        sampling_params.append(sampling_param)

    # return if no messages
    if not messages:
        return

    # run LLM
    outputs = llm.chat(messages, sampling_params=sampling_params)

    # get responses
    for i, output in zip(indices, outputs):
        response = batch.responses[i][: 2 * turn - 1]
        if turn > 0:
            user_prompt = get_user_prompt(batch.observations[i], batch.questions[i])
            if use_solution:
                user_prompt = append_solution(user_prompt, DUMMY_SOLUTION)
            response += [{"role": "user", "content": user_prompt}]
        llm_solution = output.outputs[0].text
        response += [{"role": "assistant", "content": llm_solution}]
        response_length = get_response_length(batch.prompts[i], response, tokenizer)
        if response_length >= config.max_response_length:
            continue
        reward = rllm_reward_fn_code(batch.data_sources[i], llm_solution, batch.ground_truths[i])
        observation = reward.metadata
        score = reward.reward
        done = is_done(observation, batch.questions[i])
        batch.responses[i] = response
        batch.observations[i] = observation
        batch.scores[i] = score
        batch.dones[i] = done


@hydra.main(config_path="config", config_name="gen", version_base=None)
def main(config: DictConfig):
    # resolve config
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # load LLM
    max_model_len = config.max_prompt_length + config.max_response_length
    llm = LLM(**config.llm, max_model_len=max_model_len)
    tokenizer = llm.get_tokenizer()

    # load dataset
    dataset = load_dataset(**config.dataset)
    start = config.start
    end = min(start + config.num_examples, len(dataset))
    dataset = dataset.select(range(start, end))
    dataset = dataset.filter(lambda x: get_sequence_length(x["prompt"], tokenizer) <= config.max_prompt_length)

    # initialize batch
    data_sources = [dataset[i]["data_source"] for i in range(len(dataset))]
    prompts = [dataset[i]["prompt"] for i in range(len(dataset))]
    questions = [dataset[i]["prompt"][-1]["content"] for i in range(len(dataset))]
    ground_truths = [dataset[i]["reward_model"]["ground_truth"] for i in range(len(dataset))]
    solutions = [dataset[i]["extra_info"]["solution"] for i in range(len(dataset))]
    responses = [[] for _ in range(len(dataset))]
    observations = [{} for _ in range(len(dataset))]
    scores = [0.0 for _ in range(len(dataset))]
    dones = [False for _ in range(len(dataset))]
    batch = Batch(
        data_sources=data_sources,
        prompts=prompts,
        questions=questions,
        ground_truths=ground_truths,
        solutions=solutions,
        responses=responses,
        observations=observations,
        scores=scores,
        dones=dones,
    )

    # run generation
    for turn in range(config.num_turns):
        chat(batch, llm, tokenizer, config, turn, use_solution=False)

    # run algorithm
    for start_turn in range(1, config.num_iters + 1):
        for turn in range(start_turn, config.num_turns):
            chat(batch, llm, tokenizer, config, turn, use_solution=turn == start_turn)

    # update dataset
    dataset = dataset.add_column("response", batch.responses)
    dataset = dataset.add_column("score", batch.scores)
    dataset = dataset.remove_columns("reward_model")

    # save dataset
    output_path = os.path.join(config.output_dir, f"{config.split}_{start}_{end}_{config.seed}.parquet")
    dataset.to_parquet(output_path)


if __name__ == "__main__":
    main()
