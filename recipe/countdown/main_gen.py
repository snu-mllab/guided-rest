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
from pprint import pprint

import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer

from recipe.countdown.core_algos import augment_search_path
from recipe.countdown.reward_function import compute_score


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

    # get messages
    messages = []
    sampling_params = []
    for i in range(len(dataset)):
        message = dataset[i]["prompt"]
        sampling_param = SamplingParams(**config.sampling_params, max_tokens=config.max_response_length)
        messages.append(message)
        sampling_params.append(sampling_param)

    # run LLM
    outputs = llm.chat(messages, sampling_params=sampling_params)

    # get responses
    responses = []
    response_lengths = []
    scores = []
    for i in range(len(dataset)):
        data_source = dataset[i]["data_source"]
        prompt = dataset[i]["prompt"]
        ground_truth = dataset[i]["reward_model"]["ground_truth"]
        extra_info = dataset[i]["extra_info"]
        search_path = outputs[i].outputs[0].text
        response = [{"role": "assistant", "content": search_path}]
        response_length = get_response_length(prompt, response, tokenizer)
        score = compute_score(data_source, search_path, ground_truth, extra_info)
        responses.append(response)
        response_lengths.append(response_length)
        scores.append(score)

    # run algorithm
    for _ in range(config.num_iters):
        # get messages
        indices = []
        messages = []
        sampling_params = []
        for i in range(len(dataset)):
            if scores[i] > 0.0:
                continue
            data_source = dataset[i]["data_source"]
            prompt = dataset[i]["prompt"]
            ground_truth = dataset[i]["reward_model"]["ground_truth"]
            extra_info = dataset[i]["extra_info"]
            search_path = responses[i][-1]["content"]
            search_path = augment_search_path(search_path, extra_info)
            response = [{"role": "assistant", "content": search_path}]
            response_length = get_response_length(prompt, response, tokenizer)
            if response_length >= config.max_response_length:
                continue
            message = prompt + response
            max_tokens = config.max_response_length - response_length
            sampling_param = SamplingParams(**config.sampling_params, max_tokens=max_tokens)
            indices.append(i)
            messages.append(message)
            sampling_params.append(sampling_param)

        # continue if no messages
        if not messages:
            continue

        # run LLM
        outputs = llm.chat(
            messages, sampling_params=sampling_params, add_generation_prompt=False, continue_final_message=True
        )

        # get responses
        for i, message, output in zip(indices, messages, outputs):
            data_source = dataset[i]["data_source"]
            prompt = dataset[i]["prompt"]
            ground_truth = dataset[i]["reward_model"]["ground_truth"]
            extra_info = dataset[i]["extra_info"]
            search_path = message[-1]["content"] + output.outputs[0].text
            response = [{"role": "assistant", "content": search_path}]
            response_length = get_response_length(prompt, response, tokenizer)
            score = compute_score(data_source, search_path, ground_truth, extra_info)
            responses[i] = response
            response_lengths[i] = response_length
            scores[i] = score

    # update dataset
    dataset = dataset.add_column("response", responses)
    dataset = dataset.add_column("response_length", response_lengths)
    dataset = dataset.add_column("score", scores)

    # save dataset
    output_path = os.path.join(config.output_dir, f"{config.split}_{start}_{end}_{config.seed}.parquet")
    dataset.to_parquet(output_path)


if __name__ == "__main__":
    main()
