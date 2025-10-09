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

from recipe.countdown.core_algos import augment_search_path
from recipe.countdown.reward_function import compute_score


@hydra.main(config_path="config", config_name="gen", version_base=None)
def main(config: DictConfig):
    # resolve config
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # load dataset
    dataset = load_dataset(**config.dataset)
    start = config.start
    end = min(start + config.num_examples, len(dataset))
    dataset = dataset.select(range(start, end))

    # create LLM
    llm = LLM(**config.llm)
    tokenizer = llm.get_tokenizer()

    # get messages
    messages = [example["prompt"] for example in dataset]
    sampling_params = [SamplingParams(**config.sampling_params) for _ in messages]

    # generate outputs
    outputs = llm.chat(messages, sampling_params=sampling_params)

    # get responses
    responses = []
    response_lengths = []
    scores = []
    for i in range(len(dataset)):
        example = dataset[i]
        data_source = example["data_source"]
        ground_truth = example["reward_model"]["ground_truth"]
        extra_info = example["extra_info"]
        response = outputs[i].outputs[0].text
        response_length = len(tokenizer.encode(response, add_special_tokens=False))
        score = compute_score(data_source, response, ground_truth, extra_info)
        responses.append(response)
        response_lengths.append(response_length)
        scores.append(score)

    # run algorithm
    for _ in range(config.iteration):
        # get messages
        indices = []
        messages = []
        sampling_params = []
        for i in range(len(dataset)):
            # augment response
            if scores[i] == 0.0:
                example = dataset[i]
                data_source = example["data_source"]
                ground_truth = example["reward_model"]["ground_truth"]
                extra_info = example["extra_info"]
                response = augment_search_path(responses[i], extra_info)
                response_length = len(tokenizer.encode(response, add_special_tokens=False))
                score = compute_score(data_source, response, ground_truth, extra_info)
                if response_length >= config.sampling_params.max_tokens:
                    continue
                responses[i] = response
                response_lengths[i] = response_length
                scores[i] = score
            if scores[i] == 0.0:
                message = example["prompt"] + [{"role": "assistant", "content": responses[i]}]
                sampling_param = SamplingParams(**config.sampling_params)
                sampling_param.max_tokens = config.sampling_params.max_tokens - response_lengths[i]
                indices.append(i)
                messages.append(message)
                sampling_params.append(sampling_param)
        if len(indices) == 0:
            continue

        # generate outputs
        outputs = llm.chat(
            messages, sampling_params=sampling_params, add_generation_prompt=False, continue_final_message=True
        )

        # get responses
        for i, output in zip(indices, outputs):
            example = dataset[i]
            data_source = example["data_source"]
            ground_truth = example["reward_model"]["ground_truth"]
            extra_info = example["extra_info"]
            response = responses[i] + output.outputs[0].text
            response_length = len(tokenizer.encode(response, add_special_tokens=False))
            score = compute_score(data_source, response, ground_truth, extra_info)
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
