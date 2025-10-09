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
from typing import Any

import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="config", config_name="data", version_base=None)
def main(config: DictConfig):
    # resolve config
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # load dataset
    data_files = [file for file in os.listdir(config.data_dir) if file.startswith(config.split)]
    dataset = load_dataset(**config.dataset, data_files=data_files)

    # process dataset
    def process_fn(example: dict[str, Any]) -> dict[str, Any]:
        prompt = example["prompt"]
        response = example["response"]
        score = example["score"]
        message = prompt + [{"role": "assistant", "content": response}]
        continue_final_message = score == 0.0
        example = {"messages": message, "continue_final_message": continue_final_message}
        return example

    dataset = dataset.filter(lambda x: x["score"] > 0.0)
    dataset = dataset.map(process_fn, remove_columns=dataset.column_names)

    # save dataset
    output_path = os.path.join(config.output_dir, f"{config.split}.parquet")
    dataset.to_parquet(output_path)


if __name__ == "__main__":
    main()
