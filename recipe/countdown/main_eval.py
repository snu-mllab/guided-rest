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
from collections import defaultdict
from pprint import pprint

import hydra
import numpy as np
import pandas as pd
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from scipy.special import beta


def get_pass_at_k_mean(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def get_pass_at_k_var(n: int, c: int, k: int) -> float:
    a = c + 1
    b = n - c + 1
    return beta(a, b + 2 * k) / beta(a, b) - (beta(a, b + k) / beta(a, b)) ** 2


@hydra.main(config_path="config", config_name="eval", version_base=None)
def main(config: DictConfig):
    # resolve config
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # load dataset
    data_files = [file for file in os.listdir(config.data_dir) if file.startswith(config.split)]
    dataset = load_dataset(**config.dataset, data_files=data_files)

    # compute accuracies
    budgets = [0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]
    accuracies = defaultdict(lambda: defaultdict(list))
    for example in dataset:
        response_length = example["response_length"]
        score = example["score"]
        index = example["extra_info"]["index"]
        for budget in budgets:
            accuracy = 1.0 if score > 0.0 and response_length <= budget else 0.0
            accuracies[index][budget].append(accuracy)

    # compute pass@k
    pass_at_k_means = defaultdict(lambda: defaultdict(list))
    pass_at_k_vars = defaultdict(lambda: defaultdict(list))
    for index in accuracies.keys():
        for budget in budgets:
            accuracy = accuracies[index][budget]
            n = len(accuracy)
            c = sum(accuracy)
            for k in [1, 2, 4, 8, 16, 32]:
                pass_at_k_mean = get_pass_at_k_mean(n, c, k)
                pass_at_k_var = get_pass_at_k_var(n, c, k)
                pass_at_k_means[budget][k].append(pass_at_k_mean)
                pass_at_k_vars[budget][k].append(pass_at_k_var)

    # print stats
    stats = {}
    stats["budget"] = budgets

    print("--------------------------------------mean--------------------------------------")
    print("Budget: ", end="")
    for budget in budgets:
        print(f"{budget: 8d}", end="")
    print()
    for k in [1, 2, 4, 8, 16, 32]:
        means = []
        print(f"Pass@{k:2d}:", end="")
        for budget in budgets:
            mean = np.mean(pass_at_k_means[budget][k])
            means.append(mean * 100)
            print(f"{mean * 100:8.2f}", end="")
        stats[f"pass_at_{k}_mean"] = means
        print()

    print("--------------------------------------std---------------------------------------")
    print("Budget: ", end="")
    for budget in budgets:
        print(f"{budget: 8d}", end="")
    print()
    for k in [1, 2, 4, 8, 16, 32]:
        stds = []
        print(f"Pass@{k:2d}:", end="")
        for budget in budgets:
            std = np.sqrt(np.sum(pass_at_k_vars[budget][k])) / len(pass_at_k_vars[budget][k])
            stds.append(std * 100)
            print(f"{std * 100:8.2f}", end="")
        stats[f"pass_at_{k}_std"] = stds
        print()

    # save stats
    stats = pd.DataFrame.from_dict(stats)
    stats = stats.round(2)
    os.makedirs(config.output_dir, exist_ok=True)
    output_path = os.path.join(config.output_dir, f"{config.split}_pass_at_k.csv")
    stats.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
