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

from datasets import load_dataset


def main():
    # download sft dataset
    sft_datasets = load_dataset("symoon11/countdown-sft")
    for split in sft_datasets.keys():
        sft_dataset = sft_datasets[split]
        sft_dataset.to_parquet(f"data/countdown/sft/{split}.parquet")

    # download rl dataset
    rl_datasets = load_dataset("symoon11/countdown-rl")
    for split in rl_datasets.keys():
        rl_dataset = rl_datasets[split]
        rl_dataset.to_parquet(f"data/countdown/rl/{split}.parquet")


if __name__ == "__main__":
    main()
