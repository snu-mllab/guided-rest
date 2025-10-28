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
    sft_train_dataset = load_dataset("symoon11/countdown-sft", split="train")
    sft_valid_dataset = load_dataset("symoon11/countdown-sft", split="validation")

    # save sft dataset
    sft_train_dataset.to_parquet(f"data/countdown/sft/train.parquet")
    sft_valid_dataset.to_parquet(f"data/countdown/sft/valid.parquet")

    # download rl dataset
    rl_train_dataset = load_dataset("symoon11/countdown-rl", split="train")
    rl_valid_dataset = load_dataset("symoon11/countdown-rl", split="validation")
    rl_test_seen_dataset = load_dataset("symoon11/countdown-rl", split="test_seen")
    rl_test_unseen_dataset = load_dataset("symoon11/countdown-rl", split="test_unseen")

    # save rl dataset
    rl_train_dataset.to_parquet(f"data/countdown/rl/train.parquet")
    rl_valid_dataset.to_parquet(f"data/countdown/rl/valid.parquet")
    rl_test_seen_dataset.to_parquet(f"data/countdown/rl/test_seen.parquet")
    rl_test_unseen_dataset.to_parquet(f"data/countdown/rl/test_unseen.parquet")


if __name__ == "__main__":
    main()
