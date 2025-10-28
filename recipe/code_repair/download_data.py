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
    # download rl dataset
    rl_train_dataset = load_dataset("symoon11/code-repair-rl-pi", split="train")
    rl_valid_dataset = load_dataset("symoon11/code-repair-rl-pi", split="validation")
    rl_test_cc_dataset = load_dataset("symoon11/code-repair-rl-cc", split="test")
    rl_test_cf_dataset = load_dataset("symoon11/code-repair-rl-cf", split="test")

    # save rl dataset
    rl_train_dataset.to_parquet(f"data/code_repair/rl/train.parquet")
    rl_valid_dataset.to_parquet(f"data/code_repair/rl/valid.parquet")
    rl_test_cc_dataset.to_parquet(f"data/code_repair/rl/test_cc.parquet")
    rl_test_cf_dataset.to_parquet(f"data/code_repair/rl/test_cf.parquet")


if __name__ == "__main__":
    main()
