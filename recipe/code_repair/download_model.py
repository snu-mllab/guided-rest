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


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    # Download the model
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16)
    model.save_pretrained("checkpoints/code_repair/qwen.2.5_7b/huggingface")

    # Download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    tokenizer.save_pretrained("checkpoints/code_repair/qwen.2.5_7b/huggingface")


if __name__ == "__main__":
    main()
