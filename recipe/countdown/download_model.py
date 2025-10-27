import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    # Download the model
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", dtype=torch.bfloat16)
    model.save_pretrained("checkpoints/countdown/llama_3.2_1b/huggingface")

    # Download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer.save_pretrained("checkpoints/countdown/llama_3.2_1b/huggingface")


if __name__ == "__main__":
    main()
