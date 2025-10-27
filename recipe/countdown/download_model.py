import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    # Download the model
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.bfloat16)
    model.save_pretrained("checkpoints/countdown/llama_3.2_1b")

    # Download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer.save_pretrained("models/countdown/llama_3.2_1b")


if __name__ == "__main__":
    main()
