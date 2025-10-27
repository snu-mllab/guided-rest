# Learning to Better Search with Language Models via Guided Reinforced Self-Training (NeurIPS 2025)


## Setup

```bash
# conda
conda create --name guided-rest python=3.12

# uv
pip install uv

# vllm
uv pip install vllm[flashinfer] --torch-backend=cu128

# verl
uv pip install -e .[gpu] --no-build-isolation --torch-backend=cu128

# fix packages
uv pip uninstall pynvml
```

## Countdown

### 1. Download the SFT and RL datasets
```bash
python -m recipe.countdown.download_data
```
The datasets will be saved under `data/countdown`.

### 2. Download the model and tokenizer
```bash
python -m recipe.countdown.download_model
```
The model and tokenizer will be save under `checkpoints/countdown/llama_3.2_1b`

### 3. Remove `trim` in the chat template since it causes issues during partial rollouts.

### 4. Train the base model
```bash
sh recipe/countdown/scripts/llama_3.2_1b/base/run_sft.sh
sh recipe/countdown/scripts/run_merge.sh model_name=llama_3.2_1b_base_sft/global_step_3906
```

### 5. Run Guided-ReST
```bash
# Generate trajectories
sh recipe/countdown/scripts/run_gen.sh model_name=llama_3.2_1b_base_sft/global_step_3906 temperature=1.0 num_iters=3 split=train start=0 num_examples=200000
sh recipe/countdown/scripts/run_gen.sh model_name=llama_3.2_1b_base_sft/global_step_3906 temperature=1.0 num_iters=3 split=validation start=0 num_examples=1000

# Prepare data
sh recipe/countdown/scripts/run_data.sh model_name=llama_3.2_1b_base_sft/global_step_3906 temperature=1.0 num_iters=3 split=train
sh recipe/countdown/scripts/run_data.sh model_name=llama_3.2_1b_base_sft/global_step_3906 temperature=1.0 num_iters=3 split=valid

# Run SFT
sh recipe/countdown/scripts/llama_3.2_1b/guided_rest/run_sft_1.sh
sh recipe/countdown/scripts/run_merge.sh model_name=llama_3.2_1b_guided_rest_sft_1/global_step_1546

# Repeat the above steps for 3 iterations
```

### 6. Run RL
```bash
sh recipe/countdown/scripts/llama_3.2_1b/guided_rest/run_rl.sh
sh recipe/countdown/scripts/run_merge.sh model_name=llama_3.2_1b_guided_rest_rl/global_step_390/actor
```

### 7. Run evaluation
```bash
# Generate trajectories
sh recipe/countdown/scripts/run_gen.sh model_name=llama_3.2_1b_guided_rest_rl/global_step_390/actor temperature=0.0 num_iters=0 split=test_seen start=0 num_examples=10000
sh recipe/countdown/scripts/run_gen.sh model_name=llama_3.2_1b_guided_rest_rl/global_step_390/actor temperature=0.0 num_iters=0 split=test_unseen start=0 num_examples=10000

# Compute accuracy
sh recipe/countdown/scripts/run_eval.sh model_name=llama_3.2_1b_guided_rest_rl/global_step_390/actor temperature=0.0 num_iters=0 split=test_seen
sh recipe/countdown/scripts/run_eval.sh model_name=llama_3.2_1b_guided_rest_rl/global_step_390/actor temperature=0.0 num_iters=0 split=test_unseeen
```