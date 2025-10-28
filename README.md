## Learning to Better Search with Language Models via Guided Reinforced Self-Training (NeurIPS 2025)


### Setup

```bash
# conda
conda create --name guided-rest python=3.12
conda activate guided-rest

# uv
pip install uv

# vllm
uv pip install vllm[flashinfer] --torch-backend=cu128

# verl
uv pip install -e .[gpu] --no-build-isolation --torch-backend=cu128

# fix packages
uv pip uninstall pynvml
```

### Countdown

1. Download the SFT and RL datasets
```bash
python -m recipe.countdown.download_data
```

2. Download the model and tokenizer
```bash
python -m recipe.countdown.download_model
```

3. Remove `trim` in the chat template to avoid inconsistent encoding

4. Train the base model
```bash
sh recipe/countdown/scripts/llama_3.2_1b/base/run_sft.sh
sh recipe/countdown/scripts/run_merge.sh model_name=llama_3.2_1b_base_sft/global_step_3906
```

5. Run Guided-ReST
```bash
# Generate trajectories
sh recipe/countdown/scripts/run_gen.sh model_name=llama_3.2_1b_base_sft/global_step_3906 temperature=1.0 num_iters=3 split=train start=0 num_examples=200000
sh recipe/countdown/scripts/run_gen.sh model_name=llama_3.2_1b_base_sft/global_step_3906 temperature=1.0 num_iters=3 split=valid start=0 num_examples=1000

# Prepare data
sh recipe/countdown/scripts/run_data.sh model_name=llama_3.2_1b_base_sft/global_step_3906 temperature=1.0 num_iters=3 split=train
sh recipe/countdown/scripts/run_data.sh model_name=llama_3.2_1b_base_sft/global_step_3906 temperature=1.0 num_iters=3 split=valid

# Run SFT
sh recipe/countdown/scripts/llama_3.2_1b/guided_rest/run_sft_1.sh
sh recipe/countdown/scripts/run_merge.sh model_name=llama_3.2_1b_guided_rest_sft_1/global_step_1546

# Repeat the above steps for 3 iterations
```

6. Run PPO
```bash
sh recipe/countdown/scripts/llama_3.2_1b/guided_rest/run_rl.sh
sh recipe/countdown/scripts/run_merge.sh model_name=llama_3.2_1b_guided_rest_rl/global_step_390/actor
```

7. Run evaluation

- Greedy sampling
```bash
# Generate trajectories
sh recipe/countdown/scripts/run_gen.sh model_name=llama_3.2_1b_guided_rest_rl/global_step_390/actor temperature=0.0 num_iters=0 split=test_seen start=0 num_examples=10000
sh recipe/countdown/scripts/run_gen.sh model_name=llama_3.2_1b_guided_rest_rl/global_step_390/actor temperature=0.0 num_iters=0 split=test_unseen start=0 num_examples=10000

# Compute accuracy
sh recipe/countdown/scripts/run_eval.sh model_name=llama_3.2_1b_guided_rest_rl/global_step_390/actor temperature=0.0 num_iters=0 split=test_seen
sh recipe/countdown/scripts/run_eval.sh model_name=llama_3.2_1b_guided_rest_rl/global_step_390/actor temperature=0.0 num_iters=0 split=test_unseen
```

- Random sampling
```bash
# Generate trajectories with seeds from 0 to 32
sh recipe/countdown/scripts/run_gen.sh model_name=llama_3.2_1b_guided_rest_rl/global_step_390/actor temperature=1.0 num_iters=0 split=test_seen start=0 num_examples=10000 seed=[seed]
sh recipe/countdown/scripts/run_gen.sh model_name=llama_3.2_1b_guided_rest_rl/global_step_390/actor temperature=1.0 num_iters=0 split=test_unseen start=0 num_examples=10000 seed=[seed]

# Compute accuracy
sh recipe/countdown/scripts/run_eval.sh model_name=llama_3.2_1b_guided_rest_rl/global_step_390/actor temperature=1.0 num_iters=0 split=test_seen
sh recipe/countdown/scripts/run_eval.sh model_name=llama_3.2_1b_guided_rest_rl/global_step_390/actor temperature=1.0 num_iters=0 split=test_unseen
```

### Code self-repair

1. Download the SFT and RL datasets
```bash
python -m recipe.code_repair.download_data
```

2. Download the model and tokenizer
```bash
python -m recipe.code_repair.download_model
```

3. Run Guided-ReST
```bash
# Generate trajectories 
sh recipe/code_repair/scripts/run_gen.sh model_name=qwen2.5_7b temperature=1.0 num_turns=4 num_iters=3 split=train start=0 num_examples=16000
sh recipe/code_repair/scripts/run_gen.sh model_name=qwen2.5_7b temperature=1.0 num_turns=4 num_iters=3 split=valid start=0 num_examples=300

# Prepare data
sh recipe/code_repair/scripts/run_data.sh model_name=qwen2.5_7b temperature=1.0 num_iters=3 split=train
sh recipe/code_repair/scripts/run_data.sh model_name=qwen2.5_7b temperature=1.0 num_iters=3 split=valid

# Run SFT
sh recipe/code_repair/scripts/qwen2.5_7b/guided_rest/run_sft_1.sh
sh recipe/code_repair/scripts/run_merge.sh model_name=qwen2.5_7b_guided_rest_sft_1/global_step_348

# Repeat the above steps for 3 iterations
```

4. Run evaluation

```bash
# Generate trajectories with seeds from 0 to 128
sh recipe/code_repair/scripts/run_gen.sh model_name=qwen2.5_7b_guided_rest_sft_3/global_step_348 temperature=1.0 num_turns=4 num_iters=0 split=test_cc start=0 num_examples=200 seed=[seed]
sh recipe/code_repair/scripts/run_gen.sh model_name=qwen2.5_7b_guided_rest_sft_3/global_step_348 temperature=1.0 num_turns=4 num_iters=0 split=test_cf start=0 num_examples=500 seed=[seed]

# Compute accuracy
sh recipe/code_repair/scripts/run_eval.sh model_name=qwen2.5_7b_guided_rest_sft_3/global_step_348 temperature=1.0 num_turns=4 num_iters=0 split=test_cc
sh recipe/code_repair/scripts/run_eval.sh model_name=qwen2.5_7b_guided_rest_sft_3/global_step_348 temperature=1.0 num_turns=4 num_iters=0 split=test_cf
```
