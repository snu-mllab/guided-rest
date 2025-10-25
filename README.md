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
