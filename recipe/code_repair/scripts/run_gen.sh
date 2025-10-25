set -x

export TOKENIZERS_PARALLELISM=false

python -m recipe.code_repair.main_gen $@
