set -x

model_name=llama_3.2_1b_base_sft/global_step_3906

python -m recipe.countdown.main_gen \
    model_name=${model_name} $@
