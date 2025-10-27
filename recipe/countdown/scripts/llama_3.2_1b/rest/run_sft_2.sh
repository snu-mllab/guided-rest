set -x

train_path=checkpoints/countdown/llama_3.2_1b_rest_sft_1/global_step_1287/data/temp_1.0/iter_0/train.parquet
valid_path=checkpoints/countdown/llama_3.2_1b_rest_sft_1/global_step_1287/data/temp_1.0/iter_0/validation.parquet
model_path=checkpoints/countdown/llama_3.2_1b_base_sft/global_step_3906/huggingface

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${train_path} \
    data.val_files=${valid_path} \
    data.train_batch_size=256 \
    data.micro_batch_size_per_gpu=64 \
    data.multiturn.enable=True \
    data.multiturn.messages_key=messages \
    data.max_length=5120 \
    data.truncation=right \
    model.partial_pretrain=${model_path} \
    model.enable_gradient_checkpointing=True \
    model.use_liger=True \
    model.strategy=fsdp2 \
    optim.lr=1e-5 \
    use_remove_padding=True \
    trainer.logger=['console','wandb'] \
    trainer.project_name=countdown \
    trainer.experiment_name=llama_3.2_1b_rest_sft_2 \
    trainer.save_freq=-1 \
    trainer.test_freq=100 \
    trainer.total_epochs=3 $@
