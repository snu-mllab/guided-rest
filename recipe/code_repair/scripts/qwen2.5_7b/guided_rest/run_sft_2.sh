set -x

train_path=checkpoints/code_repair/qwen2.5_7b_guided_rest_sft_1/global_step_344/data/temp_1.0/turn_4/iter_3/train.parquet
valid_path=checkpoints/code_repair/qwen2.5_7b_guided_rest_sft_1/global_step_344/data/temp_1.0/turn_4/iter_3/valid.parquet
model_path=checkpoints/code_repair/qwen2.5_7b/huggingface

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${train_path} \
    data.val_files=${valid_path} \
    data.train_batch_size=256 \
    data.micro_batch_size_per_gpu=8 \
    data.multiturn.enable=True \
    data.multiturn.messages_key=messages \
    data.max_length=18432 \
    data.truncation=right \
    model.partial_pretrain=${model_path} \
    model.enable_gradient_checkpointing=True \
    model.use_liger=True \
    model.strategy=fsdp2 \
    optim.lr=1e-5 \
    use_remove_padding=True \
    trainer.logger=['console','wandb'] \
    trainer.project_name=code_repair \
    trainer.experiment_name=qwen2.5_7b_guided_rest_sft_2 \
    trainer.save_freq=1000 \
    trainer.test_freq=100 \
    trainer.total_epochs=2 $@
