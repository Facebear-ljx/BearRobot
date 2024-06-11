bs=2
ws=2
torchrun --standalone --nnodes=1 --nproc-per-node=2 train_rt1_example.py \
    --ddp True \
    --img_size 224 \
    --frames 1 \
    --batch_size $bs \
    --world_size $ws \
    --lr 0.00001 \
    --val_freq 1000000000 \
    --resume None \
    --wandb False \
    --steps 100000 \
    --save True \
    --save_freq 10000 \
    --save_path ../experiments/rt1/test \
    --log_path ../experiments/rt1/test \
    --port 2255 \