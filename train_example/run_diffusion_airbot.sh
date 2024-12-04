bs=64
ws=2
export WANDB_API_KEY=265854b7b5037e14094d66ddd65204191693217d
torchrun --standalone --nnodes=1 --nproc-per-node=1 train_airbot_diffusion.py \
    --dataset_name 'airbot' \
    --algo_name 'diffusion visual motor' \
    --ddp True \
    --img_size 224 \
    --visual_encoder resnet34 \
    --visual_pretrain True \
    --ft_vision True \
    --film_fusion False \
    --ac_num 4 \
    --norm minmax \
    --norm_type bn \
    --add_spatial_coordinates False \
    --discretize_actions False \
    --encode_s False \
    --encode_a False \
    --s_dim 0 \
    --batch_size $bs \
    --world_size $ws \
    --lr 0.0001 \
    --val_freq 1000000000000000000 \
    --eval_freq 1000000000000000000 \
    --resume /home/dodo/ljx/BearRobot/experiments/airbot/diffusion/241202_resume_from_1130_200k/latest.pth \
    --wandb True \
    --steps 600000 \
    --save True \
    --save_freq 20000 \
    --T 25 \
    --save_path ../experiments/airbot/diffusion/241203_resume_from_1202_400k \
    --log_path ../experiments/airbot/diffusion/241203_resume_from_1202_400k \
    --port 2050 \
    --num_workers 8
