bs=64
ws=2
export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=cfbf81ce9bd7daca9d32f4bd1dbf26e8c93310c3
torchrun --standalone --nnodes=1 --nproc-per-node=1 train_diffusion_policy_decisionNCE_example.py \
    --seed 42 \
    --dataset 'libero_goal' \
    --algo_name 'dnce diffusion visual motor' \
    --ddp False \
    --mm_encoder DecisionNCE-T \
    --ft_mmencoder True \
    --film_fusion False \
    --ac_num 6 \
    --norm minmax \
    --discretize_actions False \
    --encode_s False \
    --encode_a False \
    --s_dim 9 \
    --batch_size $bs \
    --world_size $ws \
    --lr 0.0003 \
    --val_freq 10000000 \
    --eval_freq 25000 \
    --resume None \
    --wandb True \
    --steps 2000000 \
    --save True \
    --save_freq 25000 \
    --T 25 \
    --save_path ../experiments/libero/libero_goal/diffusion_dnce/test_0613_ftimg \
    --log_path ../experiments/libero/libero_goal/diffusion_dnce/test_0613_ftimg \
    --port 2052 \