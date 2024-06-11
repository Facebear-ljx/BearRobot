bs=64
ws=2
torchrun --standalone --nnodes=1 --nproc-per-node=2 train_diffusion_policy_decisionNCE_example.py \
    --dataset 'libero30' \
    --algo_name 'dnce diffusion visual motor' \
    --ddp True \
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
    --resume None \
    --wandb True \
    --steps 2000000 \
    --save True \
    --save_freq 20000 \
    --T 25 \
    --save_path ../experiments/libero/libero30/diffusion_dnce/test_0611 \
    --log_path ../experiments/libero/libero30/diffusion_dnce/test_0611 \
    --port 2050 \