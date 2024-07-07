bs=64
ws=2
text_encoder=DecisionNCE-T
for seed in 42; do
export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=cfbf81ce9bd7daca9d32f4bd1dbf26e8c93310c3
torchrun --standalone --nnodes=1 --nproc-per-node=1 train_diffusion_policy_example_libero.py \
    --dataset_name 'libero_goal' \
    --algo_name 'diffusion visual motor' \
    --ddp False \
    --img_size 128 \
    --seed $seed \
    --visual_encoder resnet34 \
    --visual_pretrain True \
    --text_encoder $text_encoder \
    --ft_vision True \
    --film_fusion False \
    --ac_num 6 \
    --norm minmax \
    --norm_type bn \
    --add_spatial_coordinates False \
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
    --steps 200000 \
    --save True \
    --save_freq 50000 \
    --T 25 \
    --save_path ../experiments/libero/libero_goal/diffusion/resnet34_wstate_0613_$text_encoder \
    --log_path ../experiments/libero/libero_goal/diffusion/resnet34_wstate_0613_$text_encoder \
    --port 2050
done
