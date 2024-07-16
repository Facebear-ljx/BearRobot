bs=64
ws=2
text_encoder=DecisionNCE-V
file_name=collapse_corrupt_libero_norm_0716_02

for seed in 42; do
export MUJOCO_GL="osmesa"
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=cdad4c82816c6da591e611ae4ac5a07ee7f4611a
torchrun --standalone --nnodes=1 --nproc_per_node=1 train_diffusion_policy_example_libero.py \
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
    --eval_freq 99999999999 \
    --resume None \
    --wandb True \
    --steps 200000 \
    --save True \
    --save_freq 20000 \
    --T 25 \
    --save_path ../experiments/libero/libero_goal/diffusion/$file_name \
    --log_path ../experiments/libero/libero_goal/diffusion/$file_name \
    --port 2077 \
    --add_noise True \
    --noise_std 0.16 \
    --minus_mean True \
    --mean_data_path /home/dodo/.zh1hao_space/bear_branch/BearRobot/analysis/libero130/mean_DecisionNCE-T_libero_360ep.npz
done

# wait

# file_name3=collapse_corrupt_0713_01

# for seed in 42; do
# export MUJOCO_GL="osmesa"
# export CUDA_VISIBLE_DEVICES=1
# export WANDB_API_KEY=0d4d8e6f87ec9508a673bb4f0d117bf6a79a9945
# torchrun --standalone --nnodes=1 --nproc-per-node=1 train_diffusion_policy_example_libero.py \
#     --dataset_name 'libero_goal' \
#     --algo_name 'diffusion visual motor' \
#     --ddp False \
#     --img_size 128 \
#     --seed $seed \
#     --visual_encoder resnet34 \
#     --visual_pretrain True \
#     --text_encoder $text_encoder \
#     --ft_vision True \
#     --film_fusion False \
#     --ac_num 6 \
#     --norm minmax \
#     --norm_type bn \
#     --add_spatial_coordinates False \
#     --discretize_actions False \
#     --encode_s False \
#     --encode_a False \
#     --s_dim 9 \
#     --batch_size $bs \
#     --world_size $ws \
#     --lr 0.0003 \
#     --val_freq 10000000 \
#     --eval_freq 50000 \
#     --resume None \
#     --wandb True \
#     --steps 200000 \
#     --save True \
#     --save_freq 50000 \
#     --T 25 \
#     --save_path ../experiments/libero/libero_goal/diffusion/$file_name3 \
#     --log_path ../experiments/libero/libero_goal/diffusion/$file_name3 \
#     --port 2077 \
#     --add_noise True \
#     --minus_mean True \
#     --mean_data_path /home/dodo/.zh1hao_space/bear_branch/BearRobot/analysis/libero130/mean_embeddings_final.npz
# done