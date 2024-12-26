bs=64
ws=2
text_encoder=T5
dataset_name=libero_goal
for seed in 42; do
export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=265854b7b5037e14094d66ddd65204191693217d
torchrun --standalone --nnodes=1 --nproc-per-node=1 train_flow_matching.py \
    --base_dir /data \
    --datalist /home/dodo/ljx/BearRobot/data/libero/libero_goal-ac-10.json \
    --view_list D435_image wrist_image \
    --dataset_name $dataset_name \
    --algo_name 'flow matching' \
    --ddp False \
    --img_size 128 \
    --seed $seed \
    --visual_encoder resnet18 \
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
    --eval_freq 100 \
    --resume None \
    --wandb True \
    --steps 200000 \
    --save True \
    --save_freq 50000 \
    --T 10 \
    --save_path ../experiments/libero/$dataset_name/fm/1226_$text_encoder \
    --log_path ../experiments/libero/$dataset_name/fm/1226_$text_encoder \
    --port 2050
done
