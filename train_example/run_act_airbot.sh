bs=64
ws=2
export WANDB_API_KEY=265854b7b5037e14094d66ddd65204191693217d
torchrun --standalone --nnodes=1 --nproc-per-node=2 train_ACT_airbot_example.py \
    --dataset_name 'airbot' \
    --datalist /home/dodo/ljx/BearRobot/data/airbot/newair_rel_eef_1206.json \
    --algo_name 'act' \
    --ddp True \
    --img_size 224 \
    --view_list top_image wrist_image side_image \
    --visual_encoder resnet18 \
    --visual_pretrain True \
    --ft_vision True \
    --ac_num 4 \
    --s_dim 7 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --norm mean \
    --discretize_actions False \
    --batch_size $bs \
    --world_size $ws \
    --lr 0.00003 \
    --val_freq 200000000 \
    --eval_freq 200000000 \
    --resume None \
    --wandb True \
    --steps 1000000 \
    --save True \
    --save_freq 100000 \
    --save_path ../experiments/airbot/ACT/1206_3view_scratch \
    --log_path ../experiments/airbot/ACT/1206_3view_scratch \
    --port 2051 \
