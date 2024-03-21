export CUDA_VISIBLE_DEVICES=1
for SEED in 150 100 50 0; do
for ENV in 'antmaze'; do
#ENV='halfcheetah'
for DATASET in 'large-diverse' 'medium-play' 'medium-diverse'; do
ENV_NAME=$ENV'-'$DATASET'-v2'
echo $ENV_NAME

python train_main.py \
       --env_name $ENV_NAME \
       --expectile 0.9\
       --seed $SEED\

done
done
done
