export CUDA_VISIBLE_DEVICES=4

model_name=ICMamba

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/din/ \
  --model_id din \
  --model $model_name \
  --data din \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/ss/ \
  --model_id ss \
  --model $model_name \
  --data ss \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

