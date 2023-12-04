export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="../../data/triplets/100photos/meta.csv"
accumulation_steps=5
learning_rate=1e-05
steps=1000
train_batch_size=1
warmup_steps=200
weight_tune=0.1
weight_preserve=0.9
defend_type="nudity_advtrain"
current_time=$(date +'%m%d_%H%M')
output_dir="mos_model_sal_"${defend_type}-"step_"${steps}-"lr_"${learning_rate}-"warm_"${warmup_steps}-${weight_tune}+${weight_preserve}"gacc_"${accumulation_steps}-${current_time}

export CUDA_VISIBLE_DEVICES=0
accelerate launch --mixed_precision="fp16" "--num_processes=1" train_text_to_image_adjust_sal.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop\
  --train_batch_size ${train_batch_size} \
  --gradient_accumulation_steps ${accumulation_steps} \
  --gradient_checkpointing \
  --max_train_steps ${steps} \
  --learning_rate ${learning_rate} \
  --image_column="nude_image_path" \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps ${warmup_steps} \
  --output_dir ${output_dir} \
  --weight_tune ${weight_tune} \
  --weight_preserve ${weight_preserve} \
  --checkpoints_total_limit 0 \
  --validation_prompts "woman, nudity" "a running dog on grass" "a woman standing aside the colorful table" "a nude sexy man, blue eyes"
