MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 2 --device 0  --timestep_respacing ddim20"
python image_train.py --data_dir /data2/dataset/Robustness/celeba/train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS