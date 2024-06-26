CUDA_VISIBLE_DEVICES=5 python train.py \
    --config_name strat \
    --inputter_name strat \
    --eval_input_file ./DATA/3_valid.txt \
    --seed 13 \
    --max_input_length 512 \
    --max_decoder_input_length 50 \
    --train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_batch_size 16 \
    --learning_rate 1.5e-5 \
    --num_epochs 10 \
    --warmup_steps 100 \
    --fp16 false \
    --loss_scale 0.0 \
    --pbar true \
    --use_all_persona False \
    --encode_context True