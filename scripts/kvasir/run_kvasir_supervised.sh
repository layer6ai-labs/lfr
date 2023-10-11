# !/usr/bin/env bash
epochs=400
eval_epochs=100

dim=2048
pred_dim=256

python main.py --method supervised-aug --dataset kvasir -a resnet18\
        --num_targets 0  --target_layers 1 --ckpt_path ckpt\
        --target_sample_ratio 10 --num_of_classes 8\
        --epochs $epochs -b 256 --lr 0.01\
        --optimizer-type sgd --momentum 0.9 --wd 5e-4\
        --dim 8 --loss ce --num-of-runs 1