epochs=100
bs=256
dim=2
pred_dim=256
lr=0.001


python main.py --method supervised --dataset theorem -a mlp \
        --epochs $epochs -b 128 --lr $lr \
        --fix_pred_lr --num_targets 0\
        --pred_epochs 1 --target_layers\
        --optimizer-type adam --wd 0 --fix_lr \
        --dim $dim --loss ce --pred_layers 1\
        --pred_dim $pred_dim --num_of_classes 6 --num-of-runs 1 \
