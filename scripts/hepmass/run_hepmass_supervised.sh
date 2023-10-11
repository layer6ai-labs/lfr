epochs=20
dim=2
pred_dim=32
lr=1e-6
python main.py --method supervised --dataset hepmass -a mlp \
    --epochs $epochs -b 512 --lr $lr \
    --fix_pred_lr   --num_targets 0\
    --pred_epochs 1 --target_layers\
    --optimizer-type adam --wd 0 --fix_lr \
    --dim 2 --loss ce --pred_layers 1\
    --pred_dim $pred_dim --num_of_classes 2 --num-of-runs 1 \