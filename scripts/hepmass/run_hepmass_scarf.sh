epochs=20
dim=32
pred_dim=16
lr=1e-6

python main.py --method simclr --dataset hepmass -a mlp\
        --num_targets 0  --target_layers\
        --epochs $epochs -b 512 --lr $lr\
        --optimizer-type adam --wd 0 --fix_lr \
        --fix_pred_lr --pred_layers 1 --num_of_classes 2\
        --dim $dim --pred_dim $pred_dim --loss ce --proj_layers 2