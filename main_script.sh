#!/bin/bash

# 定义参数模板
BASE_PATH="/mnt/nfs-ssd/data/moyuanhuan/DiffDockData/diffDock_dataset_matching/train_cache_splits/protein_ligand_split_"
SAVE_PATH="/mnt/nfs-ssd/data/moyuanhuan/DiffDockData/diffDock_dataset_matching/train_node_embedding"

# 循环范围
START=20
END=20

# 循环执行主脚本并修改参数
for i in $(seq $START $END); do
    echo "Running iteration $i"
    LBA_DATA_PATH="${BASE_PATH}${i}.pkl"
    
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
        --nproc_per_node 1 \
        --master_port 10090 unicore_train.py /mnt/nfs-ssd/data/moyuanhuan/BindNet \
        --user-dir ./unimol \
        --train-subset train \
        --valid-subset valid,test \
        --num-workers 0 \
        --ddp-backend c10d \
        --mode train \
        --task affinity_regres \
        --loss affinity_regres \
        --arch affinity_regres \
        --task-name LBA_affinity \
        --optimizer adam \
        --adam-betas \(0.9,\ 0.99\) \
        --adam-eps 1e-6 \
        --clip-norm 1.0 \
        --weight-decay 1e-4 \
        --lr-scheduler polynomial_decay \
        --lr 0.0001 \
        --warmup-updates 100 \
        --total-num-update 5000 \
        --update-freq 1 \
        --seed 1 \
        --tensorboard-logdir ./results/LBA30_frad_pretrained_1_lr0.0001_bz8_crnetFreeze0_seed1//tsb \
        --max-update 5000 \
        --log-interval 10 \
        --log-format simple \
        --save-interval-updates 100000000 \
        --validate-interval-updates 100 \
        --keep-interval-updates 1 \
        --no-epoch-checkpoints \
        --x-norm-loss 0.01 \
        --delta-pair-repr-norm-loss -1 \
        --mask-prob 0.15 \
        --noise-type uniform \
        --noise 1.0 \
        --batch-size 1 \
        --save-dir ./results/LBA30_frad_pretrained_1_lr0.0001_bz8_crnetFreeze0_seed1/ \
        --only-polar 0 \
        --run-name LBA30_frad_pretrained_1_lr0.0001_bz8_crnetFreeze0_seed1 \
        --dict-name dict_protein.txt \
        --ligdict-name dict_ligand.txt \
        --net complex_crnet \
        --recycling 1 \
        --CLS-use seperate_CLS \
        --lig-pretrained /mnt/nfs-ssd/data/moyuanhuan/BindNetData/mol_pre_no_h_220816.pt \
        --proc-pretrained /mnt/nfs-ssd/data/moyuanhuan/BindNetData/pocket_pre_220816.pt \
        --complex-pretrained-model /mnt/nfs-ssd/data/moyuanhuan/BindNetData/checkpoint_44_270000.pt \
        --proc-freeze 1 \
        --ligand-freeze 1 \
        --keep-best-checkpoints 1 \
        --best-checkpoint-metric valid_pearson \
        --maximize-best-checkpoint-metric \
        --LBA_data $LBA_DATA_PATH \
        --no-last-checkpoints \
        --patience 5 \
        --freeze-pretrained-transformer 0 \
        --tmp-save-dir ./new_pretrain_tmp/ \
        --extract-feat 1 \
        --save-path $SAVE_PATH \
        --extract-pairwise-feat 0 \
        #> bowenfeat_crossdock.log 2>&1 &
    
    # 检查主脚本是否执行成功
    if [ $? -ne 0 ]; then
        echo "Error running script with file $LBA_DATA_PATH"
        exit 1
    fi

    echo "Completed iteration $i"
done

echo "All iterations completed successfully."
