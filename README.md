BindNet: Protein-ligand binding representation learning from fine-grained interactions
===================================================================

The code architecture used in BindNet is the same as Uni-Mol, which means using BindNet need the same enviroment as Uni-Mol


# Data Process
The Biolip data should be downloaded from https://zhanggroup.org/BioLiP/index.cgi and pocessed by the following command:

`python3 ./data/BioLip/BioLip.py`

The processed data of LBA, LEP, DUD-E and AD could be downloaded from: https://drive.google.com/drive/folders/1NfXpsbbsR7FNIEoU88EA04Jmgnio0q_O?usp=drive_link

The data of docking is the same as Uni-Mol, which could be downloaded from the following link: https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/finetune/protein_ligand_binding_pose_prediction.tar.gz

# Pretrain

`bash ./pretrain.sh`


``` bash
# pretrain.sh
MASTER_PORT=10078
lr=1e-4
wd=1e-4

update_freq=1
masked_token_loss=0
masked_coord_loss=0
masked_dist_loss=0
x_norm_loss=0.01
delta_pair_repr_norm_loss=-1
mask_prob=0.15
only_polar=0
noise_type="uniform"
noise=1.0
seed=1
warmup_steps=10000
max_steps=1000000
n_gpu=4
batch_size=8
run_name="pretrain"
ctl_2d=0
save_dir=./results/pretrain

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export TORCH_DISTRIBUTED_DEBUG="DETAIL"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) .  --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task unimol --loss unimol --arch unimol_base  \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
       --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $max_steps \
       --update-freq $update_freq --seed $seed \
       --tensorboard-logdir $save_dir/tsb \
       --max-update $max_steps --log-interval 10 --log-format simple \
       --save-interval-updates 10000 --validate-interval-updates 10000 --keep-interval-updates 10 --no-epoch-checkpoints  \
       --masked-token-loss $masked_token_loss --masked-coord-loss $masked_coord_loss --masked-dist-loss $masked_dist_loss \
       --x-norm-loss $x_norm_loss --delta-pair-repr-norm-loss $delta_pair_repr_norm_loss \
       --mask-prob $mask_prob --noise-type $noise_type --noise $noise --batch-size $batch_size \
       --save-dir $save_dir  --only-polar $only_polar --run-name $run_name --ctl-2d $ctl_2d \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --online-ligfeat 1 \
       --dict-name dict_protein.txt --ligdict-name dict_ligand.txt \
       --lig-pretrained ligand_pretrain_model.pt \
       --proc-pretrained pocket_pretrain_model.pt \
       --proc-freeze 1 --complex-crnet 1 --find-unused-parameters --pocket-data ./data/BioLip/lmdb \
       --cr-regression 1 --rdkit-random 1 --mask-feature 1 \
```

The pretrained model could be downloaded from the following link: [https://drive.google.com/drive/folders/1KllcdJOuoMJedJBWvfGMLo6OMsg9Dk4V?usp=sharing](https://drive.google.com/drive/folders/1KllcdJOuoMJedJBWvfGMLo6OMsg9Dk4V?usp=sharing)

# Finetune
``` bash
# Finetune on LBA
bash ./LBA30_exp.sh
bash ./LBA60_exp.sh

# Finetune on LEP
bash ./LEP_exp.sh

# Finetune on DUD-E
bash ./DUDE_exp.sh

# Finetune on AD
bash ./AD_exp.sh

# Finetune on docking dataset
bash ./docking_exp.sh
```
