MASTER_PORT=10049
lr=1e-4
wd=1e-4
recycling=1
batch_size=32
freeze_pretrained_transformer=0
proc_freeze=1
ligand_freeze=1
seed=0
pretrained_model=./results/pretrain/checkpoint_21_260000.pt
model_name=LBA30
LBA_data=./data/LBA30/lba
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
warmup_steps=100
max_steps=5000
n_gpu=1

run_name="${model_name}_pretrained_${recycling}_lr${lr}_bz${batch_size}_crnetFreeze${freeze_pretrained_transformer}_seed${seed}"
ctl_2d=0
save_dir=./results/${model_name}_pretrained_${recycling}_lr${lr}_bz${batch_size}_crnetFreeze${freeze_pretrained_transformer}_seed${seed}/


export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

wandb disabled

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) .  \
       --user-dir ./unimol --train-subset train --valid-subset valid,test \
       --num-workers 8 --ddp-backend=c10d --mode train \
       --task affinity_regres --loss affinity_regres --arch affinity_regres  \
       --task-name LBA_affinity \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
       --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $max_steps \
       --update-freq $update_freq --seed $seed \
       --tensorboard-logdir $save_dir/tsb \
       --max-update $max_steps --log-interval 10 --log-format simple \
       --save-interval-updates 100000000 --validate-interval-updates 100 --keep-interval-updates 1 --no-epoch-checkpoints  \
       --x-norm-loss $x_norm_loss --delta-pair-repr-norm-loss $delta_pair_repr_norm_loss \
       --mask-prob $mask_prob --noise-type $noise_type --noise $noise --batch-size $batch_size \
       --save-dir $save_dir  --only-polar $only_polar --run-name $run_name \
       --dict-name dict_protein.txt --ligdict-name dict_ligand.txt \
       --net complex_crnet --recycling ${recycling} --CLS-use seperate_CLS \
       --complex-pretrained-model  $pretrained_model \
       --proc-freeze ${proc_freeze} --ligand-freeze ${ligand_freeze} \
       --keep-best-checkpoints 1 --best-checkpoint-metric valid_pearson --maximize-best-checkpoint-metric \
       --LBA_data $LBA_data \
       --no-last-checkpoints \
       --patience 5 \
       --freeze-pretrained-transformer ${freeze_pretrained_transformer} \
       --tmp-save-dir ./tmp/LBA30/ \
       --use-BindNet 1 \
