# for dataset in EN_FR_15K EN_DE_15K D_W_15K  D_Y_15K; do
# python infer.py --simulate --dataset $dataset
# done


datasets=("D_W_15K" "D_Y_15K" "EN_DE_15K" "EN_FR_15K")
tprs=(0.56 0.61 0.50 0.56)

for i in {0..3}; do
    dataset=${datasets[$i]}
    tpr=${tprs[$i]}
    python infer.py --dataset $dataset --tpr $tpr 2>&1 | tee infer_${dataset}.log
done