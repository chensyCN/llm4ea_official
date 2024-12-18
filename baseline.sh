# datasets=("D_W_15K" "D_Y_15K" "EN_DE_15K" "EN_FR_15K")
# tprs=(0.86 1.0 0.8888 0.905)

# datasets=("D_W_15K" "D_Y_15K" "EN_DE_15K" "EN_FR_15K")
# V1

datasets=("D_W_15K_V1" "D_Y_15K_V1" "EN_DE_15K_V1" "EN_FR_15K_V1")
tprs=(0.86 1.0 0.8888 0.905)

for exp in rebuttal; do
    length=${#datasets[@]}
    for ((i=0; i<$length; i++)); do
        timestamp=$(date +%Y%m%d)  # 获取当前日期，格式为YYYYMMDD
        # if the folder does not exist, create it
        if [ ! -d "logs/${timestamp}/${exp}" ]; then
            mkdir -p logs/${timestamp}/${exp}
        fi
        CUDA_VISIBLE_DEVICES=1 python infer-baseline.py --dataset ${datasets[i]} --tpr ${tprs[i]} 2>&1 | tee logs/${timestamp}/${exp}/${datasets[i]}_baseline.log
    done
done
