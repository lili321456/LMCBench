#!/bin/bash

# 定义一个数组，包含所有需要执行的配置文件名
configs=("asqa_qwen2_57B_shot2_ndoc5_gtr_oracle_default.yaml" "eli5_qwen2_57B_shot2_ndoc5_bm25_oracle_default.yaml" "asqa_qwen2_57B_shot2_ndoc20_dpr_default.yaml" "eli5_qwen2_57B_shot2_ndoc20_bm25_default.yaml" "asqa_qwen2_57B_shot2_ndoc20_gtr_default.yaml")

# 遍历数组中的每个配置文件
for config in "${configs[@]}"; do
    echo "正在执行: python run.py --config configs/$config --temperature 1.0 --top_p 0.95"
    # 执行命令，并捕获返回值
    python run.py --config configs/"$config" --temperature 1.0 --top_p 0.95
    # 检查命令是否成功执行
    if [ $? -ne 0 ]; then
        echo "命令执行失败: python run.py --config configs/$config --temperature 1.0 --top_p 0.95"
        echo "跳过当前命令，继续执行下一条命令。"
    else
        echo "命令执行成功: python run.py --config configs/$config --temperature 1.0 --top_p 0.95"
    fi
done

echo "所有命令执行完毕。"