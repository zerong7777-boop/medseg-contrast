#!/usr/bin/env bash

#chmod +x batch_pred2gray.sh
# ./batch_pred2gray.sh
set -e

# PH2 结果根目录
ROOT="/home/jgzn/PycharmProjects/RZ/danzi/seg/result/ISIC18"

# 需要批量处理的模型列表
MODELS=(
  "UNet"
  "UNetPP"
  "UKAN"
  "SwinUNet"
  "AttentionUNet"
  "UTNet"
  "DCSAUNet"
  "MALUNet"
  "EGEUNet"
)

for m in "${MODELS[@]}"; do
  IN_DIR="${ROOT}/${m}/test_pred"
  OUT_DIR="${ROOT}/${m}/test_pred_gray"

  if [ -d "${IN_DIR}" ]; then
    echo ">>> 处理模型 ${m}: ${IN_DIR} -> ${OUT_DIR}"
    python mask_convert.py \
      --mode class2gray \
      --input-dir "${IN_DIR}" \
      --output-dir "${OUT_DIR}"
  else
    echo "[WARN] 跳过 ${m}，未找到目录: ${IN_DIR}"
  fi
done

echo "全部处理完成。"
