import os
import json
import csv
import re

from PIL import Image
import matplotlib.pyplot as plt

# ================== 基本路径配置 ==================
# 结果根目录：下面有 ISIC16 / ISIC17 / ISIC18 / PH2 等
ROOT = r"/home/jgzn/PycharmProjects/RZ/danzi/seg/result"

# 数据集原始图像与标注所在根目录
DATASETS_ROOT = r"/home/jgzn/PycharmProjects/RZ/danzi/seg/datasets"

# 数据集文件夹名
DATASETS = ["ISIC16", "ISIC17", "ISIC18", "PH2"]

# 相对路径
METRICS_REL_PATH = os.path.join("test_pred", "metrics.json")
TRAIN_LOG_REL_PATH = "train.log"
SPLITS_REL_PATH = "splits.json"
PRED_GRAY_DIR = "test_pred_gray"

# 统一的显示尺寸（与训练 img_size 一致）
TARGET_SIZE = (256, 256)


# ================== 工具函数 ==================
def safe_float(x):
    """把 None/缺失值安全地转成字符串。"""
    if x is None:
        return ""
    try:
        return f"{float(x):.6f}"
    except Exception:
        return str(x)


def get_total_train_time_from_log(log_path):
    """
    从 train.log 中解析每一轮的 time，求和得到总训练时间（秒）
    日志行格式示例：
    [Epoch 001/100] ... time=19.0s
    """
    if not os.path.isfile(log_path):
        return None

    pattern = re.compile(r"time=([\d\.]+)s")
    total = 0.0
    found_any = False

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                sec = float(m.group(1))
                total += sec
                found_any = True

    return total if found_any else None


def find_label_file(label_dir, base_name, ds):
    """
    在 label_ori 目录下寻找与 base_name 对应的 GT。
    例如 base_name='ISIC_0010009.jpg'：
    - 优先匹配 stem 完全相同：ISIC_0010009.*
    - 再匹配 stem 出现在文件名中的：ISIC_0010009_segmentation.png 等。
    """
    if not os.path.isdir(label_dir):
        print(f"[DBG] GT 目录不存在: {label_dir}")
        return None

    stem = os.path.splitext(base_name)[0]
    files = os.listdir(label_dir)

    exact = [f for f in files if os.path.splitext(f)[0] == stem]
    contains = [f for f in files if stem in os.path.splitext(f)[0]]

    chosen = None
    if exact:
        chosen = sorted(exact)[0]
        mode = "exact"
    elif contains:
        chosen = sorted(contains)[0]
        mode = "contains"
    else:
        sample_list = files[:5]
        print(f"[WARN] [{ds}] GT 未找到: base={base_name}, stem={stem}, "
              f"label_dir={label_dir}, 示例文件={sample_list}")
        return None

    gt_path = os.path.join(label_dir, chosen)
    print(f"[DBG] [{ds}] GT 匹配: base={base_name}, mode={mode}, file={chosen}")
    return gt_path


def get_pred_path_by_testpos(pred_dir, test_pos, ds, model_name):
    """
    根据“测试集中第几个样本”的序号 test_pos 来定位预测图。
    预测图命名为 pred_000035.png 这种形式，因此这里按多种
    补零长度尝试：
        pred_%06d.png, pred_%05d.png, ..., pred_%d.png
    """
    if not os.path.isdir(pred_dir):
        print(f"[DBG] pred_dir 不存在: {pred_dir}")
        return None

    cand_names = [
        f"pred_{test_pos:06d}.png",
        f"pred_{test_pos:05d}.png",
        f"pred_{test_pos:04d}.png",
        f"pred_{test_pos:03d}.png",
        f"pred_{test_pos:02d}.png",
        f"pred_{test_pos:01d}.png",
        f"pred_{test_pos}.png",
    ]

    for name in cand_names:
        path = os.path.join(pred_dir, name)
        if os.path.isfile(path):
            print(f"[DBG] [{ds}/{model_name}] 用索引 {test_pos} 匹配到预测图: {name}")
            return path

    sample_list = os.listdir(pred_dir)[:5]
    print(f"[WARN] [{ds}/{model_name}] 用索引 {test_pos} 未找到预测图, "
          f"尝试名={cand_names}, 示例文件={sample_list}")
    return None


# ================== 1. 汇总指标 + 训练时间 ==================
rows = []

for ds in DATASETS:
    ds_path = os.path.join(ROOT, ds)
    if not os.path.isdir(ds_path):
        print(f"[WARN] 数据集目录不存在，跳过：{ds_path}")
        continue

    for model_name in sorted(os.listdir(ds_path)):
        model_dir = os.path.join(ds_path, model_name)
        if not os.path.isdir(model_dir):
            continue

        metrics_path = os.path.join(model_dir, METRICS_REL_PATH)
        if not os.path.isfile(metrics_path):
            print(f"[WARN] 找不到 metrics.json，跳过：{metrics_path}")
            continue

        # 读取 metrics.json
        with open(metrics_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        overall = meta.get("overall", {})

        # 读取 train.log 里的时间
        log_path = os.path.join(model_dir, TRAIN_LOG_REL_PATH)
        total_seconds = get_total_train_time_from_log(log_path)
        total_hours = total_seconds / 3600.0 if total_seconds is not None else None

        rows.append({
            "dataset": ds,
            "model": model_name,
            "dice": overall.get("dice"),
            "iou": overall.get("iou"),
            "precision": overall.get("precision"),
            "recall": overall.get("recall"),
            "train_time_sec": total_seconds,
            "train_time_hours": total_hours,
        })

# 控制台打印
print("dataset,model,dice,iou,precision,recall,train_time_sec,train_time_hours")
for r in rows:
    print(
        f"{r['dataset']},{r['model']},"
        f"{safe_float(r['dice'])},"
        f"{safe_float(r['iou'])},"
        f"{safe_float(r['precision'])},"
        f"{safe_float(r['recall'])},"
        f"{safe_float(r['train_time_sec'])},"
        f"{safe_float(r['train_time_hours'])}"
    )

# 保存为 CSV
out_csv = os.path.join(ROOT, "metrics_summary_with_time.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "dataset", "model",
        "dice", "iou", "precision", "recall",
        "train_time_sec", "train_time_hours"
    ])
    for r in rows:
        writer.writerow([
            r["dataset"],
            r["model"],
            safe_float(r["dice"]),
            safe_float(r["iou"]),
            safe_float(r["precision"]),
            safe_float(r["recall"]),
            safe_float(r["train_time_sec"]),
            safe_float(r["train_time_hours"]),
        ])

print(f"\n[INFO] 指标 + 训练时间汇总已保存到: {out_csv}")


# ================== 2. 选取每个数据集的 3 张对比样本 + 生成论文式大图 ==================
qualitative = {}  # {dataset: [ {filename, image, gt, preds{model: path}} ] }

for ds in DATASETS:
    ds_result_path = os.path.join(ROOT, ds)
    if not os.path.isdir(ds_result_path):
        print(f"[WARN] 数据集目录不存在（定性对比），跳过：{ds_result_path}")
        continue

    models = sorted([
        d for d in os.listdir(ds_result_path)
        if os.path.isdir(os.path.join(ds_result_path, d))
    ])
    if not models:
        print(f"[WARN] 数据集 {ds} 下没有模型目录，跳过定性对比。")
        continue

    # 选一个模型的 splits.json 作为参考（假设各模型的划分一致）
    splits = None
    ref_model_for_split = None
    for m in models:
        splits_path = os.path.join(ds_result_path, m, SPLITS_REL_PATH)
        if os.path.isfile(splits_path):
            with open(splits_path, "r", encoding="utf-8") as f:
                splits = json.load(f)
            ref_model_for_split = m
            break

    if not splits:
        print(f"[WARN] 数据集 {ds} 未找到 splits.json，跳过定性对比。")
        continue

    base_filenames = splits.get("base_filenames", [])
    test_indices = splits.get("test_indices", [])
    if not base_filenames or not test_indices:
        print(f"[WARN] 数据集 {ds} 的 splits.json 中 test_indices/base_filenames 为空，跳过。")
        continue

    print(f"[INFO] 数据集 {ds} 使用 {ref_model_for_split}/splits.json 作为测试索引参考。")

    # 建立 “全局索引 -> 测试子集位置” 的映射，用于反推 pred_XXXXXX.png 序号
    index_to_testpos = {global_idx: pos for pos, global_idx in enumerate(test_indices)}

    # 选测试集前三张（对应全局 index）
    selected_indices = test_indices[:3]
    ds_samples = []

    for global_idx in selected_indices:
        if global_idx < 0 or global_idx >= len(base_filenames):
            continue
        fname = base_filenames[global_idx]

        # 找图像 & GT
        img_path = os.path.join(DATASETS_ROOT, ds, "train", "image", fname)
        label_dir = os.path.join(DATASETS_ROOT, ds, "train", "label_ori")
        gt_path = find_label_file(label_dir, fname, ds)

        sample = {
            "filename": fname,
            "image": img_path,
            "gt": gt_path,
            "preds": {}
        }

        # 当前样本在“测试集序列”中的位置
        test_pos = index_to_testpos.get(global_idx, None)

        for m in models:
            pred_dir = os.path.join(ds_result_path, m, PRED_GRAY_DIR)
            if test_pos is None:
                pred_path = None
            else:
                pred_path = get_pred_path_by_testpos(pred_dir, test_pos, ds, m)
            sample["preds"][m] = pred_path

        ds_samples.append(sample)

    qualitative[ds] = ds_samples

    # ===== 生成对比图 =====
    if not ds_samples:
        print(f"[WARN] 数据集 {ds} 没有可用样本，跳过绘图。")
        continue

    n_rows = len(ds_samples)
    n_models = len(models)
    n_cols = 2 + n_models  # image + gt + 各模型

    fig_width = 1.8 * n_cols
    fig_height = 1.8 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_rows == 1:
        axes = [axes]

    col_titles = ["Image", "GT"] + models

    for r, sample in enumerate(ds_samples):
        # 原图
        try:
            img = Image.open(sample["image"]).convert("RGB")
        except Exception as e:
            print(f"[WARN] 打不开原图 {sample['image']}: {e}")
            img = Image.new("RGB", TARGET_SIZE, (128, 128, 128))
        # 统一 resize 到 256×256（双线性）
        img = img.resize(TARGET_SIZE, Image.BILINEAR)

        # GT
        if sample["gt"] is not None and os.path.isfile(sample["gt"]):
            try:
                gt = Image.open(sample["gt"]).convert("L")
            except Exception as e:
                print(f"[WARN] 打不开 GT {sample['gt']}: {e}")
                gt = Image.new("L", TARGET_SIZE, 0)
        else:
            gt = Image.new("L", TARGET_SIZE, 0)
        # 统一 resize 到 256×256（最近邻，避免灰度插值污染标签）
        gt = gt.resize(TARGET_SIZE, Image.NEAREST)

        # 放到子图
        ax = axes[r][0]
        ax.imshow(img)
        ax.axis("off")
        if r == 0:
            ax.set_title(col_titles[0], fontsize=10)

        ax = axes[r][1]
        ax.imshow(gt, cmap="gray")
        ax.axis("off")
        if r == 0:
            ax.set_title(col_titles[1], fontsize=10)

        # 各模型预测
        for c, m in enumerate(models, start=2):
            pred_path = sample["preds"].get(m)
            if pred_path is not None and os.path.isfile(pred_path):
                try:
                    pred_img = Image.open(pred_path).convert("L")
                except Exception as e:
                    print(f"[WARN] 打不开预测图 {pred_path}: {e}")
                    pred_img = Image.new("L", TARGET_SIZE, 0)
            else:
                pred_img = Image.new("L", TARGET_SIZE, 0)

            # 预测图也统一 resize（多数已经是 256×256，这里再调一次没问题）
            pred_img = pred_img.resize(TARGET_SIZE, Image.NEAREST)

            ax = axes[r][c]
            ax.imshow(pred_img, cmap="gray")
            ax.axis("off")
            if r == 0:
                ax.set_title(m, fontsize=9)

    plt.tight_layout()
    out_fig = os.path.join(ROOT, f"{ds}_qualitative.png")
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f"[INFO] 数据集 {ds} 的对比图已保存到: {out_fig}")

# 保存定性对比信息 JSON（可选）
qual_json = os.path.join(ROOT, "qualitative_samples_debug.json")
with open(qual_json, "w", encoding="utf-8") as f:
    json.dump(qualitative, f, indent=2, ensure_ascii=False)
print(f"[INFO] 定性对比样本信息已保存到: {qual_json}")
