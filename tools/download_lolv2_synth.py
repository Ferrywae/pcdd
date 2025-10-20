from datasets import load_dataset
import os, shutil

BASE = r"D:\cobaa\FourLLIE-main"
in_dir = os.path.join(BASE, "data_test", "input", "0002")
gt_dir = os.path.join(BASE, "data_test", "gt", "0002")
os.makedirs(in_dir, exist_ok=True)
os.makedirs(gt_dir, exist_ok=True)

# ambil split "test" biar cepat; ganti ke "train" kalau mau lebih banyak
ds = load_dataset("okhater/lolv2-synthetic", split="test")

n = 0
for ex in ds:
    # kolom nama file bisa bervariasi; coba beberapa kemungkinan
    low  = ex.get("low") or ex.get("low_path") or ex.get("low_img")
    high = ex.get("normal") or ex.get("high") or ex.get("normal_path") or ex.get("gt_img")
    if not (low and high):
        continue
    n += 1
    shutil.copy(low,  os.path.join(in_dir, f"{n:05d}.png"))
    shutil.copy(high, os.path.join(gt_dir, f"{n:05d}.png"))

print(f"Copied {n} pairs to:")
print(" -", in_dir)
print(" -", gt_dir)
