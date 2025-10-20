from huggingface_hub import snapshot_download
import os, glob, shutil

BASE = r"D:\cobaa\FourLLIE-main"
DL   = os.path.join(BASE, "data_lolv2_synth")

# download folder Test saja (lebih kecil & cepat)
snapshot_download(
    repo_id="okhater/lolv2-synthetic",
    local_dir=DL,
    allow_patterns=["Test/Input/*", "Test/GT/*"]
)

in_dir = os.path.join(BASE, "data_test", "input", "0002")
gt_dir = os.path.join(BASE, "data_test", "gt",    "0002")
os.makedirs(in_dir, exist_ok=True)
os.makedirs(gt_dir, exist_ok=True)

# salin ke struktur FourLLIE
for src in glob.glob(os.path.join(DL, "Test", "Input", "*")):
    shutil.copy(src, os.path.join(in_dir, os.path.basename(src)))

for src in glob.glob(os.path.join(DL, "Test", "GT", "*")):
    shutil.copy(src, os.path.join(gt_dir, os.path.basename(src)))

print("Selesai copy.")
print("Input:", len(glob.glob(os.path.join(in_dir, "*"))), "file")
print("GT   :", len(glob.glob(os.path.join(gt_dir, "*"))), "file")
