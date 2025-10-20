import os, argparse, numpy as np, cv2, random

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def make_colorful(h, w, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    # pola warna acak (gradien + patch)
    base = np.zeros((h, w, 3), np.uint8)
    # gradien
    gx = np.linspace(0, 255, w, dtype=np.uint8)
    gy = np.linspace(0, 255, h, dtype=np.uint8)
    Gx = np.tile(gx, (h,1))
    Gy = np.tile(gy[:,None], (1,w))
    base[:,:,0] = Gx
    base[:,:,1] = Gy
    base[:,:,2] = ((Gx.astype(int)+Gy.astype(int))%256).astype(np.uint8)
    # tambahkan 3 patch warna acak
    for _ in range(3):
        ph, pw = random.randint(h//8, h//3), random.randint(w//8, w//3)
        y = random.randint(0, h-ph); x = random.randint(0, w-pw)
        color = np.random.randint(0,256,(1,1,3),dtype=np.uint8)
        base[y:y+ph, x:x+pw] = color
    return base

def darken(img, min_factor=0.1, max_factor=0.35):
    factor = random.uniform(min_factor, max_factor)
    # gamma + noise kecil
    g = random.uniform(1.8, 2.6)
    f = np.float32(img)/255.0
    f = np.clip((f**g)*factor, 0, 1)
    # noise
    noise = np.random.normal(0, 0.02, f.shape).astype(np.float32)
    f = np.clip(f + noise, 0, 1)
    out = (f*255.0 + 0.5).astype(np.uint8)
    return out

def jitter(img):
    # sedikit perubahan per-frame (optional)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:,:,0] = (hsv[:,:,0] + np.random.randint(-6,7)) % 180
    hsv[:,:,1] = np.clip(hsv[:,:,1] + np.random.randint(-20,21), 0, 255)
    hsv[:,:,2] = np.clip(hsv[:,:,2] + np.random.randint(-15,16), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=20, help="jumlah folder pasangan input/gt")
    ap.add_argument("--frames", type=int, default=1, help="gambar per folder (sequence)")
    ap.add_argument("--size", type=int, default=256, help="ukuran sisi (square)")
    ap.add_argument("--out", type=str, default="data_test", help="root output")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    H = W = args.size
    root_in  = os.path.join(args.out, "input")
    root_gt  = os.path.join(args.out, "gt")
    ensure_dir(root_in); ensure_dir(root_gt)

    random.seed(args.seed); np.random.seed(args.seed)

    for i in range(1, args.pairs+1):
        folder = f"{i:04d}"
        din = os.path.join(root_in, folder)
        dgt = os.path.join(root_gt, folder)
        ensure_dir(din); ensure_dir(dgt)

        # GT warna-warni
        gt_base = make_colorful(H, W, seed=args.seed + i)
        cv2.imwrite(os.path.join(dgt, "img.png"), gt_base)

        # Input gelap (bisa multi-frame)
        base_in = darken(gt_base)
        for fidx in range(1, args.frames+1):
            frame = jitter(base_in) if args.frames > 1 else base_in
            if args.frames == 1:
                name = "img.png"
            else:
                name = f"img_{fidx:03d}.png"
            cv2.imwrite(os.path.join(din, name), frame)

    print(f"Done. Created {args.pairs} pairs in {args.out} (frames per pair: {args.frames}).")

if __name__ == "__main__":
    main()
