import cv2, glob, os
for p in sorted(glob.glob(r'.\data_test\input\**\*.png', recursive=True)):
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    shape = None if img is None else img.shape
    print(os.path.relpath(p), '->', shape)
