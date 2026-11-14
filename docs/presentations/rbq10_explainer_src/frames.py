import sys, numpy as np, imageio_ffmpeg as f
from PIL import Image
out, frac = sys.argv[1], float(sys.argv[2])
for p in sys.argv[3:]:
    n, sec = f.count_frames_and_secs(p)
    rd = f.read_frames(p); m = next(rd); w, h = m["size"]
    k = int(n * frac)
    for i, fr in enumerate(rd):
        if i == k:
            img = Image.frombytes("RGB", (w, h), fr)
            stem = p.split("/")[-1].rsplit(".", 1)[0]
            name = stem if not stem.startswith("walk") else p.split("/")[-2].replace("65_bprime_stride_s2_861_hangang_", "")
            img.save(f"{out}/{name}.jpg", quality=88); break
