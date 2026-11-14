import sys, imageio.v3 as iio, imageio_ffmpeg as f
out = sys.argv[1]
for p in sys.argv[2:]:
    meta = iio.immeta(p, plugin="pyav") if False else None
    n, sec = f.count_frames_and_secs(p)
    rd = f.read_frames(p); m = next(rd)
    name = p.split("/")[-2]
    print(name, p.split("/")[-1], m["size"], round(sec, 1), "s", round(m["fps"], 1), "fps")
