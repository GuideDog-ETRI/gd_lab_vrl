"""Tensor-shaped input/output figure for the RBQ10 vision policy (it861).

Shapes follow the code: gd_lab/gast/temporal.py (terrain encoder), gd_lab/gast_narrow/model.py (LiDAR CNN),
DreamWaQ CENet / Actor. Blocks are drawn as cuboids (channels = depth); sizes are schematic, the labels are exact.
  python3 make_tensor_fig.py out.svg
"""
import sys

W, H = 2400, 1350
FONT = "Noto Sans CJK KR, Malgun Gothic, sans-serif"
PAL = {  # same families as docs/figures/vrl_*.svg
    "in": ("#E3EEFB", "#3B6FB6"), "enc": ("#E3F2E1", "#3E8E41"), "pol": ("#FBE7D0", "#D9822B"),
    "ext": ("#EDE3F6", "#7D4CB0"), "out": ("#FCF3C6", "#B8921E"), "train": ("#FBE3E3", "#C94A4A"), "obs": ("#EEF0F1", "#6E7F80"),
}
out = []


def shade(hexcol, f):
    h = hexcol.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    c = lambda v: max(0, min(255, int(v * f)))  # noqa: E731
    return f"#{c(r):02X}{c(g):02X}{c(b):02X}"


def text(x, y, s, size=20, weight="normal", fill="#1B2430", anchor="middle"):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    out.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
               f'text-anchor="{anchor}">{s}</text>')


def cuboid(x, y, w, h, d, fam, grid=None, label=None, dims=None):
    """Front face at (x, y, w, h); depth d drawn up-right. grid = (cols, rows) draws cells on the front face."""
    fill, stroke = PAL[fam]
    dx, dy = d * 0.6, -d * 0.45
    out.append(f'<polygon points="{x},{y} {x + dx},{y + dy} {x + w + dx},{y + dy} {x + w},{y}" fill="{shade(fill, 1.03)}" stroke="{stroke}" stroke-width="1.6"/>')
    out.append(f'<polygon points="{x + w},{y} {x + w + dx},{y + dy} {x + w + dx},{y + h + dy} {x + w},{y + h}" fill="{shade(fill, 0.86)}" stroke="{stroke}" stroke-width="1.6"/>')
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    if grid:
        cols, rows = grid
        for i in range(1, cols):
            out.append(f'<line x1="{x + w * i / cols}" y1="{y}" x2="{x + w * i / cols}" y2="{y + h}" stroke="{stroke}" stroke-opacity="0.25" stroke-width="0.8"/>')
        for j in range(1, rows):
            out.append(f'<line x1="{x}" y1="{y + h * j / rows}" x2="{x + w}" y2="{y + h * j / rows}" stroke="{stroke}" stroke-opacity="0.25" stroke-width="0.8"/>')
    if dims:  # (depth, height, width) labels on the edges
        dd, hh, ww = dims
        text(x + w / 2, y + h + 22, str(ww), 16, fill="#55606A")
        text(x - 10, y + h / 2 + 6, str(hh), 16, fill="#55606A", anchor="end")
        text(x + w + dx / 2 + 8, y + dy / 2 - 4, str(dd), 16, fill="#55606A", anchor="start")
    if label:
        for k, line in enumerate(label):
            text(x + (w + dx) / 2, y + h + 50 + 24 * k, line, 18 if k else 19, "bold" if k == 0 else "normal",
                 stroke if k == 0 else "#1B2430")


def bar(x, y, w, h, fam, label=None, n=None, vertical_label=False):
    fill, stroke = PAL[fam]
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    if n is not None:
        text(x + w / 2, y - 10, str(n), 18, "bold", stroke)
    if label:
        for k, line in enumerate(label):
            text(x + w / 2, y + h + 26 + 22 * k, line, 16, fill="#1B2430")


def box(x, y, w, h, fam, lines, dashed=False):
    fill, stroke = PAL[fam]
    dash = ' stroke-dasharray="8,6"' if dashed else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>')
    n = len(lines)
    for k, line in enumerate(lines):
        text(x + w / 2, y + h / 2 + (k - (n - 1) / 2) * 24 + 7, line, 19 if k == 0 else 16, "bold" if k == 0 else "normal",
             stroke if k == 0 else "#1B2430")


def arrow(x1, y1, x2, y2, label=None, color="#55606A", dy_label=30, dashed=False):
    dash = ' stroke-dasharray="8,6"' if dashed else ""
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.2" marker-end="url(#ah)"{dash}/>')
    if label:
        text((x1 + x2) / 2, (y1 + y2) / 2 + dy_label, label, 15, fill="#55606A")


def elbow(pts, color="#55606A"):
    d = " ".join(f"{x},{y}" for x, y in pts)
    out.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2.2" marker-end="url(#ah)"/>')


def section(x, y, s, fam):
    text(x, y, s, 22, "bold", PAL[fam][1], anchor="start")


# ------------------------------------------------------------------ terrain row
section(40, 60, "① 하방 지형 (D4xx 4대 → 매퍼) — 프레임 8장을 기억", "in")
for i in range(8):  # oldest at the back
    fx, fy = 70 + i * 13, 120 + i * 11
    cuboid(fx, fy, 153, 99, 24, "in", grid=(17, 11))
fx, fy = 70 + 7 * 13, 120 + 7 * 11
for c in (10, 11):  # gap columns on the current frame
    out.append(f'<rect x="{fx + 153 * c / 17}" y="{fy}" width="{153 / 17}" height="99" fill="#0E3B43" fill-opacity="0.85"/>')
text(70 + 92, fy + 135, "지형 [8, 3, 11, 17]", 19, "bold", PAL["in"][1])
text(70 + 92, fy + 160, "8 프레임 × (높이, 유효, IR) × 11 × 17칸", 16)
text(70 + 92, fy + 182, "t−48 … t−1, t (0.1 s마다 캡처)", 15, fill="#55606A")
arrow(345, 250, 385, 250)
cuboid(400, 200, 153, 99, 32, "enc", grid=(17, 11), label=["[8, 11, 17]", "Conv 3→8 (프레임마다 같은 CNN)"])
arrow(580, 250, 610, 250)
cuboid(625, 200, 153, 99, 48, "enc", grid=(17, 11), label=["[16, 11, 17]", "Conv 8→16"])
arrow(812, 250, 842, 250)
cuboid(855, 228, 72, 45, 48, "enc", grid=(8, 5), label=["[16, 5, 8]", "AvgPool", "→ 640 → FC 64"])
arrow(962, 250, 995, 250)
# per-frame feature matrix [8, 64]
mx, my = 1005, 205
out.append(f'<rect x="{mx}" y="{my}" width="128" height="88" fill="{PAL["enc"][0]}" stroke="{PAL["enc"][1]}" stroke-width="1.8"/>')
for j in range(1, 8):
    out.append(f'<line x1="{mx}" y1="{my + 11 * j}" x2="{mx + 128}" y2="{my + 11 * j}" stroke="{PAL["enc"][1]}" stroke-opacity="0.45"/>')
text(mx + 64, my + 120, "[8, 64]", 19, "bold", PAL["enc"][1])
text(mx + 64, my + 144, "8 프레임 특징", 16)
text(mx + 64, my + 166, "+ age 임베딩 (2→64)", 16)
arrow(1140, 250, 1170, 250)
box(1175, 205, 170, 90, "enc", ["MHA 4 head", "+ 잔차 + LayerNorm", "[8, 64]"])
arrow(1350, 250, 1378, 250)
box(1383, 205, 110, 90, "enc", ["GRU", "→ [64]"])
arrow(1498, 250, 1525, 250)
bar(1532, 202, 22, 96, "enc", n=32, label=["FC tanh", "지형 latent"])

# ------------------------------------------------------------------ LiDAR row
section(40, 470, "② 머리 LiDAR 점유 지도 (통로 벽)", "in")
cuboid(70, 560, 240, 150, 20, "in", grid=(48, 30), dims=(2, 30, 48))
for (r0, r1) in ((4, 7), (23, 26)):  # walls
    out.append(f'<rect x="70" y="{560 + 150 * r0 / 30}" width="240" height="{150 * (r1 - r0) / 30}" fill="#3B6FB6" fill-opacity="0.55"/>')
text(70 + 126, 560 + 150 + 50, "LiDAR [2, 30, 48]", 19, "bold", PAL["in"][1])
text(70 + 126, 560 + 150 + 74, "이전·현재 2장 × 30 × 48칸 (5 cm)", 16)
arrow(335, 635, 395, 635)
cuboid(405, 598, 120, 75, 32, "enc", grid=(24, 15), label=["[8, 15, 24]", "Conv 5×5 s2"])
arrow(550, 635, 590, 635)
cuboid(600, 615, 60, 40, 48, "enc", grid=(12, 8), label=["[16, 8, 12]", "Conv 3×3 s2"])
arrow(692, 635, 730, 635)
cuboid(742, 625, 30, 20, 64, "enc", grid=(6, 4), label=["[32, 4, 6]", "Conv 3×3 s2", "→ 768"])
arrow(815, 635, 860, 635)
bar(868, 587, 20, 96, "enc", n=64)
arrow(895, 635, 935, 635)
bar(942, 611, 20, 48, "enc", n=16, label=["FC → FC", "LiDAR latent"])

# ------------------------------------------------------------------ proprioception row
section(40, 860, "③ 몸 감각 (IMU + 관절) — 최근 스텝 히스토리", "in")
px, py, cw, ch = 70, 905, 7, 16
groups = [(3, "#BFD7F2"), (3, "#9FC3EC"), (3, "#EDE3F6"), (12, "#CFE2F7"), (12, "#B3D0F0"), (12, "#DCE9F8"), (1, "#EDE3F6")]
col = 0
for n, c in groups:
    out.append(f'<rect x="{px + col * cw}" y="{py}" width="{n * cw}" height="{5 * ch}" fill="{c}" stroke="#3B6FB6" stroke-width="1"/>')
    col += n
for j in range(1, 5):
    out.append(f'<line x1="{px}" y1="{py + j * ch}" x2="{px + 46 * cw}" y2="{py + j * ch}" stroke="#3B6FB6" stroke-opacity="0.5"/>')
out.append(f'<rect x="{px}" y="{py}" width="{46 * cw}" height="{5 * ch}" fill="none" stroke="#3B6FB6" stroke-width="1.8"/>')
labels = ["각속도 3", "자세 3", "명령 3", "관절 각 12", "관절 속도 12", "직전 행동 12", "적재 1"]
col = 0
for (n, _), lab in zip(groups, labels):
    xm = px + (col + n / 2) * cw
    if n >= 12:
        text(xm, py + 5 * ch + 22, lab, 14, fill="#55606A")
    col += n
text(px + 9 * cw / 2, py - 12, "각속도·자세·명령 3씩", 13, fill="#55606A")
text(px + 46 * cw / 2, py + 5 * ch + 50, "고유감각 [5, 46] (최근 5 스텝, 100 Hz)", 19, "bold", PAL["in"][1])
# brackets: CENet uses 5 rows, actor the last 4
arrow(px + 46 * cw + 10, py + 8, 470, py + 8, "5 스텝 230", dy_label=-12)
box(475, py - 32, 190, 80, "enc", ["CENet (β-VAE)", "230 → 128 → 64"])
arrow(670, py + 8, 700, py + 8)
bar(705, py - 30, 20, 76, "enc", n=19)
text(735, py + 2, "z: 속도 3 + 맥락 16", 15, anchor="start")
arrow(px + 46 * cw + 10, py + 66, 905, py + 66, "최근 4 스텝 → flatten 184", dy_label=26)
bar(910, py - 14, 20, 160, "in", n=184)

# ------------------------------------------------------------------ external condition row
section(40, 1170, "④ 외부 조건 (gdm)", "ext")
cuboid(80, 1205, 34, 34, 14, "ext")
text(220, 1230, "여유 m [1] = (m − 0.10) / 0.05", 17, anchor="start")
cuboid(80, 1265, 34, 34, 14, "obs")
text(220, 1290, "회전 차단 [1] (이 계보는 항상 0)", 17, anchor="start", fill="#55606A")

# ------------------------------------------------------------------ concat + actor
cx, cy0, scale = 1660, 230, 3.0
segs = [("지형 32", 32, "enc"), ("LiDAR 16", 16, "enc"), ("z 19", 19, "enc"), ("몸 감각 184", 184, "in"), ("여유 1", 1, "ext"), ("회전 1", 1, "obs")]
y = cy0
mids = {}
for name, n, fam in segs:
    h = max(n * scale, 16)
    fill, stroke = PAL[fam]
    out.append(f'<rect x="{cx}" y="{y}" width="34" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    text(cx + 44, y + h / 2 + 6, name, 15, anchor="start", fill="#55606A")
    mids[name] = y + h / 2
    y += h
text(cx + 17, cy0 - 40, "Actor 입력", 19, "bold", PAL["pol"][1])
text(cx + 17, cy0 - 16, "[253]", 19, "bold", PAL["pol"][1])
elbow([(1560, 250), (1590, 250), (1590, mids["지형 32"]), (cx - 2, mids["지형 32"])])
elbow([(968, 635), (1560, 635), (1560, mids["LiDAR 16"]), (cx - 2, mids["LiDAR 16"])])
elbow([(715, py - 58), (715, py - 80), (1590, py - 80), (1590, mids["z 19"]), (cx - 2, mids["z 19"])])
elbow([(935, py + 66), (1620, py + 66), (1620, mids["몸 감각 184"]), (cx - 2, mids["몸 감각 184"])])
elbow([(560, 1222), (1635, 1222), (1635, mids["여유 1"]), (cx - 2, mids["여유 1"])])
# MLP
mlp_x, mid = 1830, cy0 + (y - cy0) / 2
for k, (n, hpx) in enumerate(((512, 560), (256, 300), (128, 170))):
    bx = mlp_x + k * 62
    bar(bx, mid - hpx / 2, 26, hpx, "pol", n=n)
    if k == 0:
        arrow(cx + 150, mid, bx - 2, mid)
    else:
        arrow(bx - 34, mid, bx - 2, mid)
text(mlp_x + 75, mid + 330, "Actor MLP (ELU)", 19, "bold", PAL["pol"][1])
arrow(mlp_x + 150, mid, mlp_x + 175, mid)
bar(mlp_x + 182, mid - 48, 22, 96, "out", label=["행동 a [12]"])

# ------------------------------------------------------------------ outputs as [4 legs x 3 joints]
gx, gy, gw, gh = 2140, 330, 52, 38
legs, joints = ["FL", "FR", "HL", "HR"], ["HIP", "THIGH", "KNEE"]
elbow([(mlp_x + 193, mid - 72), (mlp_x + 193, gy + 2 * gh), (gx - 44, gy + 2 * gh)])
text(gx + 1.5 * gw, gy - 62, "관절 목표 q* [4, 3]", 19, "bold", PAL["out"][1])
text(gx + 1.5 * gw, gy - 38, "q* = q₀ + 0.25 · a  (100 Hz)", 16)
for j, jn in enumerate(joints):
    text(gx + (j + 0.5) * gw, gy - 8, jn, 13, fill="#55606A")
for i, ln in enumerate(legs):
    text(gx - 10, gy + (i + 0.5) * gh + 6, ln, 15, "bold", "#55606A", anchor="end")
    for j in range(3):
        out.append(f'<rect x="{gx + j * gw}" y="{gy + i * gh}" width="{gw}" height="{gh}" fill="{PAL["out"][0]}" stroke="{PAL["out"][1]}" stroke-width="1.6"/>')
arrow(gx + 1.5 * gw, gy + 4 * gh + 8, gx + 1.5 * gw, gy + 4 * gh + 70)
box(gx - 70, gy + 4 * gh + 75, 296, 92, "out", ["PD 제어 (500 Hz)", "τ = Kp (q* − q) − Kd q̇", "Kp 123.39 / 127.77 · Kd 2.4"])
ty = gy + 4 * gh + 210
arrow(gx + 1.5 * gw, gy + 4 * gh + 170, gx + 1.5 * gw, ty - 30)
text(gx + 1.5 * gw, ty - 40, "", 10)
for i, ln in enumerate(legs):
    text(gx - 10, ty + (i + 0.5) * gh + 6, ln, 15, "bold", "#55606A", anchor="end")
    for j in range(3):
        out.append(f'<rect x="{gx + j * gw}" y="{ty + i * gh}" width="{gw}" height="{gh}" fill="#F6E2A8" stroke="{PAL["out"][1]}" stroke-width="1.6"/>')
text(gx + 1.5 * gw, ty + 4 * gh + 28, "모터 토크 τ [4, 3]", 19, "bold", PAL["out"][1])
text(gx + 1.5 * gw, ty + 4 * gh + 52, "한계 93.6 / 126 N·m", 16)

# training-only critic
box(1960, 1170, 420, 100, "train", ["Critic (학습 전용)", "특권 관측 [298 + 9] → 512-256-128 → V(s)", "로봇에는 실리지 않음"], dashed=True)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
       '<defs><marker id="ah" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">'
       '<path d="M0,0 L10,5 L0,10 z" fill="#55606A"/></marker></defs>'
       f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>' + "".join(out) + "</svg>")
open(sys.argv[1], "w").write(svg)
print("wrote", sys.argv[1])
