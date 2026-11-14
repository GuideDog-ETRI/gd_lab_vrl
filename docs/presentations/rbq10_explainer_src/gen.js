// RBQ10 vision-RL explainer deck (source: docs/GD_VRL_VISION_RL_GUIDE.md).
// Builds deck.pptx and, from the same coordinates, preview/slide-NN.png with measured text (overflow check).
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const React = require("react");
const RDS = require("react-dom/server");
const lu = require("react-icons/lu");
const md = require("react-icons/md");
const fs = require("fs");
const path = require("path");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const THEME = {
  name: "RBQ10 Guide Teal",
  headFontFace: "Malgun Gothic",
  bodyFontFace: "Malgun Gothic",
  colors: {
    dk1: "1B2430", lt1: "FFFFFF", dk2: "0E3B43", lt2: "EAF2F1",
    accent1: "0E7C7B", accent2: "F2A900", accent3: "3C6E71", accent4: "C8453B",
    accent5: "6E7F80", accent6: "9CC9C5", hlink: "0E7C7B", folHlink: "3C6E71",
  },
};
const HEX = { tx1: "dk1", tx2: "dk2", bg1: "lt1", bg2: "lt2" };
const hex = (c) => (c && THEME.colors[HEX[c] || c]) || c || "000000";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "RBQ10 비전 보행 학습";
pres.author = "bsseo";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333, H = 7.5;

// ---------------------------------------------------------------- layouts
const LAYOUT_PH = {}; // name -> placeholder name -> {x,y,w,h,fontSize,bold,color}
function layout(name, bg, phs, extra = []) {
  LAYOUT_PH[name] = { bg, phs: {} };
  const objects = [...extra];
  for (const p of phs) {
    LAYOUT_PH[name].phs[p.name] = p;
    objects.push({ placeholder: { options: { name: p.name, type: p.type || "body", x: p.x, y: p.y, w: p.w, h: p.h,
      fontSize: p.fontSize, bold: !!p.bold, color: p.color, valign: p.valign || "top", margin: 0, fontFace: p.fontFace },
      text: "" } });
  }
  pres.defineSlideMaster({ title: name, background: { color: bg }, objects,
    slideNumber: name.startsWith("DARK") ? undefined : { x: 12.25, y: 6.95, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5, align: "right" } });
}
const KICK = { name: "kicker", x: 0.6, y: 0.32, w: 9, h: 0.32, fontSize: 13, bold: true, color: C.accent1 };
const TITLE = { name: "title", type: "title", x: 0.6, y: 0.64, w: 12.1, h: 0.72, fontSize: 30, bold: true, color: C.text2 };
layout("DARK_TITLE", C.text2, [
  { name: "title", type: "title", x: 0.8, y: 2.2, w: 7.6, h: 1.7, fontSize: 40, bold: true, color: C.background1, valign: "bottom" },
  { name: "body", x: 0.8, y: 4.15, w: 7.6, h: 1.0, fontSize: 20, color: C.accent6 }]);
layout("CONTENT", C.background1, [KICK, TITLE], [
  { text: { text: "RBQ10 비전 보행 학습", options: { x: 0.6, y: 6.95, w: 4, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } }]);
layout("DARK_CONTENT", C.text2, [{ ...KICK, color: C.accent2 }, { ...TITLE, color: C.background1 }]);

// ---------------------------------------------------------------- recorder
const REC = [];
const clone = (o) => JSON.parse(JSON.stringify(o));
function slide(master, section, kicker, title, notes) {
  const s = pres.addSlide({ masterName: master, sectionTitle: section });
  const r = { master, items: [] };
  REC.push(r);
  s._r = r;
  const L = LAYOUT_PH[master].phs;
  if (kicker !== null && L.kicker) put(s, "ph", kicker, L.kicker), s.addText(kicker, { placeholder: "kicker" });
  if (title !== null) put(s, "ph", title, L.title), s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function put(s, k, text, o) { s._r.items.push({ k, text: clone(text), o: clone(o) }); }
let _n = 0;
function T(s, text, o) {
  const opt = { margin: 0, valign: "top", fontSize: 14, color: C.text1, objectName: "t" + _n++, ...o, isTextBox: true };
  put(s, "text", text, opt);
  s.addText(text, opt);
}
function S(s, type, o) {
  const opt = { objectName: "s" + _n++, ...o };
  put(s, "shape", type, opt);
  s.addShape(type, opt);
}
function IMG(s, o) { put(s, "img", null, o); s.addImage({ ...o }); }
function TAB(s, rows, o) { put(s, "table", rows, o); s.addTable(clone(rows), { ...o }); }
function MEDIA(s, o) {
  put(s, "img", null, { ...o, path: o.coverPath });
  s.addMedia({ type: "video", path: o.path, cover: "data:image/jpeg;base64," + fs.readFileSync(o.coverPath).toString("base64"),
    x: o.x, y: o.y, w: o.w, h: o.h, objectName: o.objectName });
}
function CHART(s, type, data, o) { put(s, "chart", { type, data }, o); s.addChart(type, clone(data), { ...o }); }

// ---------------------------------------------------------------- icons
const ICON_CACHE = {};
async function iconPng(name, color) {
  const key = name + color;
  if (ICON_CACHE[key]) return ICON_CACHE[key];
  const Comp = lu[name] || md[name];
  const svg = RDS.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
  return (ICON_CACHE[key] = "image/png;base64," + buf.toString("base64"));
}
async function ICON(s, name, x, y, d, circle = C.accent1, fg = "FFFFFF") {
  S(s, pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: circle }, line: { color: circle } });
  const p = d * 0.22;
  IMG(s, { data: await iconPng(name, fg), x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, altText: name });
}
function ARROW(s, x1, y1, x2, y2, color = C.accent5, w = 2) {
  S(s, pres.ShapeType.line, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001,
    flipH: x2 < x1, flipV: y2 < y1, line: { color, width: w, endArrowType: "triangle" } });
}
function CARD(s, x, y, w, h, fill = C.background2) {
  S(s, pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill } });
}
const FIG = __dirname + "/";

// ================================================================= slides
async function build() {
  const SEC = ["문제와 개념", "구조", "학습 방법", "결과", "영상 예제", "다음 단계", "부록"];

  // 1 title
  pres.addSection({ title: SEC[0] });
  let s = slide("DARK_TITLE", SEC[0], null, "안내견 로봇 EDDI는 어떻게\n'보면서' 걷는 법을 배웠나",
    "이 자료는 RBQ10 안내견 로봇의 시각 보행 정책을 처음 보는 분께 설명하기 위한 것입니다. 개념, 구조, 학습 방법, 결과, 다음 단계 순서로 갑니다. 기준 모델은 2026-11-07에 최종 선정한 it861입니다.");
  put(s, "ph", "비전 강화학습 개념부터 최종 모델 it861까지 · 2026-11", LAYOUT_PH.DARK_TITLE.phs.body);
  s.addText("비전 강화학습 개념부터 최종 모델 it861까지 · 2026-11", { placeholder: "body" });
  await ICON(s, "LuFootprints", 0.8, 0.9, 1.0, C.accent2, THEME.colors.dk2);
  CARD(s, 9.0, 1.45, 3.75, 4.75, "15505A");
  T(s, "순서", { x: 9.35, y: 1.7, w: 3.1, h: 0.4, fontSize: 14, bold: true, color: C.accent2 });
  ["문제와 개념", "구조", "학습 방법", "결과", "영상 예제", "다음 단계"].forEach((t, i) => {
    T(s, [{ text: String(i + 1) + "   ", options: { bold: true, color: C.accent2 } }, { text: t }],
      { x: 9.35, y: 2.25 + i * 0.62, w: 3.1, h: 0.45, fontSize: 18, color: C.background1 });
  });

  // 2 problem
  s = slide("CONTENT", SEC[0], "1 · 문제와 개념", "지하철 승강장과 계단을 안전하게 지나가야 한다",
    "로봇이 시각장애인을 하네스로 이끌며 지나야 하는 곳입니다. 승강장 갭은 폭과 단차가 다양하고, 계단에서는 사람이 당기거나 미는 힘도 받습니다. 관절 감각만 쓰는 블라인드 정책은 갭에서 발이 빠지고 계단에서 걸려서, 지형을 보는 정책이 필요했습니다.");
  const cards = [
    ["LuTrainFront", "2–26 cm", "승강장 갭", "단차 0–16 cm, 전진·후진\n(후진은 20 cm 이하)"],
    ["MdStairs", "5–25 cm", "계단", "사람이 당기거나 밀어도\n(200 N 이하) 버틴다"],
    ["LuDoorOpen", "0.5–1.2 m", "좁은 통로", "막혀 있으면\n앞에서 멈춘다"],
    ["LuGauge", "0.6–1.0 m/s", "실제 운용 속도", "적재 0 kg과 5 kg에서\n같은 성능"]];
  for (let i = 0; i < 4; i++) {
    const x = 0.6 + i * 3.1;
    CARD(s, x, 1.65, 2.85, 3.2);
    await ICON(s, cards[i][0], x + 0.3, 1.85, 0.75);
    T(s, cards[i][1], { x: x + 0.3, y: 2.8, w: 2.4, h: 0.65, fontSize: 30, bold: true, color: C.accent1 });
    T(s, cards[i][2], { x: x + 0.3, y: 3.5, w: 2.4, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
    T(s, cards[i][3], { x: x + 0.3, y: 3.95, w: 2.45, h: 1.0, fontSize: 14, color: C.text1 });
  }
  CARD(s, 0.6, 5.2, 12.15, 1.2, C.text2);
  await ICON(s, "LuEye", 0.85, 5.4, 0.8, C.accent2, THEME.colors.dk2);
  T(s, [{ text: "관절·자세 감각만 쓰는 정책은 갭에서 발이 빠지고 계단에서 걸린다", options: { bold: true, breakLine: true } },
        { text: "→ 몸 앞·아래 지형을 '보는' 정책이 필요하다", options: { color: C.accent2 } }],
    { x: 1.9, y: 5.35, w: 10.6, h: 0.9, fontSize: 17, color: C.background1, valign: "middle" });

  // 3 RL loop
  s = slide("CONTENT", SEC[0], "1 · 문제와 개념", "강화학습: 시뮬레이터에서 수없이 넘어지며 배운다",
    "강화학습은 정책 신경망이 시뮬레이터에서 행동하고, 잘하면 점수를 받고, 점수가 오르는 쪽으로 신경망을 고치는 과정의 반복입니다. 수만 마리 로봇을 동시에 돌려서 짧은 시간에 엄청난 양의 경험을 모읍니다.");
  const loop = [
    ["LuBrain", "① 정책이 행동", "관절 목표 12개를\n1초에 100번 정한다"],
    ["LuCpu", "② 시뮬레이터", "Isaac Lab에서 로봇\n수만 마리가 동시에"],
    ["LuTarget", "③ 보상", "잘 걸으면 +,\n넘어지거나 빠지면 −"],
    ["LuTrendingUp", "④ PPO 갱신", "점수가 오르는 쪽으로\n신경망을 조금 고친다"]];
  for (let i = 0; i < 4; i++) {
    const x = 0.6 + i * 3.15;
    CARD(s, x, 1.65, 2.7, 2.2);
    await ICON(s, loop[i][0], x + 0.25, 1.85, 0.65);
    T(s, loop[i][1], { x: x + 0.25, y: 2.6, w: 2.3, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
    T(s, loop[i][2], { x: x + 0.25, y: 3.0, w: 2.35, h: 0.75, fontSize: 14 });
    if (i < 3) ARROW(s, x + 2.75, 2.75, x + 3.1, 2.75, C.accent1, 2.5);
  }
  S(s, pres.ShapeType.line, { x: 1.95, y: 4.2, w: 9.45, h: 0, line: { color: C.accent1, width: 2, dashType: "dash" } });
  ARROW(s, 11.4, 3.88, 11.4, 4.2, C.accent1, 2);
  ARROW(s, 1.95, 4.2, 1.95, 3.9, C.accent1, 2);
  T(s, "반복: 단계마다 300–2,000번 (기본 보행은 38,000번)", { x: 4.2, y: 4.27, w: 5, h: 0.35, fontSize: 13, color: C.accent1, align: "center" });
  const stats = [["131,072", "한강 서버에서 동시에 학습한 로봇 수"], ["0.01 s", "정책이 한 번 판단하는 간격 (100 Hz)"], ["1,310만", "한 번 갱신에 쓰는 transition 수"]];
  for (let i = 0; i < 3; i++) {
    const x = 0.6 + i * 4.1;
    T(s, stats[i][0], { x, y: 4.9, w: 3.8, h: 0.9, fontSize: 44, bold: true, color: C.accent1 });
    T(s, stats[i][1], { x, y: 5.85, w: 3.8, h: 0.6, fontSize: 14, color: C.accent5 });
  }

  // 4 map instead of images
  s = slide("CONTENT", SEC[0], "1 · 문제와 개념", "카메라 영상 대신 '작은 지형 지도'를 본다",
    "카메라 영상을 정책에 그대로 넣지 않습니다. 사람이 만든 지형처리기가 깊이 영상을 11×17칸 지도로 바꾸고, 정책은 이 지도를 봅니다. 학습할 때도 같은 형식의 지도를 쓰기 때문에, 시뮬레이션에서 배운 것이 실제 로봇에서도 같은 모양의 입력으로 이어집니다.");
  const flow = [["LuCamera", "카메라 4대", "D4xx 깊이 + IR, 몸 아래·앞뒤"], ["LuFilter", "지형처리기", "카메라 깊이 → 지도"], ["LuGrid3X3", "11×17 지도", "10 cm 칸, 1.6 × 1.0 m 범위"], ["LuBrain", "정책", "지도 + 몸 감각 → 관절"]];
  for (let i = 0; i < 4; i++) {
    const x = 0.6 + i * 3.15;
    CARD(s, x, 1.6, 2.7, 1.35, i === 2 ? C.text2 : C.background2);
    await ICON(s, flow[i][0], x + 0.18, 1.95, 0.62, i === 2 ? C.accent2 : C.accent1, i === 2 ? THEME.colors.dk2 : "FFFFFF");
    T(s, flow[i][1], { x: x + 0.92, y: 1.75, w: 1.7, h: 0.42, fontSize: 14, bold: true, color: i === 2 ? C.background1 : C.text2 });
    T(s, flow[i][2], { x: x + 0.92, y: 2.2, w: 1.72, h: 0.62, fontSize: 12, color: i === 2 ? C.accent6 : C.accent5 });
    if (i < 3) ARROW(s, x + 2.75, 2.27, x + 3.1, 2.27, C.accent1, 2.5);
  }
  // grid illustration: a gap band across x and a few cells the cameras cannot see
  const gx = 0.75, gy = 3.45, cs = 0.165;
  for (let r = 0; r < 11; r++) for (let c = 0; c < 17; c++) {
    const gap = c === 10 || c === 11;
    const hidden = (r < 2 && c > 13) || (r > 8 && c < 3) || (r === 5 && c === 7);
    const col = hidden ? "D5DCDC" : gap ? THEME.colors.dk2 : c > 11 ? "7FB9B4" : "BFDDDA";
    S(s, pres.ShapeType.rect, { x: gx + c * cs, y: gy + r * cs, w: cs - 0.02, h: cs - 0.02, fill: { color: col }, line: { color: col } });
  }
  T(s, [{ text: "■ ", options: { color: THEME.colors.dk2 } }, { text: "갭(깊게 빈 곳)  " }, { text: "■ ", options: { color: "D5DCDC" } }, { text: "카메라에 안 보이는 칸 = 0" }],
    { x: 0.75, y: 5.35, w: 3.6, h: 0.3, fontSize: 11, color: C.accent5 });
  T(s, "몸 기준으로 정렬된 지도. 0.1 s마다 새로 만들고 8장을 기억한다", { x: 0.75, y: 5.7, w: 3.3, h: 0.7, fontSize: 12, color: C.accent5 });
  const why = [["LuCircleCheck", "학생 증류 손실이 없다", "학습한 정책이 그대로 로봇에 올라간다 (예전엔 '학생'을 따로 가르치다 성능을 잃었다)"],
    ["LuPlug", "센서가 바뀌면 처리기만 바꾼다", "정책은 지도 형식만 알면 된다"],
    ["LuLayers", "학습 때도 같은 지도 형식", "안 보이는 칸 0, 잡음·끊김까지 넣어 학습 → 시뮬과 실기의 입력이 같은 모양"]];
  for (let i = 0; i < 3; i++) {
    const y = 3.35 + i * 1.08;
    await ICON(s, why[i][0], 4.75, y, 0.6, C.accent1);
    T(s, why[i][1], { x: 5.55, y: y - 0.02, w: 7.2, h: 0.36, fontSize: 16, bold: true, color: C.text2 });
    T(s, why[i][2], { x: 5.55, y: y + 0.36, w: 7.2, h: 0.62, fontSize: 14 });
  }

  // 5 timeline
  s = slide("CONTENT", SEC[0], "1 · 문제와 개념", "방법을 네 번 바꿔 지금 구조에 도착했다",
    "처음에는 카메라 영상을 보는 학생 정책을 따로 가르쳤는데, 학생이 교사만큼 배우지 못했습니다. 10월 14일부터는 배포 처리기가 만들 수 있는 지도로 교사를 직접 학습시켜 학생 단계를 없앴고, 이것이 현재 방식입니다.");
  const tl = [["09-23 ~ 10-01", "CVTT", "Camera-Visible Terrain Teacher", "렌더 카메라 교사 + 카메라 학생", "학생이 갭·계단을 못 지나감, 렌더 비용 큼"],
    ["09-30 ~ 10-04", "BIVT-Ray", "Blind-Initialized Visual Teacher, Raycast visibility", "레이캐스트 높이맵 교사 + 카메라 학생", "학생 증류가 병목"],
    ["10-01 ~ 10-13", "GAST v1 → v2.1", "Gap-Aware Spatiotemporal Teacher", "8장을 기억하는 지형 인코더", "학생 성능 상한 72.6 %"],
    ["10-14 ~ 현행", "Ray+IR", "Raycast visibility + IR 밝기 채널", "배포 매퍼가 만들 수 있는 지도로 교사가 직접 학습", "학생 없음 · 카메라+필터 갭 96.2 %"]];
  S(s, pres.ShapeType.line, { x: 1.2, y: 2.55, w: 10.9, h: 0, line: { color: C.accent6, width: 3 } });
  for (let i = 0; i < 4; i++) {
    const x = 0.6 + i * 3.1, cur = i === 3;
    T(s, tl[i][0], { x, y: 1.6, w: 2.9, h: 0.35, fontSize: 13, color: C.accent5 });
    S(s, pres.ShapeType.ellipse, { x: x + 0.38, y: 2.33, w: 0.44, h: 0.44, fill: { color: cur ? C.accent2 : C.accent1 }, line: { color: C.background1, width: 2 } });
    CARD(s, x, 2.95, 2.9, 2.85, cur ? C.text2 : C.background2);
    T(s, tl[i][1], { x: x + 0.22, y: 3.07, w: 2.5, h: 0.42, fontSize: 18, bold: true, color: cur ? C.accent2 : C.text2 });
    T(s, tl[i][2], { x: x + 0.22, y: 3.49, w: 2.5, h: 0.48, fontSize: 11, color: cur ? C.accent6 : C.accent5 });
    T(s, tl[i][3], { x: x + 0.22, y: 4.0, w: 2.5, h: 0.75, fontSize: 14, color: cur ? C.background1 : C.text1 });
    T(s, [{ text: cur ? "결과 " : "버린 이유 ", options: { bold: true } }, { text: tl[i][4] }],
      { x: x + 0.22, y: 4.85, w: 2.5, h: 0.85, fontSize: 13, color: cur ? C.accent6 : C.accent4 });
  }
  CARD(s, 0.6, 5.98, 12.15, 0.72);
  T(s, [{ text: "핵심 결정 '교사 직결'  ", options: { bold: true, color: C.accent1 } },
        { text: "센서 → 처리기 → 11×17 지도를 고정하고, 정책(교사)이 그 지도 형식으로 바로 학습한다" }],
    { x: 0.9, y: 6.0, w: 11.6, h: 0.68, fontSize: 16, valign: "middle" });

  // 6 architecture figure
  pres.addSection({ title: SEC[1] });
  s = slide("CONTENT", SEC[1], "2 · 구조", "전체 구조 한 장: 학습과 배포에 같은 Actor",
    "위가 학습, 아래가 배포입니다. 가운데 주황색 Actor가 양쪽에 똑같이 있습니다. 빨간 점선의 Critic과 PPO는 학습에만 있고 로봇에는 실리지 않습니다. 학습 때 지형 입력은 정답 지형에 카메라 가시성과 잡음을 입혀 만들고, 배포 때는 실제 카메라 깊이가 처리기를 거쳐 같은 형식이 됩니다.");
  IMG(s, { path: FIG + "vrl_architecture.png", x: 0.6, y: 1.5, w: 8.35, h: 8.35 * 3117 / 4889, altText: "학습과 배포 전체 구조도" });
  const arch = [["LuBrain", "같은 Actor가 위(학습)와 아래(배포)에 있다"], ["LuShield", "빨간 점선(Critic·PPO)은 학습에만 있고 로봇에는 없다"],
    ["LuLayers", "학습용 지형 입력과 배포 처리기 출력이 같은 모양 → 시뮬에서 실기로 넘어가는 핵심"]];
  for (let i = 0; i < 3; i++) {
    const y = 1.6 + i * 1.6;
    await ICON(s, arch[i][0], 9.3, y, 0.55);
    T(s, arch[i][1], { x: 10.0, y: y - 0.05, w: 2.75, h: 1.4, fontSize: 14 });
  }

  // 7 four networks
  s = slide("CONTENT", SEC[1], "2 · 구조", "신경망 네 개가 역할을 나눠 맡는다",
    "Actor만 로봇에서 돕니다. Critic은 시뮬레이터만 아는 정답을 보고 지금 상태가 얼마나 좋은지 점수를 예측해서 학습을 안정시킵니다. CENet은 몸의 속도와 미끄러움 같은 맥락을 추정하고, GAST 지형 인코더는 지도 여러 장을 기억해서 몸에 가려진 갭 가장자리도 놓치지 않게 합니다.");
  const nets = [["LuBrain", "Actor", true, "모든 입력을 받아 관절 12개의 목표를 낸다. 로봇에서 실제로 도는 신경망", "MLP 512-256-128 · 입력 253"],
    ["LuScale", "Critic", false, "시뮬레이터만 아는 정답(정확한 지형·마찰·접촉력)으로 지금 상태의 점수를 예측해 학습을 안정시킨다", "특권 관측 298 + 통로 9 → 점수 V(s)"],
    ["LuActivity", "CENet", true, "최근 관절·자세 기록으로 몸 속도와 미끄러움 같은 맥락을 추정한다", "46 × 5 프레임 → 속도 3 + 맥락 16"],
    ["LuMap", "GAST 지형 인코더", true, "지형 지도 8장을 기억해, 몸에 가려 지금 안 보이는 갭 가장자리도 이전 장면에서 가져온다", "CNN → 어텐션(MHA) → GRU → 32"]];
  for (let i = 0; i < 4; i++) {
    const x = 0.6 + (i % 2) * 6.2, y = 1.55 + Math.floor(i / 2) * 2.7;
    CARD(s, x, y, 5.95, 2.45);
    await ICON(s, nets[i][0], x + 0.3, y + 0.3, 0.75, nets[i][2] ? C.accent1 : C.accent5);
    T(s, nets[i][1], { x: x + 1.25, y: y + 0.3, w: 2.9, h: 0.45, fontSize: 20, bold: true, color: C.text2 });
    S(s, pres.ShapeType.roundRect, { x: x + 4.15, y: y + 0.33, w: 1.5, h: 0.38, rectRadius: 0.19, fill: { color: nets[i][2] ? C.accent2 : C.accent6 }, line: { color: nets[i][2] ? C.accent2 : C.accent6 } });
    T(s, nets[i][2] ? "로봇에 실림" : "학습 전용", { x: x + 4.15, y: y + 0.33, w: 1.5, h: 0.38, fontSize: 12, bold: true, color: C.text2, align: "center", valign: "middle" });
    T(s, nets[i][3], { x: x + 1.25, y: y + 0.85, w: 4.45, h: 1.0, fontSize: 14 });
    T(s, nets[i][4], { x: x + 1.25, y: y + 1.9, w: 4.45, h: 0.35, fontSize: 12, color: C.accent5 });
  }

  // 7b Actor-Critic / PPO / Storage / Runner
  s = slide("CONTENT", SEC[1], "2 · 구조", "Actor-Critic은 '무엇을', PPO는 '어떻게' 학습하나",
    "PPO는 신경망이 아니라 학습 알고리즘입니다. Actor-Critic은 학습되는 신경망이고, 로봇이 환경에서 움직인 기록은 Storage에 쌓입니다. 100스텝이 모이면 PPO가 그 데이터로 Actor와 Critic을 20번 고칩니다. Runner는 이 순환을 정해진 횟수만큼 돌리고 저장, 로그, Top-5를 맡습니다. 학습이 끝나면 PPO, Storage, Runner, Critic은 모두 버리고 Actor만 ONNX로 로봇에 올라갑니다.");
  CARD(s, 0.6, 1.5, 8.35, 5.15, C.background2);
  T(s, [{ text: "Runner ", options: { bold: true, color: C.accent1 } }, { text: "(DreamwaqRunner): 아래 순환을 갱신 횟수만큼 반복 + 저장·로그·Top-5" }],
    { x: 0.85, y: 1.6, w: 7.9, h: 0.4, fontSize: 13, color: C.text2 });
  const node = async (x, y, w, h, icon, title, sub, dark) => {
    S(s, pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.1, fill: { color: dark ? C.text2 : C.background1 }, line: { color: dark ? C.text2 : C.accent6, width: 1.5 } });
    await ICON(s, icon, x + 0.18, y + 0.22, 0.55, dark ? C.accent2 : C.accent1, dark ? THEME.colors.dk2 : "FFFFFF");
    T(s, title, { x: x + 0.85, y: y + 0.15, w: w - 0.95, h: 0.42, fontSize: 15, bold: true, color: dark ? C.accent2 : C.text2 });
    T(s, sub, { x: x + 0.85, y: y + 0.58, w: w - 0.95, h: 0.75, fontSize: 11, color: dark ? C.background1 : C.text1 });
  };
  await node(0.9, 2.2, 2.75, 1.45, "LuCpu", "환경 (Isaac Lab)", "로봇 수천–수만 마리\n100 Hz 시뮬레이션");
  await node(5.55, 2.2, 3.15, 1.45, "LuBrain", "Actor-Critic (신경망)", "Actor: 관측 → 행동 12\nCritic: 관측 → 점수 V(s)", true);
  await node(0.9, 4.85, 2.75, 1.45, "LuDatabase", "Storage (저장소)", "env × 100 스텝의 transition\n(관측·행동·보상·V·log π)");
  await node(5.55, 4.85, 3.15, 1.45, "LuTrendingUp", "PPO (학습 알고리즘)", "어드밴티지 계산 → 확률비 0.8–1.2로 잘라 20번 갱신");
  ARROW(s, 3.7, 2.6, 5.5, 2.6, C.accent1, 2.2);
  T(s, "관측", { x: 3.9, y: 2.25, w: 1.4, h: 0.3, fontSize: 12, color: C.accent1, align: "center" });
  ARROW(s, 5.5, 3.25, 3.7, 3.25, C.accent1, 2.2);
  T(s, "행동 a", { x: 3.9, y: 3.3, w: 1.4, h: 0.3, fontSize: 12, color: C.accent1, align: "center" });
  ARROW(s, 2.27, 3.7, 2.27, 4.8, C.accent5, 2.2);
  T(s, "매 스텝 기록", { x: 2.4, y: 4.05, w: 1.6, h: 0.3, fontSize: 12, color: C.accent5 });
  ARROW(s, 3.7, 5.55, 5.5, 5.55, C.accent5, 2.2);
  T(s, "100스텝 모이면", { x: 3.75, y: 5.6, w: 1.7, h: 0.3, fontSize: 12, color: C.accent5, align: "center" });
  ARROW(s, 7.12, 4.8, 7.12, 3.7, C.accent2, 2.6);
  T(s, "신경망 수정\n(5 epoch × 4 = 20번)", { x: 7.25, y: 3.85, w: 1.65, h: 0.75, fontSize: 11, color: C.text2, bold: true });
  const parts = [["Actor-Critic", "학습되는 신경망", "Actor만 실림", true], ["PPO", "고치는 규칙 (알고리즘)", "학습 전용", false],
    ["Storage", "한 번 갱신분의 데이터", "학습 전용", false], ["Runner", "반복 일정표 + 저장·로그", "학습 전용", false]];
  for (let i = 0; i < 4; i++) {
    const y = 1.5 + i * 1.08;
    CARD(s, 9.2, y, 3.55, 0.95);
    T(s, parts[i][0], { x: 9.38, y: y + 0.1, w: 2.0, h: 0.38, fontSize: 15, bold: true, color: C.text2 });
    S(s, pres.ShapeType.roundRect, { x: 11.3, y: y + 0.13, w: 1.3, h: 0.32, rectRadius: 0.16, fill: { color: parts[i][3] ? C.accent2 : C.accent6 }, line: { color: parts[i][3] ? C.accent2 : C.accent6 } });
    T(s, parts[i][2], { x: 11.3, y: y + 0.13, w: 1.3, h: 0.32, fontSize: 10, bold: true, color: C.text2, align: "center", valign: "middle" });
    T(s, parts[i][1], { x: 9.38, y: y + 0.52, w: 3.25, h: 0.36, fontSize: 12 });
  }
  CARD(s, 9.2, 5.85, 3.55, 0.8, C.text2);
  T(s, "학습이 끝나면 Actor만 ONNX로 로봇에. 같은 Actor-Critic을 다른 알고리즘(SAC 등)으로도 학습할 수 있다",
    { x: 9.35, y: 5.88, w: 3.3, h: 0.74, fontSize: 11, color: C.background1, valign: "middle" });

  // 8 inputs
  s = slide("CONTENT", SEC[1], "2 · 구조", "정책이 받는 입력은 네 종류",
    "몸 감각, 하방 지형 지도, 머리 LiDAR 지도, 그리고 gdm이 주는 여유 값입니다. 모두 실제 로봇에서도 같은 값을 만들 수 있는 것만 넣었습니다. 시뮬레이터만 아는 정답은 Critic에만 줍니다.");
  const ins = [["LuActivity", "몸 감각 (고유감각)", "46 × 최근 4장", "몸 기울기·회전, 관절 각도·속도, 직전 행동, 속도 명령, 적재"],
    ["LuGrid3X3", "하방 지형 지도", "11×17 × 3채널 × 8장", "높이 · 유효 여부 · IR 밝기. 0.1 s마다 새 장면을 현재 자세에 맞춰 정렬"],
    ["LuRadar", "머리 LiDAR 지도", "30×48 × 2장", "발 위 0.08–1.0 m에 무언가 있으면 1 → 통로 벽. 0.3 s 끊기면 'LiDAR 꺼짐'으로 학습된 상태"],
    ["LuSlidersHorizontal", "여유 m (외부 조건)", "숫자 1개", "gdm이 정하는 '벽까지 한쪽 최소 거리' 0.05–0.15 m"]];
  for (let i = 0; i < 4; i++) {
    const y = 1.55 + i * 1.3;
    CARD(s, 0.6, y, 8.6, 1.15);
    await ICON(s, ins[i][0], 0.82, y + 0.22, 0.7);
    T(s, ins[i][1], { x: 1.75, y: y + 0.14, w: 3.6, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
    T(s, ins[i][2], { x: 5.4, y: y + 0.16, w: 3.6, h: 0.36, fontSize: 13, bold: true, color: C.accent1, align: "right" });
    T(s, ins[i][3], { x: 1.75, y: y + 0.55, w: 7.25, h: 0.55, fontSize: 13 });
  }
  CARD(s, 9.55, 1.55, 3.2, 5.05, C.text2);
  T(s, "253", { x: 9.8, y: 1.8, w: 2.7, h: 1.0, fontSize: 60, bold: true, color: C.accent2 });
  T(s, "Actor가 한 번에 받는 숫자 개수", { x: 9.8, y: 2.85, w: 2.75, h: 0.6, fontSize: 14, color: C.background1 });
  const brk = [["184", "몸 감각 46 × 4"], ["19", "CENet 추정"], ["32", "지형 latent"], ["16", "LiDAR latent"], ["2", "여유 + 회전 차단"]];
  for (let i = 0; i < 5; i++) {
    T(s, [{ text: brk[i][0] + "  ", options: { bold: true, color: C.accent6 } }, { text: brk[i][1] }],
      { x: 9.8, y: 3.6 + i * 0.52, w: 2.75, h: 0.4, fontSize: 14, color: C.background1 });
  }

  // 8b tensor view of inputs and outputs
  s = slide("CONTENT", SEC[1], "2 · 구조", "입력과 출력을 텐서 모양 그대로 보면",
    "앞 장의 입력은 결국 한 줄짜리 벡터 253으로 합쳐지지만, 합쳐지기 전에는 모양이 서로 다릅니다. 지형은 8장 × 3채널 × 11 × 17의 4차원 텐서이고, 프레임마다 같은 CNN을 거쳐 [8, 64]가 된 뒤 어텐션과 GRU로 32 하나로 줄어듭니다. LiDAR는 2 × 30 × 48 지도가 stride 2 합성곱 세 번으로 32 × 4 × 6까지 줄어 16이 됩니다. 몸 감각은 5 스텝 × 46의 2차원 행렬이고, CENet은 5 스텝 전체를, Actor는 최근 4 스텝을 펼쳐 씁니다. 출력 12는 다리 4개 × 관절 3개의 표로 보면 이해가 쉽고, PD 제어가 이것을 토크 [4, 3]으로 바꿉니다. 블록 크기는 개념도이고 괄호 안의 숫자가 실제 크기입니다.");
  { const fw = 9.65, fh = fw * 2700 / 4800;
    IMG(s, { path: FIG + "vrl_tensors.png", x: (W - fw) / 2, y: 1.45, w: fw, h: fh, altText: "입력과 출력의 텐서 모양 그림" }); }

  // 9 outputs
  s = slide("CONTENT", SEC[1], "2 · 구조", "출력은 관절 12개의 목표 각도",
    "정책은 관절 12개의 목표 각도를 1초에 100번 냅니다. PD 게인은 Kp 고관절·허벅지 123.39, 무릎 127.77, Kd 2.4로, 모터 드라이버가 실제로 받을 수 있는 값으로 맞췄고 학습과 배포에서 같습니다. 모터 제어기는 그 목표를 1초에 500번 따라갑니다. 배포 때는 학습 코드와 ONNX 출력이 거의 같은지 검사하고, 지형 정보가 1초 넘게 끊기면 로봇을 멈춥니다.");
  CARD(s, 0.6, 1.6, 6.6, 2.75);
  T(s, "행동 → 토크", { x: 0.9, y: 1.78, w: 6, h: 0.4, fontSize: 16, bold: true, color: C.accent1 });
  T(s, [{ text: "목표 각도 = 기본자세 + 0.25 × 행동", options: { breakLine: true } },
        { text: "토크 = Kp × (목표 − 현재) − Kd × 관절 속도" }],
    { x: 0.9, y: 2.25, w: 6.1, h: 0.95, fontSize: 18, bold: true, color: C.text2, paraSpaceAfter: 6 });
  T(s, [{ text: "PD 게인  ", options: { bold: true, color: C.accent1 } }, { text: "Kp 123.39 (고관절·허벅지) / 127.77 (무릎) · Kd 2.4", options: { bold: true } }],
    { x: 0.9, y: 3.25, w: 6.1, h: 0.4, fontSize: 15, color: C.text2 });
  T(s, "모터 드라이버가 받는 게인 단위의 정수배로 맞춘 값 (학습·배포 동일) · 토크 한계 93.6 / 126 N·m", { x: 0.9, y: 3.7, w: 6.1, h: 0.5, fontSize: 12, color: C.accent5 });
  const hz = [["100 Hz", "정책이 판단"], ["500 Hz", "모터가 목표를 따라감"]];
  for (let i = 0; i < 2; i++) {
    T(s, hz[i][0], { x: 0.6 + i * 3.3, y: 4.75, w: 3.1, h: 0.9, fontSize: 44, bold: true, color: C.accent1 });
    T(s, hz[i][1], { x: 0.6 + i * 3.3, y: 5.65, w: 3.1, h: 0.4, fontSize: 14, color: C.accent5 });
  }
  const dep = [["LuPlug", "배포 묶음", "policy_vrl.onnx(Actor) + gast_encoder.onnx(지형) — 로봇 CPU에서 실행"],
    ["LuCircleCheck", "학습 코드와 같은지 검사", "ONNX 출력 차이: Actor 2.7e-5, 지형 1.6e-3 (허용 2e-3)"],
    ["LuShield", "입력이 끊기면 멈춤", "지형 정보가 0.25 s 이내면 정상, 1 s 넘게 끊기면 Damp. 빈 지도로 걷지 않는다"]];
  for (let i = 0; i < 3; i++) {
    const y = 1.65 + i * 1.6;
    await ICON(s, dep[i][0], 7.65, y, 0.62);
    T(s, dep[i][1], { x: 8.45, y: y - 0.02, w: 4.3, h: 0.38, fontSize: 16, bold: true, color: C.text2 });
    T(s, dep[i][2], { x: 8.45, y: y + 0.38, w: 4.3, h: 0.95, fontSize: 14 });
  }

  // 9b deployment pipeline: checkpoint -> bundle -> CAMEL-Pilot
  s = slide("CONTENT", SEC[1], "2 · 구조", "정책이 로봇까지 가는 길: 체크포인트 → 묶음 → CAMEL-Pilot",
    "학습이 끝난 체크포인트는 export 스크립트로 ONNX 두 개(Actor와 지형 인코더)와 기록 파일로 바뀝니다. export는 학습 코드와 ONNX 출력이 같은지(parity) 검사합니다. 이 묶음을 배포 저장소에 넣으면, 우리 C++ 프로그램 CAMEL-Pilot이 ONNX 안에 실린 정책 계약을 읽고 그대로 관측을 만들어 추론합니다. MuJoCo 시험은 run_sim.sh가 같은 바이너리를 띄우고, 실기는 deploy.sh가 빌드한 바이너리를 컨트롤러 PC로 보냅니다. 로봇 컴퓨터에 직접 굽는 것이 아니라, 별도 PC에서 네트워크로 로봇을 모는 구조입니다.");
  const dp = [["LuFileCode", "① 체크포인트", "gd_lab_vrl\n학습 .pt (Actor + Critic + CENet + 인코더)"],
    ["LuPackage", "② export", "policy_vrl.onnx (Actor·CENet·LiDAR)\ngast_encoder.onnx (지형 기억)\n학습 코드와 parity ≤ 2e-3"],
    ["LuFolderOpen", "③ 배포 묶음", "resources/policy/gast/<이름>/\n+ deployment_manifest.json (SHA)"],
    ["LuCpu", "④ CAMEL-Pilot", "ONNX의 정책 계약(camel.policy.v1)을 읽고 관측 구성 → 100 Hz 추론"]];
  for (let i = 0; i < 4; i++) {
    const x = 0.6 + i * 3.15, last = i === 3;
    CARD(s, x, 1.5, 2.8, 2.55, last ? C.text2 : C.background2);
    await ICON(s, dp[i][0], x + 0.22, 1.7, 0.6, last ? C.accent2 : C.accent1, last ? THEME.colors.dk2 : "FFFFFF");
    T(s, dp[i][1], { x: x + 0.95, y: 1.75, w: 1.8, h: 0.45, fontSize: 16, bold: true, color: last ? C.accent2 : C.text2 });
    T(s, dp[i][2], { x: x + 0.22, y: 2.45, w: 2.45, h: 1.5, fontSize: 12, color: last ? C.background1 : C.text1 });
    if (i < 3) ARROW(s, x + 2.85, 2.75, x + 3.1, 2.75, C.accent1, 2.5);
  }
  const two = [["LuMonitor", "MuJoCo 시험 (지금)", ["deploy/mujoco/<단계>/<모델>/run_sim.sh", "호스트: CAMEL-Pilot + 센서 sidecar", "컨테이너: 벤더 RBQ 스택 (Motion --sim, Mujoco)", "정답 pose 사용, 같은 Pilot 바이너리"]],
    ["LuBot", "실제 로봇 (포팅 예정)", ["scripts/common/deploy.sh: 빌드 → SSH로 컨트롤러 PC에", "컨트롤러 PC: CAMEL-Pilot (소스 없이 바이너리만)", "로봇 컴퓨터: 벤더 Motion이 실제 모터 구동", "할 일: /odom 구독, 실제 D4xx → 매퍼 연결"]]];
  for (let i = 0; i < 2; i++) {
    const x = 0.6 + i * 6.2;
    CARD(s, x, 4.35, 5.95, 2.3);
    await ICON(s, two[i][0], x + 0.25, 4.55, 0.55, i ? C.accent3 : C.accent1);
    T(s, two[i][1], { x: x + 0.95, y: 4.6, w: 4.8, h: 0.42, fontSize: 16, bold: true, color: C.text2 });
    T(s, two[i][2].map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < 3 } })), { x: x + 0.3, y: 5.15, w: 5.5, h: 1.4, fontSize: 12, paraSpaceAfter: 2 });
  }

  // 9c communication: CAMEL-Pilot <-> RBQ SDK (DDS) <-> vendor stack
  s = slide("CONTENT", SEC[1], "2 · 구조", "CAMEL-Pilot과 RBQ SDK의 통신 구조 (MuJoCo 배포)",
    "MuJoCo 배포에서 호스트에는 우리 프로그램 CAMEL-Pilot이, 컨테이너에는 레인보우로보틱스의 벤더 스택(Motion, Mujoco)이 돕니다. 둘은 RBQ SDK의 DDS(CycloneDDS)로 연결됩니다. Pilot은 관절, IMU, 로봇 상태를 구독하고, 서기·앉기·걷기 같은 상위 명령을 보냅니다. 걷기 구간에만 관절 소유권 _20을 잡고 500 Hz로 관절 목표와 게인을 보냅니다. 이 동안 벤더 QuadWalk의 낙상 감지가 꺼지기 때문에, Pilot의 자체 워치독이 이상 시 Damp로 떨어뜨립니다. 지형 지도와 LiDAR 지도는 sidecar가 UDP로 따로 보내고, 콘솔과 조이스틱은 TCP로 명령을 보냅니다. 실제 로봇에서는 같은 DDS가 네트워크를 넘어가고, 로봇의 Motion이 실제 모터를 구동합니다.");
  // host frame
  S(s, pres.ShapeType.roundRect, { x: 0.6, y: 1.5, w: 6.3, h: 5.15, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.accent6, width: 1.5, dashType: "dash" } });
  T(s, "호스트 PC (실기: 컨트롤러 PC)", { x: 0.8, y: 1.57, w: 5.8, h: 0.35, fontSize: 12, bold: true, color: C.accent3 });
  S(s, pres.ShapeType.roundRect, { x: 9.0, y: 1.5, w: 3.75, h: 5.15, rectRadius: 0.08, fill: { color: "F6F1E4" }, line: { color: C.accent2, width: 1.5, dashType: "dash" } });
  T(s, "벤더 RBQ 스택 (= 로봇)", { x: 9.15, y: 1.57, w: 3.5, h: 0.35, fontSize: 12, bold: true, color: "9A6A00" });
  // Pilot box
  S(s, pres.ShapeType.roundRect, { x: 2.75, y: 2.05, w: 4.0, h: 3.0, rectRadius: 0.08, fill: { color: C.text2 }, line: { color: C.text2 } });
  T(s, "CAMEL-Pilot", { x: 2.95, y: 2.12, w: 3.6, h: 0.42, fontSize: 17, bold: true, color: C.accent2 });
  T(s, [{ text: "RlWalker  500 Hz 루프: 매 틱 관절 ref 발행", options: { breakLine: true } },
        { text: "PolicyBackend  5틱마다 추론 = 100 Hz", options: { breakLine: true } },
        { text: "  policy_vrl.onnx · gast_encoder.onnx", options: { breakLine: true, color: C.accent6 } },
        { text: "지형 기억 8장 (VisionStudentThread)", options: { breakLine: true } },
        { text: "RbqLink  DDS 구독·상위 명령", options: { breakLine: true } },
        { text: "워치독: 피드백 두절·NaN·60° tilt·is_fall → Damp", options: { color: "F2B8B3" } }],
    { x: 2.95, y: 2.6, w: 3.7, h: 2.35, fontSize: 11, color: C.background1, paraSpaceAfter: 3 });
  // sidecars + console
  const side = [["센서 sidecar", "D4xx 4대 depth+IR → camfilter → 매퍼", "UDP GSC2 · 10 Hz", 2.05],
    ["head_lidar", "Ouster + Mid-360 → 점유 지도", "UDP NLM1 · 10 Hz", 3.35],
    ["CAMEL-Console", "운용 화면 · 조이스틱", "TCP 명령 (STAND/SIT/WALK)", 4.65]];
  for (const [t, d, l, y] of side) {
    CARD(s, 0.8, y, 1.75, 1.15, C.background1);
    T(s, t, { x: 0.9, y: y + 0.05, w: 1.6, h: 0.35, fontSize: 12, bold: true, color: C.text2 });
    T(s, d, { x: 0.9, y: y + 0.4, w: 1.6, h: 0.7, fontSize: 9, color: C.text1 });
    ARROW(s, 2.58, y + 0.55, 2.72, y + 0.55, C.accent5, 2);
    T(s, l, { x: 0.8, y: y + 1.13, w: 1.95, h: 0.22, fontSize: 8, color: C.accent3 });
  }
  T(s, "실기 pose: 로봇 쪽 camel_pilot(ROS2)의 /odom 50 Hz (우리 코드 밖) · MuJoCo는 정답 pose", { x: 0.8, y: 6.12, w: 5.95, h: 0.45, fontSize: 9, color: C.accent5 });
  // vendor boxes
  const ven = [["Motion (--sim)", "QuadWalk 상위 보행 · 소유권 중재\n실기: 실제 모터 드라이브", 2.05], ["Mujoco (GastSync)", "물리 시뮬레이션 + 동기 카메라 렌더\n실기: 실제 로봇 몸·센서", 3.75], ["GUI", "벤더 상태 화면", 5.45]];
  for (const [t, d, y] of ven) {
    CARD(s, 9.2, y, 3.35, y === 5.45 ? 0.95 : 1.5, C.background1);
    T(s, t, { x: 9.35, y: y + 0.08, w: 3.1, h: 0.38, fontSize: 13, bold: true, color: "9A6A00" });
    T(s, d, { x: 9.35, y: y + 0.48, w: 3.1, h: 0.9, fontSize: 10, color: C.text1 });
  }
  // DDS bus
  S(s, pres.ShapeType.roundRect, { x: 7.05, y: 1.5, w: 1.8, h: 5.15, rectRadius: 0.08, fill: { color: C.accent1 }, line: { color: C.accent1 } });
  T(s, [{ text: "RBQ SDK", options: { bold: true, breakLine: true } }, { text: "DDS (CycloneDDS)", options: { fontSize: 10 } }],
    { x: 7.1, y: 1.6, w: 1.7, h: 0.6, fontSize: 13, color: C.background1, align: "center" });
  const topics = [["← 구독", "rt/rbq/leg_joint\nrt/rbq/imu\nrt/rbq/robot_status\nbattery · system_log", 2.3], ["→ 항상", "rt/rbq/cmd/high_level\n+ sport RPC (서기·앉기·보행)", 3.75],
    ["→ WALK만", "owner/_20 (소유권)\nref/motion/_20\n관절 목표 + Kp/Kd 500 Hz", 4.75]];
  for (const [h, b, y] of topics) {
    T(s, [{ text: h, options: { bold: true, color: h === "→ WALK만" ? C.accent2 : C.background1, breakLine: true } }, { text: b, options: { fontSize: 8.5 } }],
      { x: 7.12, y, w: 1.66, h: 1.05, fontSize: 10, color: C.background1, align: "center" });
  }
  ARROW(s, 6.75, 3.0, 7.0, 3.0, C.accent1, 2.5);
  ARROW(s, 7.0, 3.55, 6.75, 3.55, C.accent1, 2.5);
  ARROW(s, 8.9, 3.0, 9.15, 3.0, C.accent1, 2.5);
  ARROW(s, 9.15, 4.2, 8.9, 4.2, C.accent1, 2.5);
  T(s, "MuJoCo: 호스트–컨테이너가 lo로 DDS 공유 · 실기: configs/cyclonedds.xml로 네트워크 DDS", { x: 7.05, y: 6.68, w: 5.7, h: 0.25, fontSize: 9, color: C.accent5 });

  // 10 environment
  pres.addSection({ title: SEC[2] });
  s = slide("CONTENT", SEC[2], "3 · 학습 방법", "학습장: 44종 지형과 일부러 흔드는 조건",
    "지형 타일 44열 중 절반이 좁은 통로이고, 승강장 갭이 10열입니다. 마찰, 질량, 모터 게인, 적재, 밀기, 계단 손잡이 힘까지 매번 무작위로 바꿔서 시뮬레이션과 실제의 차이를 견디게 합니다. 잘하면 더 어려운 레벨을, 실패하면 쉬운 레벨을 줍니다.");
  CHART(s, pres.ChartType.doughnut, [{ name: "지형 열", labels: ["좁은 통로", "승강장 갭", "보존 지형", "계단"], values: [20, 10, 10, 4] }],
    { x: 0.5, y: 1.5, w: 5.6, h: 4.9, holeSize: 55, showValue: true, showPercent: false, showLegend: true, legendPos: "b",
      legendFontSize: 13, legendFontFace: "+mn-lt", dataLabelColor: "FFFFFF", dataLabelFontSize: 14, dataLabelFontFace: "+mn-lt",
      chartColors: [THEME.colors.accent1, THEME.colors.dk2, THEME.colors.accent6, THEME.colors.accent2],
      showTitle: true, title: "지형 44열의 구성", titleFontSize: 15, titleColor: THEME.colors.dk2, titleFontFace: "+mn-lt" });
  T(s, "보존 지형 = 계단 변형·거친 지면·경사(최대 26.6°)·울타리", { x: 0.6, y: 6.4, w: 5.5, h: 0.35, fontSize: 11, color: C.accent5 });
  T(s, "매번 무작위로 바꾸는 조건", { x: 6.6, y: 1.55, w: 6, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  const rnd = [["마찰", "0.2–1.25"], ["몸통 질량", "±2 kg"], ["모터 게인", "±15 %"], ["적재", "0 / 6 kg"], ["밀기", "±0.5 m/s"], ["계단 손잡이 힘", "≤ 200 N"]];
  for (let i = 0; i < 6; i++) {
    const x = 6.6 + (i % 3) * 2.07, y = 2.05 + Math.floor(i / 3) * 1.35;
    CARD(s, x, y, 1.9, 1.15);
    T(s, rnd[i][1], { x: x + 0.15, y: y + 0.12, w: 1.65, h: 0.5, fontSize: 20, bold: true, color: C.accent1 });
    T(s, rnd[i][0], { x: x + 0.15, y: y + 0.65, w: 1.65, h: 0.35, fontSize: 13, color: C.text1 });
  }
  await ICON(s, "LuTrendingUp", 6.6, 4.95, 0.6, C.accent2, THEME.colors.dk2);
  T(s, [{ text: "커리큘럼 (curriculum)", options: { bold: true, color: C.text2, breakLine: true } },
        { text: "잘 건너면 더 넓은 갭·높은 계단·좁은 통로로, 실패하면 쉬운 레벨로 돌아간다" }],
    { x: 7.4, y: 4.9, w: 5.35, h: 1.1, fontSize: 14 });


  // 10a terrain grid: levels x columns with robot counts (Hangang it846)
  s = slide("CONTENT", SEC[2], "3 · 학습 방법", "학습장 격자: 15 레벨 × 44 지형 열, 로봇은 어디에 있나",
    "이 장은 좁은 통로까지 학습한 최종 단계, ⑥ B′ 2단계 한강 학습(job 1228779)의 it846 시점 예시입니다. 학습장은 8 m × 8 m 지형 타일을 세로 15줄(레벨, 위로 갈수록 어려움)과 가로 44열(지형 종류)로 깐 격자입니다. 로봇은 처음에 열마다 똑같이 나눠 배치되고, 그 열(지형 종류)은 학습 내내 바뀌지 않습니다. 오르내리는 것은 레벨(줄)뿐입니다. 한강은 16노드 × GPU 4장 = 64 GPU이고, GPU 한 장마다 시뮬레이션 장면이 따로 하나씩 있고, 그 장면의 격자에서 2,048마리가 걷습니다(합계 131,072). GPU 하나에서 열당 2,048 ÷ 44 ≈ 46.5마리이고, 64개 장면을 합치면 열당 약 2,979마리입니다. 칸 숫자는 64개 장면의 합이라, GPU 한 장의 장면으로 보려면 64로 나눕니다. 로봇이 4 m 이상 걸어가면 한 레벨 올라가고, 명령 거리의 절반도 못 가면 내려갑니다. 그래서 학습이 진행되면 대부분 위쪽 레벨에 모입니다. 칸의 숫자와 색은 최종 학습 it846 갱신 때 끝난 에피소드 분포로 추정한 로봇 수입니다.");
  const counts = JSON.parse(fs.readFileSync(FIG + "terrain_grid_counts.json", "utf8"));
  S(s, pres.ShapeType.roundRect, { x: 8.85, y: 0.2, w: 3.9, h: 0.36, rectRadius: 0.18, fill: { color: C.accent2 }, line: { color: C.accent2 } });
  T(s, "예시: ⑥ B′ 2단계 최종 학습 · 한강 it846", { x: 8.85, y: 0.2, w: 3.9, h: 0.36, fontSize: 12, bold: true, color: C.text2, align: "center", valign: "middle" });
  const groups = [
    { name: "계단·보존 지형", fams: [["pyramid_stairs", 2], ["pyramid_stairs_inv", 2], ["random_rough", 1], ["hf_pyramid_slope", 1], ["hf_pyramid_slope_inv", 1], ["pyramid_stairs_wide", 1], ["pyramid_stairs_inv_wide", 1], ["pyramid_stairs_nose", 1], ["pyramid_stairs_inv_nose", 1], ["ring_fence", 1]], diff: "계단 5→25 cm · 경사 → 26.6°" },
    { name: "승강장 갭", fams: [["platform_gap", 10]], diff: "폭 2→26 cm · 단차 0→16 cm" },
    { name: "좁은 통로", fams: [["narrow_corridor", 4], ["narrow_door_pos", 3], ["narrow_door_neg", 3], ["narrow_bend_left", 4], ["narrow_bend_right", 4], ["narrow_dead_end", 4]], diff: "폭 1.20 → 0.50 m" }];
  const PER_COL = 131072 / 44, NL = 15;
  const gx0 = 1.15, gy0 = 1.95, cw = 0.218, rh = 0.248, gapG = 0.12;
  const cells = []; let x = gx0, maxv = 1;
  const rowTot = new Array(NL).fill(0);
  for (const g of groups) {
    g.x0 = x;
    for (const [fam, ncol] of g.fams) {
      const c = counts[fam] || new Array(NL).fill(0); const tot = c.reduce((a, b) => a + b, 0) || 1;
      for (let k = 0; k < ncol; k++) {
        for (let l = 0; l < NL; l++) {
          const v = PER_COL * c[l] / tot; maxv = Math.max(maxv, v); rowTot[l] += v;
          cells.push({ x, y: gy0 + (NL - 1 - l) * rh, v });
        }
        x += cw;
      }
    }
    g.x1 = x; x += gapG;
  }
  const shade = (v) => {  // light teal -> dark teal
    const t = Math.min(1, Math.sqrt(v / maxv)), a = [0xEA, 0xF2, 0xF1], b = [0x0E, 0x3B, 0x43];
    return a.map((ai, i) => Math.round(ai + (b[i] - ai) * t).toString(16).padStart(2, "0")).join("").toUpperCase();
  };
  for (const c of cells) S(s, pres.ShapeType.rect, { x: c.x, y: c.y, w: cw - 0.02, h: rh - 0.02, fill: { color: shade(c.v) }, line: { color: shade(c.v) } });
  for (const c of cells) T(s, String(Math.round(c.v)), { x: c.x - 0.02, y: c.y, w: cw + 0.02, h: rh - 0.02, fontSize: 6.5, align: "center", valign: "middle",
    color: Math.sqrt(c.v / maxv) > 0.55 ? C.background1 : C.text2 });
  for (let l = 0; l < NL; l++) {
    T(s, String(l), { x: gx0 - 0.42, y: gy0 + (NL - 1 - l) * rh, w: 0.32, h: rh, fontSize: 10, color: C.accent5, align: "right", valign: "middle" });
    T(s, Math.round(rowTot[l]).toLocaleString("en-US"), { x: x - gapG + 0.1, y: gy0 + (NL - 1 - l) * rh, w: 0.85, h: rh, fontSize: 10, color: C.text2, valign: "middle" });
  }
  T(s, "레벨", { x: gx0 - 0.55, y: gy0 - 0.33, w: 0.5, h: 0.28, fontSize: 10, bold: true, color: C.accent5, align: "right" });
  T(s, "레벨별 로봇", { x: x - gapG + 0.05, y: gy0 - 0.33, w: 1.1, h: 0.28, fontSize: 10, bold: true, color: C.accent5 });
  T(s, "↑ 어려움", { x: gx0 - 0.95, y: gy0 + 0.05, w: 0.55, h: 0.5, fontSize: 9, color: C.accent4 });
  T(s, "쉬움 ↓", { x: gx0 - 0.95, y: gy0 + (NL - 1) * rh - 0.15, w: 0.55, h: 0.4, fontSize: 9, color: C.accent3 });
  for (const g of groups) {
    const ncol = Math.round((g.x1 - g.x0) / cw);
    T(s, [{ text: g.name, options: { bold: true, breakLine: true } }, { text: `${ncol}열 × ${Math.round(PER_COL).toLocaleString("en-US")} = ${Math.round(ncol * PER_COL).toLocaleString("en-US")}마리`, options: { fontSize: 10, color: C.accent5 } }],
      { x: g.x0, y: gy0 - 0.62, w: g.x1 - g.x0, h: 0.58, fontSize: 12, color: C.text2, align: "center", valign: "bottom" });
    T(s, g.diff, { x: g.x0, y: gy0 + NL * rh + 0.05, w: g.x1 - g.x0, h: 0.3, fontSize: 10, color: C.accent3, align: "center" });
  }
  const lx = gx0, ly = gy0 + NL * rh + 0.38;
  T(s, [{ text: "읽는 법  ", options: { bold: true, color: C.accent1 } },
        { text: "64개 GPU의 학습장을 한 장에 겹쳐 그린 그림 · 칸 숫자 = 그 타일 위의 로봇 수 (64장 합)", options: { bold: true, breakLine: true } },
        { text: "가로(지형 종류)는 처음 나눈 그대로 고정 → 열마다 약 2,979마리 · 세로(레벨)는 잘 걸을수록 위로 → it846에는 약 3분의 2가 레벨 9–14 (그 시점에 끝난 에피소드로 추정)", options: { fontSize: 10, color: C.accent5 } }],
    { x: lx, y: ly, w: 11.6, h: 0.56, fontSize: 11, color: C.text2 });
  const facts = [["로봇", "64 GPU × 2,048 = 131,072"], ["열당", "2,048 ÷ 44 ≈ 46.5 × 64 ≈ 2,979"], ["레벨 ↑", "4 m 이상 걸으면"], ["레벨 ↓", "명령 거리 절반 미만"]];
  for (let i = 0; i < 4; i++) {
    T(s, [{ text: facts[i][0] + "  ", options: { bold: true, color: C.accent1 } }, { text: facts[i][1] }],
      { x: lx + i * 2.95, y: ly + 0.6, w: 2.9, h: 0.3, fontSize: 11, color: C.text1 });
  }

  // 10b rollout data per transition
  s = slide("CONTENT", SEC[2], "3 · 학습 방법", "롤아웃 한 칸(transition) = 로봇 한 마리의 한 스텝",
    "transition 한 칸은 로봇 한 마리가 한 스텝(0.01 s) 움직일 때 얻는 기본 데이터입니다. 로봇마다 이것이 100스텝 모이면 롤아웃 하나가 되고, 지금 설정에서 롤아웃 하나는 100스텝입니다. 모든 로봇의 롤아웃을 모은 것(로봇 수 × 100칸)이 갱신 한 번에 쓰입니다. PPO는 로봇이 한 스텝 움직일 때마다 관측, 행동, 보상 등을 한 칸씩 저장합니다. 최종 B′ 과제에서 한 칸은 float 약 9,000개, float32로 약 36 KB입니다. 이 가운데 절반이 지형 기억 8장이고, 다음이 LiDAR 지도입니다. 크기는 B′ 환경을 실제로 만들어 관측 그룹 크기를 읽은 값입니다.");
  const dh2 = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.accent1 }, fontSize: 13 } });
  const dc2 = (t, o = {}) => ({ text: t, options: { fontSize: 13, color: C.text1, ...o } });
  const num = { align: "right" };
  const defs = [["한 칸 (transition)", "로봇 1마리 × 1스텝 (0.01 s)", "이 장의 표"], ["롤아웃 하나", "로봇 1마리 × 100스텝 = 100칸", "num_steps_per_env = 100"], ["갱신 한 번에 쓰는 양", "로봇 수 × 100칸", "다음 장 (한강 1,311만 칸)"]];
  for (let i = 0; i < 3; i++) {
    const x = 0.6 + i * 4.1;
    CARD(s, x, 1.42, 3.85, 0.84, i === 0 ? C.text2 : C.background2);
    T(s, [{ text: defs[i][0] + "  ", options: { bold: true, color: i === 0 ? C.accent2 : C.accent1, breakLine: true } }, { text: defs[i][1], options: { bold: true } }, { text: "  · " + defs[i][2], options: { fontSize: 10, color: i === 0 ? C.accent6 : C.accent5 } }],
      { x: x + 0.15, y: 1.44, w: 3.6, h: 0.8, fontSize: 13, color: i === 0 ? C.background1 : C.text2, valign: "middle" });
    if (i < 2) ARROW(s, x + 3.88, 1.84, x + 4.07, 1.84, C.accent1, 2);
  }
  TAB(s, [[dh2("데이터 (관측 그룹)"), dh2("float 수"), dh2("어디에 쓰나")],
    [dc2("몸 감각 46 × 최근 5 스텝 (policy)"), dc2("230", num), dc2("Actor(최근 4 스텝)·CENet 입력")],
    [dc2("지형 기억 8장 × 562 (gast_history)"), dc2("4,496", num), dc2("GAST 지형 인코더 입력")],
    [dc2("현재 지형 프레임: 높이 187 + 유효 187 (terrain)"), dc2("374", num), dc2("지형 입력 (현재 장면)")],
    [dc2("지형 정답 (gast_clean)"), dc2("748", num), dc2("지형 복원 보조 손실의 정답")],
    [dc2("LiDAR 점유 지도 2 × 30 × 48"), dc2("2,880", num), dc2("LiDAR CNN 입력")],
    [dc2("여유 1 · 회전 차단 1 · 통로 특권 9"), dc2("11", num), dc2("Actor / Critic")],
    [dc2("Critic 특권 관측 (critic)"), dc2("298", num), dc2("Critic 입력 (학습 전용)")],
    [dc2("행동 · 평균 μ · 표준편차 σ"), dc2("36", num), dc2("PPO 확률비 r = π새 / π옛")],
    [dc2("보상 · 종료 · V(s) · log π · 리턴 · 어드밴티지"), dc2("6", num), dc2("GAE와 손실 계산")],
    [dc2("합계", { bold: true }), dc2("≈ 9,079", { ...num, bold: true, color: C.accent1 }), dc2("float32 ≈ 36 KB / transition", { bold: true, color: C.accent1 })]],
    { x: 0.6, y: 2.42, w: 8.3, colW: [3.9, 1.1, 3.3], rowH: 0.38, border: { type: "solid", pt: 1, color: "D7E3E2" }, fill: { color: "FFFFFF" }, valign: "middle", margin: 0.06 });
  CARD(s, 9.2, 2.42, 3.55, 4.2, C.text2);
  T(s, "36 KB", { x: 9.45, y: 2.6, w: 3.1, h: 0.85, fontSize: 40, bold: true, color: C.accent2 });
  T(s, "한 칸 (float32)", { x: 9.45, y: 3.42, w: 3.1, h: 0.4, fontSize: 14, color: C.background1 });
  T(s, [{ text: "50 %", options: { bold: true, color: C.accent6, breakLine: true } }, { text: "지형 기억 8장", options: { breakLine: true } },
        { text: "32 %", options: { bold: true, color: C.accent6, breakLine: true } }, { text: "LiDAR 지도", options: { breakLine: true } },
        { text: "18 %", options: { bold: true, color: C.accent6, breakLine: true } }, { text: "나머지 (몸 감각, Critic, 행동 등)" }],
    { x: 9.45, y: 3.9, w: 3.1, h: 2.65, fontSize: 14, color: C.background1, paraSpaceAfter: 2 });

  // 10c one PPO update: collect, reuse 5x, 20 gradient steps
  s = slide("CONTENT", SEC[2], "3 · 학습 방법", "갱신 한 번: 모으고, 5번 다시 보며, 20번 고친다",
    "갱신 한 번은 모든 로봇이 100스텝을 모으는 것으로 시작합니다. 모은 데이터를 4묶음(미니배치)으로 나누고 이것을 5번(epoch) 반복해서 기울기 갱신을 20번 합니다. 그러면 transition 하나가 5번 쓰이고, 좌우 반전 데이터를 더하면 10번 쓰입니다. 그다음 데이터는 버리고 새로 모읍니다(on-policy). CENet도 같은 데이터로 따로 20번 갱신합니다. 한강은 A6000보다 25.6배 많이 모으지만 갱신 횟수는 똑같이 20번입니다.");
  const flow2 = [["LuDatabase", "① 수집", "env × 100 스텝\n(정책 고정)"], ["LuSigma", "② GAE", "리턴과 어드밴티지\n계산"], ["LuRepeat", "③ 5 epoch × 4 미니배치", "기울기 갱신 20번\n(DDP는 GPU 평균)"], ["LuTrash2", "④ 버리고 다시 수집", "on-policy:\n다음 갱신은 새 데이터"]];
  for (let i = 0; i < 4; i++) {
    const x = 0.6 + i * 3.15;
    CARD(s, x, 1.5, 2.75, 1.55);
    await ICON(s, flow2[i][0], x + 0.2, 1.72, 0.55);
    T(s, flow2[i][1], { x: x + 0.85, y: 1.62, w: 1.85, h: 0.6, fontSize: 14, bold: true, color: C.text2 });
    T(s, flow2[i][2], { x: x + 0.85, y: 2.22, w: 1.85, h: 0.75, fontSize: 12 });
    if (i < 3) ARROW(s, x + 2.8, 2.27, x + 3.1, 2.27, C.accent1, 2.5);
  }
  const vh = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.accent1 }, fontSize: 13, align: "center" } });
  const vc = (t, o = {}) => ({ text: t, options: { fontSize: 13, color: C.text1, align: "center", ...o } });
  TAB(s, [[vh("한 번의 갱신 (B′ 과제)"), vh("A6000 (5,120 env, GPU 3)"), vh("한강 (131,072 env, GPU 64)")],
    [vc("모으는 transition (env × 100)", { align: "left" }), vc("51만 2천"), vc("1,311만", { bold: true })],
    [vc("저장량 (× 36 KB)", { align: "left" }), vc("18.6 GB (GPU당 6.2)"), vc("476 GB (GPU당 7.4)", { bold: true })],
    [vc("미니배치 (GPU당)", { align: "left" }), vc("4만 2,667"), vc("5만 1,200")],
    [vc("기울기 갱신 횟수", { align: "left" }), vc("20"), vc("20")],
    [vc("표본 사용 (5 epoch / + 좌우 반전)", { align: "left" }), vc("256만 / 512만"), vc("6,554만 / 1억 3,107만", { bold: true })],
    [vc("1,000번 갱신 누적 수집", { align: "left" }), vc("5억 1,200만 (로봇 59일)"), vc("131억 (로봇 4.2년)", { bold: true, color: C.accent1 })]],
    { x: 0.6, y: 3.35, w: 12.15, colW: [4.15, 4.0, 4.0], rowH: 0.42, border: { type: "solid", pt: 1, color: "D7E3E2" }, fill: { color: "FFFFFF" }, valign: "middle", margin: 0.06 });
  T(s, "float32 기준 추정 (실제 GPU 메모리에는 네트워크·중간값이 더 있음). 같은 데이터로 CENet도 20번 따로 갱신 · 좌우 반전(미러) 증강은 미니배치를 2배로", { x: 0.6, y: 6.45, w: 12.1, h: 0.4, fontSize: 11, color: C.accent5 });
  // 11 policy vs gdm
  s = slide("CONTENT", SEC[2], "3 · 학습 방법", "정책은 '어떻게', gdm은 '어디로·언제'",
    "어디로 갈지, 언제 멈출지는 내비게이션 스택 gdm이 정합니다. 정책은 주어진 명령을 안전하게 걷는 방법만 정합니다. 그래서 정책에는 계단이나 문 같은 장소 이름을 주지 않고, 여유 m처럼 gdm이 숫자로 바꾼 조건만 줍니다.");
  const roles = [["LuNavigation", "gdm (내비게이션)", "어디로 · 언제", ["경로와 회전 지점", "정지·출발, 막다른 길 정지", "후진 25 cm 금지 같은 운용 규칙", "장소별 여유 m 결정"]],
    ["LuFootprints", "정책 (이 학습의 대상)", "어떻게 걷나", ["받은 명령으로 갭·계단·통로를 안전하게", "막힌 통로 앞에서는 선다", "넘어지지 않고 조용히 걷는다", "장소 이름은 모른다 — 숫자 조건만 받는다"]]];
  for (let i = 0; i < 2; i++) {
    const x = 0.6 + i * 6.2;
    CARD(s, x, 1.55, 5.95, 3.15);
    await ICON(s, roles[i][0], x + 0.3, 1.8, 0.75, i ? C.accent1 : C.accent3);
    T(s, roles[i][1], { x: x + 1.25, y: 1.78, w: 4.5, h: 0.42, fontSize: 18, bold: true, color: C.text2 });
    T(s, roles[i][2], { x: x + 1.25, y: 2.2, w: 4.5, h: 0.35, fontSize: 14, color: C.accent1, bold: true });
    T(s, roles[i][3].map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < 3 } })),
      { x: x + 0.4, y: 2.8, w: 5.3, h: 1.9, fontSize: 14, paraSpaceAfter: 4 });
  }
  CARD(s, 0.6, 4.95, 12.15, 1.6, C.text2);
  T(s, [{ text: "예: 여유 m (gdm이 장소마다 정하는 숫자)", options: { bold: true, color: C.accent2, breakLine: true } },
        { text: "통로 폭 < 로봇 폭 0.48 m + 2m 이면 멈추고, 아니면 양쪽 벽에서 m 이상 떨어져 지나간다", options: { breakLine: true } },
        { text: "같은 폭이라도 m에 따라 행동이 갈리도록, 경계 폭에서 '막힘/통과'를 50 : 50으로 뽑아 학습했다", options: { color: C.accent6 } }],
    { x: 0.95, y: 5.05, w: 11.6, h: 1.4, fontSize: 15, color: C.background1, valign: "middle" });

  // 12 lineage
  s = slide("CONTENT", SEC[2], "3 · 학습 방법", "기술을 하나씩 이어 붙인 일곱 단계 = 로봇 한 마리의 약 18년",
    "각 단계는 바로 앞 단계 모델에서 이어 학습했고, 한 단계에 기술 하나씩만 더했습니다. 갱신 한 번은 로봇마다 100스텝, 즉 1초의 경험이라서, 동시 로봇 수 × 갱신 횟수가 곧 로봇 한 마리 기준의 경험 시간(초)입니다. 일곱 단계를 모두 더하면 약 5억 6천만 초, 로봇 한 마리가 쉬지 않고 걸으면 약 17.7년 걸리는 분량입니다. 그중 한강에서 돌린 두 단계가 12.5년을 차지합니다.");
  const lin = [["0", "DWB-38000", "블라인드 기본 보행 (관절 감각만)", "A6000 · 4,096 × 38,000회", 4096 * 38000],
    ["1", "3906", "지형을 보는 교사, 갭 비중 50 %", "5090 · 1,024 × 5,000회", 1024 * 5000],
    ["2", "A′ a1993", "전·후진 갭 + 조용한 걸음", "한강 · 131,072 × 2,000회", 131072 * 2000],
    ["3", "A18L", "후진 갭에서 긴 대각 트롯", "로컬 5090 · 1,024 × 300회", 1024 * 300],
    ["4", "bps_cont 299", "좁은 통로 + 서기 seed 1–2.5 %", "A6000 · 5,120 × 600회", 5120 * 600],
    ["5", "v22 299", "1단계: 막히면 서기부터", "A6000 · 5,120 × 300회", 5120 * 300],
    ["6", "it861 (최종)", "2단계: 여유 경계 다듬기", "한강 · 131,072 × 1,000회", 131072 * 1000]];
  const YEAR = 365.25 * 86400;
  const span = (sec) => (sec >= YEAR ? (sec / YEAR).toFixed(1) + "년" : Math.round(sec / 86400) + "일");
  const hdr = [["모델", 1.6, 2.3, "left"], ["배운 것", 3.95, 4.35, "left"], ["서버 · 로봇 수 × 갱신", 8.35, 2.55, "right"], ["한 로봇 환산", 11.05, 1.6, "right"]];
  for (const [t, x, w, al] of hdr) T(s, t, { x, y: 1.45, w, h: 0.3, fontSize: 11, bold: true, color: C.accent5, align: al });
  S(s, pres.ShapeType.line, { x: 0.95, y: 2.1, w: 0, h: 3.6, line: { color: C.accent6, width: 3 } });
  let total = 0;
  for (let i = 0; i < 7; i++) {
    const y = 1.85 + i * 0.6, last = i === 6;
    total += lin[i][4];
    if (last) CARD(s, 1.45, y - 0.03, 11.3, 0.56, C.background2);
    S(s, pres.ShapeType.ellipse, { x: 0.7, y, w: 0.5, h: 0.5, fill: { color: last ? C.accent2 : C.accent1 }, line: { color: C.background1, width: 2 } });
    T(s, lin[i][0], { x: 0.7, y, w: 0.5, h: 0.5, fontSize: 15, bold: true, color: last ? C.text2 : C.background1, align: "center", valign: "middle" });
    T(s, lin[i][1], { x: 1.6, y: y + 0.06, w: 2.3, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
    T(s, lin[i][2], { x: 3.95, y: y + 0.07, w: 4.35, h: 0.4, fontSize: 15 });
    T(s, lin[i][3], { x: 8.35, y: y + 0.1, w: 2.55, h: 0.36, fontSize: 12, color: C.accent5, align: "right" });
    T(s, span(lin[i][4]), { x: 11.05, y: y + 0.06, w: 1.6, h: 0.4, fontSize: 16, bold: true, color: C.accent1, align: "right" });
  }
  CARD(s, 1.45, 6.08, 11.3, 0.55, C.text2);
  T(s, [{ text: "합계 ≈ " + (total / YEAR).toFixed(1) + "년  ", options: { bold: true, color: C.accent2 } },
        { text: "로봇 한 마리가 쉬지 않고 걸으면 이만큼 걸리는 경험 (갱신 1회 = 로봇마다 100스텝 × 0.01 s = 1초)" }],
    { x: 1.7, y: 6.1, w: 10.9, h: 0.5, fontSize: 14, color: C.background1, valign: "middle" });
  T(s, "현재 계보 일곱 단계만 합산 (탐색 세대·사전 검증 run 제외) · 번호 n은 n−1 모델에서 이어 학습", { x: 0.6, y: 6.68, w: 10, h: 0.25, fontSize: 10, color: C.accent5 });

  // 13 rewards table
  s = slide("CONTENT", SEC[2], "3 · 학습 방법", "문제가 보이면 보상을 하나씩 더했다",
    "보상은 여러 항목의 점수 합입니다. 영상과 평가에서 나쁜 습관이 보이면, 그 습관을 막는 항목을 하나씩 추가했습니다. 벌점 하나가 다른 나쁜 습관을 만들기도 해서, 과속 벌점에는 정체 벌점을 짝으로 넣는 식의 규칙이 생겼습니다.");
  const hd = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.accent1 }, fontSize: 14 } });
  const cell = (t, o = {}) => ({ text: t, options: { fontSize: 14, color: C.text1, ...o } });
  const rows = [[hd("문제 (영상·평가에서 본 습관)"), hd("추가한 보상"), hd("결과 / 교훈")],
    [cell("갭에 발이 빠진다"), cell("발 빠짐 −2, 갭 침범 −3, 깨끗이 건너면 +"), cell("카메라 지도로 갭 96 % 이상")],
    [cell("평지에서 머리를 든다"), cell("머리 들림 벌점 (지면 기준, 계단에서는 끔)"), cell("경사에서 잘못 벌받지 않게 지형 기준으로")],
    [cell("후진 갭 뒤 3발로 버틴다"), cell("대각 트롯 + 걸음 하한 0.5 m + 착지 여유"), cell("3발 지지 1–4 %로 감소")],
    [cell("통로를 질주해 빠져나간다"), cell("과속 벌점 → 그러자 멈춰 서서 정체 벌점을 짝으로"), cell("벌점은 짝으로 넣는다")],
    [cell("막힌 통로로 돌진한다"), cell("앞에서 서면 +, 차단선을 넘으면 벌점 + 종료"), cell("막힘 서기 100 %, 침범 0", { bold: true, color: C.accent1 })]];
  TAB(s, rows, { x: 0.6, y: 1.6, w: 12.15, colW: [3.55, 5.0, 3.6], rowH: 0.8, border: { type: "solid", pt: 1, color: "D7E3E2" },
    fill: { color: "FFFFFF" }, valign: "middle", margin: 0.08 });

  // 13b reward families
  s = slide("CONTENT", SEC[2], "3 · 학습 방법", "보상은 여섯 종류로 나뉜다",
    "보상은 매 스텝 여러 항목의 값에 가중치를 곱해 더한 점수입니다. 앞으로 나아가게 하는 것은 추종 보상 하나뿐이고, 나머지 대부분은 나쁜 습관을 깎는 벌점입니다. 가중치 숫자가 크다고 중요한 것은 아닙니다. 항목마다 값의 크기가 달라서, 예를 들어 몸통 높이 −15는 오차 제곱(아주 작은 값)에 곱해집니다.");
  const fam = [["LuTarget", "추종 (+)", "명령 속도·회전을 따라가면 점수. 걷게 만드는 유일한 동력", "선속도 +1.0 · 회전 +0.8"],
    ["LuMountain", "자세 (−)", "몸통 출렁임·기울기·처짐·머리 들림", "수직 속도 −2 · 높이 −15 · 머리 들림 −4"],
    ["LuFootprints", "보행 품질 (−)", "발 끌림·쿵 착지·미끄러짐·너무 짧은 걸음", "발 들기 7 cm 미만 −8 · 미끄러짐 −0.3"],
    ["LuCpu", "모터 (−, 아주 작게)", "토크·에너지·가속·행동 급변·관절 한계", "토크 −2.5e-5 · 관절 한계 −2"],
    ["LuShield", "안전 (−)", "허벅지·종아리가 닿음, 넘어져 에피소드 종료", "접촉 −1 · 종료 −2"],
    ["LuMap", "과제별 (+/−, 게이트)", "갭·계단·통로에서만 켜짐: 빠짐·침범·통과·서기", "갭 빠짐 −2 · 통로 통과 +8 · 침범 −4"]];
  for (let i = 0; i < 6; i++) {
    const x = 0.6 + (i % 3) * 4.1, y = 1.55 + Math.floor(i / 3) * 2.45;
    CARD(s, x, y, 3.85, 2.25, i === 0 ? C.text2 : C.background2);
    await ICON(s, fam[i][0], x + 0.25, y + 0.25, 0.6, i === 0 ? C.accent2 : C.accent1, i === 0 ? THEME.colors.dk2 : "FFFFFF");
    T(s, fam[i][1], { x: x + 1.0, y: y + 0.3, w: 2.7, h: 0.45, fontSize: 18, bold: true, color: i === 0 ? C.accent2 : C.text2 });
    T(s, fam[i][2], { x: x + 0.25, y: y + 0.95, w: 3.4, h: 0.7, fontSize: 14, color: i === 0 ? C.background1 : C.text1 });
    T(s, fam[i][3], { x: x + 0.25, y: y + 1.7, w: 3.4, h: 0.4, fontSize: 12, color: i === 0 ? C.accent6 : C.accent5 });
  }
  T(s, "보상 = Σ 항목 값 × 가중치 × dt (0.01 s). 기본 보행 21항목 + 단계마다 추가 (보행 v3 R1–R20, 갭·통로 항목)", { x: 0.6, y: 6.5, w: 12, h: 0.3, fontSize: 11, color: C.accent5 });

  // 13c reward interactions
  s = slide("CONTENT", SEC[2], "3 · 학습 방법", "보상끼리는 서로 밀고 당긴다",
    "보상 항목은 따로 놀지 않습니다. 하나를 세게 하면 다른 습관이 생기거나, 두 항목이 같은 상황에서 정반대를 요구합니다. 그래서 짝으로 넣거나, 상황별로 끄거나, 두 항목으로 허용 범위의 띠를 만들었습니다. 이 여섯 가지가 실제로 겪은 관계입니다.");
  const rel = [["연쇄", "통과 성공 +8 → 과속 → 정체", "성공 보상이 '빨리 끝내기'를 부름 → 과속 벌점 → 그러자 멈춰 섬 → 정체 벌점 + 성공 보상 속도 감쇠"],
    ["충돌", "속도 추종 ↔ 막힘 앞 서기", "명령은 계속 '가라'인데 서야 함 → 막힌 구역에서는 추종 보상을 끈다"],
    ["band", "발 들기 하한 ↔ 상한", "7 cm 미만 벌점(끌림)과 9 cm 초과 벌점(과하게 듦)이 함께 적당한 높이의 band를 만든다"],
    ["게이트", "머리 들림 ↔ 계단·경사", "몸통 기준이면 경사에서 잘못 벌받음 → 지면 기준으로 정의하고 계단에서는 끈다"],
    ["충돌", "보행 품질 ↔ 후진 갭 버티기", "착지·걸음 시간 항목이 후진 갭의 안전한 버티기를 막음 → 후진 갭에서는 끈다(R19)"],
    ["보존", "새 보상 ↔ 이미 되던 기술", "새 항목을 처음부터 세게 켜면 갭·계단 동작을 잃음 → ramp 0.2→1 + 고정 앵커"]];
  const relCol = { "연쇄": C.accent2, "충돌": C.accent4, "band": C.accent1, "게이트": C.accent3, "보존": C.accent5 };
  for (let i = 0; i < 6; i++) {
    const x = 0.6 + (i % 2) * 6.2, y = 1.55 + Math.floor(i / 2) * 1.7;
    CARD(s, x, y, 5.95, 1.55);
    S(s, pres.ShapeType.roundRect, { x: x + 0.25, y: y + 0.22, w: 0.95, h: 0.38, rectRadius: 0.19, fill: { color: relCol[rel[i][0]] }, line: { color: relCol[rel[i][0]] } });
    T(s, rel[i][0], { x: x + 0.25, y: y + 0.22, w: 0.95, h: 0.38, fontSize: 12, bold: true, color: rel[i][0] === "연쇄" ? C.text2 : C.background1, align: "center", valign: "middle" });
    T(s, rel[i][1], { x: x + 1.35, y: y + 0.2, w: 4.45, h: 0.42, fontSize: 16, bold: true, color: C.text2 });
    T(s, rel[i][2], { x: x + 0.25, y: y + 0.72, w: 5.5, h: 0.75, fontSize: 13 });
  }

  // 14 retention
  s = slide("CONTENT", SEC[2], "3 · 학습 방법", "새 기술을 배워도 옛 기술을 잃지 않게",
    "이어 학습할 때 새 기술을 넣으면 이미 되던 갭이나 계단이 무너지는 일이 반복됐습니다. 그래서 다섯 가지 장치를 씁니다. 아래 세 칸이 핵심 개념입니다. 앵커는 고정해 둔 이전 모델의 행동과 지금 정책의 차이를 손실로 더해 정책이 멀리 떠내려가지 않게 묶는 것이고, 반드시 고정 복사본을 목표로 해야 합니다. 게이트는 조건이 맞을 때만 보상을 켜는 스위치, ramp는 새 보상을 0.2배에서 1배로 천천히 올리는 것입니다.");
  const ret = [["LuMountain", "보존 지형", "새 단계에도 갭 8–10열·계단 열을 남긴다", "갭을 3열로 줄이자 퇴화"],
    ["LuAnchor", "보존 앵커", "고정한 이전 모델 행동 쪽으로 묶어 둔다", "살아 있는 정책을 목표로 하면 발산"],
    ["LuToggleRight", "보상 게이트", "조건이 맞을 때만 보상을 켠다", "계단에선 머리 들림 벌점 끔"],
    ["LuTrendingUp", "ramp", "새 보상을 0.2배 → 1배로 천천히", "처음부터 전부 켜면 안전 동작 상실"],
    ["LuStepForward", "단계 분리", "한 단계에 기술 하나, 쉬운 조건 먼저", "경계 다듬기는 그다음"]];
  for (let i = 0; i < 5; i++) {
    const x = 0.6 + i * 2.47;
    CARD(s, x, 1.5, 2.28, 2.45);
    await ICON(s, ret[i][0], x + 0.2, 1.68, 0.55);
    T(s, ret[i][1], { x: x + 0.85, y: 1.75, w: 1.38, h: 0.42, fontSize: 15, bold: true, color: C.text2 });
    T(s, ret[i][2], { x: x + 0.2, y: 2.38, w: 1.95, h: 0.85, fontSize: 13 });
    T(s, [{ text: "주의 ", options: { bold: true } }, { text: ret[i][3] }], { x: x + 0.2, y: 3.25, w: 1.95, h: 0.6, fontSize: 11, color: C.accent4, valign: "bottom" });
  }
  const cpt = [["앵커 (anchor)", "PPO 갱신 뒤, 지정한 표본에서 지금 정책의 행동을 고정해 둔 이전 모델 사본의 행동 쪽으로 당기는 보조 손실. 배의 닻처럼 이미 되던 동작에서 멀리 떠내려가지 않게 한다",
      "손실 = w · ‖μ지금 − μ고정‖²   (w ≈ 1.0)"],
    ["게이트 (gate)", "조건이 맞을 때만 보상 항목을 켜는 0/1 스위치. 조건은 지형·명령 방향·센서 값으로 판정한다",
      "보상 = g × 항목 값,  g ∈ {0, 1}\n예: 계단 위 → 머리 들림 g = 0"],
    ["ramp", "새 보상을 처음부터 전부 켜지 않고, 0.2배에서 시작해 20,000 step 동안 1배까지 선형으로 올린다",
      "κ = 0.2 + 0.8 · min(1, step / 20,000)"]];
  for (let i = 0; i < 3; i++) {
    const x = 0.6 + i * 4.1;
    CARD(s, x, 4.15, 3.85, 2.5, i === 0 ? C.text2 : C.background2);
    T(s, cpt[i][0], { x: x + 0.25, y: 4.3, w: 3.4, h: 0.42, fontSize: 17, bold: true, color: i === 0 ? C.accent2 : C.text2 });
    T(s, cpt[i][1], { x: x + 0.25, y: 4.75, w: 3.4, h: 1.15, fontSize: 12, color: i === 0 ? C.background1 : C.text1 });
    T(s, cpt[i][2], { x: x + 0.25, y: 5.95, w: 3.4, h: 0.6, fontSize: 12, bold: true, color: i === 0 ? C.accent6 : C.accent1 });
  }

  // 15 two stages
  s = slide("CONTENT", SEC[2], "3 · 학습 방법", "'서기'는 보상만으로는 생기지 않아 두 단계로 나눴다",
    "막힌 통로 앞에서 서는 행동이 아직 없을 때 강한 정지 벌점을 켜면, 서 있는 것의 기대 점수가 너무 낮아 정책이 '언제나 간다'로 굳습니다. 통과 가능한 통로에서 20초 서 있으면 −38점인데, 서기가 이득이 되려면 막힐 확률이 86 % 이상이어야 합니다. 그래서 약한 벌점으로 서기를 먼저 만들고, 그다음 경계를 다듬었습니다.");
  const st = [["−38", "통과 가능한 통로에서 20 s 서 있을 때 점수 (강한 정지 벌점)"], ["86 %", "그때 서기가 이득이 되려면 필요한 막힘 확률"]];
  for (let i = 0; i < 2; i++) {
    T(s, st[i][0], { x: 0.6, y: 1.55 + i * 1.75, w: 3.6, h: 1.0, fontSize: 54, bold: true, color: i ? C.accent4 : C.accent1 });
    T(s, st[i][1], { x: 0.6, y: 2.55 + i * 1.75, w: 3.7, h: 0.65, fontSize: 14, color: C.accent5 });
  }
  const stage = [["1단계 · 서기 먼저", "약한 정지 벌점 (1.5 s, −1)", "막힘 서기 92.2 %"], ["2단계 · 경계 다듬기", "경계 폭 50 : 50 추출 + 강한 벌점 (1.0 s, −2)", "막힘 서기 100 %, 침범 0"]];
  for (let i = 0; i < 2; i++) {
    const x = 4.75 + i * 4.15;
    CARD(s, x, 1.6, 3.85, 2.9, i ? C.text2 : C.background2);
    T(s, stage[i][0], { x: x + 0.25, y: 1.8, w: 3.4, h: 0.45, fontSize: 18, bold: true, color: i ? C.accent2 : C.text2 });
    T(s, stage[i][1], { x: x + 0.25, y: 2.35, w: 3.4, h: 0.95, fontSize: 14, color: i ? C.background1 : C.text1 });
    T(s, stage[i][2], { x: x + 0.25, y: 3.45, w: 3.4, h: 0.8, fontSize: 20, bold: true, color: i ? C.accent6 : C.accent1 });
  }
  ARROW(s, 8.62, 3.05, 8.88, 3.05, C.accent1, 2.5);
  CARD(s, 4.75, 4.8, 8.0, 1.75, "FBEDEB");
  await ICON(s, "LuTriangleAlert", 5.0, 5.05, 0.6, C.accent4);
  T(s, [{ text: "출발 모델에 서기 seed가 있어야 한다", options: { bold: true, color: C.accent4, breakLine: true } },
        { text: "서기가 전혀 없는 모델(A18L)에서 1단계를 시작하자 135번 갱신 동안 서기 0 % → 취소. 서기가 1–2.5 %라도 있는 모델(④)에서만 작동했다" }],
    { x: 5.8, y: 4.95, w: 6.75, h: 1.5, fontSize: 14 });

  // 16 selection
  pres.addSection({ title: SEC[3] });
  s = slide("CONTENT", SEC[3], "4 · 결과", "모델은 세 번 걸러서 고른다",
    "학습 중 매 갱신마다 점수를 매겨 상위 5개를 남깁니다. 그 후보들을 Isaac에서 조건별로 정해진 시험을 보게 하고, 마지막으로 다른 물리엔진인 MuJoCo에서 실제 로봇에 올라가는 프로그램 그대로 시험합니다. 최종 861은 학습 끝 모델이 아니라 중간 후보에서 골랐습니다.");
  const sel = [["LuFilter", "① 매 갱신 Top-5", "학습 통계로 점수를 매겨 상위 5개 저장. 관문: 서기 ≥ 0.85, 침범 ≤ 0.05 등"],
    ["LuTarget", "② E2 시험 (Isaac)", "갭·계단·통로·막힘을 조건 칸마다 정해진 방식으로 판정. 기준: 갭 ≥ 90 %"],
    ["LuCpu", "③ MuJoCo 검증", "다른 물리엔진에서 실제 배포 프로그램(Pilot + ONNX)으로 코스 완주"],
    ["LuFlag", "④ 최종 it861", "학습 끝 모델(999)이 아니라 중간 후보에서 선정"]];
  for (let i = 0; i < 4; i++) {
    const x = 0.6 + i * 3.15, last = i === 3;
    CARD(s, x, 1.7, 2.75, 3.6, last ? C.text2 : C.background2);
    await ICON(s, sel[i][0], x + 0.25, 1.95, 0.75, last ? C.accent2 : C.accent1, last ? THEME.colors.dk2 : "FFFFFF");
    T(s, sel[i][1], { x: x + 0.25, y: 2.9, w: 2.35, h: 0.45, fontSize: 17, bold: true, color: last ? C.accent2 : C.text2 });
    T(s, sel[i][2], { x: x + 0.25, y: 3.4, w: 2.35, h: 1.7, fontSize: 14, color: last ? C.background1 : C.text1 });
    if (i < 3) ARROW(s, x + 2.8, 3.5, x + 3.1, 3.5, C.accent1, 2.5);
  }
  S(s, pres.ShapeType.roundRect, { x: 3.95, y: 4.35, w: 2.35, h: 0.82, rectRadius: 0.06, fill: { color: C.background1 }, line: { color: C.accent6 } });
  T(s, [{ text: "이름: ", options: { bold: true, color: C.accent1 } }, { text: "E = Experiment. 실험 번호 E1–E5에서 시작, E1 다음 세대 평가라 E2 (자세히는 다음 장)" }],
    { x: 4.02, y: 4.37, w: 2.22, h: 0.78, fontSize: 10, color: C.text1, valign: "middle" });
  CARD(s, 0.6, 5.6, 12.15, 1.0);
  T(s, [{ text: "왜 MuJoCo까지? ", options: { bold: true, color: C.accent1 } },
        { text: "Isaac에서 좋아진 것이 다른 물리엔진에서 그대로 이어지지 않은 사례가 있었다. 최종 선택은 반드시 여기를 통과해야 한다" }],
    { x: 0.9, y: 5.65, w: 11.6, h: 0.9, fontSize: 15, valign: "middle" });


  // 16b E2 explained + example
  s = slide("CONTENT", SEC[3], "4 · 결과", "E2: 조건을 하나씩 고정해 결정론적으로 시험한다",
    "E2의 E는 Experiment입니다. 10월 9일 원인 분석 계획에서 실험에 E1부터 E5까지 번호를 붙였는데, 그중 E1(Isaac에서 같은 상태로 latent를 바꿔 끼우는 폐루프 실험)이 모델 평가 도구로 굳었습니다. 10월 27일 그다음 세대 평가 도구를 만들면서 E2라고 이름 붙였습니다. 처음의 E2(카메라가 갭을 처음 보는 시점 분석)와는 다른 것입니다. E2는 학습 환경을 그대로 쓰되, 각 로봇을 시험 칸 하나에 고정합니다. 칸은 YAML 파일로 정합니다. 예를 들어 갭 후진, 레벨 11, 0.3 m/s 같은 식입니다. 커리큘럼은 끄고 명령도 고정하며, 정책은 탐색 잡음 없이 평균 행동만 냅니다. 로봇마다 첫 에피소드만 기록해서 칸별로 건넘, 넘어짐, 미도달, 머리 들림, 통과, 서기, 침범을 셉니다. 오른쪽은 최종 후보 비교 결과이고, 이 표로 학습 끝 모델 999가 아니라 861을 골랐습니다.");
  const e2s = [["LuSlidersHorizontal", "① YAML로 시험 칸 정의", "suite × 방향 × 레벨 × 속도 (예: gap / backward / L11 / 0.3)"],
    ["LuGrid3X3", "② 칸마다 로봇을 고정", "커리큘럼 끔, 명령 고정, 같은 seed의 같은 타일"],
    ["LuTarget", "③ 결정론적 정책, 첫 에피소드만", "탐색 잡음 없는 평균 행동, 20 s"],
    ["LuCircleCheck", "④ 칸별 지표 → 모델 비교 표", "건넘·clean·넘어짐·미도달·머리 들림 / 통과·서기·침범"]];
  for (let i = 0; i < 4; i++) {
    const y = 1.55 + i * 1.22;
    await ICON(s, e2s[i][0], 0.6, y, 0.55);
    T(s, e2s[i][1], { x: 1.3, y: y - 0.03, w: 4.6, h: 0.4, fontSize: 15, bold: true, color: C.text2 });
    T(s, e2s[i][2], { x: 1.3, y: y + 0.38, w: 4.6, h: 0.7, fontSize: 13 });
  }
  const eh = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.accent1 }, fontSize: 13, align: "center" } });
  const ec = (t, o = {}) => ({ text: t, options: { fontSize: 13, color: C.text1, align: "center", ...o } });
  const best = { bold: true, color: C.accent1 };
  TAB(s, [[eh("E2 (%)"), eh("999 (끝)"), eh("846 (온라인 1위)"), eh("861 (선정)")],
    [ec("갭 건넘 (후진 L14 제외)", { align: "left" }), ec("97.5"), ec("97.8"), ec("98.3", best)],
    [ec("느린 후진 갭 L11 0.3 m/s", { align: "left" }), ec("85.4", { color: C.accent4 }), ec("87.5", { color: C.accent4 }), ec("91.7", best)],
    [ec("계단", { align: "left" }), ec("95.7"), ec("94.7"), ec("95.7")],
    [ec("통로 / 굽은 통로", { align: "left" }), ec("96.3 / 94.5"), ec("96.6 / 95.3"), ec("95.8 / 94.5")],
    [ec("막힘 서기 / 침범", { align: "left" }), ec("100 / 0"), ec("100 / 0"), ec("100 / 0")],
    [ec("막다른 길 서기", { align: "left" }), ec("96.9"), ec("98.5"), ec("98.5")]],
    { x: 6.2, y: 1.55, w: 6.55, colW: [2.65, 1.2, 1.5, 1.2], rowH: 0.43, border: { type: "solid", pt: 1, color: "D7E3E2" }, fill: { color: "FFFFFF" }, valign: "middle", margin: 0.05 });
  CARD(s, 6.2, 4.72, 6.55, 1.0, C.text2);
  T(s, [{ text: "이름 유래  ", options: { bold: true, color: C.accent2 } },
        { text: "E = Experiment. 10-09 원인 분석 계획의 실험 번호 E1–E5에서 왔다. E1(Isaac 폐루프 latent 교체)이 평가 도구로 굳었고, 10-27 그다음 세대인 조건 고정 평가를 E2로 이름 붙였다 (처음 E2는 '카메라가 갭을 처음 보는 시점' 분석)" }],
    { x: 6.4, y: 4.74, w: 6.2, h: 0.96, fontSize: 11, color: C.background1, valign: "middle" });
  CARD(s, 6.2, 5.85, 6.55, 0.78);
  T(s, [{ text: "왜 필요한가  ", options: { bold: true, color: C.accent1 } },
        { text: "온라인 점수는 탐색 잡음이 섞인 학습 중 평균이라 드문 칸이 묻힌다. 온라인 1위 846은 느린 후진 갭이 90 % 미달" }],
    { x: 6.4, y: 5.87, w: 6.2, h: 0.74, fontSize: 11, valign: "middle" });
  T(s, "scripts/gast_narrow/e2/run_e2.py + configs/gast_narrow/e2/*.yaml · 칸당 48–512 env · A6000 4 GPU", { x: 0.6, y: 6.45, w: 5.4, h: 0.45, fontSize: 10, color: C.accent5 });

  // 16c blind Top-5 and blind E2 (added 11-12)
  s = slide("CONTENT", SEC[3], "4 · 결과", "블라인드 학습에도 Top-5와 E2를 붙였다 (11-12)",
    "블라인드 DWB를 한강에서 새로 학습하기 위해 두 가지를 추가했습니다. 첫째, 블라인드 학습에도 매 갱신마다 점수를 매겨 상위 5개를 남기는 온라인 Top-5를 붙였습니다. 기존 Top-5는 갭이 있어야 점수를 매길 수 있어서 블라인드에는 쓸 수 없었기 때문입니다. 둘째, 평지 추종, 밀기, 거친 지면, 경사, 계단 26칸으로 된 블라인드 E2를 만들었습니다. 기존 DWB-38000으로 시험하면 거의 모든 칸에서 성공 100 %이고, 새 블라인드 모델은 이 결과와 나란히 비교합니다.");
  T(s, "온라인 Top-5 점수 (매 갱신)", { x: 0.6, y: 1.5, w: 6, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  const sc = [["속도 추종", 0.35], ["생존 (넘어지지 않음)", 0.30], ["지형 레벨", 0.15], ["진행 거리", 0.10], ["에너지", 0.05], ["보상 합", 0.05]];
  for (let i = 0; i < 6; i++) {
    const y = 2.0 + i * 0.48;
    T(s, sc[i][0], { x: 0.6, y, w: 2.3, h: 0.36, fontSize: 13 });
    S(s, pres.ShapeType.rect, { x: 2.95, y: y + 0.06, w: sc[i][1] * 8, h: 0.26, fill: { color: i < 2 ? C.accent1 : C.accent6 }, line: { color: i < 2 ? C.accent1 : C.accent6 } });
    T(s, sc[i][1].toFixed(2), { x: 3.0 + sc[i][1] * 8, y, w: 0.6, h: 0.36, fontSize: 12, color: C.accent5 });
  }
  T(s, "관문: 지형 레벨 ≥ 6, 넘어짐 ≤ 10 %, 계단 ≤ 12 %, 경사 ≤ 8 %, 추종 ≥ 0.55, Top-1 대비 생존 −2 %p · 추종 −3 %p 이내 (200 it부터)",
    { x: 0.6, y: 4.95, w: 5.6, h: 0.75, fontSize: 12, color: C.accent5 });
  T(s, "로컬 시험: 512 env × 30 it, 매 갱신 Top-5 저장 + 중단 시 저장 확인", { x: 0.6, y: 5.75, w: 5.6, h: 0.5, fontSize: 12, color: C.accent1, bold: true });
  T(s, "블라인드 E2 26칸 (DWB-38000 예시, 칸당 16 env)", { x: 6.7, y: 1.5, w: 6, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  TAB(s, [[eh("묶음"), eh("칸"), eh("성공 %"), eh("넘어짐 %")],
    [ec("평지 추종 (전·후·옆·회전·정지, 0.3–1.2 m/s)", { align: "left" }), ec("8"), ec("100"), ec("0")],
    [ec("밀기 (0.5 / 1.0 m/s 옆 충격)", { align: "left" }), ec("2"), ec("100"), ec("0")],
    [ec("거친 지면 L7 / L14", { align: "left" }), ec("2"), ec("100"), ec("0")],
    [ec("경사 오르기·내리기 L7 / L14", { align: "left" }), ec("4"), ec("98.4"), ec("0")],
    [ec("계단 오르기·내리기 L5 / 9 / 14 + 넓은·코 계단", { align: "left" }), ec("10"), ec("100"), ec("0")]],
    { x: 6.7, y: 2.0, w: 6.05, colW: [3.65, 0.6, 0.9, 0.9], rowH: 0.52, border: { type: "solid", pt: 1, color: "D7E3E2" }, fill: { color: "FFFFFF" }, valign: "middle", margin: 0.05 });
  T(s, "경사 오르기 L14 한 칸만 93.8 %(넘어짐 없이 시간 안에 못 오름). scripts/blind/e2_blind.py + configs/blind/e2_blind.yaml", { x: 6.7, y: 5.25, w: 6.05, h: 0.6, fontSize: 11, color: C.accent5 });

  // 16d rl_replay
  s = slide("CONTENT", SEC[3], "4 · 결과", "rl_replay: 한 장면을 보상 관점에서 다시 본다",
    "rl_replay는 정책을 몇 초 돌리며 매 스텝을 기록하고, 브라우저에서 로봇 움직임과 보상 구성을 같이 재생하는 분석 도구입니다. 영상은 3906 교사가 갭을 건너는 장면입니다. 왼쪽 위 3D에는 로봇과 정답 지형(파랑), 교사가 받은 지형 칸(주황)이 보이고, 가운데에는 이번 스텝 보상과 미래 보상의 항목별 막대가 나옵니다. 아래 타임라인에는 critic이 예상한 V(s), 실제 미래 보상 G, 어드밴티지 A가 그려집니다. 어떤 보상이 행동을 움직였는지, critic 예측이 어디서 빗나갔는지를 볼 때 씁니다.");
  { const vw = 6.95, vhh = vw * 928 / 1280;
    MEDIA(s, { path: FIG + "media/rl_replay_3906_gap.mp4", coverPath: FIG + "media/cover/rl_replay_3906_gap.jpg", x: 0.6, y: 1.5, w: vw, h: vhh, objectName: "rl_replay" }); }
  const rr = [["LuBox", "3D: 로봇 + 지형", "정답 지형(파랑), 교사가 받은 칸(주황), 학생 카메라 점(초록)"],
    ["LuChartBar", "보상 항목별 막대", "이번 스텝 r_t와 미래 보상 G_t를 항목별로"],
    ["LuActivity", "V(s) · G_t · A_t 타임라인", "critic 예측과 실제 결과의 차이 = 어드밴티지"],
    ["LuCamera", "기록 → 브라우저 재생", "tools/rl_replay: record_rollout.py → server.py → viewer.html"]];
  for (let i = 0; i < 4; i++) {
    const y = 1.55 + i * 1.22;
    await ICON(s, rr[i][0], 8.3, y, 0.55);
    T(s, rr[i][1], { x: 9.0, y: y - 0.03, w: 3.75, h: 0.4, fontSize: 15, bold: true, color: C.text2 });
    T(s, rr[i][2], { x: 9.0, y: y + 0.38, w: 3.75, h: 0.75, fontSize: 12 });
  }
  T(s, "예시: 3906 교사, 0.6 m/s, env 3 (갭 통과 +0.5, 깨끗한 착지 +0.5) · 6초 기록을 뷰어로 재생해 녹화", { x: 0.6, y: 6.62, w: 7.4, h: 0.3, fontSize: 10, color: C.accent5 });
  // 17 results
  s = slide("CONTENT", SEC[3], "4 · 결과", "최종 it861: 모든 항목이 실전 기준 90 %를 넘는다",
    "E2 시험 결과입니다. 갭 98.3 %, 계단 95.7 %, 통로 95.8 %, 굽은 통로 94.5 %, 느린 후진 갭 91.7 %이고, 막힌 통로 앞에서는 100 % 섰고 차단선 침범은 한 번도 없었습니다. 실전 사용 기준은 90 %입니다.");
  CHART(s, pres.ChartType.bar, [{ name: "통과·성공률 (%)", labels: ["막힌 통로 서기", "느린 후진 갭", "굽은 통로", "통로", "계단", "승강장 갭"],
    values: [100, 91.7, 94.5, 95.8, 95.7, 98.3] }],
    { x: 0.5, y: 1.5, w: 8.2, h: 5.2, barDir: "bar", valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 20,
      chartColors: [THEME.colors.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 14, dataLabelColor: THEME.colors.dk1,
      dataLabelFormatCode: "0.0", dataLabelFontFace: "+mn-lt", catAxisLabelFontSize: 14, catAxisLabelFontFace: "+mn-lt", catAxisLabelColor: THEME.colors.dk1,
      valAxisLabelFontSize: 11, valAxisLabelColor: THEME.colors.accent5, valAxisLabelFontFace: "+mn-lt",
      valGridLine: { color: "E3EAEA", size: 1 }, catGridLine: { style: "none" }, showLegend: false, barGapWidthPct: 60,
      showTitle: true, title: "E2 시험 결과 (%)", titleFontSize: 15, titleColor: THEME.colors.dk2, titleFontFace: "+mn-lt" });
  T(s, "느린 후진 갭 = 레벨 11, 0.3 m/s", { x: 0.6, y: 6.55, w: 6, h: 0.3, fontSize: 11, color: C.accent5 });
  const res = [["0", "막힌 통로 차단선 침범"], ["90 %", "실전 사용 기준 (모든 항목 통과)"], ["6", "MuJoCo 좁은 통로 코스 (실제 배포 프로그램)"]];
  for (let i = 0; i < 3; i++) {
    const y = 1.6 + i * 1.7;
    CARD(s, 9.05, y, 3.7, 1.5, i === 0 ? C.text2 : C.background2);
    T(s, res[i][0], { x: 9.3, y: y + 0.1, w: 3.3, h: 0.8, fontSize: 40, bold: true, color: i === 0 ? C.accent2 : C.accent1 });
    T(s, res[i][1], { x: 9.3, y: y + 0.9, w: 3.3, h: 0.5, fontSize: 13, color: i === 0 ? C.background1 : C.text1 });
  }

  // 18 servers
  s = slide("CONTENT", SEC[3], "4 · 결과", "큰 서버의 이득은 드문 상황에 몰렸다",
    "같은 2단계 레시피를 A6000 서버와 한강 슈퍼컴퓨터에서 돌린 비교입니다. 로봇 수를 25배로 늘리자 굽은 통로와 좁은 통로처럼 드문 상황의 성적이 크게 올랐고, 갭과 계단은 거의 같았습니다. 그래서 레시피는 A6000에서 검증하고 본 학습만 한강에서 합니다.");
  CHART(s, pres.ChartType.bar, [{ name: "A6000 (5,120 로봇 × 300회)", labels: ["승강장 갭", "계단", "통로", "굽은 통로"], values: [97.9, 96.1, 89.4, 81.2] },
    { name: "한강 (131,072 로봇 × 1,000회)", labels: ["승강장 갭", "계단", "통로", "굽은 통로"], values: [98.3, 95.7, 95.8, 94.5] }],
    { x: 0.5, y: 1.5, w: 7.9, h: 5.2, barDir: "col", barGrouping: "clustered", valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 20,
      chartColors: [THEME.colors.accent6, THEME.colors.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12,
      dataLabelFormatCode: "0.0", dataLabelColor: THEME.colors.dk1, dataLabelFontFace: "+mn-lt", catAxisLabelFontSize: 14, catAxisLabelFontFace: "+mn-lt",
      catAxisLabelColor: THEME.colors.dk1, valAxisLabelFontSize: 11, valAxisLabelColor: THEME.colors.accent5, valAxisLabelFontFace: "+mn-lt",
      valGridLine: { color: "E3EAEA", size: 1 }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b", legendFontSize: 12,
      legendFontFace: "+mn-lt", showTitle: true, title: "같은 2단계 레시피, 서버 규모만 다름 (E2 %)", titleFontSize: 15,
      titleColor: THEME.colors.dk2, titleFontFace: "+mn-lt" });
  const srv = [["LuServer", "A6000 (RTX PRO 6000 × 3)", "레시피 사전 검증: 5,120 로봇 × 300회 ≈ 2.3시간"],
    ["LuServer", "한강 (KISTI GH200, 16노드 × 4)", "본 학습: 131,072 로봇 × 1,000회 ≈ 1,200 GPU-시간"]];
  for (let i = 0; i < 2; i++) {
    const y = 1.6 + i * 1.55;
    await ICON(s, srv[i][0], 8.8, y, 0.6, i ? C.accent1 : C.accent6, i ? "FFFFFF" : THEME.colors.dk2);
    T(s, srv[i][1], { x: 9.55, y: y - 0.02, w: 3.2, h: 0.4, fontSize: 15, bold: true, color: C.text2 });
    T(s, srv[i][2], { x: 9.55, y: y + 0.4, w: 3.2, h: 0.95, fontSize: 13 });
  }
  CARD(s, 8.8, 4.85, 3.95, 1.75, C.text2);
  T(s, [{ text: "+13 %p", options: { bold: true, color: C.accent2, fontSize: 30, breakLine: true } },
        { text: "굽은 통로 (81.2 → 94.5). 갭·계단은 같은 수준", options: { fontSize: 13 } }],
    { x: 9.05, y: 4.95, w: 3.5, h: 1.55, color: C.background1, valign: "middle" });

  // 19 rules
  s = slide("CONTENT", SEC[3], "4 · 결과", "실패에서 얻은 설계 규칙 여섯 가지",
    "30개 규칙 중 다음 학습 설계에 바로 쓰이는 여섯 가지만 골랐습니다.");
  const rules = [["LuAnchor", "보존할 기술은 지형 열과 고정 앵커를 늘 함께 둔다"],
    ["LuScale", "벌점은 짝으로: 성공 보상엔 속도 감쇠, 과속 벌점엔 정체 벌점"],
    ["LuSprout", "새 '서기'는 보상만으로 생기지 않는다. 입력 플래그·앵커·표본 장치를 함께 쓴다"],
    ["LuSlidersHorizontal", "입력으로 갈리는 행동은 입력 없이는 못 맞히게 50 : 50으로 뽑는다"],
    ["LuRadar", "입력은 센서·운용자·gdm이 만들 수 있는 값만. 장소·타일 라벨은 넣지 않는다"],
    ["LuStepForward", "새 행동은 쉬운 분포·약한 벌점으로 먼저, 경계 다듬기는 그 뒤"]];
  for (let i = 0; i < 6; i++) {
    const x = 0.6 + (i % 3) * 4.1, y = 1.6 + Math.floor(i / 3) * 2.55;
    CARD(s, x, y, 3.85, 2.3);
    await ICON(s, rules[i][0], x + 0.25, y + 0.25, 0.6);
    T(s, String(i + 1).padStart(2, "0"), { x: x + 2.9, y: y + 0.25, w: 0.75, h: 0.5, fontSize: 22, bold: true, color: C.accent6, align: "right" });
    T(s, rules[i][1], { x: x + 0.25, y: y + 1.0, w: 3.4, h: 1.15, fontSize: 15, color: C.text2 });
  }

  // video examples (final it861 MuJoCo runs with the deploy binary; handstand = skill model)
  const MED = FIG + "media/";
  pres.addSection({ title: SEC[4] });
  const pair = async (title, notes, items) => {
    const sl = slide("CONTENT", SEC[4], "5 · 영상 예제", title, notes);
    for (let i = 0; i < 2; i++) {
      const y = 1.5 + i * 2.62;
      MEDIA(sl, { path: MED + items[i][0] + ".mp4", coverPath: MED + "cover/" + items[i][0] + ".jpg", x: 0.6, y, w: 9.55, h: 9.55 * 480 / 1898, objectName: items[i][0] });
      T(sl, items[i][1], { x: 10.45, y: y + 0.05, w: 2.3, h: 0.75, fontSize: 17, bold: true, color: C.text2 });
      T(sl, items[i][2], { x: 10.45, y: y + 0.85, w: 2.3, h: 1.45, fontSize: 13 });
    }
    T(sl, "왼쪽: MuJoCo 화면 · 오른쪽: 로봇 카메라 4대의 깊이·IR 영상 (실제 배포 프로그램 Pilot + ONNX로 실행)", { x: 0.6, y: 6.62, w: 11, h: 0.3, fontSize: 11, color: C.accent5 });
  };
  await pair("승강장 갭을 전진·후진으로 건넌다",
    "최종 모델 it861을 MuJoCo에서 실제 배포 프로그램으로 돌린 영상입니다. 위는 5–25 cm 갭이 이어진 코스를 0.5 m/s로 전진, 아래는 0.3 m/s로 후진하는 장면입니다. 오른쪽 화면은 로봇 아래·앞뒤 카메라 4대가 보는 깊이와 IR 영상입니다. 두 번 모두 넘어지지 않고 목표 지점까지 갔습니다.",
    [["gap_fwd", "전진 0.5 m/s", "갭 5–25 cm 코스\n목표 지점 도착"], ["gap_back", "후진 0.3 m/s", "같은 갭 코스를 뒤로\n목표 지점 도착"]]);
  await pair("계단 20 cm를 오르고 내린다",
    "20 cm 계단 10칸을 오르고, 꼭대기를 지나 10칸을 내려가는 코스입니다. 위는 0.6 m/s 전진, 아래는 180도 돈 뒤 0.4 m/s로 뒤로 걸으며 오르내립니다.",
    [["stairs_fwd", "전진 0.6 m/s", "20 cm × 10칸 오르기 → 꼭대기 → 10칸 내리기"], ["stairs_back", "후진 0.4 m/s", "180° 돌아서 같은 계단을 뒤로 오르내리기"]]);
  s = slide("CONTENT", SEC[4], "5 · 영상 예제", "좁은 통로: 위에서 보면 벽 사이를 지나는 게 보인다",
    "옆에서 찍으면 벽에 가려서, 녹화된 로봇 상태 기록으로 위에서 본 영상을 다시 그렸습니다. 왼쪽은 복도, 어긋난 문, 굽은 통로가 이어진 코스이고, 오른쪽은 폭을 더 좁힌 코스입니다. 두 코스 모두 벽에 걸리지 않고 끝까지 갔습니다.");
  const tops = [["narrow_only_top", "좁은 통로 코스", "복도 → 어긋난 문 → 굽은 통로 → L자 · 69 s"], ["tight_fwd_top", "더 좁힌 코스", "같은 구성, 폭을 줄임 · 86 s"]];
  for (let i = 0; i < 2; i++) {
    const x = 0.6 + i * 6.2;
    MEDIA(s, { path: MED + tops[i][0] + ".mp4", coverPath: MED + "cover/" + tops[i][0] + ".jpg", x, y: 1.55, w: 5.95, h: 5.95 * 9 / 16, objectName: tops[i][0] });
    T(s, tops[i][1], { x, y: 5.1, w: 5.95, h: 0.42, fontSize: 17, bold: true, color: C.text2 });
    T(s, tops[i][2], { x, y: 5.55, w: 5.95, h: 0.4, fontSize: 14 });
  }
  T(s, "위에서 본 영상은 녹화 때 저장한 로봇 상태(위치·자세·관절, 약 10 Hz)로 다시 그린 것", { x: 0.6, y: 6.55, w: 11, h: 0.3, fontSize: 11, color: C.accent5 });
  s = slide("CONTENT", SEC[4], "5 · 영상 예제", "모드 스킬 예: 물구나무 + 제자리 걸음",
    "블라인드 모델에서 따로 학습한 물구나무 기술입니다. 외부 신호를 받으면 앞발로 서서 90도까지 올라가고, 걷기 신호에 앞발로 제자리 걸음을 한 뒤 신호가 꺼지면 네 발로 돌아옵니다. Isaac과 비슷한 발 접촉 조건에서는 10번 모두 성공했지만, 실제 로봇용 발 접촉 조건에서는 절반이 넘어져서 아직 실기에 쓰지 않습니다.");
  MEDIA(s, { path: MED + "handstand_step.mp4", coverPath: MED + "cover/handstand_step.jpg", x: 0.6, y: 1.55, w: 6.6, h: 4.95, objectName: "handstand_step" });
  const hs = [["LuFlag", "요청 → 앞발 물구나무 90°", "외부 모드 신호로 시작. payload 0, LiDAR 없음"],
    ["LuFootprints", "걷기 신호 → 제자리 앞발 걸음", "앞발 25걸음, 제자리 이탈 0.15 m"],
    ["LuCircleCheck", "신호 끔 → 네 발로 복귀", "Isaac과 비슷한 발 접촉 조건 10회 중 10회 성공"],
    ["LuTriangleAlert", "아직 실기 전", "실제 로봇용 발 접촉 조건 10회 중 5회 넘어짐 → 마찰을 섞어 재학습 필요"]];
  for (let i = 0; i < 4; i++) {
    const y = 1.6 + i * 1.25;
    await ICON(s, hs[i][0], 7.6, y, 0.6, i === 3 ? C.accent4 : C.accent1);
    T(s, hs[i][1], { x: 8.4, y: y - 0.03, w: 4.35, h: 0.4, fontSize: 16, bold: true, color: i === 3 ? C.accent4 : C.text2 });
    T(s, hs[i][2], { x: 8.4, y: y + 0.38, w: 4.35, h: 0.75, fontSize: 13 });
  }

  // handstand rewards 1: gated by the mode signal
  s = slide("CONTENT", SEC[4], "5 · 영상 예제", "물구나무 보상은 '모드 신호'로 켜고 끈다",
    "보행 보상은 명령 속도를 따라가는 추종 보상이 중심인데, 물구나무에는 추종 보상이 없습니다. 대신 외부 모드 신호가 켜져 있는지에 따라 보상 묶음이 통째로 바뀝니다. 신호 전에는 가만히 서 있기, 신호가 켜지면 목표 자세로 가서 버티기, 신호가 꺼지면 네 발로 수평 복귀입니다. 시간이나 지형으로 켜고 끄지 않는 것이 핵심입니다.");
  const hsState = [["신호 전", "가만히 선다", [["대기 중 움직임", "−1"]], C.accent5],
    ["신호 켜짐 (요청)", "앞발로 서서 90°에서 버틴다", [["목표 자세 쪽 몸통 방향", "+3"], ["유지 (±10° 안, 뒷발 들림)", "+2"], ["앞발만 땅 (목표 절반 이후)", "+1"], ["앞발 하중 비율 (체중 이동)", "+1.5"], ["뒷발 닿음 개수 (30° 이후)", "−1"]], C.accent1],
    ["신호 꺼짐 / 중단", "네 발로 수평 복귀", [["수평 자세 + 네 발 접지", "+2"]], C.accent3]];
  for (let i = 0; i < 3; i++) {
    const x = 0.6 + i * 4.1, big = i === 1;
    CARD(s, x, 1.55, 3.85, 3.75, big ? C.text2 : C.background2);
    S(s, pres.ShapeType.roundRect, { x: x + 0.25, y: 1.78, w: 3.35, h: 0.45, rectRadius: 0.22, fill: { color: hsState[i][3] }, line: { color: hsState[i][3] } });
    T(s, hsState[i][0], { x: x + 0.25, y: 1.78, w: 3.35, h: 0.45, fontSize: 15, bold: true, color: C.background1, align: "center", valign: "middle" });
    T(s, hsState[i][1], { x: x + 0.25, y: 2.38, w: 3.35, h: 0.4, fontSize: 15, bold: true, color: big ? C.accent2 : C.text2 });
    hsState[i][2].forEach(([t, w], j) => {
      T(s, t, { x: x + 0.25, y: 2.95 + j * 0.44, w: 2.65, h: 0.4, fontSize: 13, color: big ? C.background1 : C.text1 });
      T(s, w, { x: x + 2.9, y: 2.95 + j * 0.44, w: 0.7, h: 0.4, fontSize: 14, bold: true, color: big ? C.accent6 : (w.startsWith("−") ? C.accent4 : C.accent1), align: "right" });
    });
    if (i < 2) ARROW(s, x + 3.88, 3.4, x + 4.07, 3.4, C.accent1, 2.5);
  }
  const hsDiff = [["LuToggleRight", "추종 보상 없음", "속도 명령은 0. 보상 묶음은 모드 신호가 바꾼다"],
    ["LuSprout", "보상만으로는 못 일어섰다", "균형 잡힌 시작 자세(물구나무 중간부터 시작)를 섞고 앞발 하중 보상을 더함"],
    ["LuAnchor", "네 발 서기는 앵커로 지킨다", "신호 없는 표본은 출발 블라인드 모델의 행동에 묶어 둔다"]];
  for (let i = 0; i < 3; i++) {
    const x = 0.6 + i * 4.1;
    await ICON(s, hsDiff[i][0], x, 5.55, 0.55, C.accent2, THEME.colors.dk2);
    T(s, hsDiff[i][1], { x: x + 0.7, y: 5.5, w: 3.25, h: 0.38, fontSize: 15, bold: true, color: C.text2 });
    T(s, hsDiff[i][2], { x: x + 0.7, y: 5.9, w: 3.3, h: 0.85, fontSize: 12 });
  }

  // handstand rewards 2: stepping and safety, by stage
  s = slide("CONTENT", SEC[4], "5 · 영상 예제", "제자리 걸음과 안전 보상은 문제마다 하나씩 더했다",
    "걷기 신호가 켜지면 앞발로 제자리 걸음을 하는 단계입니다. S2에서 걸음 보상을 넣었고, S2b에서는 MuJoCo에서 수직을 넘어 넘어지는 문제 때문에 과회전 벌점과 관절 마찰 무작위화를 넣었습니다. S2c에서는 걸으면서 옆으로 흘러가는 문제 때문에 제자리 벌점을 세게 하고 이탈 거리 벌점을 더했습니다.");
  const hh = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.accent1 }, fontSize: 13 } });
  const hc = (t, o = {}) => ({ text: t, options: { fontSize: 13, color: C.text1, ...o } });
  const neg = { color: C.accent4, bold: true }, pos = { color: C.accent1, bold: true };
  const hrows = [[hh("단계"), hh("보상 항목"), hh("가중치"), hh("무엇을 하나 / 넣은 이유")],
    [hc("S2 걸음"), hc("발 공중 시간"), hc("+4", pos), hc("앞발이 0.10–0.4 s 떠 있다가 디딜 때 점수")],
    [hc(""), hc("번갈아 딛기"), hc("+0.5", pos), hc("앞발 하나만 땅이면 +, 둘 다 공중이면 −")],
    [hc(""), hc("제자리 유지"), hc("−1 → −3", neg), hc("몸통 수평 속도·회전 (S2c에서 강화)")],
    [hc(""), hc("관절 목표 급변"), hc("−1", neg), hc("한 스텝에 10° 넘게 바뀌는 목표 (실제 모터가 따라가지 못함)")],
    [hc(""), hc("토크 포화"), hc("−0.5 → −1", neg), hc("모터 한계 90 %를 넘는 관절 수")],
    [hc("S2b 안전"), hc("과회전"), hc("−5", neg), hc("90°를 10° 넘게 넘어감 (MuJoCo에서 넘어간 원인)")],
    [hc(""), hc("수직 근처 회전 속도"), hc("−0.05", neg), hc("70° 이후에는 천천히 도착")],
    [hc(""), hc("스윙 높이"), hc("+1", pos), hc("든 앞발을 5 cm까지 올림")],
    [hc("S2c 제자리"), hc("이탈 거리"), hc("−2", neg), hc("걷기 시작점에서 5 cm 넘게 벗어난 거리")]];
  TAB(s, hrows, { x: 0.6, y: 1.5, w: 12.15, colW: [1.7, 2.6, 1.45, 6.4], rowH: 0.43, border: { type: "solid", pt: 1, color: "D7E3E2" },
    fill: { color: "FFFFFF" }, valign: "middle", margin: 0.06 });
  CARD(s, 0.6, 6.0, 12.15, 0.75);
  T(s, [{ text: "보상 밖의 장치  ", options: { bold: true, color: C.accent1 } },
        { text: "관절 쿨롱 마찰 0–1 N·m 무작위화(S2b) · 버티기 표본을 S1 모델 행동에 묶는 앵커 0.5 · 관절 한계 여유 5° → 8°" }],
    { x: 0.85, y: 6.03, w: 11.7, h: 0.7, fontSize: 13, valign: "middle" });

  // 20 next steps (dark)
  pres.addSection({ title: SEC[5] });
  s = slide("DARK_CONTENT", SEC[5], "6 · 다음 단계", "다음 단계",
    "남은 일은 네 가지입니다. 낭떠러지 앞에서 서기, 제자리 회전, 실제 로봇으로 옮기기, 그리고 물구나무 같은 모드 스킬입니다. 물구나무는 시뮬레이션에서 제자리 걸음까지 됐지만, 실제 발 접촉 조건에서 아직 절반이 넘어져 마찰을 섞어 다시 학습해야 합니다.");
  const nxt = [["LuTriangleAlert", "낭떠러지 앞 서기 (C′)", "앞·뒤 1 m 낙차 앞에서 선다. 센서에서 계산한 '지지면 끊김' 입력 사용. 시험 서기 100 %, 저속 넓은 갭 정지가 남은 과제"],
    ["LuRotateCw", "제자리 회전 (C2)", "회전 가능 공간 판단 + 회전 차단 플래그 + 서기 앵커 레시피로 설계 예정"],
    ["LuPlug", "실제 로봇으로 옮기기", "Pilot에 로봇 위치(/odom 50 Hz) 연결, 실제 D4xx 카메라 → 지형처리기 연결. 지금 지형 경로는 시뮬레이션 전용"],
    ["LuSprout", "모드 스킬 (물구나무 등)", "블라인드 모델에서 따로 학습. 물구나무 + 제자리 걸음까지 됨. 실제 발 접촉 조건 10회 중 5회 넘어짐 → 마찰 재학습 필요"]];
  for (let i = 0; i < 4; i++) {
    const x = 0.6 + (i % 2) * 6.2, y = 1.6 + Math.floor(i / 2) * 2.55;
    CARD(s, x, y, 5.95, 2.3, "15505A");
    await ICON(s, nxt[i][0], x + 0.3, y + 0.3, 0.7, C.accent2, THEME.colors.dk2);
    T(s, nxt[i][1], { x: x + 1.2, y: y + 0.3, w: 4.5, h: 0.45, fontSize: 18, bold: true, color: C.background1 });
    T(s, nxt[i][2], { x: x + 1.2, y: y + 0.82, w: 4.55, h: 1.35, fontSize: 14, color: C.accent6 });
  }

  // 21-22 appendix figures
  pres.addSection({ title: SEC[6] });
  s = slide("CONTENT", SEC[6], "부록", "입력 구조 상세",
    "통합 문서 그림 2입니다. 몸 감각 46의 구성과 히스토리, 하방 지형 지도와 GAST 인코더 내부, LiDAR 점유 지도, 외부 조건을 보여 줍니다. 오른쪽 세로 막대가 Actor 입력 253의 구성입니다.");
  const ih = 5.3, iw = ih * 4889 / 3453;
  IMG(s, { path: FIG + "vrl_inputs.png", x: (W - iw) / 2, y: 1.48, w: iw, h: ih, altText: "입력 구조 상세 그림" });
  s = slide("CONTENT", SEC[6], "부록", "출력 구조와 배포 상세",
    "통합 문서 그림 3입니다. 행동이 관절 목표와 PD 토크가 되는 과정, 100 Hz와 500 Hz의 시간 구조, 관절 배치, 배포 묶음과 입력 신선도 보호를 보여 줍니다.");
  const ow = 12.1, oh = ow * 2139 / 4889;
  IMG(s, { path: FIG + "vrl_outputs.png", x: 0.6, y: 1.55, w: ow, h: oh, altText: "출력 구조와 배포 상세 그림" });

  // appendix: dog gait analysis (docs/design/DOG_GAIT_REFERENCE.md, docs/teachers/gast_v21_reward_dog_insights_20261007.md)
  const dh = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.accent1 }, fontSize: 13, align: "center" } });
  const dc = (t, o = {}) => ({ text: t, options: { fontSize: 13, color: C.text1, align: "center", ...o } });
  s = slide("CONTENT", SEC[6], "부록 · 개 보행 분석", "개 보행 문헌을 로봇 크기로 환산했다",
    "리트리버급 개 보행 문헌 11편에서 스윙·지지 시간, 보폭, 지면 반력을 모았습니다. 개와 로봇은 크기가 달라서 Froude 수(속도² ÷ (중력 × 체고))를 맞추는 방식으로 환산했습니다. 로봇 체고 0.50 m는 개 0.6 m의 0.83배라서, 시간은 0.91배, 길이는 0.83배가 되고, 로봇 속도 v는 개 속도 1.1·v에 해당합니다. 우리 운용 속도 전체에서 같은 크기의 개는 걷기(walk)로 걷습니다. 개의 스윙 시간은 속도와 거의 무관하고, 개는 속도를 스윙이 아니라 지지 시간을 줄여서 냅니다.");
  const lit = [["11편", "개 보행 문헌 (래브라도·말리노이즈·셰퍼드 등)"], ["0.27–0.56 s", "개의 스윙 시간 — 속도와 거의 무관"], ["1.8–2.6배", "걷기 착지 수직력 (정지 시 한 발 몫 대비)"]];
  for (let i = 0; i < 3; i++) {
    const y = 1.55 + i * 1.12;
    T(s, lit[i][0], { x: 0.6, y, w: 3.4, h: 0.6, fontSize: 28, bold: true, color: C.accent1 });
    T(s, lit[i][1], { x: 0.6, y: y + 0.6, w: 3.6, h: 0.45, fontSize: 13, color: C.accent5 });
  }
  CARD(s, 0.6, 5.0, 3.6, 1.65, C.text2);
  T(s, [{ text: "Froude 수로 환산", options: { bold: true, color: C.accent2, breakLine: true } },
        { text: "Fr = v² / (g · 체고)", options: { breakLine: true } },
        { text: "로봇 0.50 m / 개 0.60 m → 길이 ×0.83, 시간 ×0.91", options: { fontSize: 12, color: C.accent6 } }],
    { x: 0.85, y: 5.1, w: 3.2, h: 1.45, fontSize: 15, color: C.background1, valign: "middle" });
  TAB(s, [[dh("로봇 속도"), dh("해당 개 속도"), dh("Fr"), dh("개의 걸음"), dh("스윙 (환산)"), dh("발당 걸음 수")],
    [dc("0.6 m/s"), dc("0.66 m/s"), dc("0.07"), dc("걷기"), dc("0.26 s"), dc("1.0 /s")],
    [dc("0.8 m/s"), dc("0.88 m/s"), dc("0.13"), dc("걷기"), dc("0.25 s"), dc("1.3 /s")],
    [dc("1.0 m/s"), dc("1.10 m/s"), dc("0.20"), dc("걷기"), dc("0.25 s"), dc("1.45 /s")],
    [dc("1.2 m/s"), dc("1.31 m/s"), dc("0.29"), dc("걷기–속보 경계"), dc("0.24 s"), dc("1.6 /s")]],
    { x: 4.55, y: 1.55, w: 8.2, colW: [1.25, 1.45, 0.9, 1.9, 1.35, 1.35], rowH: 0.55, border: { type: "solid", pt: 1, color: "D7E3E2" },
      fill: { color: "FFFFFF" }, valign: "middle", margin: 0.05 });
  const dread = [["LuFootprints", "속도를 올려도 스윙은 0.25 s 근처", "개는 지지 시간을 줄여 속도를 낸다 → 걸음 수가 1.0 → 1.6 /s로 는다"],
    ["LuTriangleAlert", "걸음 방식이 다르다", "우리 정책은 모든 속도에서 대각 trot, 개는 walk. 환산값은 '같은 크기·속도의 개가 보이는 범위'로만 쓴다"]];
  for (let i = 0; i < 2; i++) {
    const y = 4.55 + i * 1.08;
    await ICON(s, dread[i][0], 4.55, y, 0.55, i ? C.accent4 : C.accent1);
    T(s, dread[i][1], { x: 5.3, y: y - 0.03, w: 7.45, h: 0.38, fontSize: 15, bold: true, color: i ? C.accent4 : C.text2 });
    T(s, dread[i][2], { x: 5.3, y: y + 0.36, w: 7.45, h: 0.6, fontSize: 13 });
  }
  T(s, "스윙 환산 = 말리노이즈 회귀식(Maes 등 2008, J Exp Biol) × 0.91. 래브라도 느린 걷기(Light 등 2010)는 0.45–0.51 s로 범위의 위쪽 끝", { x: 0.6, y: 6.72, w: 12.1, h: 0.25, fontSize: 10, color: C.accent5 });

  s = slide("CONTENT", SEC[6], "부록 · 개 보행 분석", "개 기준으로 보면 3906은 종종걸음, gait v3(G0)는 개 범위 안",
    "평지 0.8 m/s에서 우리 정책을 개 환산값과 비교했습니다. 3906의 앞발 스윙 0.20 s는 어떤 개보다 짧은 종종걸음이고, 조용한 걸음 레시피 G0는 0.32–0.38 s로 개 범위 안입니다. 착지 힘은 정지력 대비 1.0 m/s에서 두 모델 모두 2.6–2.9배로, 개 걷기 뒷발 1.8–2.1배보다 큽니다. 발 높이는 어떤 문헌에도 수치가 없어서 로봇 기준(5–10 cm)으로만 정했습니다. 이 기준은 계단 보행이 무너지지 않는 범위에서만 적용합니다.");
  TAB(s, [[dh("항목 (평지 0.8 m/s)"), dh("개 (환산)"), dh("3906"), dh("G0 (gait v3)"), dh("판단")],
    [dc("스윙 시간 앞 / 뒤"), dc("0.24–0.51 s"), dc("0.20 / 0.24 s", { color: C.accent4, bold: true }), dc("0.32 / 0.38 s", { color: C.accent1, bold: true }), dc("3906은 하한 밖")],
    [dc("발당 걸음 수"), dc("1.1–1.3 /s"), dc("1.5 /s", { color: C.accent4, bold: true }), dc("1.3 /s", { color: C.accent1, bold: true }), dc("G0가 개에 가까움")],
    [dc("뒷발 착지 힘 (1.0 m/s, 정지력 대비)"), dc("걷기 1.8–2.1배"), dc("2.6–2.9배"), dc("2.6–2.9배"), dc("둘 다 개보다 큼")],
    [dc("발 높이 (스윙 중 최대)"), dc("문헌 수치 없음"), dc("5.5 cm"), dc("10.2 cm"), dc("로봇 기준 5–10 cm")]],
    { x: 0.6, y: 1.55, w: 12.15, colW: [3.35, 2.1, 1.9, 2.1, 2.7], rowH: 0.55, border: { type: "solid", pt: 1, color: "D7E3E2" },
      fill: { color: "FFFFFF" }, valign: "middle", margin: 0.05 });
  T(s, "이렇게 정한 gait v3 걸음 기준 (평지 전·후진)", { x: 0.6, y: 4.6, w: 8, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  const crit = [["스윙 시간", "0.24–0.45 s", "개 환산 범위의 안쪽"], ["발당 걸음 수", "0.9–1.7 /s", "속도별 개 환산 ± 0.2"], ["착지 수직력 p90", "≤ 2.0–2.2배", "개 걷기 뒷발 수준"], ["발 높이", "5–10 cm", "끌림 방지 / 착지 속도"]];
  for (let i = 0; i < 4; i++) {
    const x = 0.6 + i * 3.1;
    CARD(s, x, 5.1, 2.85, 1.5);
    T(s, crit[i][1], { x: x + 0.2, y: 5.2, w: 2.5, h: 0.55, fontSize: 22, bold: true, color: C.accent1 });
    T(s, crit[i][0], { x: x + 0.2, y: 5.78, w: 2.5, h: 0.35, fontSize: 14, bold: true, color: C.text2 });
    T(s, crit[i][2], { x: x + 0.2, y: 6.12, w: 2.5, h: 0.35, fontSize: 12, color: C.accent5 });
  }

  s = slide("CONTENT", SEC[6], "부록 · 개 보행 분석", "개 행동에서 얻은 설계 가설",
    "보행 수치 말고도 개와 동물의 행동에서 설계 가설을 얻었습니다. 가장 큰 영향은 '뒷다리는 장애물을 기억으로 넘는다'는 관찰로, 지형 기억이 있는 GAST 교사를 선택한 근거가 됐습니다. 갭 앞에서 미리 몸을 준비하는 동작은 학생 구조 개선으로 이어졌습니다. 나머지는 진단용으로만 두거나 2차 과제로 미뤘습니다. 근거는 기사와 연구 요약이고, 동물 행동과 보상식의 대응은 설계 가설입니다.");
  const hyp = [["반영", "뒷다리는 장애물을 기억으로 넘는다", "장애물이 앞뒤 다리 사이에 오면 보이지 않는다. 앞다리가 넘은 경험이 기억이 된다", "갭에서 뒷발 둘을 44–49 cm 들던 습관 → 지형 기억이 있는 GAST 교사 선택 → 23–32 cm로"],
    ["반영", "장애물 앞에서 미리 몸을 준비한다", "보이는 동안 자세와 발 들기를 바꾼다", "교사는 갭 0.7–0.4 m 앞에서 pitch +5°, 학생은 0° → 몸 진행량과 함께 갱신되는 학생 기억(student v3)"],
    ["진단만", "direct register", "여우·고양이는 뒷발을 같은 쪽 앞발이 막 디딘 자리에 놓는다", "trot에 강제할 근거가 약하고 안전한 착지보다 우선할 위험 → 가중치 0"],
    ["2차로 미룸", "끌림 + 모서리 + 뒷다리 사고", "주인이 끌자 뒷다리로 버티다 갭으로 미끄러진 실제 사고", "손잡이 당김 외력을 계단뿐 아니라 갭 앞·위에서도 → 0–100 N부터 단계적으로"]];
  const hcol = { "반영": C.accent1, "진단만": C.accent5, "2차로 미룸": C.accent3 };
  for (let i = 0; i < 4; i++) {
    const x = 0.6 + (i % 2) * 6.2, y = 1.55 + Math.floor(i / 2) * 2.55;
    CARD(s, x, y, 5.95, 2.35);
    S(s, pres.ShapeType.roundRect, { x: x + 0.25, y: y + 0.22, w: 1.25, h: 0.38, rectRadius: 0.19, fill: { color: hcol[hyp[i][0]] }, line: { color: hcol[hyp[i][0]] } });
    T(s, hyp[i][0], { x: x + 0.25, y: y + 0.22, w: 1.25, h: 0.38, fontSize: 12, bold: true, color: C.background1, align: "center", valign: "middle" });
    T(s, hyp[i][1], { x: x + 1.65, y: y + 0.2, w: 4.15, h: 0.42, fontSize: 16, bold: true, color: C.text2 });
    T(s, hyp[i][2], { x: x + 0.25, y: y + 0.75, w: 5.5, h: 0.6, fontSize: 13, color: C.accent5 });
    T(s, [{ text: "→ ", options: { bold: true, color: hcol[hyp[i][0]] } }, { text: hyp[i][3] }], { x: x + 0.25, y: y + 1.38, w: 5.5, h: 0.85, fontSize: 13 });
  }
  T(s, "근거: 연구 요약·기사 본문 (J Neurosci 2009 등). 동물 행동 → 로봇 행동 → 보상식은 설계 가설", { x: 0.6, y: 6.72, w: 12.1, h: 0.25, fontSize: 10, color: C.accent5 });
}

// ================================================================= preview / overflow check
const PX = 144; // px per inch
const FONT = "Noto Sans CJK KR";
const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
function markup(text, o) {
  const runs = typeof text === "string" ? [{ text, options: {} }] : text;
  let m = "";
  for (const r of runs) {
    const ro = r.options || {};
    const size = (ro.fontSize || o.fontSize || 14);
    const col = "#" + hex(ro.color || o.color || "tx1");
    const b = ro.bold || o.bold ? "bold" : "normal";
    const t = (ro.bullet ? "• " : "") + esc(r.text);
    m += `<span font_desc="${FONT} ${size}" weight="${b}" foreground="${col}">${t}</span>`;
    if (ro.breakLine) m += "\n";
  }
  return m;
}
async function textImg(text, o, wIn) {
  const plain = typeof text === "string" ? text : text.map((r) => r.text).join("");
  if (!plain.trim()) return { buf: await sharp({ create: { width: 1, height: 1, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 0 } } }).png().toBuffer(), w: 1, h: 1 };
  const width = Math.max(8, Math.round(wIn * PX));
  const img = sharp({ text: { text: markup(text, o), width, dpi: PX, rgba: true, wrap: "word-char",
    align: o.align === "center" ? "centre" : o.align === "right" ? "right" : "left", spacing: 0 } });
  const buf = await img.png().toBuffer();
  const meta = await sharp(buf).metadata();
  return { buf, w: meta.width, h: meta.height };
}
function shapeSvg(type, o) {
  const x = o.x * PX, y = o.y * PX, w = o.w * PX, h = o.h * PX;
  const fill = o.fill ? "#" + hex(o.fill.color) : "none";
  const stroke = o.line ? "#" + hex(o.line.color) : "none";
  const sw = o.line && o.line.width ? o.line.width * PX / 72 : 1;
  if (type === "ellipse") return `<ellipse cx="${x + w / 2}" cy="${y + h / 2}" rx="${w / 2}" ry="${h / 2}" fill="${fill}" stroke="${stroke}"/>`;
  if (type === "line") {
    const x1 = o.flipH ? x + w : x, x2 = o.flipH ? x : x + w, y1 = o.flipV ? y + h : y, y2 = o.flipV ? y : y + h;
    const dash = o.line.dashType === "dash" ? ' stroke-dasharray="10,6"' : "";
    return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${sw}"${dash}${o.line.endArrowType ? ' marker-end="url(#a)"' : ""}/>`;
  }
  const r = type === "roundRect" ? (o.rectRadius || 0.1) * PX : 0;
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}"/>`;
}
async function preview(outDir) {
  fs.mkdirSync(outDir, { recursive: true });
  const problems = [];
  for (let si = 0; si < REC.length; si++) {
    const r = REC[si];
    const bg = "#" + hex(LAYOUT_PH[r.master].bg);
    let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W * PX}" height="${H * PX}"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#6E7F80"/></marker></defs><rect width="100%" height="100%" fill="${bg}"/>`;
    const comps = [];
    for (const it of r.items) {
      const o = it.o;
      if (it.k === "shape") svg += shapeSvg(it.text, o);
      else if (it.k === "img") {
        const src = o.data ? Buffer.from(o.data.split(",")[1], "base64") : fs.readFileSync(o.path);
        comps.push({ input: await sharp(src).resize(Math.round(o.w * PX), Math.round(o.h * PX), { fit: "fill" }).png().toBuffer(), left: Math.round(o.x * PX), top: Math.round(o.y * PX) });
      } else if (it.k === "chart") {
        svg += `<rect x="${o.x * PX}" y="${o.y * PX}" width="${o.w * PX}" height="${o.h * PX}" fill="#F4F8F8" stroke="#9CC9C5" stroke-dasharray="8,6"/>`;
        const lab = `${it.text.type} chart: ` + it.text.data.map((d) => d.name + " " + d.values.join("/")).join(" | ");
        const ti = await textImg(lab, { fontSize: 12, color: "accent5" }, o.w - 0.4);
        comps.push({ input: ti.buf, left: Math.round((o.x + 0.2) * PX), top: Math.round((o.y + 0.2) * PX) });
      } else if (it.k === "table") {
        let y = o.y;
        for (const row of it.text) {
          let rowH = o.rowH || 0.4;
          const imgs = [];
          let x = o.x;
          for (let c = 0; c < row.length; c++) {
            const cw = o.colW[c], m = o.margin || 0.05;
            const ti = await textImg(row[c].text, row[c].options, cw - 2 * m);
            rowH = Math.max(rowH, ti.h / PX + 2 * m);
            imgs.push({ ti, x: x + m, fill: row[c].options.fill });
            x += cw;
          }
          let cx = o.x;
          for (let c = 0; c < row.length; c++) {
            svg += `<rect x="${cx * PX}" y="${y * PX}" width="${o.colW[c] * PX}" height="${rowH * PX}" fill="${imgs[c].fill ? "#" + hex(imgs[c].fill.color) : "#FFFFFF"}" stroke="#D7E3E2"/>`;
            comps.push({ input: imgs[c].ti.buf, left: Math.round(imgs[c].x * PX), top: Math.round((y + (rowH - imgs[c].ti.h / PX) / 2) * PX) });
            cx += o.colW[c];
          }
          y += rowH;
        }
        if (y > 6.85) problems.push(`slide ${si + 1}: table bottom ${y.toFixed(2)} in`);
      } else { // text or placeholder
        const ti = await textImg(it.text, o, o.w);
        const hIn = ti.h / PX;
        // Malgun Gothic vs the Noto preview font: demand ~10 % slack
        if (hIn * 1.1 > o.h + 0.02) problems.push(`slide ${si + 1}: "${(typeof it.text === "string" ? it.text : it.text.map((x) => x.text).join("")).slice(0, 30)}" needs ${(hIn * 1.1).toFixed(2)} in, box ${o.h} in`);
        if (o.x + o.w > W - 0.3 || o.y + o.h > H - 0.25) problems.push(`slide ${si + 1}: box past margin (${o.x}+${o.w}, ${o.y}+${o.h})`);
        let top = o.y;
        if (o.valign === "middle") top = o.y + (o.h - hIn) / 2;
        else if (o.valign === "bottom") top = o.y + o.h - hIn;
        let left = o.x;
        comps.push({ input: ti.buf, left: Math.round(left * PX), top: Math.max(0, Math.round(top * PX)) });
      }
    }
    svg += "</svg>";
    await sharp(Buffer.from(svg)).composite(comps).jpeg({ quality: 82 }).toFile(path.join(outDir, `slide-${String(si + 1).padStart(2, "0")}.jpg`));
  }
  return problems;
}

(async () => {
  await build();
  const out = process.argv[2] || "deck.pptx";
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  const probs = await preview(process.argv[3] || "preview");
  console.log("slides", REC.length, "| problems", probs.length);
  for (const p of probs) console.log("  " + p);
})();
