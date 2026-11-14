const puppeteer = require('puppeteer-core');
(async () => {
  const out = process.argv[2], nFrames = +process.argv[3], stepPer = +process.argv[4];
  const browser = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: 'new',
    args: ['--no-sandbox', '--use-gl=angle', '--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist', '--window-size=1600,1160'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 1160 });
  await page.evaluateOnNewDocument(() => {  // keep the WebGL frame for screenshots (viewer file unchanged)
    const orig = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, attrs) {
      if (type === 'webgl' || type === 'webgl2' || type === 'experimental-webgl') attrs = Object.assign({}, attrs, { preserveDrawingBuffer: true });
      return orig.call(this, type, attrs);
    };
  });
  page.on('console', m => { if (m.type() === 'error') console.log('console:', m.text()); });
  await page.goto('http://127.0.0.1:8767/tools/rl_replay/viewer.html?data=/logs/rl_replay/deck_3906_gap.json', { waitUntil: 'load' });
  await page.waitForFunction(() => typeof D !== 'undefined' && D && E, { timeout: 120000 });
  await new Promise(r => setTimeout(r, 4000));
  await page.select('#env', process.argv[5] || '0');
  await new Promise(r => setTimeout(r, 1000));
  for (let k = 0; k < 2; k++) await page.keyboard.press('ArrowRight');
  await page.evaluate(() => {  // the viewer camera starts at the world origin; put it beside the robot (follow keeps the offset)
    const r = R3.root.position; R3.camera.position.set(r.x - 1.3, r.y - 1.6, r.z + 0.8);
    R3.controls.target.set(r.x, r.y, r.z - 0.1); R3.controls.update(); R3.last = null; draw();
  });
  await new Promise(r => setTimeout(r, 1000));
  for (let i = 0; i < nFrames; i++) {
    await page.screenshot({ path: `${out}/f${String(i).padStart(4, '0')}.jpg`, type: 'jpeg', quality: 85 });
    for (let k = 0; k < stepPer; k++) await page.keyboard.press('ArrowRight');
    await new Promise(r => setTimeout(r, 30));
  }
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
