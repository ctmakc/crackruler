// Record one live scene of the CrackRuler UI. Usage: node record.cjs '<json spec>'
// spec: {base, action, sample, dur, out_dir}. Prints JSON {video, mark} (mark = seconds into the
// recording where the scene starts).
const { chromium } = require('playwright');
const spec = JSON.parse(process.argv[2]);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
(async () => {
  const b = await chromium.launch();
  const ctx = await b.newContext({ viewport: { width: 1536, height: 864 }, deviceScaleFactor: 1, colorScheme: 'light',
    recordVideo: { dir: spec.out_dir, size: { width: 1536, height: 864 } } });
  const t0 = Date.now();
  const p = await ctx.newPage();
  const smooth = (y) => p.evaluate((y) => window.scrollTo({ top: y, behavior: 'smooth' }), y);
  const yOf = (sel) => p.evaluate((s) => document.querySelector(s).getBoundingClientRect().top + window.scrollY - 24, sel);
  await p.goto(spec.base + '/', { waitUntil: 'networkidle' });
  let mark, cut = null;
  const dur = spec.dur * 1000;
  if (spec.action === 'intro') {
    mark = (Date.now() - t0) / 1000;
    await sleep(1600);
    await p.hover('#drop'); await sleep(1500);
    await p.hover('.samples li:nth-child(1) button'); await sleep(900);
    await p.hover('.samples li:nth-child(2) button'); await sleep(900);
    await smooth(await yOf('#exp') - 120); await sleep(1400);
    for (const n of [1, 2, 3, 4, 5]) { await p.hover(`#exp label:nth-child(${n})`); await sleep(500); }
    await smooth(await yOf('.go') - 300); await sleep(1200);
    await p.hover('a[href="/static/marker-50mm.svg"]'); await sleep(800);
  } else if (spec.action === 'run' || spec.action === 'trace') {
    await p.click(`.samples button:has(img[src$="${spec.sample}"])`);
    if (spec.action === 'trace') {
      await p.click('#run');
      await p.waitForSelector('#result:not([hidden])', { timeout: 90000 });
      await sleep(1500);
      await p.evaluate(() => window.scrollTo(0, document.querySelector('.trace-wrap').getBoundingClientRect().top + window.scrollY - 40));
      await sleep(400);
      mark = (Date.now() - t0) / 1000;
      await sleep(700);
      await p.click('#replay'); await sleep(Math.max(1500, dur * 0.7));
    } else {
      mark = (Date.now() - t0) / 1000;
      await sleep(700);
      await smooth(await yOf('.go') - 520); await sleep(500);
      await p.click('#run');
      const tClick = Date.now();
      await p.waitForSelector('#result:not([hidden])', { timeout: 90000 });
      const tRes = Date.now();
      // the Lambda run takes ~15 s: keep 1.5 s of the busy state, cut the rest of the wait,
      // and extend the scene by the cut so the narration still fits
      if (tRes - tClick > 3000) {
        cut = [(tClick - t0) / 1000 + 1.5, (tRes - t0) / 1000 - 0.2];
      }
      await smooth(await yOf('#evidence') - 10);
      // stay on verdict + photo, then comparator, then cracks table
      const left = dur - (Date.now() - t0 - mark * 1000) + (cut ? (cut[1] - cut[0]) * 1000 : 0);
      await sleep(Math.max(1500, left * 0.42));
      await smooth(await yOf('.card-gauge') - 90);
      await sleep(Math.max(1200, left * 0.3));
      await smooth(await yOf('.table-wrap') - 330);
    }
  }
  const spent = Date.now() - t0 - mark * 1000 - (cut ? (cut[1] - cut[0]) * 1000 : 0);
  if (spent < dur + 600) await sleep(dur + 600 - spent);
  const video = await p.video().path();
  await ctx.close(); await b.close();
  console.log(JSON.stringify({ video, mark, cut }));
})();
