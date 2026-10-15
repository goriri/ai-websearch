"""Records scripted demo videos of the running app with headless Chrome.

It drives the UI over the Chrome DevTools Protocol (typing, clicking,
uploading, scrolling), captures frames with ``Page.startScreencast`` and
encodes them with ffmpeg into MP4 (and optionally GIF) at real-time speed.

Requirements: Google Chrome/Chromium, ffmpeg, and ``pip install websockets``.

Usage:
  python -m scripts.record_demo --url http://localhost:8080 --out demo
  python -m scripts.record_demo --only search-mode --gif
"""

import argparse
import asyncio
import base64
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import time
import urllib.request

import websockets


class Browser:
  """Minimal CDP client for one page."""

  def __init__(self, ws):
    self.ws = ws
    self._id = 0
    self._pending: dict[int, asyncio.Future] = {}
    self.frames: list[tuple[float, bytes]] = []
    self.recording = False
    self._reader = asyncio.create_task(self._read())

  async def _read(self):
    async for raw in self.ws:
      msg = json.loads(raw)
      if "id" in msg and msg["id"] in self._pending:
        self._pending.pop(msg["id"]).set_result(msg.get("result", msg))
      elif msg.get("method") == "Page.screencastFrame":
        p = msg["params"]
        if self.recording:
          self.frames.append((time.monotonic(), base64.b64decode(p["data"])))
        asyncio.create_task(self.call("Page.screencastFrameAck", sessionId=p["sessionId"]))
      elif msg.get("method") == "Runtime.exceptionThrown":
        print("  page exception:", msg["params"]["exceptionDetails"].get("text"))

  async def call(self, method, **params):
    self._id += 1
    fut = asyncio.get_running_loop().create_future()
    self._pending[self._id] = fut
    await self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
    return await fut

  async def js(self, expr, await_promise=True):
    r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True,
                        awaitPromise=await_promise)
    return r.get("result", {}).get("value")

  async def type(self, text, delay=0.045):
    for ch in text:
      await self.call("Input.insertText", text=ch)
      await asyncio.sleep(delay)

  async def wait_for(self, expr, timeout=180, interval=0.5):
    t = time.monotonic()
    while time.monotonic() - t < timeout:
      if await self.js(expr):
        return True
      await asyncio.sleep(interval)
    return False

  async def sleep(self, s):
    await asyncio.sleep(s)


# JS helpers ---------------------------------------------------------------
BUSY = "!!document.querySelector('.turn-meta .status')"
IDLE = "!document.querySelector('.turn-meta .status') && !!document.querySelector('.turn')"
FOLLOW = """(() => { const t=[...document.querySelectorAll('.turn')].pop(); if(!t) return;
  const bottom = t.getBoundingClientRect().bottom + scrollY;
  const target = Math.max(0, bottom - innerHeight + 220);
  if (target > scrollY + 4) scrollTo({top: target, behavior: 'smooth'}); })()"""


async def follow_until_done(b, timeout=180):
  t = time.monotonic()
  while time.monotonic() - t < timeout:
    await b.js(FOLLOW, await_promise=False)
    if await b.js(IDLE):
      return
    await asyncio.sleep(0.6)


async def smooth_scroll(b, start_selector=None, speed=520, fps=30):
  """Scrolls from the last turn's top to the page bottom at `speed` px/s."""
  start = await b.js(
      "(() => { const t=[...document.querySelectorAll('.turn')].pop();"
      " return t ? t.getBoundingClientRect().top + scrollY - 70 : 0; })()")
  end = await b.js("document.documentElement.scrollHeight - innerHeight")
  await b.js(f"scrollTo({{top:{start}, behavior:'instant'}})")
  await b.sleep(1.2)
  y = start
  step = speed / fps
  while y < end:
    y = min(end, y + step)
    await b.js(f"scrollTo({{top:{y}, behavior:'instant'}})", await_promise=False)
    await b.sleep(1 / fps)
  await b.sleep(1.5)


async def submit_query(b, text, mode=None):
  await b.js("document.querySelector('#q').focus()")
  if mode:
    await b.js(f"document.querySelector('.seg button[data-mode=\"{mode}\"]').click()")
    await b.sleep(0.6)
  await b.type(text)
  await b.sleep(0.5)
  await b.js("document.querySelector('#composer').requestSubmit()")


# Scenarios ----------------------------------------------------------------
async def scenario_search(b, url):
  await b.call("Page.navigate", url=url)
  await b.sleep(1.5)
  await submit_query(b, "Best beginner hikes in Zion National Park")
  await follow_until_done(b)
  await b.sleep(1)
  await smooth_scroll(b)


async def scenario_visual(b, url):
  await b.call("Page.navigate", url=url)
  await b.sleep(1.5)
  await submit_query(b, "How does a jet engine work?", mode="visual")
  await follow_until_done(b, timeout=240)
  await b.sleep(1)
  await smooth_scroll(b)


async def scenario_followup(b, url):
  await b.call("Page.navigate", url=url)
  await b.sleep(1.5)
  await submit_query(b, "Most iconic buildings designed by Zaha Hadid")
  await follow_until_done(b)
  await b.sleep(2)
  await b.js("document.querySelector('#q').focus()")
  await b.type("Which of these can I visit in Europe, and when are they open?")
  await b.sleep(0.4)
  await b.js("document.querySelector('#composer').requestSubmit()")
  await b.sleep(1.5)
  await follow_until_done(b)
  await b.sleep(1)
  await smooth_scroll(b)


async def scenario_image_query(b, url, image_path):
  await b.call("Page.navigate", url=url)
  await b.sleep(1.5)
  doc = await b.call("DOM.getDocument")
  node = await b.call("DOM.querySelector", nodeId=doc["root"]["nodeId"], selector="#fileInput")
  await b.call("DOM.setFileInputFiles", nodeId=node["nodeId"], files=[str(image_path)])
  await b.wait_for("!document.querySelector('#attachPreview').hidden", timeout=10)
  await b.sleep(1)
  await submit_query(b, "What is this building and can I go to the top?")
  await follow_until_done(b)
  await b.sleep(1)
  await smooth_scroll(b)


async def scenario_mobile(b, url):
  await b.call("Page.navigate", url=url)
  await b.sleep(1.5)
  await b.js("document.querySelector('.example').click()")  # First example chip.
  await follow_until_done(b)
  await b.sleep(1)
  await smooth_scroll(b, speed=700)


COMPARE_QUERY = "Most beautiful libraries in the world"
COMPARE_DONE = ("[...document.querySelectorAll('[data-metric=total]')]"
                ".every(e => /\\d+(\\.\\d+)?s/.test(e.textContent))")
FRAMES_FOLLOW = """(() => { for (const f of document.querySelectorAll('iframe.compare-frame')) {
  try { const w = f.contentWindow, d = w.document.documentElement;
        const t = Math.max(0, d.scrollHeight - w.innerHeight - 120);
        if (t > w.scrollY + 4) w.scrollTo({top: t, behavior: 'smooth'}); } catch (e) {} } })()"""


async def scenario_compare(b, url):
  """The /compare portal: one query, three architectures streaming side by side."""
  await b.call("Page.navigate", url=url.rstrip("/") + "/compare")
  await b.sleep(2)
  await b.js("document.querySelector('#compareInput').focus()")
  await b.type(COMPARE_QUERY)
  await b.sleep(0.6)
  await b.js("document.querySelector('#compareForm').requestSubmit()")
  t = time.monotonic()
  while time.monotonic() - t < 150:
    await b.js(FRAMES_FOLLOW, await_promise=False)
    if await b.js(COMPARE_DONE):
      break
    await asyncio.sleep(0.8)
  await b.sleep(2.5)  # Let the "best" badges and last images settle.
  # Synchronized smooth scroll through all three answers.
  await b.js("document.querySelectorAll('iframe.compare-frame').forEach(f => { try { f.contentWindow.scrollTo(0, 0); } catch (e) {} })")
  await b.sleep(1.5)
  fps, speed, y = 30, 650, 0
  while True:
    moved = await b.js(f"""(() => {{ let any = false; for (const f of document.querySelectorAll('iframe.compare-frame')) {{
        try {{ const w = f.contentWindow, max = w.document.documentElement.scrollHeight - w.innerHeight;
              if ({y} < max) {{ w.scrollTo(0, Math.min({y}, max)); any = true; }} }} catch (e) {{}} }} return any; }})()""")
    if not moved:
      break
    y += speed / fps
    await b.sleep(1 / fps)
  await b.sleep(2)


SCENARIOS = {
    "search-mode": (scenario_search, (1280, 800)),
    "visual-mode": (scenario_visual, (1280, 800)),
    "follow-up": (scenario_followup, (1280, 800)),
    "image-query": (scenario_image_query, (1280, 800)),
    "mobile": (scenario_mobile, (390, 844)),
    "compare": (scenario_compare, (1600, 900)),
}


def encode(frames, out_mp4, gif=None, width=None):
  """Writes frames with their real timing using ffmpeg's concat demuxer."""
  tmp = pathlib.Path(tempfile.mkdtemp())
  lines = []
  for i, (ts, data) in enumerate(frames):
    p = tmp / f"f{i:05d}.jpg"
    p.write_bytes(data)
    dur = (frames[i + 1][0] - ts) if i + 1 < len(frames) else 1.0
    lines += [f"file '{p}'", f"duration {max(dur, 0.001):.3f}"]
  lines.append(f"file '{tmp / f'f{len(frames) - 1:05d}.jpg'}'")
  (tmp / "list.txt").write_text("\n".join(lines))
  vf = "fps=30,pad=ceil(iw/2)*2:ceil(ih/2)*2"
  subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                  "-i", str(tmp / "list.txt"), "-vf", vf, "-c:v", "libx264",
                  "-preset", "slow", "-crf", "26", "-pix_fmt", "yuv420p",
                  "-movflags", "+faststart", str(out_mp4)], check=True)
  if gif:
    w = width or 720
    # 2x speed, palette-optimized GIF for embedding in README.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out_mp4), "-vf",
                    f"setpts=0.5*PTS,fps=10,scale={w}:-1:flags=lanczos,split[a][b];"
                    "[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    str(gif)], check=True)
  shutil.rmtree(tmp)


async def record(name, url, out_dir, image_path, gif, port):
  fn, (w, h) = SCENARIOS[name]
  prof = tempfile.mkdtemp()
  chrome = os.environ.get("CHROME", "google-chrome")
  proc = subprocess.Popen(
      [chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
       "--force-device-scale-factor=1", f"--remote-debugging-port={port}",
       f"--user-data-dir={prof}", f"--window-size={w},{h}", "about:blank"],
      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  try:
    for _ in range(100):
      try:
        tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
        break
      except OSError:
        await asyncio.sleep(0.1)
    ws_url = [t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"]
    async with websockets.connect(ws_url, max_size=2**28) as ws:
      b = Browser(ws)
      await b.call("Page.enable")
      await b.call("Runtime.enable")
      await b.call("Emulation.setDeviceMetricsOverride", width=w, height=h,
                   deviceScaleFactor=1 if w > 600 else 2, mobile=w < 600)
      await b.call("Page.startScreencast", format="jpeg", quality=82,
                   maxWidth=w * (1 if w > 600 else 2), maxHeight=h * (1 if w > 600 else 2))
      b.recording = True
      t0 = time.monotonic()
      if name == "image-query":
        await fn(b, url, image_path)
      else:
        await fn(b, url)
      b.recording = False
      await b.call("Page.stopScreencast")
      print(f"  {name}: {len(b.frames)} frames, {time.monotonic() - t0:.1f}s")
      out_dir.mkdir(parents=True, exist_ok=True)
      encode(b.frames, out_dir / f"{name}.mp4",
             gif=(out_dir / f"{name}.gif") if gif else None,
             width=960 if w > 1400 else (720 if w > 600 else 320))
  finally:
    proc.terminate()
    shutil.rmtree(prof, ignore_errors=True)


async def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--url", default="http://localhost:8080/")
  ap.add_argument("--out", default="demo")
  ap.add_argument("--only", nargs="*", choices=list(SCENARIOS))
  ap.add_argument("--image", default="demo/assets/query-photo.jpg",
                  help="Photo used by the image-query scenario.")
  ap.add_argument("--gif", action="store_true", help="Also write 2x-speed GIFs.")
  args = ap.parse_args()
  names = args.only or list(SCENARIOS)
  image = pathlib.Path(args.image).resolve()
  # Each scenario gets its own browser so they can run in parallel.
  await asyncio.gather(*(
      record(n, args.url, pathlib.Path(args.out), image, args.gif, 9400 + i)
      for i, n in enumerate(names)))


if __name__ == "__main__":
  asyncio.run(main())
