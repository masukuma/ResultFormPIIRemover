// The read-only page viewer (rfpii_nn/view.py): one page at a time, its
// labels drawn over it. Nothing here writes anything anywhere.
"use strict";

const COLORS = {
  signature: "#e0201b", id_number: "#18a018", phone_number: "#1f4fe0", name: "#f08c00",
  stamp: "#b000c0", results_region: "#00a8b0", agents_table: "#7a7a00", declaration: "#6e6e6e",
  comments: "#8a4b00",
};
const LEAK = "#ff0078";
const MASK_COLORS = [[1, [150, 150, 150]], [4, [190, 0, 200]], [2, [20, 60, 255]], [8, [0, 0, 0]]];

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const state = {
  pages: [], shown: [], at: 0, labels: null, img: null, mask: null,
  zoom: 1, panX: 0, panY: 0, hidden: new Set(), selected: -1, filter: "all",
};

function api(path) {
  return fetch(path, { credentials: "same-origin" }).then((r) => {
    if (!r.ok) throw new Error(path + ": " + r.status);
    return r;
  });
}

function loadImage(url) {
  return new Promise((ok, fail) => {
    const im = new Image();
    im.onload = () => ok(im);
    im.onerror = () => fail(new Error("cannot load " + url));
    im.src = url;
  });
}

function leakOf(b) {
  const s = b.redaction && b.redaction.status;
  return s && s !== "masked_ok" ? s : null;
}

function buildFilter() {
  const sel = $("filter");
  const opts = [["all", "all"], ["leaks", "with a leak"], ["held", "held"], ["flags", "with flags"]];
  for (const c of Object.keys(COLORS)) opts.push(["class:" + c, "with " + c]);
  for (const [v, t] of opts) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = t;
    sel.appendChild(o);
  }
  sel.addEventListener("change", () => {
    state.filter = sel.value;
    applyFilter();
    show(0);
  });
}

function applyFilter() {
  const f = state.filter;
  state.shown = state.pages.filter((p) => {
    if (f === "all") return true;
    if (f === "leaks") return p.leaks > 0;
    if (f === "held") return p.held;
    if (f === "flags") return p.flags > 0;
    if (f.startsWith("class:")) return (p.classes[f.slice(6)] || 0) > 0;
    return true;
  });
}

function buildClasses() {
  const box = $("classes");
  for (const [c, col] of Object.entries(COLORS)) {
    const l = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = true;
    cb.addEventListener("change", () => {
      if (cb.checked) state.hidden.delete(c); else state.hidden.add(c);
      draw();
    });
    const sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = col;
    l.append(cb, sw, document.createTextNode(" " + c));
    box.appendChild(l);
  }
}

function fit() {
  const st = $("stage");
  canvas.width = st.clientWidth;
  canvas.height = st.clientHeight;
  if (!state.img) return;
  state.zoom = Math.min(canvas.width / state.img.width, canvas.height / state.img.height);
  state.panX = (canvas.width - state.img.width * state.zoom) / 2;
  state.panY = (canvas.height - state.img.height * state.zoom) / 2;
  draw();
}

function colorMask(im) {
  const off = document.createElement("canvas");
  off.width = im.width;
  off.height = im.height;
  const c = off.getContext("2d");
  c.drawImage(im, 0, 0);
  const d = c.getImageData(0, 0, off.width, off.height);
  const px = d.data;
  for (let i = 0; i < px.length; i += 4) {
    const v = px[i];
    let rgb = null;
    for (const [bit, col] of MASK_COLORS) if (v & bit) rgb = col;
    if (rgb) { px[i] = rgb[0]; px[i + 1] = rgb[1]; px[i + 2] = rgb[2]; px[i + 3] = 255; } else px[i + 3] = 0;
  }
  c.putImageData(d, 0, 0);
  return off;
}

function draw() {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.img) return;
  ctx.setTransform(state.zoom, 0, 0, state.zoom, state.panX, state.panY);
  ctx.drawImage(state.img, 0, 0);
  if ($("mask").checked && state.mask) {
    ctx.globalAlpha = 0.55;
    ctx.drawImage(state.mask, 0, 0);
    ctx.globalAlpha = 1;
  }
  if (!$("boxes").checked || !state.labels) return;
  const boxes = state.labels.boxes || [];
  boxes.forEach((b, i) => {
    if (state.hidden.has(b.class)) return;
    const [x0, y0, x1, y1] = b.box;
    const leak = leakOf(b);
    ctx.strokeStyle = leak ? LEAK : (COLORS[b.class] || "#000");
    ctx.lineWidth = (i === state.selected ? 5 : leak ? 4 : b.empty ? 1 : 2) / state.zoom;
    ctx.setLineDash(b.empty ? [6 / state.zoom, 4 / state.zoom] : []);
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    if (b.slots) {
      ctx.setLineDash([]);
      ctx.lineWidth = 2 / state.zoom;
      for (const [a, z] of b.slots) {
        ctx.beginPath();
        ctx.moveTo(a, y1 + 3 / state.zoom);
        ctx.lineTo(z, y1 + 3 / state.zoom);
        ctx.stroke();
      }
    }
  });
  for (const f of state.labels.flags || []) {
    if (!f.box) continue;
    const [x0, y0, x1, y1] = f.box;
    ctx.setLineDash([3 / state.zoom, 3 / state.zoom]);
    ctx.strokeStyle = "#ffbf00";
    ctx.lineWidth = 3 / state.zoom;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  }
  ctx.setLineDash([]);
}

function fact(dl, k, v) {
  if (v === null || v === undefined || v === "") return;
  const dt = document.createElement("dt");
  dt.textContent = k;
  const dd = document.createElement("dd");
  dd.textContent = typeof v === "object" ? JSON.stringify(v) : String(v);
  dl.append(dt, dd);
}

function describe(meta) {
  const l = state.labels;
  const pg = l.page || {};
  const dl = $("facts");
  dl.replaceChildren();
  fact(dl, "id", l.id);
  fact(dl, "dataset", meta.dataset);
  fact(dl, "kind", pg.kind);
  fact(dl, "template", pg.template);
  fact(dl, "size", pg.width + " x " + pg.height);
  fact(dl, "turn", pg.turn);
  fact(dl, "capture", pg.capture);
  fact(dl, "split", meta.split);
  fact(dl, "held", pg.held ? (pg.reason || "yes") : null);
  if (l.redaction) fact(dl, "redaction", l.redaction.leaks + " leak(s), " + (l.redaction.perturbed || []).length + " mask(s) cut back");
  const fl = $("flags");
  fl.replaceChildren();
  const flags = l.flags || [];
  if (!flags.length) {
    const li = document.createElement("li");
    li.className = "none";
    li.textContent = "none";
    fl.appendChild(li);
  }
  for (const f of flags) {
    const li = document.createElement("li");
    li.textContent = (f.kind || "flag") + (f.note ? ": " + f.note : "");
    fl.appendChild(li);
  }
  const ol = $("list");
  ol.replaceChildren();
  (l.boxes || []).forEach((b, i) => {
    const li = document.createElement("li");
    const leak = leakOf(b);
    if (leak) li.className = "leak";
    li.textContent = b.class + (b.zone ? " " + b.zone : "") + (b.role ? " (" + b.role + ")" : "") + (b.empty ? " empty" : "") + (leak ? " " + leak : "") + (b.source && b.source !== "synthetic" ? " [" + b.source + "]" : "");
    li.addEventListener("click", () => {
      state.selected = state.selected === i ? -1 : i;
      for (const x of ol.children) x.classList.remove("sel");
      if (state.selected === i) li.classList.add("sel");
      draw();
    });
    ol.appendChild(li);
  });
}

async function show(k) {
  if (!state.shown.length) {
    $("where").textContent = "0 / 0";
    state.img = null;
    draw();
    return;
  }
  state.at = (k + state.shown.length) % state.shown.length;
  const meta = state.shown[state.at];
  $("where").textContent = state.at + 1 + " / " + state.shown.length;
  state.selected = -1;
  const [labels, img] = await Promise.all([
    api("/api/page/" + meta.n + "/labels").then((r) => r.json()),
    loadImage("/api/page/" + meta.n + "/image"),
  ]);
  state.labels = labels;
  state.img = img;
  state.mask = null;
  if (meta.has_mask) {
    loadImage("/api/page/" + meta.n + "/mask").then((m) => { state.mask = colorMask(m); draw(); }).catch(() => {});
  }
  describe(meta);
  fit();
}

function zoomAt(f, cx, cy) {
  state.panX = cx - (cx - state.panX) * f;
  state.panY = cy - (cy - state.panY) * f;
  state.zoom *= f;
  draw();
}

function wire() {
  $("prev").addEventListener("click", () => show(state.at - 1));
  $("next").addEventListener("click", () => show(state.at + 1));
  $("fit").addEventListener("click", fit);
  $("mask").addEventListener("change", draw);
  $("boxes").addEventListener("change", draw);
  window.addEventListener("resize", fit);
  const st = $("stage");
  let drag = null;
  st.addEventListener("mousedown", (e) => { drag = [e.clientX, e.clientY]; st.classList.add("dragging"); });
  window.addEventListener("mouseup", () => { drag = null; st.classList.remove("dragging"); });
  window.addEventListener("mousemove", (e) => {
    if (!drag) return;
    state.panX += e.clientX - drag[0];
    state.panY += e.clientY - drag[1];
    drag = [e.clientX, e.clientY];
    draw();
  });
  st.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = canvas.getBoundingClientRect();
    zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - r.left, e.clientY - r.top);
  }, { passive: false });
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    const k = e.key;
    if (k === "ArrowRight" || k === "j") show(state.at + 1);
    else if (k === "ArrowLeft" || k === "k") show(state.at - 1);
    else if (k === "+" || k === "=") zoomAt(1.25, canvas.width / 2, canvas.height / 2);
    else if (k === "-") zoomAt(0.8, canvas.width / 2, canvas.height / 2);
    else if (k === "0") fit();
    else if (k === "m") { $("mask").checked = !$("mask").checked; draw(); }
    else if (k === "b") { $("boxes").checked = !$("boxes").checked; draw(); }
  });
}

async function start() {
  buildClasses();
  buildFilter();
  wire();
  const r = await api("/api/pages").then((x) => x.json());
  $("title").textContent = r.title;
  document.title = r.title;
  state.pages = r.pages;
  applyFilter();
  show(0);
}

start().catch((e) => { $("title").textContent = "error: " + e.message; });
