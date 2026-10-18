// Session analysis panel: stat cards, SVG charts, and per-browser run history (localStorage).
const Analysis = (() => {
  const STORE = "assessmentVision.runs.v1";
  const MAX_RUNS = 200;
  const CUBE = { blue: "#3f7ee8", green: "#34a853", red: "#e5484d", yellow: "#f2c94c" };
  const C = { text: "#ddd", muted: "#8a8a8a", grid: "#2c2c2c", axis: "#555", stack: "#4a7fd0", unstack: "#e39b3b", line: "#7fb3ff" };
  const PHASES = [["search_ms", "search", "#8f9bb0"], ["grasp_ms", "grasp", "#6c8ebf"], ["transport_ms", "transport", "#3f6fb0"], ["dwell_ms", "dwell", "#e39b3b"]];
  const NS = "http://www.w3.org/2000/svg";
  let current = null;

  const $ = (id) => document.getElementById(id);
  const sec = (ms) => (ms == null ? "–" : `${(ms / 1000).toFixed(1)} s`);
  const clock = (ms) => {
    if (ms == null) return "–";
    const s = ms / 1000;
    return `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
  };
  const median = (xs) => {
    const v = xs.filter((x) => x != null).sort((a, b) => a - b);
    if (!v.length) return null;
    const m = Math.floor(v.length / 2);
    return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
  };
  const shortTarget = (target) => target.map((c) => c[0].toUpperCase()).join("-");

  function el(tag, attrs = {}, parent = null, text = null) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text != null) node.textContent = text;
    if (parent) parent.append(node);
    return node;
  }

  function svg(width, height) {
    return el("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", role: "img", "font-family": "system-ui, sans-serif", "font-size": 12 });
  }

  function niceMax(value) {
    if (!(value > 0)) return 1;
    const step = 10 ** Math.floor(Math.log10(value));
    for (const m of [1, 2, 2.5, 5, 10]) if (m * step >= value) return m * step;
    return 10 * step;
  }

  function tickCount(max) {
    const round = (step) => {
      const unit = step / 10 ** Math.floor(Math.log10(step));
      return [1, 2, 2.5, 5].some((n) => Math.abs(unit - n) < 1e-9);
    };
    return [4, 5, 2].find((t) => round(max / t)) || 4;
  }

  function yAxis(root, x0, x1, y0, y1, max, ticks = tickCount(max), unit = "s") {
    for (let i = 0; i <= ticks; i++) {
      const y = y1 - ((y1 - y0) * i) / ticks;
      el("line", { x1: x0, x2: x1, y1: y, y2: y, stroke: C.grid }, root);
      el("text", { x: x0 - 6, y: y + 4, "text-anchor": "end", fill: C.muted }, root, `${+(max * i / ticks).toFixed(2)}${i === ticks ? " " + unit : ""}`);
    }
  }

  function empty(container, text) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = text;
    container.replaceChildren(p);
  }

  function legend(root, items, x, y) {
    let cx = x;
    for (const [label, color] of items) {
      el("rect", { x: cx, y: y - 9, width: 10, height: 10, rx: 2, fill: color }, root);
      el("text", { x: cx + 14, y, fill: C.muted }, root, label);
      cx += 22 + label.length * 6.5;
    }
  }

  // Horizontal stacked bars per round: stack | unstack, with target chips.
  function roundsChart(container, rounds) {
    if (!rounds.length) return empty(container, "Rounds appear here as you play.");
    const W = 460, rowH = 34, top = 26, left = 112, right = 54;
    const H = top + rounds.length * rowH + 26;
    const max = niceMax(Math.max(...rounds.map((r) => ((r.stack_ms || 0) + (r.unstack_ms || 0)) / 1000)));
    const scale = (v) => left + ((W - left - right) * v) / max;
    const root = svg(W, H);
    legend(root, [["stack", C.stack], ["unstack", C.unstack]], left, 14);
    const ticks = tickCount(max);
    for (let i = 0; i <= ticks; i++) {
      const x = scale((max * i) / ticks);
      el("line", { x1: x, x2: x, y1: top, y2: H - 22, stroke: C.grid }, root);
      el("text", { x, y: H - 6, "text-anchor": "middle", fill: C.muted }, root, `${+(max * i / ticks).toFixed(2)}${i === ticks ? " s" : ""}`);
    }
    rounds.forEach((r, i) => {
      const y = top + i * rowH + 6;
      el("text", { x: 4, y: y + 15, fill: C.text }, root, `R${r.round}`);
      r.target.forEach((color, k) => el("rect", { x: 30 + k * 18, y: y + 4, width: 14, height: 14, rx: 3, fill: CUBE[color] || "#888" }, root));
      const stack = (r.stack_ms || 0) / 1000, unstack = (r.unstack_ms || 0) / 1000;
      el("rect", { x: left, y, width: Math.max(0, scale(stack) - left), height: 22, rx: 3, fill: C.stack }, root);
      el("rect", { x: scale(stack), y, width: Math.max(0, scale(stack + unstack) - scale(stack)), height: 22, rx: 3, fill: C.unstack }, root);
      const label = r.round_ms != null ? sec(r.round_ms) : "in progress";
      el("text", { x: scale(stack + unstack) + 6, y: y + 15, fill: C.text }, root, label);
    });
    container.replaceChildren(root);
  }

  // Step time per tower position: placements S1..Sn-1 then removals Un-1..U1, one dot per round.
  function stepChart(container, steps) {
    const keyed = steps.filter((s) => s.phase && s.step_ms != null);
    if (!keyed.length) return empty(container, "Step times appear after the first placement.");
    const keys = [...new Set(keyed.map((s) => `${s.phase}:${s.index}`))].sort((a, b) => {
      const [pa, ia] = a.split(":"), [pb, ib] = b.split(":");
      if (pa !== pb) return pa === "stack" ? -1 : 1;
      return pa === "stack" ? ia - ib : ib - ia;
    });
    const W = 460, H = 220, left = 46, right = 12, top = 14, bottom = 190;
    const max = niceMax(Math.max(...keyed.map((s) => s.step_ms / 1000)));
    const root = svg(W, H);
    yAxis(root, left, W - right, top, bottom, max);
    const band = (W - left - right) / keys.length;
    keys.forEach((key, i) => {
      const [phase, index] = key.split(":");
      const cx = left + band * (i + 0.5);
      const values = keyed.filter((s) => `${s.phase}:${s.index}` === key);
      values.forEach((s, j) => {
        const y = bottom - ((bottom - top) * s.step_ms) / 1000 / max;
        const dot = el("circle", { cx: cx + (j - (values.length - 1) / 2) * 7, cy: y, r: 5, fill: CUBE[s.color] || "#888", stroke: "#111" }, root);
        el("title", {}, dot, `Round ${s.round}, ${s.color}: ${sec(s.step_ms)}`);
      });
      const m = median(values.map((s) => s.step_ms / 1000));
      const my = bottom - ((bottom - top) * m) / max;
      el("line", { x1: cx - band * 0.32, x2: cx + band * 0.32, y1: my, y2: my, stroke: C.text, "stroke-width": 2 }, root);
      el("text", { x: cx, y: H - 10, "text-anchor": "middle", fill: C.text }, root, `${phase === "stack" ? "S" : "U"}${index}`);
    });
    container.replaceChildren(root);
  }

  // Stacked phase bars per step (only steps with hand-tracked phases).
  function phaseChart(container, steps) {
    const rows = steps.filter((s) => PHASES.some(([k]) => s[k] != null));
    if (!rows.length) return empty(container, "Turn on hand tracking to see search, grasp, transport and dwell times.");
    const W = 460, H = 230, left = 46, right = 12, top = 30, bottom = 196;
    const total = (s) => PHASES.reduce((sum, [k]) => sum + (s[k] || 0), 0) / 1000;
    const max = niceMax(Math.max(...rows.map(total)));
    const root = svg(W, H);
    legend(root, PHASES.map(([, label, color]) => [label, color]), left, 14);
    yAxis(root, left, W - right, top, bottom, max);
    const band = (W - left - right) / rows.length;
    rows.forEach((s, i) => {
      let y = bottom;
      const x = left + band * i + band * 0.15;
      for (const [k, label, color] of PHASES) {
        const h = ((bottom - top) * (s[k] || 0)) / 1000 / max;
        const rect = el("rect", { x, y: y - h, width: band * 0.7, height: h, fill: color }, root);
        el("title", {}, rect, `${label}: ${sec(s[k])}`);
        y -= h;
      }
      if (rows.length <= 16 || i % 2 === 0) {
        const name = s.phase ? `${s.round}${s.phase === "stack" ? "S" : "U"}${s.index}` : (s.color || "")[0];
        el("text", { x: x + band * 0.35, y: H - 18, "text-anchor": "middle", fill: C.muted, "font-size": 10 }, root, name);
      }
    });
    container.replaceChildren(root);
  }

  // Strip plot of step times per finger pair, plus a compact table.
  function pairChart(container, pairs, ruleUsed) {
    if (!pairs.length) return empty(container, "No finger-pair data yet. Turn on hand tracking; pairs are detected automatically, or use the finger-pair rule to fix one.");
    const W = 460, rowH = 30, left = 104, right = 16, top = 8;
    const H = top + pairs.length * rowH + 24;
    const all = pairs.flatMap((p) => p.step_values || []);
    const max = niceMax(Math.max(1, ...all) / 1000);
    const scale = (ms) => left + ((W - left - right) * ms) / 1000 / max;
    const root = svg(W, H);
    const ticks = tickCount(max);
    for (let i = 0; i <= ticks; i++) {
      const x = scale((max * 1000 * i) / ticks);
      el("line", { x1: x, x2: x, y1: top, y2: H - 20, stroke: C.grid }, root);
      el("text", { x, y: H - 6, "text-anchor": "middle", fill: C.muted }, root, `${+(max * i / ticks).toFixed(2)}${i === ticks ? " s" : ""}`);
    }
    pairs.forEach((p, i) => {
      const y = top + i * rowH + rowH / 2;
      el("text", { x: 4, y: y + 4, fill: C.text }, root, p.pair);
      (p.step_values || []).forEach((v) => el("circle", { cx: scale(v), cy: y, r: 4.5, fill: C.line, "fill-opacity": 0.75 }, root));
      if (p.step_ms != null) el("line", { x1: scale(p.step_ms), x2: scale(p.step_ms), y1: y - 10, y2: y + 10, stroke: C.text, "stroke-width": 2 }, root);
    });
    const table = document.createElement("table");
    table.className = "pairs";
    table.innerHTML = "<thead><tr><th>pair</th><th>n</th><th>step</th><th>transport</th><th>dwell</th><th>rejected</th></tr></thead>";
    const body = document.createElement("tbody");
    for (const p of pairs) {
      const tr = document.createElement("tr");
      for (const value of [p.pair, p.n, sec(p.step_ms), sec(p.transport_ms), sec(p.dwell_ms), p.rejected]) {
        const td = document.createElement("td");
        td.textContent = value;
        tr.append(td);
      }
      body.append(tr);
    }
    table.append(body);
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = ruleUsed ? "Pairs are the ones required by the finger-pair rule." : "Pairs are the ones detected from your grip.";
    container.replaceChildren(root, table, note);
  }

  // Line chart of a metric across stored runs of the same game mode and difficulty.
  function progressionChart(container, runs, metric) {
    const points = runs.map((r, i) => ({ i, run: r, v: metric.value(r.summary) })).filter((p) => p.v != null);
    if (points.length < 2) return empty(container, `Play ${points.length ? "one more game" : "two games"} in this mode to see a progression curve.`);
    const W = 940, H = 230, left = 52, right = 16, top = 16, bottom = 196;
    const max = niceMax(Math.max(...points.map((p) => p.v / 1000)));
    const root = svg(W, H);
    yAxis(root, left, W - right, top, bottom, max);
    const x = (k) => left + ((W - left - right) * k) / Math.max(1, points.length - 1);
    const y = (v) => bottom - ((bottom - top) * v) / 1000 / max;
    const rolling = points.map((_, k) => median(points.slice(Math.max(0, k - 2), k + 1).map((p) => p.v)));
    el("polyline", { points: points.map((p, k) => `${x(k)},${y(p.v)}`).join(" "), fill: "none", stroke: C.axis, "stroke-width": 1.5 }, root);
    el("polyline", { points: rolling.map((v, k) => `${x(k)},${y(v)}`).join(" "), fill: "none", stroke: C.line, "stroke-width": 2.5 }, root);
    points.forEach((p, k) => {
      const dot = el("circle", { cx: x(k), cy: y(p.v), r: p.run.session_id === current?.session_id ? 6 : 4, fill: p.run.session_id === current?.session_id ? C.unstack : C.line }, root);
      el("title", {}, dot, `${new Date(p.run.started_at).toLocaleString()}: ${sec(p.v)}`);
      if (points.length <= 12 || k % Math.ceil(points.length / 12) === 0) {
        const d = new Date(p.run.started_at);
        el("text", { x: x(k), y: H - 12, "text-anchor": "middle", fill: C.muted, "font-size": 10 }, root, `${d.getDate()}.${d.getMonth() + 1}`);
      }
    });
    legend(root, [["each run", C.axis], ["rolling median (3)", C.line]], W - 260, 14);
    container.replaceChildren(root);
  }

  function cards(summary) {
    const q = summary.quality || {};
    const list = [["Total time", clock(summary.total_ms)]];
    if (summary.mode === "screwing") {
      const pct = (value) => (value == null ? "–" : `${Math.round(value * 100)}%`);
      list.push(
        ["Pick up", sec(summary.pickup_ms)],
        ["Screw on", sec(summary.screw_ms)],
        ["Unscrew", sec(summary.unscrew_ms)],
        ["Put down", sec(summary.putdown_ms)],
        ["Depth", pct(summary.depth)],
        ["Target", pct(summary.target)],
        ["Turns", summary.turns == null ? "–" : String(summary.turns)],
      );
    } else if (summary.mode === "random") {
      const done = summary.rounds.filter((r) => r.round_ms != null);
      list.push(
        ["Rounds", `${done.length}/${current.game.rounds}`],
        ["Median round", sec(summary.round_ms_median)],
        ["Best round", sec(summary.best_round_ms)],
        ["Wrong cubes", summary.rounds.reduce((s, r) => s + r.wrong_cubes, 0)],
        ["Rejected moves", summary.rounds.reduce((s, r) => s + r.rejected_moves, 0)],
      );
    } else {
      list.push(["First stack", sec(summary.stacks[0]?.achieved_ms)], ["Stacks", summary.stacks.length], ["Moves", q.moves]);
    }
    if (q.hand_coverage != null) list.push(["Hand seen", `${Math.round(q.hand_coverage * 100)}%`]);
    if (q.fps != null) list.push(["Frame rate", `${q.fps.toFixed(1)} fps`]);
    $("an-cards").replaceChildren(...list.map(([label, value]) => {
      const div = document.createElement("div");
      div.className = "tile";
      const b = document.createElement("b");
      b.textContent = label;
      div.append(b, document.createTextNode(value));
      return div;
    }));
  }

  function loadRuns() {
    try { return JSON.parse(localStorage.getItem(STORE)) || []; } catch { return []; }
  }

  function saveRun(record) {
    const runs = loadRuns().filter((r) => r.session_id !== record.session_id);
    runs.push(record);
    localStorage.setItem(STORE, JSON.stringify(runs.slice(-MAX_RUNS)));
  }

  function sameGame(a, b) {
    return a.game.mode === b.game.mode && a.game.difficulty === b.game.difficulty;
  }

  function metrics(record) {
    const list = [{ name: "Total time", value: (s) => (s.completed ? s.total_ms : null) }];
    if (record.game.mode === "random") list.push({ name: "Median round time", value: (s) => s.round_ms_median });
    const pairs = new Set(loadRuns().filter((r) => sameGame(r, record)).flatMap((r) => r.summary.pairs.map((p) => p.pair)));
    for (const pair of pairs) list.push({ name: `Median step, ${pair}`, value: (s) => s.pairs.find((p) => p.pair === pair)?.step_ms ?? null });
    return list;
  }

  function renderProgression() {
    if (!current) return;
    const list = metrics(current);
    const select = $("an-metric");
    const previous = select.value;
    select.replaceChildren(...list.map((m, i) => new Option(m.name, String(i))));
    select.value = previous !== "" && list[+previous] ? previous : "0";
    const runs = loadRuns().filter((r) => sameGame(r, current) && r.summary.completed);
    $("an-history").textContent = `${runs.length} stored ${current.game.mode} / ${current.game.difficulty.toLowerCase()} run${runs.length === 1 ? "" : "s"} in this browser`;
    progressionChart($("an-progress"), runs, list[+select.value]);
  }

  function update(record, final) {
    current = record;
    const s = record.summary;
    $("analysis").hidden = false;
    const modeTitle = s.mode === "screwing" ? "Screwing" : `${s.mode === "random" ? "Random" : "Simple"} stacking`;
    $("an-title").textContent = `${modeTitle} · ${record.game.difficulty.toLowerCase()}${s.completed ? " · finished" : " · in progress"}`;
    cards(s);
    const random = s.mode === "random";
    $("an-rounds-card").hidden = !random;
    $("an-steps-card").hidden = !random;
    if (random) {
      roundsChart($("an-rounds"), s.rounds);
      stepChart($("an-steps"), s.steps);
    }
    phaseChart($("an-phases"), s.steps);
    pairChart($("an-pairs"), s.pairs, s.finger_rule_used);
    if (final) saveRun(record);
    renderProgression();
  }

  function init() {
    $("an-metric").onchange = renderProgression;
    $("an-export").onclick = () => {
      const blob = new Blob([JSON.stringify(loadRuns(), null, 1)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `assessment-runs-${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      URL.revokeObjectURL(a.href);
    };
    $("an-clear").onclick = () => {
      if (confirm("Delete all stored runs in this browser?")) {
        localStorage.removeItem(STORE);
        renderProgression();
      }
    };
  }

  return { init, update, loadRuns };
})();
