const T = 0.8;
const TRAIN = [
  [-0.85, -0.60], [-0.75, 0.30], [-0.60, 0.65], [-0.50, -0.15],
  [-0.35, 0.10], [-0.25, -0.80], [-0.10, -0.25], [0.10, -0.65],
  [-0.45, 0.85], [-0.15, 0.55], [0.05, 0.20], [0.25, -0.10],
  [0.45, -0.35], [0.55, 0.30], [0.75, -0.45], [0.85, 0.70]
].map(([x, v]) => ({ x, v, y: x + T * v >= 0 ? 1 : -1 }));
const TEST = [
  [-0.90, 0.75], [-0.55, 0.45], [-0.15, -0.45], [0.15, -0.55],
  [-0.30, 0.65], [0.15, 0.10], [0.50, -0.20], [0.85, -0.60]
].map(([x, v]) => ({ x, v, y: x + T * v >= 0 ? 1 : -1 }));
const INITIAL_WEIGHTS = [0, -0.35, 0];
const $ = id => document.getElementById(id);
const ui = {
  plot: $('plot'), epoch: $('epoch-label'), sample: $('sample-label'),
  trainAccuracy: $('train-accuracy'), testAccuracy: $('test-accuracy'), updates: $('update-count'),
  status: $('example-status'), delta: $('update-delta'),
  wx: $('weight-x'), wv: $('weight-v'), bias: $('weight-b'),
  rate: $('rate'), rateValue: $('rate-value'), speed: $('speed'), speedValue: $('speed-value'),
  step: $('step'), continuous: $('continuous'), reset: $('reset')
};
const state = { weights: INITIAL_WEIGHTS.slice(), previous: null, index: 0, epoch: 1, updates: 0, continuous: false, timer: null };
const svgNS = 'http://www.w3.org/2000/svg';

function svg(parent, name, attributes = {}, label) {
  const node = document.createElementNS(svgNS, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (label !== undefined) node.textContent = label;
  parent.appendChild(node);
  return node;
}
function signed(n, digits = 2) { return `${n < 0 ? '−' : '+'}${Math.abs(n).toFixed(digits)}`; }
function plotX(x) { return 55 + (x + 1) * 210; }
function plotY(v) { return 315 - (v + 1) * 145; }
function predict(point) {
  const [wx, wv, b] = state.weights;
  return wx * point.x + wv * point.v + b >= 0 ? 1 : -1;
}
function accuracy(points) { return Math.round(100 * points.filter(point => predict(point) === point.y).length / points.length); }

function lineEnds(weights) {
  const [wx, wv, b] = weights;
  const points = [];
  if (Math.abs(wv) > 1e-9) for (const x of [-1, 1]) {
    const v = (-b - wx * x) / wv;
    if (v >= -1 && v <= 1) points.push({ x, v });
  }
  if (Math.abs(wx) > 1e-9) for (const v of [-1, 1]) {
    const x = (-b - wv * v) / wx;
    if (x >= -1 && x <= 1 && !points.some(p => Math.abs(p.x - x) < 1e-8 && Math.abs(p.v - v) < 1e-8)) points.push({ x, v });
  }
  return points.length >= 2 ? points : null;
}
function decisionRegion(side) {
  const [wx, wv, b] = state.weights;
  const score = point => side * (wx * point.x + wv * point.v + b);
  const square = [{ x: -1, v: -1 }, { x: 1, v: -1 }, { x: 1, v: 1 }, { x: -1, v: 1 }];
  const clipped = [];
  for (let i = 0; i < square.length; i++) {
    const a = square[i], next = square[(i + 1) % square.length];
    const sa = score(a), sb = score(next);
    if (sa >= 0) clipped.push(a);
    if ((sa < 0 && sb > 0) || (sa > 0 && sb < 0)) {
      const t = sa / (sa - sb);
      clipped.push({ x: a.x + t * (next.x - a.x), v: a.v + t * (next.v - a.v) });
    }
  }
  return clipped.map(point => `${plotX(point.x)},${plotY(point.v)}`).join(' ');
}
function drawPlot() {
  ui.plot.replaceChildren();
  svg(ui.plot, 'rect', { x: 55, y: 25, width: 420, height: 290, fill: '#fff', stroke: '#adb5bd' });
  svg(ui.plot, 'polygon', { points: decisionRegion(-1), fill: '#f7f4fc' });
  svg(ui.plot, 'polygon', { points: decisionRegion(1), fill: '#f2faf9' });
  svg(ui.plot, 'line', { x1: plotX(0), y1: 25, x2: plotX(0), y2: 315, stroke: '#dee2e6' });
  svg(ui.plot, 'line', { x1: 55, y1: plotY(0), x2: 475, y2: plotY(0), stroke: '#dee2e6' });
  for (const n of [-1, 0, 1]) {
    svg(ui.plot, 'text', { x: plotX(n), y: 336, 'text-anchor': 'middle', fill: '#6c757d', 'font-size': 11 }, String(n));
    svg(ui.plot, 'text', { x: 47, y: plotY(n) + 4, 'text-anchor': 'end', fill: '#6c757d', 'font-size': 11 }, String(n));
  }
  svg(ui.plot, 'text', { x: 485, y: 337, fill: '#6c757d', 'font-size': 12 }, 'x');
  svg(ui.plot, 'text', { x: 25, y: 18, fill: '#6c757d', 'font-size': 12 }, 'v');
  if (state.previous) {
    const previous = lineEnds(state.previous);
    if (previous) svg(ui.plot, 'line', { x1: plotX(previous[0].x), y1: plotY(previous[0].v), x2: plotX(previous[1].x), y2: plotY(previous[1].v), stroke: '#adb5bd', 'stroke-width': 2, 'stroke-dasharray': '4 4' });
  }
  const ends = lineEnds(state.weights);
  if (ends) svg(ui.plot, 'line', { x1: plotX(ends[0].x), y1: plotY(ends[0].v), x2: plotX(ends[1].x), y2: plotY(ends[1].v), stroke: '#0d6efd', 'stroke-width': 2 });
  for (const point of TRAIN) {
    const dot = svg(ui.plot, 'circle', { cx: plotX(point.x), cy: plotY(point.v), r: 5, fill: point.y < 0 ? '#6f42c1' : '#0f766e' });
    svg(dot, 'title', {}, `Train · x = ${signed(point.x)} · v = ${signed(point.v)} · ${point.y < 0 ? 'left' : 'right'}`);
  }
  for (const point of TEST) {
    const dot = svg(ui.plot, 'circle', { cx: plotX(point.x), cy: plotY(point.v), r: 5, fill: '#fff', stroke: point.y < 0 ? '#6f42c1' : '#0f766e', 'stroke-width': 2 });
    svg(dot, 'title', {}, `Test · x = ${signed(point.x)} · v = ${signed(point.v)} · ${point.y < 0 ? 'left' : 'right'}`);
  }
  const next = TRAIN[state.index];
  svg(ui.plot, 'circle', { cx: plotX(next.x), cy: plotY(next.v), r: 10, fill: 'none', stroke: '#0d6efd', 'stroke-width': 1.5 });
}
function render() {
  ui.epoch.textContent = `Epoch ${state.epoch}`;
  ui.sample.textContent = `Example ${state.index + 1} / ${TRAIN.length}`;
  ui.trainAccuracy.textContent = `${accuracy(TRAIN)}%`;
  ui.testAccuracy.textContent = `${accuracy(TEST)}%`;
  ui.updates.textContent = String(state.updates);
  ui.wx.textContent = `wₓ = ${signed(state.weights[0], 3)}`;
  ui.wv.textContent = `wᵥ = ${signed(state.weights[1], 3)}`;
  ui.bias.textContent = `b = ${signed(state.weights[2], 3)}`;
  drawPlot();
}
function step() {
  clearTimeout(state.timer);
  const point = TRAIN[state.index];
  const choice = predict(point);
  state.previous = null;
  if (choice !== point.y) {
    const rate = Number(ui.rate.value);
    const delta = [rate * point.y * point.x, rate * point.y * point.v, rate * point.y];
    state.previous = state.weights.slice();
    state.weights = state.weights.map((weight, i) => weight + delta[i]);
    state.updates++;
    ui.delta.textContent = `Δ(wₓ, wᵥ, b) = (${delta.map(value => signed(value, 3)).join(', ')})`;
  } else ui.delta.textContent = '';
  ui.status.textContent = `Example ${state.index + 1}: ${choice < 0 ? 'left' : 'right'} · ${choice === point.y ? 'correct' : 'mistake → updated'}`;
  state.index++;
  if (state.index === TRAIN.length) { state.index = 0; state.epoch++; }
  render();
  if (state.continuous) state.timer = setTimeout(step, 700 / Number(ui.speed.value));
}

ui.step.addEventListener('click', step);
ui.continuous.addEventListener('click', () => {
  state.continuous = !state.continuous;
  ui.continuous.textContent = state.continuous ? 'Pause' : 'Run continuously';
  ui.continuous.setAttribute('aria-pressed', String(state.continuous));
  if (state.continuous) { clearTimeout(state.timer); step(); }
  else clearTimeout(state.timer);
});
ui.reset.addEventListener('click', () => {
  clearTimeout(state.timer);
  state.weights = INITIAL_WEIGHTS.slice();
  state.previous = null;
  state.index = 0;
  state.epoch = 1;
  state.updates = 0;
  state.continuous = false;
  ui.continuous.textContent = 'Run continuously';
  ui.continuous.setAttribute('aria-pressed', 'false');
  ui.status.textContent = '';
  ui.delta.textContent = '';
  render();
});
ui.rate.addEventListener('input', () => { ui.rateValue.textContent = Number(ui.rate.value).toFixed(2); });
ui.speed.addEventListener('input', () => { ui.speedValue.textContent = `${Number(ui.speed.value).toFixed(1)}×`; });

render();
