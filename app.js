const T = 0.8;
const DECISION_PROGRESS = 0.44;
const $ = (id) => document.getElementById(id);
const ui = {
  ball: $('ball'), cart: $('cart'), trail: $('trail'),
  status: $('round-status'), round: $('round-label'), count: $('count-label'), plot: $('plot'),
  wx: $('weight-x'), wv: $('weight-v'), bias: $('weight-b'), updateDelta: $('update-delta'),
  rate: $('rate'), rateValue: $('rate-value'), speed: $('speed'), speedValue: $('speed-value'),
  step: $('step'), continuous: $('continuous'), reset: $('reset')
};
const state = { weights: [0.12, -0.45, 0.16], updateFrom: null, rounds: [], next: null, playing: false, continuous: false, timer: null, run: 0 };
const svgNS = 'http://www.w3.org/2000/svg';

function svg(parent, name, attributes = {}, label) {
  const node = document.createElementNS(svgNS, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (label !== undefined) node.textContent = label;
  parent.appendChild(node);
  return node;
}

function formatted(n, digits = 2) { return `${n < 0 ? '−' : '+'}${Math.abs(n).toFixed(digits)}`; }
function sample() { return (Math.floor(Math.random() * 201) - 100) / 100; }
function ballX(x, v, progress) { return 130 + 100 * x + (300 + 100 * v) * T * progress; }
function ballY(progress) { return 32 + 217 * progress * progress; }
function plotX(x) { return 55 + (x + 1) / 2 * 420; }
function plotY(v) { return 315 - (v + 1) / 2 * 290; }

function newBall() {
  state.next = { x: sample(), v: sample() };
  ui.ball.setAttribute('cx', ballX(state.next.x, state.next.v, 0));
  ui.ball.setAttribute('cy', ballY(0));
  ui.trail.setAttribute('d', '');
  ui.cart.setAttribute('transform', 'translate(370 0)');
  ui.updateDelta.textContent = '';
  ui.status.textContent = `x = ${formatted(state.next.x)} · v = ${formatted(state.next.v)} · Ready`;
}

function decision(x, v) {
  const [wx, wv, b] = state.weights;
  return wx * x + wv * v + b >= 0 ? 1 : -1;
}

function showWeights() {
  const [wx, wv, b] = state.weights;
  ui.wx.textContent = `wₓ = ${formatted(wx, 3)}`;
  ui.wv.textContent = `wᵥ = ${formatted(wv, 3)}`;
  ui.bias.textContent = `b = ${formatted(b, 3)}`;
}

function completeRound(round, caught) {
  ui.status.textContent = `x = ${formatted(round.x)} · v = ${formatted(round.v)} · ${round.choice < 0 ? 'Left' : 'Right'} · ${caught ? 'Caught' : 'Missed → updated'}`;
  ui.round.textContent = `Round ${state.rounds.length + 1}`;
  ui.count.textContent = `${state.rounds.length} ${state.rounds.length === 1 ? 'round' : 'rounds'}`;
  state.updateFrom = null;
  state.next = null;
  state.playing = false;
  ui.step.disabled = false;
  drawPlot();
  showWeights();
  if (state.continuous) state.timer = setTimeout(() => { newBall(); play(); }, 400 / Number(ui.speed.value));
}

function finish(round) {
  const correct = round.landing >= 0 ? 1 : -1;
  const caught = round.choice === correct;
  state.rounds.push({ ...round, correct, caught });
  if (caught) { completeRound(round, true); return; }

  const rate = Number(ui.rate.value);
  const before = state.weights.slice();
  const delta = [rate * correct * round.x, rate * correct * round.v, rate * correct];
  const after = before.map((weight, i) => weight + delta[i]);
  state.updateFrom = before;
  ui.updateDelta.textContent = `Δ(wₓ, wᵥ, b) = (${delta.map(value => formatted(value, 3)).join(', ')})`;
  ui.status.textContent = `x = ${formatted(round.x)} · v = ${formatted(round.v)} · Missed · updating`;
  const run = state.run;
  const duration = matchMedia('(prefers-reduced-motion: reduce)').matches ? 100 : 650 / Number(ui.speed.value);
  const start = performance.now();
  function animate(now) {
    if (run !== state.run) return;
    const progress = Math.min(1, (now - start) / duration);
    state.weights = before.map((weight, i) => weight + delta[i] * progress);
    drawPlot();
    showWeights();
    if (progress < 1) requestAnimationFrame(animate);
    else { state.weights = after; completeRound(round, false); }
  }
  requestAnimationFrame(animate);
}

function play() {
  if (state.playing) return;
  clearTimeout(state.timer);
  if (!state.next) newBall();
  const { x, v } = state.next;
  const landing = x + T * v;
  let choice = null;
  ui.step.disabled = true;
  state.playing = true;
  const run = ++state.run;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const speed = Number(ui.speed.value);
  const duration = reduced ? 200 : 1800 / speed;
  const missFall = reduced ? 60 : 250 / speed;
  const start = performance.now();
  function frame(now) {
    if (run !== state.run) return;
    const elapsed = now - start;
    const p = Math.min(1, elapsed / duration);
    if (choice === null && p >= DECISION_PROGRESS) {
      choice = decision(x, v);
      ui.status.textContent = `x = ${formatted(x)} · v = ${formatted(v)} · ${choice < 0 ? 'Left' : 'Right'}`;
    }
    ui.ball.setAttribute('cx', ballX(x, v, p));
    ui.ball.setAttribute('cy', ballY(p));
    const segments = Math.max(1, Math.ceil(p * 30));
    const points = [];
    for (let i = 0; i <= segments; i++) {
      const t = p * i / segments;
      points.push(`${i === 0 ? 'M' : 'L'} ${ballX(x, v, t)} ${ballY(t)}`);
    }
    ui.trail.setAttribute('d', points.join(' '));
    const cartProgress = choice === null ? 0 : Math.min(1, (p - DECISION_PROGRESS) / 0.36);
    ui.cart.setAttribute('transform', `translate(${370 + (choice < 0 ? -90 : 90) * cartProgress} 0)`);
    if (p === 1 && choice !== (landing >= 0 ? 1 : -1)) {
      const missP = Math.min(1, (elapsed - duration) / missFall);
      ui.ball.setAttribute('cy', ballY(1) + 25 * missP);
      if (missP === 1) { finish({ x, v, landing, choice }); return; }
    } else if (p === 1) { finish({ x, v, landing, choice }); return; }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function lineEnds(wx, wv, b) {
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
  const score = (point) => side * (wx * point.x + wv * point.v + b);
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
  if (state.updateFrom) {
    const previous = lineEnds(...state.updateFrom);
    if (previous) svg(ui.plot, 'line', { x1: plotX(previous[0].x), y1: plotY(previous[0].v), x2: plotX(previous[1].x), y2: plotY(previous[1].v), stroke: '#adb5bd', 'stroke-width': 2, 'stroke-dasharray': '4 4' });
  }
  const ends = lineEnds(...state.weights);
  if (ends) svg(ui.plot, 'line', { x1: plotX(ends[0].x), y1: plotY(ends[0].v), x2: plotX(ends[1].x), y2: plotY(ends[1].v), stroke: '#0d6efd', 'stroke-width': 2 });
  for (const round of state.rounds) {
    const cx = plotX(round.x), cy = plotY(round.v);
    if (!round.caught) svg(ui.plot, 'circle', { cx, cy, r: 8, fill: 'none', stroke: '#dc3545', 'stroke-width': 2 });
    const dot = svg(ui.plot, 'circle', { cx, cy, r: 4, fill: round.correct < 0 ? '#6f42c1' : '#0f766e' });
    svg(dot, 'title', {}, `${round.correct < 0 ? 'Left' : 'Right'} landing; ${round.caught ? 'caught' : 'missed'}`);
  }
}

ui.step.addEventListener('click', play);
ui.continuous.addEventListener('click', () => {
  state.continuous = !state.continuous;
  ui.continuous.textContent = state.continuous ? 'Pause' : 'Run continuously';
  ui.continuous.setAttribute('aria-pressed', state.continuous);
  if (state.continuous && !state.playing) play();
  if (!state.continuous) clearTimeout(state.timer);
});
ui.reset.addEventListener('click', () => {
  ++state.run;
  clearTimeout(state.timer);
  state.weights = [0.12, -0.45, 0.16];
  state.updateFrom = null;
  state.rounds = [];
  state.playing = false;
  state.continuous = false;
  ui.step.disabled = false;
  ui.continuous.textContent = 'Run continuously';
  ui.continuous.setAttribute('aria-pressed', 'false');
  ui.round.textContent = 'Round 1';
  ui.count.textContent = '0 rounds';
  newBall();
  drawPlot();
  showWeights();
});
ui.rate.addEventListener('input', () => { ui.rateValue.textContent = Number(ui.rate.value).toFixed(2); });
ui.speed.addEventListener('input', () => { ui.speedValue.textContent = `${Number(ui.speed.value).toFixed(1)}×`; });

newBall();
drawPlot();
showWeights();
