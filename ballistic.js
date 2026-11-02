const G = 9.8;
const RANGE_SPLIT = 13.5;
const PIXELS_PER_UNIT = 310 / RANGE_SPLIT;
const LAUNCH_X = 40;
const GROUND_Y = 249;
const DECISION_X = 98;
const $ = (id) => document.getElementById(id);
const ui = {
  ball: $('ball'), barrel: $('barrel'), cart: $('cart'), trail: $('trail'),
  status: $('round-status'), round: $('round-label'), count: $('count-label'), plot: $('plot'),
  wa: $('weight-angle'), wu: $('weight-speed'), bias: $('weight-bias'), delta: $('update-delta'),
  rate: $('rate'), rateValue: $('rate-value'), speed: $('speed'), speedValue: $('speed-value'),
  step: $('step'), continuous: $('continuous'), reset: $('reset')
};
const state = { weights: [0, 0.45, -0.05], updateFrom: null, rounds: [], next: null, playing: false, continuous: false, timer: null, run: 0 };
const svgNS = 'http://www.w3.org/2000/svg';

function svg(parent, name, attributes = {}, label) {
  const node = document.createElementNS(svgNS, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (label !== undefined) node.textContent = label;
  parent.appendChild(node);
  return node;
}

function signed(n, digits = 2) { return `${n < 0 ? '−' : '+'}${Math.abs(n).toFixed(digits)}`; }
function radians(degrees) { return degrees * Math.PI / 180; }
function range(angle, speed) { return speed * speed * Math.sin(2 * radians(angle)) / G; }
function features(angle, speed) { return [(angle - 45) / 25, (speed - 12) / 2]; }
function plotX(angle) { return 55 + (angle - 20) / 50 * 420; }
function plotY(speed) { return 315 - (speed - 10) / 4 * 290; }
function landingX(round) { return LAUNCH_X + PIXELS_PER_UNIT * round.range; }
function position(round, progress) {
  const t = round.flight * progress;
  return {
    x: LAUNCH_X + PIXELS_PER_UNIT * round.speed * Math.cos(round.radians) * t,
    y: GROUND_Y - PIXELS_PER_UNIT * (round.speed * Math.sin(round.radians) * t - G * t * t / 2)
  };
}

function newBall() {
  const angle = 20 + Math.floor(Math.random() * 101) / 2;
  const speed = 10 + Math.floor(Math.random() * 81) / 20;
  const theta = radians(angle);
  state.next = { angle, speed, radians: theta, flight: 2 * speed * Math.sin(theta) / G, range: range(angle, speed) };
  ui.ball.setAttribute('cx', LAUNCH_X);
  ui.ball.setAttribute('cy', GROUND_Y);
  ui.barrel.setAttribute('x2', LAUNCH_X + 25 * Math.cos(theta));
  ui.barrel.setAttribute('y2', GROUND_Y - 25 * Math.sin(theta));
  ui.cart.setAttribute('transform', 'translate(350 0)');
  ui.trail.setAttribute('d', '');
  ui.delta.textContent = '';
  ui.status.textContent = `θ = ${angle.toFixed(1)}° · v = ${speed.toFixed(2)} · Ready`;
}

function decision(angle, speed) {
  const [a, u] = features(angle, speed);
  const [wa, wu, bias] = state.weights;
  return wa * a + wu * u + bias >= 0 ? 1 : -1;
}

function showWeights() {
  const [wa, wu, bias] = state.weights;
  ui.wa.textContent = `wₐ = ${signed(wa, 3)}`;
  ui.wu.textContent = `wᵤ = ${signed(wu, 3)}`;
  ui.bias.textContent = `b = ${signed(bias, 3)}`;
}

function completeRound(round, caught) {
  state.updateFrom = null;
  state.next = null;
  state.playing = false;
  ui.step.disabled = false;
  ui.status.textContent = `θ = ${round.angle.toFixed(1)}° · v = ${round.speed.toFixed(2)} · ${round.choice < 0 ? 'Left' : 'Right'} · ${caught ? 'Caught' : 'Missed → updated'}`;
  ui.round.textContent = `Round ${state.rounds.length + 1}`;
  ui.count.textContent = `${state.rounds.length} ${state.rounds.length === 1 ? 'round' : 'rounds'}`;
  drawPlot();
  showWeights();
  if (state.continuous) state.timer = setTimeout(() => { newBall(); play(); }, 350 / Number(ui.speed.value));
}

function finish(round) {
  const correct = round.range >= RANGE_SPLIT ? 1 : -1;
  const caught = round.choice === correct;
  state.rounds.push({ ...round, correct, caught });
  if (caught) { completeRound(round, true); return; }

  const rate = Number(ui.rate.value);
  const [a, u] = features(round.angle, round.speed);
  const before = state.weights.slice();
  const delta = [rate * correct * a, rate * correct * u, rate * correct];
  const after = before.map((weight, i) => weight + delta[i]);
  state.updateFrom = before;
  ui.delta.textContent = `Δ(wₐ, wᵤ, b) = (${delta.map(value => signed(value, 3)).join(', ')})`;
  ui.status.textContent = `θ = ${round.angle.toFixed(1)}° · v = ${round.speed.toFixed(2)} · Missed · updating`;
  const run = state.run;
  const duration = matchMedia('(prefers-reduced-motion: reduce)').matches ? 100 : 650 / Number(ui.speed.value);
  const start = performance.now();
  function animate(now) {
    if (run !== state.run) return;
    const p = Math.min(1, (now - start) / duration);
    state.weights = before.map((weight, i) => weight + delta[i] * p);
    drawPlot();
    showWeights();
    if (p < 1) requestAnimationFrame(animate);
    else { state.weights = after; completeRound(round, false); }
  }
  requestAnimationFrame(animate);
}

function play() {
  if (state.playing) return;
  clearTimeout(state.timer);
  if (!state.next) newBall();
  const round = state.next;
  let choice = null;
  state.playing = true;
  ui.step.disabled = true;
  const run = ++state.run;
  const speed = Number(ui.speed.value);
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const duration = reduced ? 200 : (600 + round.flight * 650) / speed;
  const missFall = reduced ? 60 : 220 / speed;
  const decisionProgress = (DECISION_X - LAUNCH_X) / (landingX(round) - LAUNCH_X);
  const start = performance.now();
  function frame(now) {
    if (run !== state.run) return;
    const elapsed = now - start;
    const p = Math.min(1, elapsed / duration);
    if (choice === null && p >= decisionProgress) {
      choice = decision(round.angle, round.speed);
      ui.status.textContent = `θ = ${round.angle.toFixed(1)}° · v = ${round.speed.toFixed(2)} · ${choice < 0 ? 'Left' : 'Right'}`;
    }
    const current = position(round, p);
    ui.ball.setAttribute('cx', current.x);
    ui.ball.setAttribute('cy', current.y);
    const segments = Math.max(1, Math.ceil(p * 36));
    const points = [];
    for (let i = 0; i <= segments; i++) {
      const point = position(round, p * i / segments);
      points.push(`${i === 0 ? 'M' : 'L'} ${point.x} ${point.y}`);
    }
    ui.trail.setAttribute('d', points.join(' '));
    const cartProgress = choice === null ? 0 : Math.min(1, (p - decisionProgress) / ((1 - decisionProgress) * 0.55));
    ui.cart.setAttribute('transform', `translate(${350 + (choice < 0 ? -80 : 80) * cartProgress} 0)`);
    if (p === 1 && choice !== (round.range >= RANGE_SPLIT ? 1 : -1)) {
      const missP = Math.min(1, (elapsed - duration) / missFall);
      ui.ball.setAttribute('cy', GROUND_Y + 25 * missP);
      if (missP === 1) { finish({ ...round, choice }); return; }
    } else if (p === 1) { finish({ ...round, choice }); return; }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function lineEnds(wa, wu, bias) {
  const points = [];
  if (Math.abs(wu) > 1e-9) for (const a of [-1, 1]) {
    const u = (-bias - wa * a) / wu;
    if (u >= -1 && u <= 1) points.push({ a, u });
  }
  if (Math.abs(wa) > 1e-9) for (const u of [-1, 1]) {
    const a = (-bias - wu * u) / wa;
    if (a >= -1 && a <= 1 && !points.some(point => Math.abs(point.a - a) < 1e-8 && Math.abs(point.u - u) < 1e-8)) points.push({ a, u });
  }
  return points.length >= 2 ? points : null;
}

function decisionRegion(side) {
  const [wa, wu, bias] = state.weights;
  const score = (point) => side * (wa * point.a + wu * point.u + bias);
  const square = [{ a: -1, u: -1 }, { a: 1, u: -1 }, { a: 1, u: 1 }, { a: -1, u: 1 }];
  const clipped = [];
  for (let i = 0; i < square.length; i++) {
    const a = square[i], next = square[(i + 1) % square.length];
    const sa = score(a), sb = score(next);
    if (sa >= 0) clipped.push(a);
    if ((sa < 0 && sb > 0) || (sa > 0 && sb < 0)) {
      const t = sa / (sa - sb);
      clipped.push({ a: a.a + t * (next.a - a.a), u: a.u + t * (next.u - a.u) });
    }
  }
  return clipped.map(point => `${plotX(45 + 25 * point.a)},${plotY(12 + 2 * point.u)}`).join(' ');
}

function drawPlot() {
  ui.plot.replaceChildren();
  svg(ui.plot, 'rect', { x: 55, y: 25, width: 420, height: 290, fill: '#fff' });
  svg(ui.plot, 'polygon', { points: decisionRegion(-1), fill: '#f7f4fc' });
  svg(ui.plot, 'polygon', { points: decisionRegion(1), fill: '#f2faf9' });
  svg(ui.plot, 'rect', { x: 55, y: 25, width: 420, height: 290, fill: 'none', stroke: '#adb5bd' });
  svg(ui.plot, 'line', { x1: plotX(45), y1: 25, x2: plotX(45), y2: 315, stroke: '#dee2e6' });
  svg(ui.plot, 'line', { x1: 55, y1: plotY(12), x2: 475, y2: plotY(12), stroke: '#dee2e6' });
  for (const angle of [20, 45, 70]) svg(ui.plot, 'text', { x: plotX(angle), y: 336, 'text-anchor': 'middle', fill: '#6c757d', 'font-size': 11 }, `${angle}°`);
  for (const speed of [10, 12, 14]) svg(ui.plot, 'text', { x: 47, y: plotY(speed) + 4, 'text-anchor': 'end', fill: '#6c757d', 'font-size': 11 }, String(speed));
  svg(ui.plot, 'text', { x: 478, y: 338, fill: '#6c757d', 'font-size': 12 }, 'θ');
  svg(ui.plot, 'text', { x: 27, y: 17, fill: '#6c757d', 'font-size': 12 }, 'v');

  let curve = '';
  let drawing = false;
  for (let angle = 20; angle <= 70; angle += 0.25) {
    const speed = Math.sqrt(RANGE_SPLIT * G / Math.sin(2 * radians(angle)));
    if (speed >= 10 && speed <= 14) {
      curve += `${drawing ? 'L' : 'M'} ${plotX(angle)} ${plotY(speed)} `;
      drawing = true;
    } else drawing = false;
  }
  svg(ui.plot, 'path', { d: curve, fill: 'none', stroke: '#6c757d', 'stroke-width': 1.5, 'stroke-dasharray': '5 5' });

  function boundary(weights, color, dash) {
    const ends = lineEnds(...weights);
    if (!ends) return;
    svg(ui.plot, 'line', {
      x1: plotX(45 + 25 * ends[0].a), y1: plotY(12 + 2 * ends[0].u),
      x2: plotX(45 + 25 * ends[1].a), y2: plotY(12 + 2 * ends[1].u),
      stroke: color, 'stroke-width': 2, ...(dash ? { 'stroke-dasharray': '4 4' } : {})
    });
  }
  if (state.updateFrom) boundary(state.updateFrom, '#adb5bd', true);
  boundary(state.weights, '#0d6efd', false);
  for (const round of state.rounds) {
    const cx = plotX(round.angle), cy = plotY(round.speed);
    if (!round.caught) svg(ui.plot, 'circle', { cx, cy, r: 8, fill: 'none', stroke: '#dc3545', 'stroke-width': 2 });
    const dot = svg(ui.plot, 'circle', { cx, cy, r: 4, fill: round.correct < 0 ? '#6f42c1' : '#0f766e' });
    svg(dot, 'title', {}, `${round.angle.toFixed(1)}°, ${round.speed.toFixed(2)}; ${round.correct < 0 ? 'left' : 'right'} landing; ${round.caught ? 'caught' : 'missed'}`);
  }
}

ui.step.addEventListener('click', play);
ui.continuous.addEventListener('click', () => {
  state.continuous = !state.continuous;
  ui.continuous.textContent = state.continuous ? 'Pause' : 'Run continuously';
  ui.continuous.setAttribute('aria-pressed', String(state.continuous));
  if (state.continuous && !state.playing) play();
  if (!state.continuous) clearTimeout(state.timer);
});
ui.reset.addEventListener('click', () => {
  ++state.run;
  clearTimeout(state.timer);
  state.weights = [0, 0.45, -0.05];
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
