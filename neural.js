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
  inputWeights: [1, 2, 3].map(j => [$(`input-weight-${j}-a`), $(`input-weight-${j}-u`)]),
  hiddenBiases: [1, 2, 3].map(j => $(`hidden-bias-${j}`)),
  outputWeights: [1, 2, 3].map(j => $(`output-weight-${j}`)),
  neurons: Object.fromEntries(['a', 'u', 'h1', 'h2', 'h3', 'p'].map(name => [name, $(`node-${name}`)])),
  outputBias: $('output-bias'), activation: $('activation'),
  rate: $('rate'), rateValue: $('rate-value'), speed: $('speed'), speedValue: $('speed-value'),
  step: $('step'), continuous: $('continuous'), reset: $('reset')
};
const svgNS = 'http://www.w3.org/2000/svg';

function initialNetwork() {
  return { W: [[1, 0], [-1, 0], [0, 1]], b: [-0.4, -0.4, 1], q: [0, 0, 0.4], c: -0.6 };
}
function cloneNetwork(net) {
  return { W: net.W.map(row => row.slice()), b: net.b.slice(), q: net.q.slice(), c: net.c };
}
const state = { net: initialNetwork(), updateFrom: null, selectedNeuron: null, rounds: [], next: null, playing: false, continuous: false, timer: null, run: 0 };

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

function activation(z, kind) { return kind === 'relu' ? Math.max(0, z) : 1 / (1 + Math.exp(-z)); }
function derivative(z, h, kind) { return kind === 'relu' ? (z > 0 ? 1 : 0) : h * (1 - h); }
function sigmoid(z) { return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z)))); }
function forward(net, angle, speed, kind = ui.activation.value) {
  const input = features(angle, speed);
  const z = net.W.map((row, j) => row[0] * input[0] + row[1] * input[1] + net.b[j]);
  const h = z.map(value => activation(value, kind));
  const score = net.q.reduce((sum, weight, j) => sum + weight * h[j], net.c);
  return { input, z, h, score, probability: sigmoid(score) };
}
function trainExample(net, round, rate, kind) {
  const result = forward(net, round.angle, round.speed, kind);
  const target = round.correct > 0 ? 1 : 0;
  const error = target - result.probability;
  const oldOutputWeights = net.q.slice();
  for (let j = 0; j < 3; j++) {
    net.q[j] += rate * error * result.h[j];
    const factor = rate * error * oldOutputWeights[j] * derivative(result.z[j], result.h[j], kind);
    net.W[j][0] += factor * result.input[0];
    net.W[j][1] += factor * result.input[1];
    net.b[j] += factor;
  }
  net.c += rate * error;
}
function trainRecent(net, rounds) {
  const rate = Number(ui.rate.value), kind = ui.activation.value;
  for (const round of rounds.slice(-120)) trainExample(net, round, rate, kind);
}
function interpolateNetwork(before, after, p) {
  const mix = (a, b) => a + (b - a) * p;
  return {
    W: before.W.map((row, j) => row.map((weight, k) => mix(weight, after.W[j][k]))),
    b: before.b.map((weight, j) => mix(weight, after.b[j])),
    q: before.q.map((weight, j) => mix(weight, after.q[j])),
    c: mix(before.c, after.c)
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
  ui.status.textContent = `θ = ${angle.toFixed(1)}° · v = ${speed.toFixed(2)} · Ready`;
}

function showNetwork() {
  const net = state.net;
  for (let j = 0; j < 3; j++) {
    ui.inputWeights[j][0].textContent = signed(net.W[j][0]);
    ui.inputWeights[j][1].textContent = signed(net.W[j][1]);
    ui.hiddenBiases[j].textContent = `b:${signed(net.b[j])}`;
    ui.outputWeights[j].textContent = signed(net.q[j]);
  }
  ui.outputBias.textContent = `c:${signed(net.c)}`;
}
function completeRound(round, caught) {
  state.updateFrom = null;
  state.next = null;
  state.playing = false;
  ui.step.disabled = false;
  ui.activation.disabled = false;
  ui.status.textContent = `θ = ${round.angle.toFixed(1)}° · v = ${round.speed.toFixed(2)} · ${round.choice < 0 ? 'Left' : 'Right'} · ${caught ? 'Caught' : 'Missed'} · trained`;
  ui.round.textContent = `Round ${state.rounds.length + 1}`;
  ui.count.textContent = `${state.rounds.length} ${state.rounds.length === 1 ? 'round' : 'rounds'}`;
  drawPlot();
  showNetwork();
  if (state.continuous) state.timer = setTimeout(() => { newBall(); play(); }, 350 / Number(ui.speed.value));
}
function finish(round) {
  const correct = round.range >= RANGE_SPLIT ? 1 : -1;
  const caught = round.choice === correct;
  state.rounds.push({ ...round, correct, caught });
  const before = cloneNetwork(state.net);
  const after = cloneNetwork(state.net);
  trainRecent(after, state.rounds);
  state.updateFrom = before;
  ui.status.textContent = `θ = ${round.angle.toFixed(1)}° · v = ${round.speed.toFixed(2)} · ${caught ? 'Caught' : 'Missed'} · training`;
  const run = state.run;
  const duration = matchMedia('(prefers-reduced-motion: reduce)').matches ? 100 : 650 / Number(ui.speed.value);
  const start = performance.now();
  function animate(now) {
    if (run !== state.run) return;
    const p = Math.min(1, (now - start) / duration);
    state.net = interpolateNetwork(before, after, p);
    drawPlot();
    showNetwork();
    if (p < 1) requestAnimationFrame(animate);
    else { state.net = after; completeRound(round, caught); }
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
  ui.activation.disabled = true;
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
      choice = forward(state.net, round.angle, round.speed).probability >= 0.5 ? 1 : -1;
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

function fillRegions() {
  const cols = 54, rows = 38;
  const paths = ['', ''];
  for (let row = 0; row < rows; row++) {
    const y = 25 + row * 290 / rows;
    const speed = 14 - (row + 0.5) * 4 / rows;
    let runSide = -1, runStart = 0;
    for (let col = 0; col <= cols; col++) {
      const side = col === cols ? -1 : (forward(state.net, 20 + (col + 0.5) * 50 / cols, speed).score >= 0 ? 1 : 0);
      if (side !== runSide) {
        if (runSide >= 0) {
          const x = 55 + runStart * 420 / cols;
          const width = (col - runStart) * 420 / cols;
          paths[runSide] += `M ${x.toFixed(2)} ${y.toFixed(2)} h ${width.toFixed(2)} v ${(290 / rows).toFixed(2)} h ${(-width).toFixed(2)} Z `;
        }
        runSide = side;
        runStart = col;
      }
    }
  }
  svg(ui.plot, 'path', { d: paths[0], fill: '#f7f4fc' });
  svg(ui.plot, 'path', { d: paths[1], fill: '#f2faf9' });
}
function neuronOutput(result, name) {
  if (name === 'a') return result.input[0];
  if (name === 'u') return result.input[1];
  if (name === 'p') return result.probability;
  return result.h[Number(name[1]) - 1];
}
function fillNeuronOutput() {
  const cols = 54, rows = 38, name = state.selectedNeuron;
  const values = [];
  let max = name === 'a' || name === 'u' || name === 'p' || ui.activation.value === 'sigmoid' ? 1 : 0;
  const min = name === 'a' || name === 'u' ? -1 : 0;
  for (let row = 0; row < rows; row++) {
    const speeds = 14 - (row + 0.5) * 4 / rows;
    const line = [];
    for (let col = 0; col < cols; col++) {
      const angle = 20 + (col + 0.5) * 50 / cols;
      const value = neuronOutput(forward(state.net, angle, speeds), name);
      line.push(value);
      max = Math.max(max, value);
    }
    values.push(line);
  }
  const scaleMax = Math.max(max, min + 0.01);
  const paths = Array(8).fill('');
  for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
    const level = Math.min(7, Math.floor(8 * (values[row][col] - min) / (scaleMax - min)));
    const x = 55 + col * 420 / cols, y = 25 + row * 290 / rows;
    const width = 420 / cols, height = 290 / rows;
    paths[level] += `M ${x.toFixed(2)} ${y.toFixed(2)} h ${width.toFixed(2)} v ${height.toFixed(2)} h ${(-width).toFixed(2)} Z `;
  }
  paths.forEach((d, level) => svg(ui.plot, 'path', { d, fill: '#0d6efd', 'fill-opacity': (0.03 + level * 0.045).toFixed(3) }));
  svg(ui.plot, 'text', { x: 475, y: 17, 'text-anchor': 'end', fill: '#6c757d', 'font-size': 11 }, `${name === 'p' ? 'p' : name.replace(/\d/, digit => '₁₂₃'[Number(digit) - 1])} output: ${min.toFixed(2)} → ${max.toFixed(2)}`);
}
function networkContour(net, color, dashed = false) {
  const cols = 80, rows = 56;
  const values = [];
  for (let y = 0; y <= rows; y++) {
    const row = [];
    for (let x = 0; x <= cols; x++) row.push(forward(net, 20 + 50 * x / cols, 14 - 4 * y / rows).score);
    values.push(row);
  }
  let path = '';
  for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
    const corners = [
      { x, y, s: values[y][x] }, { x: x + 1, y, s: values[y][x + 1] },
      { x: x + 1, y: y + 1, s: values[y + 1][x + 1] }, { x, y: y + 1, s: values[y + 1][x] }
    ];
    const crossings = [];
    for (let edge = 0; edge < 4; edge++) {
      const a = corners[edge], b = corners[(edge + 1) % 4];
      if ((a.s >= 0) === (b.s >= 0)) continue;
      const t = a.s / (a.s - b.s);
      crossings.push({ x: 55 + (a.x + (b.x - a.x) * t) * 420 / cols, y: 25 + (a.y + (b.y - a.y) * t) * 290 / rows });
    }
    for (let i = 0; i + 1 < crossings.length; i += 2) path += `M ${crossings[i].x.toFixed(2)} ${crossings[i].y.toFixed(2)} L ${crossings[i + 1].x.toFixed(2)} ${crossings[i + 1].y.toFixed(2)} `;
  }
  svg(ui.plot, 'path', { d: path, fill: 'none', stroke: color, 'stroke-width': 2, ...(dashed ? { 'stroke-dasharray': '4 4' } : {}) });
}
function drawPlot() {
  ui.plot.replaceChildren();
  svg(ui.plot, 'rect', { x: 55, y: 25, width: 420, height: 290, fill: '#fff' });
  if (state.selectedNeuron) fillNeuronOutput();
  else fillRegions();
  svg(ui.plot, 'rect', { x: 55, y: 25, width: 420, height: 290, fill: 'none', stroke: '#adb5bd' });
  svg(ui.plot, 'line', { x1: plotX(45), y1: 25, x2: plotX(45), y2: 315, stroke: '#dee2e6' });
  svg(ui.plot, 'line', { x1: 55, y1: plotY(12), x2: 475, y2: plotY(12), stroke: '#dee2e6' });
  for (const angle of [20, 45, 70]) svg(ui.plot, 'text', { x: plotX(angle), y: 336, 'text-anchor': 'middle', fill: '#6c757d', 'font-size': 11 }, `${angle}°`);
  for (const speed of [10, 12, 14]) svg(ui.plot, 'text', { x: 47, y: plotY(speed) + 4, 'text-anchor': 'end', fill: '#6c757d', 'font-size': 11 }, String(speed));
  svg(ui.plot, 'text', { x: 478, y: 338, fill: '#6c757d', 'font-size': 12 }, 'θ');
  svg(ui.plot, 'text', { x: 27, y: 17, fill: '#6c757d', 'font-size': 12 }, 'v');
  let curve = '', drawing = false;
  for (let angle = 20; angle <= 70; angle += 0.25) {
    const speed = Math.sqrt(RANGE_SPLIT * G / Math.sin(2 * radians(angle)));
    if (speed >= 10 && speed <= 14) { curve += `${drawing ? 'L' : 'M'} ${plotX(angle)} ${plotY(speed)} `; drawing = true; }
    else drawing = false;
  }
  svg(ui.plot, 'path', { d: curve, fill: 'none', stroke: '#6c757d', 'stroke-width': 1.5, 'stroke-dasharray': '5 5' });
  if (state.updateFrom) networkContour(state.updateFrom, '#adb5bd', true);
  networkContour(state.net, '#0d6efd');
  for (const round of state.rounds) {
    const cx = plotX(round.angle), cy = plotY(round.speed);
    if (!round.caught) svg(ui.plot, 'circle', { cx, cy, r: 8, fill: 'none', stroke: '#dc3545', 'stroke-width': 2 });
    const dot = svg(ui.plot, 'circle', { cx, cy, r: 4, fill: round.correct < 0 ? '#6f42c1' : '#0f766e' });
    svg(dot, 'title', {}, `${round.angle.toFixed(1)}°, ${round.speed.toFixed(2)}; ${round.correct < 0 ? 'left' : 'right'} landing; ${round.caught ? 'caught' : 'missed'}`);
  }
}

for (const [name, node] of Object.entries(ui.neurons)) {
  const select = () => {
    state.selectedNeuron = state.selectedNeuron === name ? null : name;
    for (const [key, item] of Object.entries(ui.neurons)) {
      const selected = key === state.selectedNeuron;
      item.setAttribute('class', `selectable-node${key === 'p' ? ' output-node' : ''}${selected ? ' selected' : ''}`);
      item.setAttribute('aria-pressed', String(selected));
    }
    drawPlot();
  };
  node.addEventListener('click', select);
  node.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(); }
  });
}

ui.step.addEventListener('click', play);
ui.continuous.addEventListener('click', () => {
  state.continuous = !state.continuous;
  ui.continuous.textContent = state.continuous ? 'Pause' : 'Run continuously';
  ui.continuous.setAttribute('aria-pressed', String(state.continuous));
  if (state.continuous && !state.playing) play();
  if (!state.continuous) clearTimeout(state.timer);
});
ui.activation.addEventListener('change', () => {
  clearTimeout(state.timer);
  state.continuous = false;
  ui.continuous.textContent = 'Run continuously';
  ui.continuous.setAttribute('aria-pressed', 'false');
  state.net = initialNetwork();
  for (let i = 0; i < state.rounds.length; i++) trainRecent(state.net, state.rounds.slice(0, i + 1));
  state.updateFrom = null;
  drawPlot();
  showNetwork();
});
ui.reset.addEventListener('click', () => {
  ++state.run;
  clearTimeout(state.timer);
  state.net = initialNetwork();
  state.updateFrom = null;
  state.rounds = [];
  state.playing = false;
  state.continuous = false;
  ui.step.disabled = false;
  ui.activation.disabled = false;
  ui.continuous.textContent = 'Run continuously';
  ui.continuous.setAttribute('aria-pressed', 'false');
  ui.round.textContent = 'Round 1';
  ui.count.textContent = '0 rounds';
  newBall();
  drawPlot();
  showNetwork();
});
ui.rate.addEventListener('input', () => { ui.rateValue.textContent = Number(ui.rate.value).toFixed(2); });
ui.speed.addEventListener('input', () => { ui.speedValue.textContent = `${Number(ui.speed.value).toFixed(1)}×`; });

newBall();
drawPlot();
showNetwork();
