const T = 0.8;
const DECISION_PROGRESS = 0.44;
const $ = (id) => document.getElementById(id);
const ui = {
  ball: $('ball'), cart: $('cart'), trail: $('trail'), status: $('play-status'), score: $('score'),
  start: $('start'), left: $('choose-left'), right: $('choose-right'), reset: $('reset'),
  plot: $('plot'), plotPanel: $('plot-panel'), plotToggle: $('plot-toggle'), layout: $('play-layout')
};
const state = { x: 0, v: 0, choice: null, phase: 'ready', rounds: 0, caught: 0, history: [], run: 0 };
const svgNS = 'http://www.w3.org/2000/svg';

function formatted(n) { return `${n < 0 ? '−' : '+'}${Math.abs(n).toFixed(2)}`; }
function sample() { return (Math.floor(Math.random() * 201) - 100) / 100; }
function ballX(progress) { return 130 + 100 * state.x + (300 + 100 * state.v) * T * progress; }
function ballY(progress) { return 32 + 217 * progress * progress; }
function plotX(x) { return 55 + (x + 1) / 2 * 420; }
function plotY(v) { return 315 - (v + 1) / 2 * 290; }

function svg(name, attributes, label) {
  const node = document.createElementNS(svgNS, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (label !== undefined) node.textContent = label;
  ui.plot.appendChild(node);
  return node;
}

function drawPlot() {
  ui.plot.replaceChildren();
  svg('rect', { x: 55, y: 25, width: 420, height: 290, fill: '#fff', stroke: '#adb5bd' });
  svg('line', { x1: plotX(0), y1: 25, x2: plotX(0), y2: 315, stroke: '#dee2e6' });
  svg('line', { x1: 55, y1: plotY(0), x2: 475, y2: plotY(0), stroke: '#dee2e6' });
  for (const n of [-1, 0, 1]) {
    svg('text', { x: plotX(n), y: 336, 'text-anchor': 'middle', fill: '#6c757d', 'font-size': 11 }, String(n));
    svg('text', { x: 47, y: plotY(n) + 4, 'text-anchor': 'end', fill: '#6c757d', 'font-size': 11 }, String(n));
  }
  svg('text', { x: 485, y: 337, fill: '#6c757d', 'font-size': 12 }, 'x');
  svg('text', { x: 25, y: 18, fill: '#6c757d', 'font-size': 12 }, 'v');
  for (const round of state.history) {
    const cx = plotX(round.x), cy = plotY(round.v);
    if (!round.caught) svg('circle', { cx, cy, r: 8, fill: 'none', stroke: '#dc3545', 'stroke-width': 2 });
    svg('circle', { cx, cy, r: 4, fill: round.correct < 0 ? '#6f42c1' : '#0f766e' });
  }
  if (state.phase !== 'complete') svg('circle', { cx: plotX(state.x), cy: plotY(state.v), r: 5, fill: 'white', stroke: '#6c757d', 'stroke-width': 1.5 });
}

function drawBall(progress) {
  ui.ball.setAttribute('cx', ballX(progress));
  ui.ball.setAttribute('cy', ballY(progress));
  const segments = Math.max(1, Math.ceil(progress * 30));
  const points = [];
  for (let i = 0; i <= segments; i++) {
    const t = progress * i / segments;
    points.push(`${i === 0 ? 'M' : 'L'} ${ballX(t)} ${ballY(t)}`);
  }
  ui.trail.setAttribute('d', points.join(' '));
}

function newBall() {
  state.x = sample();
  state.v = sample();
  state.choice = null;
  state.phase = 'ready';
  ui.ball.setAttribute('cx', ballX(0));
  ui.ball.setAttribute('cy', ballY(0));
  ui.cart.setAttribute('transform', 'translate(370 0)');
  ui.trail.setAttribute('d', '');
  ui.status.textContent = `x = ${formatted(state.x)} · v = ${formatted(state.v)} · Ready`;
  ui.left.disabled = true;
  ui.right.disabled = true;
  drawPlot();
}

function startBall() {
  if (state.phase === 'complete') newBall();
  if (state.phase !== 'ready') return;
  state.phase = 'approach';
  ui.start.disabled = true;
  ui.status.textContent = `x = ${formatted(state.x)} · v = ${formatted(state.v)} · Incoming`;
  const run = ++state.run;
  const duration = matchMedia('(prefers-reduced-motion: reduce)').matches ? 80 : 800;
  const start = performance.now();
  function frame(now) {
    if (run !== state.run) return;
    const progress = Math.min(1, (now - start) / duration);
    drawBall(DECISION_PROGRESS * progress);
    if (progress < 1) requestAnimationFrame(frame);
    else {
      state.phase = 'waiting';
      ui.status.textContent = `x = ${formatted(state.x)} · v = ${formatted(state.v)} · Choose left or right`;
      ui.left.disabled = false;
      ui.right.disabled = false;
    }
  }
  requestAnimationFrame(frame);
}

function complete(caught) {
  state.rounds++;
  if (caught) state.caught++;
  state.phase = 'complete';
  state.history.push({ x: state.x, v: state.v, correct: state.x + T * state.v >= 0 ? 1 : -1, caught });
  drawPlot();
  ui.score.textContent = `${state.caught} caught / ${state.rounds} ${state.rounds === 1 ? 'round' : 'rounds'}`;
  ui.status.textContent = `x = ${formatted(state.x)} · v = ${formatted(state.v)} · ${state.choice < 0 ? 'Left' : 'Right'} · ${caught ? 'Caught' : 'Missed'}`;
  ui.start.disabled = false;
  ui.start.textContent = 'Next ball';
}

function choose(choice) {
  if (state.phase !== 'waiting') return;
  state.phase = 'falling';
  state.choice = choice;
  ui.left.disabled = true;
  ui.right.disabled = true;
  ui.status.textContent = `x = ${formatted(state.x)} · v = ${formatted(state.v)} · ${choice < 0 ? 'Left' : 'Right'}`;
  const run = state.run;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const duration = reduced ? 100 : 1000;
  const missFall = reduced ? 60 : 250;
  const start = performance.now();
  const caught = choice === (state.x + T * state.v >= 0 ? 1 : -1);
  function frame(now) {
    if (run !== state.run) return;
    const elapsed = now - start;
    const p = Math.min(1, elapsed / duration);
    drawBall(DECISION_PROGRESS + (1 - DECISION_PROGRESS) * p);
    ui.cart.setAttribute('transform', `translate(${370 + (choice < 0 ? -90 : 90) * Math.min(1, p / 0.64)} 0)`);
    if (p === 1 && !caught) {
      const missP = Math.min(1, (elapsed - duration) / missFall);
      ui.ball.setAttribute('cy', ballY(1) + 25 * missP);
      if (missP === 1) { complete(false); return; }
    } else if (p === 1) { complete(true); return; }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

ui.start.addEventListener('click', startBall);
ui.left.addEventListener('click', () => choose(-1));
ui.right.addEventListener('click', () => choose(1));
ui.reset.addEventListener('click', () => {
  ++state.run;
  state.rounds = 0;
  state.caught = 0;
  state.history = [];
  ui.score.textContent = '0 caught / 0 rounds';
  ui.start.disabled = false;
  ui.start.textContent = 'Start ball';
  newBall();
});
ui.plotToggle.addEventListener('click', () => {
  const show = ui.plotPanel.hidden;
  ui.plotPanel.hidden = !show;
  ui.layout.classList.toggle('show-plot', show);
  ui.plotToggle.textContent = show ? 'Hide scatterplot' : 'Show scatterplot';
  ui.plotToggle.setAttribute('aria-expanded', String(show));
});
document.addEventListener('keydown', (event) => {
  if (state.phase !== 'waiting') return;
  if (event.key === 'ArrowLeft') { event.preventDefault(); choose(-1); }
  if (event.key === 'ArrowRight') { event.preventDefault(); choose(1); }
});

newBall();
