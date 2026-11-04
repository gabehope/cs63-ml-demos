const $ = (id) => document.getElementById(id);
const ui = {
  page: $('page'), world: $('world'), grid: $('grid'), gridTitle: $('grid-title'), status: $('status'), update: $('update'), episode: $('episode-label'),
  step: $('step'), continuous: $('continuous'), episodes: $('episodes'), reset: $('reset'),
  alpha: $('alpha'), gamma: $('gamma'), epsilon: $('epsilon'), noise: $('noise'), living: $('living'), speed: $('speed'),
  scale: $('scale'), scaleValue: $('scale-value'), view: $('view'), showPolicy: $('show-policy'), showMoves: $('show-moves'),
  log: $('update-log'), chart: $('reward-chart'), count: $('count-label'), statWin: $('stat-win'), statLose: $('stat-lose'), statReward: $('stat-reward'),
  inputs: $('inputs'), hidden: $('hidden'), netSvg: $('network'), ripple: $('ripple-note')
};
// ---------- Where Q-values come from ----------
// Each learner gives every square's four Q-values (north, east, south, west) and applies one update toward a target.
// Table: Q(s, a) ← Q(s, a) + α[target − Q(s, a)], touching only that entry.
function tableLearner() {
  let Q = {};
  return {
    kind: 'table',
    reset() { Q = {}; for (const s of states) Q[key(s)] = [0, 0, 0, 0]; },
    values() { return Q; },
    update(s, i, target, rate) { const q = Q[key(s)]; q[i] += rate * (target - q[i]); }
  };
}
// Tiny network: inputs → a few ReLU units → 4 outputs (north, east, south, west). One gradient step per update:
// w ← w + α[target − Q(s, a; w)] ∇Q(s, a; w), which moves every square's values, not just Q(s, a).
function networkLearner() {
  let net = null;
  // (x, y) scaled to −1…1 across the grid, or a one-hot vector over the squares the agent acts from.
  const encode = (s) => {
    if (ui.inputs.value === 'onehot') return states.map(c => (c.x === s.x && c.y === s.y ? 1 : 0));
    const hx = (GRID.width - 1) / 2, hy = (GRID.height - 1) / 2;
    return [(s.x - hx) / hx, (s.y - hy) / hy];
  };
  return {
    kind: 'network', encode,
    get net() { return net; },
    sizes() { return [encode(GRID.start).length, Number(ui.hidden.value), 4]; },
    reset() { net = createNetwork(this.sizes()); },
    values() {
      const Q = {};
      for (const s of states) Q[key(s)] = outputs(net, encode(s));
      return Q;
    },
    update(s, i, target, rate) {
      const acts = forwardPass(net, encode(s)), out = acts[acts.length - 1];
      const delta = new Float64Array(out.length);
      delta[i] = out[i] - target;
      const grads = zeroGrads(net);
      backprop(net, acts, delta, grads);
      net.layers.forEach((layer, l) => {
        for (let k = 0; k < layer.W.length; k++) layer.W[k] -= rate * grads[l].W[k];
        for (let j = 0; j < layer.outputs; j++) layer.b[j] -= rate * grads[l].b[j];
      });
    }
  };
}
const LEARNER = document.body.dataset.learner === 'network' ? networkLearner() : tableLearner();

const state = { Q: {}, log: [], chartDirty: true, anim: null, inspect: null, agent: null, done: false, steps: 0, total: 0, returns: [], last: null, running: false, carry: 0, lastFrame: performance.now() };

// ---------- Q-learning ----------
function resetTable() {
  LEARNER.reset();
  state.Q = LEARNER.values();
}
const copyQ = (Q) => Object.fromEntries(Object.entries(Q).map(([k, q]) => [k, q.slice()]));
// Applies the update and refreshes every square's values; also counts how many other values moved.
function applyUpdate(s, i, target, alpha) {
  const before = copyQ(state.Q);
  LEARNER.update(s, i, target, alpha);
  state.Q = LEARNER.values();
  let others = 0, largest = 0;
  for (const [k, q] of Object.entries(state.Q)) q.forEach((v, j) => {
    if (k === key(s) && j === i) return;
    const change = Math.abs(v - before[k][j]);
    if (change > 0.005) others++;
    largest = Math.max(largest, change);
  });
  return { before, value: state.Q[key(s)][i], others, largest };
}
function startEpisode() {
  state.agent = { ...GRID.start };
  state.done = false;
  state.steps = 0;
  state.total = 0;
}
function maxQ(s) { return Math.max(...state.Q[key(s)]); }
function greedy(s) {
  const q = state.Q[key(s)], best = Math.max(...q);
  const ties = q.map((v, i) => i).filter(i => q[i] === best);
  return ties[Math.floor(Math.random() * ties.length)];
}
// The move goes the intended way with prob. 1 − noise, otherwise 90° to the left or right.
function slip(a) {
  const noise = Number(ui.noise.value), r = Math.random();
  if (r < 1 - noise) return a;
  return r < 1 - noise / 2 ? (a + 1) % 4 : (a + 3) % 4;
}

// One step of Q-learning. manual: an action index chosen with the arrow keys (null: ε-greedy).
// Returns a record of the update for display.
function step(manual = null) {
  if (state.done) { startEpisode(); state.last = null; return null; }
  const alpha = Number(ui.alpha.value), gamma = Number(ui.gamma.value);
  const s = state.agent, q = state.Q[key(s)];
  const explored = manual === null && Math.random() < Number(ui.epsilon.value);
  const a = manual !== null ? manual : explored ? Math.floor(Math.random() * 4) : greedy(s);
  const actual = slip(a);
  const next = moveFrom(s, actual);
  // Target R(s) + γ max Q(s′, a′); landing on an exit ends the episode, and there the exit's reward stands in for the max.
  const terminal = isExit(next);
  const r = stateReward(s, Number(ui.living.value));
  const old = q[a];
  const nextMax = terminal ? GRID.exits[key(next)] : maxQ(next);
  const nextAction = terminal ? null : state.Q[key(next)].indexOf(nextMax);
  const target = r + gamma * nextMax;
  const result = applyUpdate(s, a, target, alpha);
  state.agent = next;
  state.total += r + (terminal ? GRID.exits[key(next)] : 0);
  state.steps++;
  if (terminal) endEpisode();
  else if (state.steps >= GRID.maxSteps) endEpisode(true);
  return remember({ s, a, actual, next, r, old, nextMax, nextAction, target, terminal, alpha, gamma, explored, manual: manual !== null, ...result });
}
function remember(record) {
  state.log.unshift(record);
  state.log.length = Math.min(state.log.length, 8);
  return record;
}
function endEpisode(timedOut = false) {
  state.done = true;
  state.timedOut = timedOut;
  const exit = timedOut ? 0 : GRID.exits[key(state.agent)];
  state.returns.push({ reward: state.total, caught: exit > 0, lost: exit < 0 });
  state.chartDirty = true;
}

fillWorldMenu(ui.world);
let view = createGridView(ui.grid);

// ---------- Step animation ----------
// A step plays in two beats (fractions of its duration): the agent moves (0–0.35), then the dashed, tagged
// max Q(s′, ·) appears and the updated number changes (0.45–0.9).
const BEATS = { moved: 0.35, change: 0.45, changed: 0.9 };
const ease = (t) => t < 0 ? 0 : t > 1 ? 1 : t * t * (3 - 2 * t);
const phase = (p, from, to) => ease((p - from) / (to - from));
function animate(record, duration) {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  state.anim = record && duration >= 220 && !reduced ? { record, start: performance.now(), duration } : null;
}
function animProgress(now) {
  if (!state.anim) return 1;
  const p = (now - state.anim.start) / state.anim.duration;
  if (p >= 1) { state.anim = null; return 1; }
  return p;
}
function render(now = performance.now()) {
  const showQ = ui.view.value === 'q';
  const p = animProgress(now), last = state.last, anim = state.anim;
  // While the animation runs, every value moves from where it was before this update to where it is now.
  // (A table changes one entry; a network changes them all.)
  const t = phase(p, BEATS.change, BEATS.changed);
  const shown = (k, a, value) => {
    if (!anim) return value;
    const from = anim.record.before[k][a];
    return from + (value - from) * t;
  };
  const updatedKey = last && key(last.s), updatedIndex = last && last.a;
  // Faint orange outline on other values this update moved, scaled by how much (network only, during the change).
  const ripple = (k, a, node) => {
    const change = anim && p >= BEATS.change && !(k === updatedKey && a === updatedIndex) ? Math.abs(state.Q[k][a] - anim.record.before[k][a]) : 0;
    node.style.stroke = change > 0.005 ? `rgba(232, 89, 12, ${Math.min(1, 0.25 + change / anim.record.largest).toFixed(2)})` : '';
    node.style.strokeWidth = change > 0.005 ? '2.5' : '';
  };
  const shownQ = Object.fromEntries(states.map(s => [key(s), state.Q[key(s)].map((v, a) => shown(key(s), a, v))]));
  paintGrid(view, shownQ, showQ, { decorate: ripple, showPolicy: ui.showPolicy.checked });
  outline(view.updated, last && last.s, last && last.a);
  outline(view.used, last ? last.next : null, last && !last.terminal ? last.nextAction : null);
  // max over s′, dashed and tagged "max", once the agent has landed.
  const landed = showQ && last && p >= BEATS.moved;
  view.used.setAttribute('display', landed ? 'inline' : 'none');
  view.maxTag.setAttribute('display', landed && !last.terminal && !state.inspect ? 'inline' : 'none');
  if (landed && !last.terminal) {
    const tag = maxTagPoint(last.next, last.nextAction);
    view.maxTag.setAttribute('x', tag.x);
    view.maxTag.setAttribute('y', tag.y);
  }
  view.updated.setAttribute('display', showQ ? 'inline' : 'none');

  // Arrows: the chosen move, and the actual move when it slipped.
  const move = !!last;
  const arrows = move && !state.inspect && ui.showMoves.checked;
  drawArrow(view.chosen, arrows ? center(last.s) : null, arrows ? last.a : null);
  drawArrow(view.slipped, arrows ? center(last.s) : null, arrows && last.actual !== last.a ? last.actual : null);

  // Agent: slides to where it landed (or bumps the wall and comes back).
  // Centred in a square (just below the reward in an exit box).
  const spot = (s) => isExit(s) ? { x: cellX(s) + S / 2, y: cellY(s) + S / 2 + 10 } : center(s);
  let pos = spot(state.agent);
  if (anim) {
    const r = anim.record, t = phase(p, 0, BEATS.moved);
    const from = spot(r.s), to = spot(r.next);
    pos = { x: from.x + (to.x - from.x) * t, y: from.y + (to.y - from.y) * t };
    if (key(r.s) === key(r.next)) {
      const bump = 24 * Math.sin(Math.PI * Math.min(1, p / BEATS.moved));
      pos = { x: pos.x + ACTIONS[r.actual].dx * bump, y: pos.y - ACTIONS[r.actual].dy * bump };
    }
  }
  view.agent.setAttribute('cx', pos.x);
  view.agent.setAttribute('cy', pos.y);

  if (diagram) {
    drawNetworkWeights(diagram, LEARNER.net);
    // Exit squares are never trained, so after an episode show the square the last move came from (or the start).
    const s = state.done ? (last ? last.s : GRID.start) : state.agent;
    drawNetworkActivations(diagram, forwardPass(LEARNER.net, LEARNER.encode(s)), { chosen: last && key(last.s) === key(s) ? last.a : null });
  }
  showUnit();
  ui.episode.textContent = `Episode ${state.returns.length + (state.done ? 0 : 1)} · step ${state.steps}`;
  ui.status.textContent = describe(last);
  ui.update.innerHTML = formula(last);
  if (ui.ripple) ui.ripple.textContent = last ? `This step also moved ${last.others} other ${last.others === 1 ? 'value' : 'values'} (largest change ${last.largest.toFixed(2)}).` : '';
  showLog();
  if (state.chartDirty) showStats();
}

function showStats() {
  const episodes = state.returns, recent = episodes.slice(-20);
  const wins = episodes.filter(e => e.caught).length;
  ui.count.textContent = `${episodes.length} ${episodes.length === 1 ? 'episode' : 'episodes'}`;
  ui.statWin.textContent = String(wins);
  ui.statLose.textContent = String(episodes.filter(e => e.lost).length);
  ui.statReward.textContent = recent.length ? signedValue(recent.reduce((a, e) => a + e.reward, 0) / recent.length) : '–';
  drawRewardChart(ui.chart, episodes, { window: 20, minSpan: 20 });
  state.chartDirty = false;
}

function describe(last) {
  if (state.done) return state.timedOut ? `Stopped after ${GRID.maxSteps} steps · press Step to start the next episode` : `Reached the ${rewardText(GRID.exits[key(state.agent)])} exit · episode reward ${signedValue(state.total)} · press Step to start the next episode`;
  if (!last) return `At ${where(state.agent)} · press Step or an arrow key`;
  const who = last.manual ? 'you chose' : last.explored ? 'chose at random' : 'chose greedily';
  const slipped = last.actual !== last.a ? ` · slipped ${ACTIONS[last.actual].name}` : '';
  return `At ${where(last.s)} · ${who}: ${ACTIONS[last.a].name}${slipped} · landed in ${where(last.next)} · R(s) = ${signedValue(last.r)}`;
}
const label = (u) => `Q(${where(u.s)}, ${ACTIONS[u.a].name})`;
// The update with this step's numbers: Q(s,a) ← Q(s,a) + α[target − Q(s,a)].
// Table: the update itself. Network: the error the gradient step reduces, and where Q(s, a) ended up.
function formula(last) {
  if (!last) return '';
  const target = `${num(last.r)} + ${last.gamma.toFixed(2)}·${num(last.nextMax)}`;
  const bracket = `[<span class="target" data-label="target = ${num(last.target)}${last.terminal ? ' (episode ends)' : ''}">${target}</span> − ${num(last.old)}]`;
  if (LEARNER.kind === 'network') return `${label(last)}: ${num(last.old)} → <strong>${num(last.value)}</strong><br>error = ${bracket} = ${signedValue(last.target - last.old)}`;
  return `${label(last)} ← ${num(last.old)} + ${last.alpha.toFixed(alphaDigits())}${bracket} = <strong>${num(last.value)}</strong>`;
}
function alphaDigits() { return Number(ui.alpha.step) < 0.01 ? 3 : 2; }
function showLog() {
  ui.log.replaceChildren(...state.log.map((u, i) => {
    const row = document.createElement('tr');
    if (i === 0 && u === state.last) row.className = 'latest';
    for (const text of [label(u), num(u.old), num(u.target), num(u.value)]) {
      const cell = document.createElement('td');
      cell.textContent = text;
      row.appendChild(cell);
    }
    return row;
  }));
}

// ---------- Controls ----------
const STEP_ANIMATION = 900;
function doStep(manual = null) {
  const record = step(manual);
  if (record) state.last = record;
  animate(record, STEP_ANIMATION);
  render();
}
function setRunning(running) {
  state.running = running;
  ui.continuous.textContent = running ? 'Pause' : 'Run continuously';
  ui.continuous.setAttribute('aria-pressed', String(running));
  state.carry = 0;
}
function frame(now) {
  const elapsed = Math.min(250, now - state.lastFrame);
  state.lastFrame = now;
  if (state.running) {
    state.carry += elapsed / 1000 * Number(ui.speed.value);
    let record = null, steps = 0;
    while (state.carry >= 1) { state.carry--; record = step(); if (record) state.last = record; steps++; }
    if (steps) animate(record, Math.min(STEP_ANIMATION, 850 / Number(ui.speed.value)));
    if (steps || state.anim) render(now);
  } else if (state.anim) render(now);
  requestAnimationFrame(frame);
}

ui.step.addEventListener('click', () => { setRunning(false); doStep(); });
ui.continuous.addEventListener('click', () => setRunning(!state.running));
ui.episodes.addEventListener('click', () => {
  setRunning(false);
  if (state.done) startEpisode();
  for (let n = 0; n < 100; n++) {
    while (!state.done) step();
    if (n < 99) startEpisode();
  }
  state.last = null;
  state.anim = null;
  render();
});
function resetAll() {
  setRunning(false);
  view = createGridView(ui.grid);
  resetTable();
  if (LEARNER.kind === 'network') buildDiagram();
  startEpisode();
  state.returns = [];
  state.chartDirty = true;
  state.log = [];
  state.last = null;
  state.anim = null;
  render();
}
ui.reset.addEventListener('click', resetAll);
for (const box of [ui.showPolicy, ui.showMoves]) box.addEventListener('change', () => render());
ui.world.addEventListener('change', () => { setWorld(ui.world.value); resetAll(); });
if (ui.inputs) ui.inputs.addEventListener('change', resetAll);
if (ui.hidden) ui.hidden.addEventListener('change', resetAll);

// Network diagram (network page only), rebuilt when the inputs or hidden units change.
let diagram = null;
// While a hidden unit is hovered, the grid shows that unit's output (its ReLU activation) in every square.
function showUnit() {
  const inspect = state.inspect;
  view.unitLayer.setAttribute('display', inspect ? 'inline' : 'none');
  if (!ui.gridTitle.dataset.title) ui.gridTitle.dataset.title = ui.gridTitle.textContent;
  ui.gridTitle.textContent = inspect ? `Hidden unit ${inspect.unit + 1}: its output in each square` : ui.gridTitle.dataset.title;
  if (!inspect) return;
  const values = states.map(s => forwardPass(LEARNER.net, LEARNER.encode(s))[inspect.layer][inspect.unit]);
  const max = Math.max(1e-6, ...values);
  states.forEach((s, i) => {
    const part = view.unitParts[key(s)];
    part.text.textContent = num(values[i]);
    part.bg.style.fill = `rgba(13, 110, 253, ${(0.08 + 0.6 * values[i] / max).toFixed(3)})`;
  });
}
function bindUnitHover() {
  diagram.nodes.slice(1, -1).forEach((layer, l) => layer.forEach((node, unit) => {
    const hit = svgNode(ui.netSvg, 'circle', { cx: node.getAttribute('cx'), cy: node.getAttribute('cy'), r: 16, class: 'unit-hit', tabindex: 0, role: 'button', 'aria-label': `Show hidden unit ${unit + 1}'s output in each square` });
    const show = () => { state.inspect = { layer: l + 1, unit }; render(); };
    const hide = () => { if (state.inspect && state.inspect.unit === unit) { state.inspect = null; render(); } };
    hit.addEventListener('pointerenter', show);
    hit.addEventListener('pointerleave', hide);
    hit.addEventListener('focus', show);
    hit.addEventListener('blur', hide);
  }));
  for (let j = 0; j < diagram.nodes[1].length; j++) svgNode(ui.netSvg, 'text', { x: diagram.layout[1][j].x - 20, y: diagram.layout[1][j].y + 4, 'text-anchor': 'end', class: 'unit-number' }, String(j + 1));
}
function buildDiagram() {
  state.inspect = null;
  ui.netSvg.replaceChildren();
  const labels = ui.inputs.value === 'onehot' ? states.map(c => `${c.x},${c.y}`) : ['x', 'y'];
  diagram = createNetworkDiagram(ui.netSvg, LEARNER.sizes(), { inputLabels: labels, outputLabels: ['↑', '→', '↓', '←'], outputCaption: 'Q(s, ·)' });
  bindUnitHover();
}
for (const [input, format] of [[ui.alpha, v => v.toFixed(alphaDigits())], [ui.gamma, v => v.toFixed(2)], [ui.epsilon, v => v.toFixed(2)], [ui.noise, v => v.toFixed(2)], [ui.living, v => v.toFixed(2)], [ui.speed, v => `${v} steps/s`]]) {
  const output = $(`${input.id}-value`);
  input.addEventListener('input', () => { output.textContent = format(Number(input.value)); });
}
ui.view.addEventListener('change', () => render());

const ARROWS = { ArrowUp: 0, ArrowRight: 1, ArrowDown: 2, ArrowLeft: 3 };
// After a mouse click on a control, drop its focus so the arrow keys move the agent instead of the control.
let usedPointer = false;
document.addEventListener('pointerdown', () => { usedPointer = true; }, true);
document.addEventListener('keydown', (event) => {
  usedPointer = false;
  if (event.target.matches('input, select') || ARROWS[event.key] === undefined) return;
  event.preventDefault();
  setRunning(false);
  doStep(ARROWS[event.key]);
});
document.addEventListener('change', (event) => { if (usedPointer && event.target.matches('input, select')) event.target.blur(); });
bindGameScale(ui.scale, ui.scaleValue, ui.page);

resetTable();
if (LEARNER.kind === 'network') buildDiagram();
startEpisode();
render();
requestAnimationFrame(frame);
