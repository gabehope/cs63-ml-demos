// Value learning on the gridworld: one value V(s) per square, learned from the agent's own moves (ε-greedy).
// The agent picks its move by looking one step ahead at r + γV(s′), averaged over slips when noise > 0.
const $ = (id) => document.getElementById(id);
const ui = {
  page: $('page'), world: $('world'), grid: $('grid'), status: $('status'), update: $('update'), episode: $('episode-label'),
  step: $('step'), continuous: $('continuous'), episodes: $('episodes'), reset: $('reset'),
  alpha: $('alpha'), gamma: $('gamma'), epsilon: $('epsilon'), noise: $('noise'), living: $('living'), speed: $('speed'),
  scale: $('scale'), scaleValue: $('scale-value'), showPolicy: $('show-policy'), showMoves: $('show-moves'),
  log: $('update-log'), chart: $('reward-chart'), count: $('count-label'), statWin: $('stat-win'), statLose: $('stat-lose'), statReward: $('stat-reward'),
  reference: $('reference'), errorShade: $('error-shade'), referenceNote: $('reference-note'), errorChart: $('error-chart')
};
const state = { V: {}, version: 0, refs: null, errors: [], log: [], chartDirty: true, anim: null, agent: null, done: false, steps: 0, total: 0, returns: [], last: null, running: false, carry: 0, lastFrame: performance.now() };

// ---------- Value learning ----------
function resetValues() { state.V = {}; for (const s of states) state.V[key(s)] = 0; }
function startEpisode() {
  state.agent = { ...GRID.start };
  state.done = false;
  state.steps = 0;
  state.total = 0;
}
// R(s) + γ × expected V(s′) for each of the four moves from s, over where the move can end up
// (an exit's value is its reward). With noise 0 there is just one s′ per move.
const squareValue = (s, V) => isExit(s) ? GRID.exits[key(s)] : V[key(s)];
function lookahead(s, V = state.V) {
  const gamma = Number(ui.gamma.value), noise = Number(ui.noise.value), r = stateReward(s, Number(ui.living.value));
  return ACTIONS.map((_, a) => r + gamma * outcomes(s, a, noise).reduce((sum, o) => sum + o.p * squareValue(o.next, V), 0));
}
// The move goes the intended way with prob. 1 − noise, otherwise 90° to the left or right.
function slip(a) {
  const noise = Number(ui.noise.value), r = Math.random();
  if (r < 1 - noise) return a;
  return r < 1 - noise / 2 ? (a + 1) % 4 : (a + 3) % 4;
}
function greedy(s) {
  const q = lookahead(s), best = Math.max(...q);
  const ties = q.map((v, i) => i).filter(i => q[i] === best);
  return ties[Math.floor(Math.random() * ties.length)];
}

// One step: move (possibly slipping), then V(s) ← V(s) + α[R(s) + γV(s′) − V(s)] using the square actually reached.
// manual: an arrow-key action (null: ε-greedy).
function step(manual = null) {
  if (state.done) { startEpisode(); state.last = null; return null; }
  const alpha = Number(ui.alpha.value), gamma = Number(ui.gamma.value);
  const s = state.agent, k = key(s);
  const explored = manual === null && Math.random() < Number(ui.epsilon.value);
  const a = manual !== null ? manual : explored ? Math.floor(Math.random() * 4) : greedy(s);
  const actual = slip(a);
  const next = moveFrom(s, actual);
  // Landing on an exit ends the episode; its value is its reward, V(exit) = R(exit).
  const terminal = isExit(next);
  const r = stateReward(s, Number(ui.living.value));
  const nextV = squareValue(next, state.V);
  const target = r + gamma * nextV;
  const old = state.V[k];
  state.V[k] = old + alpha * (target - old);
  state.version++;
  state.agent = next;
  state.total += r + (terminal ? GRID.exits[key(next)] : 0);
  state.steps++;
  if (terminal) endEpisode();
  else if (state.steps >= GRID.maxSteps) endEpisode(true);
  return remember({ s, a, actual, next, r, old, nextV, target, terminal, value: state.V[k], alpha, gamma, explored, manual: manual !== null });
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
  logError();
  state.chartDirty = true;
}

// ---------- True values, computed exactly from the world's rules ----------
// Repeats V(s) ← choose(s, expected move values under V) for every square until nothing changes.
function solve(choose) {
  let V = Object.fromEntries(states.map(s => [key(s), 0]));
  for (let sweep = 0; sweep < 2000; sweep++) {
    let change = 0;
    const next = {};
    for (const s of states) {
      next[key(s)] = choose(s, lookahead(s, V));
      change = Math.max(change, Math.abs(next[key(s)] - V[key(s)]));
    }
    V = next;
    if (change < 1e-7) break;
  }
  return V;
}
// π(a | s) for the policy the agent follows: ε-greedy over the look-ahead with the learned values, ties split evenly.
function policyProbabilities(s) {
  const q = lookahead(s), best = Math.max(...q), ties = q.filter(v => v === best).length, eps = Number(ui.epsilon.value);
  return q.map(v => eps / 4 + (v === best ? (1 - eps) / ties : 0));
}
// Vπ: the current policy's true values. V*: the best possible values (always take the best move).
function references() {
  if (state.refs && state.refs.version === state.version) return state.refs;
  const probs = Object.fromEntries(states.map(s => [key(s), policyProbabilities(s)]));
  state.refs = {
    version: state.version,
    pi: solve((s, q) => q.reduce((sum, v, a) => sum + probs[key(s)][a] * v, 0)),
    star: solve((s, q) => Math.max(...q))
  };
  return state.refs;
}
const meanError = (ref) => states.reduce((sum, s) => sum + Math.abs(state.V[key(s)] - ref[key(s)]), 0) / states.length;
function logError() {
  const refs = references();
  state.errors.push({ pi: meanError(refs.pi), star: meanError(refs.star) });
}

// ---------- Drawing ----------
fillWorldMenu(ui.world);
let view, moveMarks;
// The grid, plus the move drawn between the square backgrounds and the numbers: a solid arrow from s to where the
// agent landed, a bar where a wall blocked it, and a faint dashed arrow for the intended direction when it slipped.
function buildView() {
  view = createGridView(ui.grid);
  const g = svgNode(view.vMid, 'g', { class: 'grid-step' });
  const arrow = (cls) => ({ line: svgNode(g, 'path', { class: cls }), head: svgNode(g, 'polygon', { class: cls }) });
  moveMarks = { g, intended: arrow('grid-step-intended'), actual: arrow('grid-step-actual'), bar: svgNode(g, 'path', { class: 'grid-step-actual' }) };
  // The reference value (Vπ or V*), small and grey under each learned value.
  refTexts = Object.fromEntries(states.map(s => [key(s), svgNode(view.vLayer, 'text', { x: cellX(s) + S / 2, y: cellY(s) + 54, 'text-anchor': 'middle', class: 'grid-ref' })]));
}
let refTexts = {};
buildView();
function setArrow(arrow, from, to) {
  if (!to) { arrow.line.setAttribute('d', ''); arrow.head.setAttribute('points', ''); return; }
  const dx = to.x - from.x, dy = to.y - from.y, length = Math.hypot(dx, dy) || 1, ux = dx / length, uy = dy / length;
  const base = { x: to.x - ux * 11, y: to.y - uy * 11 };
  arrow.line.setAttribute('d', `M ${from.x.toFixed(1)} ${from.y.toFixed(1)} L ${base.x.toFixed(1)} ${base.y.toFixed(1)}`);
  arrow.head.setAttribute('points', [[to.x, to.y], [base.x - uy * 6, base.y + ux * 6], [base.x + uy * 6, base.y - ux * 6]].map(p => p.map(n => n.toFixed(1)).join(',')).join(' '));
}
// t: how far through the move (0–1). The solid arrow grows with the agent.
function drawMove(record, t) {
  const show = record && t > 0;
  moveMarks.g.setAttribute('display', show ? 'inline' : 'none');
  if (!show) return;
  const c = center(record.s), step = (a, d) => ({ x: c.x + ACTIONS[a].dx * d, y: c.y - ACTIONS[a].dy * d });
  setArrow(moveMarks.intended, step(record.a, 20), record.actual !== record.a ? step(record.a, S / 2 - 8) : null);
  const blocked = key(record.s) === key(record.next);
  const end = blocked ? step(record.actual, S / 2 - 8) : (() => { const to = spot(record.next); const d = Math.hypot(to.x - c.x, to.y - c.y); return { x: c.x + (to.x - c.x) * (d - 21) / d, y: c.y + (to.y - c.y) * (d - 21) / d }; })();
  const start = step(record.actual, 20);
  setArrow(moveMarks.actual, start, { x: start.x + (end.x - start.x) * t, y: start.y + (end.y - start.y) * t });
  // A wall: a short bar across the arrow's tip.
  if (blocked && t >= 1) {
    const tip = step(record.actual, S / 2 - 4), px = -(-ACTIONS[record.actual].dy), py = ACTIONS[record.actual].dx;
    moveMarks.bar.setAttribute('d', `M ${tip.x + px * 12} ${tip.y + py * 12} L ${tip.x - px * 12} ${tip.y - py * 12}`);
  } else moveMarks.bar.setAttribute('d', '');
}

// A step plays in two beats (fractions of its duration): the agent moves (0–0.35), then the dashed V(s′) appears
// and the updated value changes (0.45–0.9).
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
// The agent sits in the centre of a square, between its value and policy arrow (just below the reward in an exit box).
function spot(s) {
  return isExit(s) ? { x: cellX(s) + S / 2, y: cellY(s) + S / 2 + 10 } : center(s);
}

function render(now = performance.now()) {
  const p = animProgress(now), last = state.last, anim = state.anim;
  // While the animation runs, the updated value moves from its old number to its new one.
  const V = { ...state.V };
  if (anim) V[key(anim.record.s)] = anim.record.old + (anim.record.value - anim.record.old) * phase(p, BEATS.change, BEATS.changed);
  paintGrid(view, Object.fromEntries(states.map(s => [key(s), lookahead(s, V)])), false, { v: V, showPolicy: ui.showPolicy.checked });
  const mode = ui.reference.value, ref = mode === 'off' ? null : references()[mode];
  for (const s of states) {
    const k = key(s);
    refTexts[k].textContent = ref ? `${mode === 'pi' ? 'Vπ' : 'V*'} ${num(ref[k])}` : '';
    if (ref && ui.errorShade.checked) view.parts[k].bg.style.fill = shade(V[k] - ref[k]);
  }
  ui.errorShade.disabled = !ref;
  ui.referenceNote.textContent = !ref ? '' : `small numbers: ${mode === 'pi' ? 'Vπ, the true value of the current ε-greedy policy' : 'V*, the best possible value'}${ui.errorShade.checked ? ' · shading: V − reference' : ''}`;

  outline(view.updated, last && last.s);
  outline(view.used, last ? last.next : null);
  view.used.setAttribute('display', last && p >= BEATS.moved ? 'inline' : 'none');
  view.maxTag.setAttribute('display', 'none');
  drawArrow(view.chosen, null, null);
  drawArrow(view.slipped, null, null);
  drawMove(ui.showMoves.checked ? last : null, anim ? phase(p, 0, BEATS.moved) : 1);

  // Agent: slides to where it landed (or bumps the wall and comes back).
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

  ui.episode.textContent = `Episode ${state.returns.length + (state.done ? 0 : 1)} · step ${state.steps}`;
  ui.status.textContent = describe(last);
  ui.update.innerHTML = formula(last);
  showLog();
  if (state.chartDirty) { showStats(); drawErrorChart(); }
}

// Average |V − Vπ| and |V − V*| after each episode.
function drawErrorChart() {
  const root = ui.errorChart, L = 46, R = 486, T = 12, B = 150;
  root.replaceChildren();
  const n = state.errors.length, span = Math.max(20, n - 1);
  const top = Math.max(0.25, ...state.errors.map(e => Math.max(e.pi, e.star)));
  const x = i => L + i / span * (R - L), y = v => B - v / top * (B - T);
  svgNode(root, 'rect', { x: L, y: T, width: R - L, height: B - T, fill: '#fff', stroke: '#adb5bd' });
  for (const v of [0, top / 2, top]) {
    svgNode(root, 'line', { x1: L, x2: R, y1: y(v), y2: y(v), stroke: '#e9ecef' });
    svgNode(root, 'text', { x: L - 8, y: y(v) + 4, 'text-anchor': 'end', class: 'chart-label' }, v.toFixed(2));
  }
  for (const tick of [0, Math.round(span / 2), span]) svgNode(root, 'text', { x: x(tick), y: B + 16, 'text-anchor': 'middle', class: 'chart-label' }, String(tick));
  svgNode(root, 'text', { x: (L + R) / 2, y: B + 32, 'text-anchor': 'middle', class: 'chart-label' }, 'episode');
  const step = Math.max(1, Math.ceil(n / 500));
  const line = (field, attributes) => {
    const points = [];
    state.errors.forEach((e, i) => { if (i % step === 0 || i === n - 1) points.push(`${x(i).toFixed(1)} ${y(e[field]).toFixed(1)}`); });
    if (points.length > 1) svgNode(root, 'path', { d: `M ${points.join(' L ')}`, fill: 'none', 'stroke-width': 2, ...attributes });
  };
  line('star', { stroke: '#6c757d', 'stroke-dasharray': '5 4' });
  line('pi', { stroke: '#0d6efd' });
}

function showStats() {
  const episodes = state.returns, recent = episodes.slice(-20);
  ui.count.textContent = `${episodes.length} ${episodes.length === 1 ? 'episode' : 'episodes'}`;
  ui.statWin.textContent = String(episodes.filter(e => e.caught).length);
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
const label = (u) => `V(${where(u.s)})`;
// The update with this step's numbers: V(s) ← V(s) + α[target − V(s)].
function formula(last) {
  if (!last) return '';
  const target = `${num(last.r)} + ${last.gamma.toFixed(2)}·${num(last.nextV)}`;
  const bracket = `[<span class="target" data-label="target = ${num(last.target)}${last.terminal ? ' (episode ends)' : ''}">${target}</span> − ${num(last.old)}]`;
  return `${label(last)} ← ${num(last.old)} + ${last.alpha.toFixed(2)}${bracket} = <strong>${num(last.value)}</strong>`;
}
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
  buildView();
  resetValues();
  startEpisode();
  state.returns = [];
  state.version++;
  state.errors = [];
  logError();
  state.chartDirty = true;
  state.log = [];
  state.last = null;
  state.anim = null;
  render();
}
ui.reset.addEventListener('click', resetAll);
for (const box of [ui.showPolicy, ui.showMoves, ui.reference, ui.errorShade]) box.addEventListener('change', () => render());
ui.world.addEventListener('change', () => { setWorld(ui.world.value); resetAll(); });
for (const [input, format] of [[ui.alpha, v => v.toFixed(2)], [ui.gamma, v => v.toFixed(2)], [ui.epsilon, v => v.toFixed(2)], [ui.noise, v => v.toFixed(2)], [ui.living, v => v.toFixed(2)], [ui.speed, v => `${v} steps/s`]]) {
  const output = $(`${input.id}-value`);
  input.addEventListener('input', () => { output.textContent = format(Number(input.value)); state.version++; render(); });
}

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

resetValues();
startEpisode();
render();
requestAnimationFrame(frame);
