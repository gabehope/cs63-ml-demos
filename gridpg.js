// REINFORCE on the gridworld: each square holds four preferences θ(s, a), and π(· | s) = softmax(θ(s, ·)).
// Moves are sampled from π; when the episode ends, every move taken is made more or less likely by G_t − b.
const $ = (id) => document.getElementById(id);
const ui = {
  page: $('page'), world: $('world'), grid: $('grid'), status: $('status'), update: $('update'), episode: $('episode-label'),
  step: $('step'), continuous: $('continuous'), episodes: $('episodes'), reset: $('reset'),
  alpha: $('alpha'), gamma: $('gamma'), noise: $('noise'), living: $('living'), speed: $('speed'), baseline: $('baseline'),
  scale: $('scale'), scaleValue: $('scale-value'), view: $('view'), showPolicy: $('show-policy'), showMoves: $('show-moves'),
  log: $('update-log'), chart: $('reward-chart'), count: $('count-label'), statWin: $('stat-win'), statLose: $('stat-lose'), statReward: $('stat-reward')
};
const BASELINE_RATE = 0.02;
const state = { theta: {}, b: 0, traj: [], update: null, changes: null, chartDirty: true, anim: null, agent: null, done: false, steps: 0, total: 0, returns: [], last: null, running: false, carry: 0, lastFrame: performance.now() };

// ---------- Policy ----------
function resetPolicy() { state.theta = {}; for (const s of states) state.theta[key(s)] = [0, 0, 0, 0]; state.b = 0; }
function probabilities(s) {
  const t = state.theta[key(s)], m = Math.max(...t), e = t.map(v => Math.exp(v - m)), z = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / z);
}
function sample(p) {
  let u = Math.random(), a = 0;
  while (a < p.length - 1 && u >= p[a]) { u -= p[a]; a++; }
  return a;
}
// The move goes the intended way with prob. 1 − noise, otherwise 90° to the left or right.
function slip(a) {
  const noise = Number(ui.noise.value), r = Math.random();
  if (r < 1 - noise) return a;
  return r < 1 - noise / 2 ? (a + 1) % 4 : (a + 3) % 4;
}
function startEpisode() {
  state.agent = { ...GRID.start };
  state.done = false;
  state.steps = 0;
  state.total = 0;
  state.traj = [];
  state.changes = null;
}

// One move, sampled from π (or chosen with the arrow keys). Nothing is learned until the episode ends.
function step(manual = null) {
  if (state.done) { startEpisode(); state.last = null; return null; }
  const s = state.agent, p = probabilities(s);
  const a = manual !== null ? manual : sample(p);
  const actual = slip(a);
  const next = moveFrom(s, actual);
  const terminal = isExit(next);
  const r = stateReward(s, Number(ui.living.value));
  state.traj.push({ s, a, r, p: p[a] });
  // Reaching an exit ends the episode; its reward is the last one collected, V(exit) = R(exit).
  state.final = terminal ? GRID.exits[key(next)] : 0;
  state.agent = next;
  state.total += r + state.final;
  state.steps++;
  const record = { s, a, actual, next, r, p, terminal, manual: manual !== null };
  if (terminal) endEpisode();
  else if (state.steps >= GRID.maxSteps) endEpisode(true);
  return record;
}

// G_t = R(s_t) + γ G_t+1 for each step (ending with R(exit)), then θ(s_t, a) += α (G_t − b)(1[a = a_t] − π(a | s_t)).
function learnFromEpisode() {
  const alpha = Number(ui.alpha.value), gamma = Number(ui.gamma.value), useBaseline = ui.baseline.checked;
  const before = Object.fromEntries(states.map(s => [key(s), probabilities(s)]));
  let G = state.final || 0;
  for (let t = state.traj.length - 1; t >= 0; t--) { G = state.traj[t].r + gamma * G; state.traj[t].G = G; }
  const b = useBaseline ? state.b : 0;
  for (const move of state.traj) {
    move.advantage = move.G - b;
    const p = probabilities(move.s), theta = state.theta[key(move.s)];
    for (let a = 0; a < 4; a++) theta[a] += alpha * move.advantage * ((a === move.a ? 1 : 0) - p[a]);
  }
  for (const move of state.traj) {
    move.after = probabilities(move.s)[move.a];
    if (useBaseline) state.b += BASELINE_RATE * (move.G - state.b);
  }
  // How much each probability moved, for the green / red outlines.
  const changes = {};
  let largest = 0;
  for (const s of states) {
    const k = key(s), now = probabilities(s);
    changes[k] = now.map((v, a) => v - before[k][a]);
    largest = Math.max(largest, ...changes[k].map(Math.abs));
  }
  state.changes = { byKey: changes, largest };
  state.update = { steps: state.traj.slice(), b, baseline: useBaseline, G0: state.traj[0].G };
}
function endEpisode(timedOut = false) {
  state.done = true;
  state.timedOut = timedOut;
  const exit = timedOut ? 0 : GRID.exits[key(state.agent)];
  state.returns.push({ reward: state.total, caught: exit > 0, lost: exit < 0 });
  learnFromEpisode();
  state.chartDirty = true;
}

// ---------- Drawing ----------
fillWorldMenu(ui.world);
let view = createGridView(ui.grid);
const blue = (p) => `rgba(13, 110, 253, ${(0.04 + 0.5 * p).toFixed(3)})`;

const BEATS = { moved: 0.35 };
const ease = (t) => t < 0 ? 0 : t > 1 ? 1 : t * t * (3 - 2 * t);
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
const spot = (s) => isExit(s) ? { x: cellX(s) + S / 2, y: cellY(s) + S / 2 + 10 } : center(s);

function render(now = performance.now()) {
  const showQ = ui.view.value === 'q', p = animProgress(now), last = state.last, anim = state.anim;
  const probs = Object.fromEntries(states.map(s => [key(s), probabilities(s)]));
  const changes = state.done && state.changes;
  // After an episode, outline every probability the update moved: green if it went up, red if down.
  const decorate = (k, a, poly) => {
    const change = changes ? changes.byKey[k][a] : 0, show = changes && Math.abs(change) > 0.005;
    poly.style.stroke = show ? (change > 0 ? `rgba(15, 118, 110, ${(0.35 + 0.65 * Math.abs(change) / changes.largest).toFixed(2)})` : `rgba(220, 53, 69, ${(0.35 + 0.65 * Math.abs(change) / changes.largest).toFixed(2)})`) : '';
    poly.style.strokeWidth = show ? '3' : '';
  };
  paintGrid(view, probs, showQ, { v: Object.fromEntries(states.map(s => [key(s), Math.max(...probs[key(s)])])), decorate, showPolicy: ui.showPolicy.checked });
  for (const s of states) {
    const k = key(s), part = view.parts[k];
    part.tris.forEach((t, a) => { t.text.textContent = `${Math.round(100 * probs[k][a])}%`; t.poly.style.fill = blue(probs[k][a]); });
    part.value.textContent = `${Math.round(100 * Math.max(...probs[k]))}%`;
    part.bg.style.fill = blue(Math.max(...probs[k]));
  }

  outline(view.updated, showQ && last && !state.done ? last.s : null, last ? last.a : null);
  view.updated.setAttribute('display', showQ && last && !state.done ? 'inline' : 'none');
  view.used.setAttribute('display', 'none');
  view.maxTag.setAttribute('display', 'none');
  const arrows = last && ui.showMoves.checked;
  drawArrow(view.chosen, arrows ? center(last.s) : null, arrows ? last.a : null);
  drawArrow(view.slipped, arrows ? center(last.s) : null, arrows && last.actual !== last.a ? last.actual : null);

  let pos = spot(state.agent);
  if (anim) {
    const r = anim.record, t = ease(p / BEATS.moved), from = spot(r.s), to = spot(r.next);
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
  ui.update.innerHTML = formula();
  showLog();
  if (state.chartDirty) showStats();
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

const plain = (v) => `${v < 0 ? '−' : ''}${Math.abs(v).toFixed(2)}`;
function describe(last) {
  if (state.done) {
    const end = state.timedOut ? `Stopped after ${GRID.maxSteps} steps` : `Reached the ${rewardText(GRID.exits[key(state.agent)])} exit`;
    return `${end} · episode reward ${signedValue(state.total)} · the policy has been updated · press Step for the next episode`;
  }
  if (!last) return `At ${where(state.agent)} · press Step or an arrow key`;
  const how = last.manual ? 'you chose' : `sampled (π = ${Math.round(100 * last.p[last.a])}%)`;
  const slipped = last.actual !== last.a ? ` · slipped ${ACTIONS[last.actual].name}` : '';
  return `At ${where(last.s)} · ${how}: ${ACTIONS[last.a].name}${slipped} · landed in ${where(last.next)} · R(s) = ${signedValue(last.r)}`;
}
const label = (m) => `${where(m.s)} ${ACTIONS[m.a].name}`;
// During an episode: how many moves so far. After it: the first move's numbers, and the biggest change.
function formula() {
  if (!state.done) return `<span class="hint">${state.traj.length} ${state.traj.length === 1 ? 'move' : 'moves'} so far this episode · the policy updates when the episode ends</span>`;
  const u = state.update, first = u.steps[0];
  const biggest = u.steps.reduce((best, m) => Math.abs(m.after - m.p) > Math.abs(best.after - best.p) ? m : best, first);
  const advantage = first.G - u.b;
  return `first move: G<sub>0</sub> − b = ${plain(first.G)} − ${u.baseline ? plain(u.b) : '0'} = <strong>${signedValue(advantage)}</strong> → made ${advantage >= 0 ? 'more' : 'less'} likely` +
    `<br>biggest change: π(${ACTIONS[biggest.a].name} | ${where(biggest.s)}) ${Math.round(100 * biggest.p)}% → <strong>${Math.round(100 * biggest.after)}%</strong>`;
}
// The last episode's moves, newest first.
function showLog() {
  const moves = state.done && state.update ? state.update.steps.slice().reverse().slice(0, 10) : [];
  ui.log.replaceChildren(...moves.map(m => {
    const row = document.createElement('tr');
    for (const text of [label(m), plain(m.G), signedValue(m.advantage), `${Math.round(100 * m.p)}% → ${Math.round(100 * m.after)}%`]) {
      const cell = document.createElement('td');
      cell.textContent = text;
      row.appendChild(cell);
    }
    return row;
  }));
}

// ---------- Controls ----------
const STEP_ANIMATION = 600;
function doStep(manual = null) {
  const record = step(manual);
  state.last = record;
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
    while (state.carry >= 1) { state.carry--; record = step(); state.last = record; steps++; }
    if (steps) animate(record, Math.min(STEP_ANIMATION, 600 / Number(ui.speed.value)));
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
  resetPolicy();
  startEpisode();
  state.returns = [];
  state.update = null;
  state.chartDirty = true;
  state.last = null;
  state.anim = null;
  render();
}
ui.reset.addEventListener('click', resetAll);
ui.world.addEventListener('change', () => { setWorld(ui.world.value); resetAll(); });
for (const box of [ui.showPolicy, ui.showMoves, ui.view]) box.addEventListener('change', () => render());
for (const [input, format] of [[ui.alpha, v => v.toFixed(2)], [ui.gamma, v => v.toFixed(2)], [ui.noise, v => v.toFixed(2)], [ui.living, v => v.toFixed(2)], [ui.speed, v => `${v} steps/s`]]) {
  const output = $(`${input.id}-value`);
  input.addEventListener('input', () => { output.textContent = format(Number(input.value)); });
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

resetPolicy();
startEpisode();
render();
requestAnimationFrame(frame);
