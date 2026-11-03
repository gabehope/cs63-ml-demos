const $ = (id) => document.getElementById(id);
const ui = {
  page: $('page'), game: $('game'), status: $('q-status'), episode: $('episode-label'), count: $('count-label'),
  qRows: [0, 1, 2].map(i => { const row = $(`q-row-${i}`); return { row, bar: row.querySelector('i'), value: row.querySelector('.q-value') }; }),
  rate: $('rate'), rateValue: $('rate-value'), gamma: $('gamma'), gammaValue: $('gamma-value'),
  epsilon: $('epsilon'), epsilonValue: $('epsilon-value'), rewardMode: $('reward-mode'),
  speed: $('speed'), speedValue: $('speed-value'), scale: $('scale'), scaleValue: $('scale-value'),
  fast: $('fast'), showLanding: $('show-landing'), step: $('step'), continuous: $('continuous'), reset: $('reset'),
  chart: $('reward-chart'), caught: $('stat-caught'), missed: $('stat-missed'), rate50: $('stat-rate'), reward50: $('stat-reward'), memory: $('memory-label'), replay: $('replay'), target: $('target'), runLegend: $('run-legend'),
  policy: $('policy'), policyPanel: $('policy-panel'), policyToggle: $('policy-toggle'), network: $('network')
};
const SIZES = [6, 24, 24, 3];
const EPSILON = { decay: 0.99, floor: 0.05 };
const RUN_COLORS = ['#e8590c', '#6f42c1', '#868e96'];
const scene = createScene(ui.game);
const state = {
  agent: createAgent({ sizes: SIZES }), game: newEpisode(), episodes: [], previous: [], epsilon: 1, running: false,
  pending: null, last: null, hold: 0, banner: '', path: [], carry: 0, lastFrame: performance.now(),
  dirty: true, drawn: { chart: 0, weights: 0, policy: 0 }
};

// ---------- Agent loop ----------
function startEpisode() {
  state.game = newEpisode();
  state.pending = null;
  state.hold = 0;
  state.banner = '';
  state.path = [];
  state.last = { q: qValues(state.agent.net, features(state.game)), action: null, explored: false };
  state.drawn.policy = 0;
}

function decide() {
  const game = state.game;
  const s = features(game);
  const choice = chooseAction(state.agent, s, state.epsilon);
  state.pending = { s, action: choice.action };
  state.last = choice;
  state.path.push({ x: game.cart.x, v: game.cart.v });
  game.motor = CART.ACTIONS[choice.action];
}

function completeStep() {
  const game = state.game;
  const reward = game.done ? terminalReward(game, ui.rewardMode.value) : 0;
  remember(state.agent, state.pending.s, state.pending.action, reward, features(game), game.done);
  learn(state.agent, { gamma: Number(ui.gamma.value), rate: Number(ui.rate.value), replay: ui.replay.checked, sync: ui.target.checked ? 300 : 1 });
  state.pending = null;
  if (game.done) endEpisode(reward);
}

function endEpisode(reward) {
  const game = state.game;
  state.path.push({ x: game.cart.x, v: game.cart.v });
  state.episodes.push({ reward, caught: game.caught, miss: game.miss, settings: settingsLabel() });
  if (state.epsilon > EPSILON.floor) state.epsilon = Math.max(EPSILON.floor, state.epsilon * EPSILON.decay);
  ui.epsilon.value = state.epsilon;
  showEpsilon();
  state.banner = outcomeText(game, reward);
  state.dirty = true;
  if (ui.fast.checked) startEpisode();
  else state.hold = 40;
}

// Runs one physics tick. Returns true when it finished a decision step.
function tick() {
  const game = state.game;
  if (game.done) {
    physicsTick(game);
    if (--state.hold <= 0) startEpisode();
    return false;
  }
  if (isDecisionTick(game)) decide();
  physicsTick(game);
  if (game.done || isDecisionTick(game)) { completeStep(); return true; }
  return false;
}

function stepOnce() {
  setRunning(false);
  if (state.game.done) startEpisode();
  while (!tick());
  state.drawn.weights = state.drawn.policy = 0;
}

function setRunning(running) {
  state.running = running;
  ui.continuous.textContent = running ? 'Pause' : 'Run continuously';
  ui.continuous.setAttribute('aria-pressed', String(running));
}

function frame(now) {
  const elapsed = Math.min(100, now - state.lastFrame);
  state.lastFrame = now;
  if (state.running) {
    if (ui.fast.checked) {
      const end = performance.now() + 14;
      do { for (let i = 0; i < 30; i++) tick(); } while (performance.now() < end);
    } else {
      state.carry += elapsed / 1000 * Number(ui.speed.value) / CART.DT;
      while (state.carry >= 1) { state.carry--; tick(); }
    }
  }
  render(now);
  requestAnimationFrame(frame);
}

// ---------- Rendering ----------
function render(now) {
  const game = state.game, fast = ui.fast.checked && state.running;
  drawScene(scene, game, { showLanding: ui.showLanding.checked, banner: fast ? '' : state.banner });
  ui.episode.textContent = `Episode ${state.episodes.length + (game.done ? 0 : 1)}`;
  const last = state.last;
  const how = last && last.action !== null ? `${CART.ACTION_NAMES[last.action]} (${last.explored ? 'random' : 'greedy'})` : 'waiting';
  ui.status.textContent = fast ? `Fast-forward · ε = ${state.epsilon.toFixed(2)}` : game.done ? state.banner : `t = ${game.time.toFixed(2)} s · ${how} · ε = ${state.epsilon.toFixed(2)}`;
  showQ(last);
  const throttle = fast ? 400 : 100;
  if (state.dirty && now - state.drawn.chart > (fast ? 250 : 0)) { showStats(); state.dirty = false; state.drawn.chart = now; }
  if (!game.done && !ui.policyPanel.hidden && now - state.drawn.policy > throttle) { drawPolicy(); state.drawn.policy = now; }
  if (now - state.drawn.weights > (state.running ? throttle * 2.5 : 100)) { drawNetworkWeights(diagram, state.agent.net); state.drawn.weights = now; }
  drawNetworkActivations(diagram, forwardPass(state.agent.net, features(game)), { chosen: last ? last.action : null });
}

function showQ(last) {
  const q = last ? last.q : [0, 0, 0];
  ui.qRows.forEach((item, i) => {
    const width = Math.min(1, Math.abs(q[i]) / 1.2) * 50;
    item.bar.style.left = `${q[i] < 0 ? 50 - width : 50}%`;
    item.bar.style.width = `${width}%`;
    item.value.textContent = signedValue(q[i], 3);
    item.row.classList.toggle('chosen', !!last && last.action === i);
    item.row.classList.toggle('explored', !!last && last.action === i && last.explored);
  });
}

function showStats() {
  const stats = episodeStats(state.episodes, 50);
  const recent = state.episodes.slice(-50);
  ui.caught.textContent = String(stats.caught);
  ui.missed.textContent = String(stats.missed);
  ui.rate50.textContent = stats.total ? `${Math.round(100 * stats.rate)}%` : '–';
  ui.reward50.textContent = recent.length ? signedValue(recent.reduce((sum, e) => sum + e.reward, 0) / recent.length) : '–';
  ui.count.textContent = `${stats.total} ${stats.total === 1 ? 'episode' : 'episodes'}`;
  ui.memory.textContent = ui.replay.checked ? `replay memory ${state.agent.memory.size.toLocaleString()} steps` : 'replay off: each step is learned from once';
  drawRewardChart(ui.chart, state.episodes, { window: 50, minSpan: 50, previous: state.previous.map((run, i) => ({ episodes: run.episodes, color: RUN_COLORS[i] })) });
  showRunLegend();
}

const SLICE = { L: 50, R: 486, T: 10, B: 200, cols: 72, rows: 36, vmax: 12 };
const ACTION_FILLS = ['#e2d9f3', '#f1f3f5', '#cfe9e5'];
function sliceX(x) { return SLICE.L + x / CART.TRACK * (SLICE.R - SLICE.L); }
function sliceY(v) { return SLICE.B - (v + SLICE.vmax) / (2 * SLICE.vmax) * (SLICE.B - SLICE.T); }
function drawPolicy() {
  const root = ui.policy, game = state.game, net = state.agent.net;
  root.replaceChildren();
  const probe = { ball: game.ball, cart: { x: 0, v: 0 } };
  const cw = (SLICE.R - SLICE.L) / SLICE.cols, rh = (SLICE.B - SLICE.T) / SLICE.rows;
  const paths = ['', '', ''];
  for (let row = 0; row < SLICE.rows; row++) {
    probe.cart.v = SLICE.vmax - (row + 0.5) * 2 * SLICE.vmax / SLICE.rows;
    const y = SLICE.T + row * rh;
    let runAction = -1, runStart = 0;
    for (let col = 0; col <= SLICE.cols; col++) {
      let action = -1;
      if (col < SLICE.cols) {
        probe.cart.x = (col + 0.5) / SLICE.cols * CART.TRACK;
        action = argmax(qValues(net, features(probe)));
      }
      if (action !== runAction) {
        if (runAction >= 0) paths[runAction] += `M ${(SLICE.L + runStart * cw).toFixed(2)} ${y.toFixed(2)} h ${((col - runStart) * cw).toFixed(2)} v ${rh.toFixed(2)} h ${(-(col - runStart) * cw).toFixed(2)} Z `;
        runAction = action;
        runStart = col;
      }
    }
  }
  paths.forEach((d, i) => svgNode(root, 'path', { d, fill: ACTION_FILLS[i] }));
  svgNode(root, 'rect', { x: SLICE.L, y: SLICE.T, width: SLICE.R - SLICE.L, height: SLICE.B - SLICE.T, fill: 'none', stroke: '#adb5bd' });
  svgNode(root, 'line', { x1: SLICE.L, x2: SLICE.R, y1: sliceY(0), y2: sliceY(0), stroke: '#adb5bd', 'stroke-opacity': 0.6 });
  for (const m of [0, 5, 10, 15, 20]) svgNode(root, 'text', { x: sliceX(m), y: SLICE.B + 15, 'text-anchor': 'middle', class: 'chart-label' }, String(m));
  for (const v of [-SLICE.vmax, 0, SLICE.vmax]) svgNode(root, 'text', { x: SLICE.L - 7, y: sliceY(v) + 4, 'text-anchor': 'end', class: 'chart-label' }, v > 0 ? `+${v}` : v < 0 ? `−${-v}` : '0');
  svgNode(root, 'text', { x: (SLICE.L + SLICE.R) / 2, y: SLICE.B + 32, 'text-anchor': 'middle', class: 'chart-label' }, 'cart position cx (m)');
  svgNode(root, 'text', { x: 12, y: (SLICE.T + SLICE.B) / 2, 'text-anchor': 'middle', class: 'chart-label', transform: `rotate(-90 12 ${(SLICE.T + SLICE.B) / 2})` }, 'cart velocity cv (m/s)');
  const lx = sliceX(game.landing);
  svgNode(root, 'line', { x1: lx, x2: lx, y1: SLICE.T, y2: SLICE.B, stroke: '#0d6efd', 'stroke-width': 1.5, 'stroke-dasharray': '4 3' });
  const clampV = v => Math.max(-SLICE.vmax, Math.min(SLICE.vmax, v));
  const points = state.path.concat([{ x: game.cart.x, v: game.cart.v }]);
  svgNode(root, 'path', { d: `M ${points.map(p => `${sliceX(p.x).toFixed(1)} ${sliceY(clampV(p.v)).toFixed(1)}`).join(' L ')}`, fill: 'none', stroke: '#495057', 'stroke-width': 1.5 });
  svgNode(root, 'circle', { cx: sliceX(game.cart.x), cy: sliceY(clampV(game.cart.v)), r: 5, fill: 'white', stroke: '#212529', 'stroke-width': 2 });
}

const diagram = createNetworkDiagram(ui.network, SIZES);

// Settings that shape learning, recorded with each episode so finished runs can be labelled.
function settingsLabel() {
  const parts = [ui.replay.checked ? 'replay' : 'no replay', ui.target.checked ? 'frozen target' : 'target every update', `lr ${Number(ui.rate.value).toFixed(4)}`, `γ ${Number(ui.gamma.value).toFixed(3)}`];
  if (ui.rewardMode.value === 'catch') parts.push('±1 reward');
  return parts.join(' · ');
}

function showRunLegend() {
  ui.runLegend.replaceChildren();
  ui.runLegend.hidden = !state.previous.length;
  state.previous.forEach((run, i) => {
    const item = document.createElement('span');
    const swatch = document.createElement('i');
    swatch.className = 'run-line';
    swatch.style.borderColor = RUN_COLORS[i];
    item.append(swatch, ` ${i ? `${i + 1} runs ago` : 'previous run'}: ${run.label}`);
    ui.runLegend.appendChild(item);
  });
  const clear = document.createElement('button');
  clear.type = 'button';
  clear.className = 'link-button';
  clear.textContent = 'Clear previous runs';
  clear.addEventListener('click', () => { state.previous = []; state.dirty = true; });
  ui.runLegend.appendChild(clear);
}

function showEpsilon() { ui.epsilonValue.textContent = state.epsilon.toFixed(2); }

// ---------- Controls ----------
ui.step.addEventListener('click', stepOnce);
ui.continuous.addEventListener('click', () => { setRunning(!state.running); state.carry = 0; });
ui.reset.addEventListener('click', () => {
  setRunning(false);
  if (state.episodes.length >= 20) {
    const labels = [...new Set(state.episodes.map(e => e.settings))];
    state.previous.unshift({ episodes: state.episodes, label: labels.length === 1 ? labels[0] : 'mixed settings' });
    state.previous.length = Math.min(state.previous.length, RUN_COLORS.length);
  }
  state.agent = createAgent({ sizes: SIZES });
  state.episodes = [];
  state.epsilon = 1;
  ui.epsilon.value = 1;
  showEpsilon();
  startEpisode();
  state.dirty = true;
  state.drawn.weights = 0;
});
ui.replay.addEventListener('change', () => { state.dirty = true; });
ui.epsilon.addEventListener('input', () => { state.epsilon = Number(ui.epsilon.value); showEpsilon(); });
ui.rate.addEventListener('input', () => { ui.rateValue.textContent = Number(ui.rate.value).toFixed(4); });
ui.gamma.addEventListener('input', () => { ui.gammaValue.textContent = Number(ui.gamma.value).toFixed(3); });
ui.speed.addEventListener('input', () => { ui.speedValue.textContent = `${Number(ui.speed.value).toFixed(1)}×`; });
ui.fast.addEventListener('change', () => { state.drawn.policy = 0; if (ui.fast.checked && state.game.done) startEpisode(); });
bindGameScale(ui.scale, ui.scaleValue, ui.page);
bindPanelToggle(ui.policyToggle, ui.policyPanel, drawPolicy);

startEpisode();
requestAnimationFrame(frame);
