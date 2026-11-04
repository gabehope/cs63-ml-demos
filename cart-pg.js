// REINFORCE on the cart game: a policy network gives π(left), π(coast), π(right); moves are sampled from it,
// and after each episode every move taken is made more (or less) likely depending on how the catch went.
const $ = (id) => document.getElementById(id);
const ui = {
  page: $('page'), game: $('game'), status: $('pg-status'), episode: $('episode-label'), count: $('count-label'),
  pRows: [0, 1, 2].map(i => { const row = $(`p-row-${i}`); return { row, bar: row.querySelector('i'), value: row.querySelector('.q-value') }; }),
  update: $('update'), rate: $('rate'), rateValue: $('rate-value'), gamma: $('gamma'), gammaValue: $('gamma-value'),
  entropy: $('entropy'), entropyValue: $('entropy-value'), baseline: $('baseline'), rewardMode: $('reward-mode'),
  speed: $('speed'), speedValue: $('speed-value'), scale: $('scale'), scaleValue: $('scale-value'),
  fast: $('fast'), showLanding: $('show-landing'), step: $('step'), continuous: $('continuous'), reset: $('reset'),
  chart: $('reward-chart'), caught: $('stat-caught'), missed: $('stat-missed'), rate50: $('stat-rate'), reward50: $('stat-reward'), runLegend: $('run-legend'),
  policy: $('policy'), policyPanel: $('policy-panel'), policyToggle: $('policy-toggle'), network: $('network')
};
const SIZES = [6, 24, 24, 3];
const BASELINE_RATE = 0.05;
const RUN_COLORS = ['#e8590c', '#6f42c1', '#868e96'];
const scene = createScene(ui.game);
const state = {
  policy: createPolicy({ sizes: SIZES }), baseline: 0, game: newEpisode(), moves: [], episodes: [], previous: [], running: false,
  last: null, update: null, hold: 0, banner: '', path: [], carry: 0, lastFrame: performance.now(),
  dirty: true, drawn: { chart: 0, weights: 0, policy: 0 }
};

// ---------- Agent loop ----------
function startEpisode() {
  state.game = newEpisode();
  state.moves = [];
  state.hold = 0;
  state.banner = '';
  state.path = [];
  state.last = { p: actionProbabilities(state.policy.net, features(state.game)), action: null };
  state.drawn.policy = 0;
}

// Sample a move from π(· | s); no ε needed, the randomness is the exploration.
function decide() {
  const game = state.game, s = features(game);
  const { action, p } = sampleAction(state.policy, s);
  state.moves.push({ s, a: action, p: p[action] });
  state.last = { p, action };
  state.path.push({ x: game.cart.x, v: game.cart.v });
  game.motor = CART.ACTIONS[action];
}

// After the episode: A_t = γ^(T−1−t) (R − b) for every move, one gradient step on Σ A_t log π(a_t | s_t).
function endEpisode() {
  const game = state.game, reward = terminalReward(game, ui.rewardMode.value);
  const gamma = Number(ui.gamma.value), T = state.moves.length, b = ui.baseline.checked ? state.baseline : 0;
  state.moves.forEach((move, t) => { move.advantage = Math.pow(gamma, T - 1 - t) * (reward - b); });
  reinforce(state.policy, state.moves, { rate: Number(ui.rate.value), entropy: Number(ui.entropy.value) });
  // How the update changed the probability of the moves this episode took.
  const before = state.moves.reduce((sum, m) => sum + m.p, 0) / T;
  const after = state.moves.reduce((sum, m) => sum + actionProbabilities(state.policy.net, m.s)[m.a], 0) / T;
  state.update = { reward, b, T, before, after, baseline: ui.baseline.checked };
  state.baseline += BASELINE_RATE * (reward - state.baseline);
  state.path.push({ x: game.cart.x, v: game.cart.v });
  state.episodes.push({ reward, caught: game.caught, miss: game.miss, settings: settingsLabel() });
  state.banner = outcomeText(game, reward);
  state.dirty = true;
  state.drawn.weights = 0;
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
  if (game.done) { endEpisode(); return true; }
  return isDecisionTick(game);
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
  const how = last && last.action !== null ? `sampled ${CART.ACTION_NAMES[last.action]} (π = ${Math.round(100 * last.p[last.action])}%)` : 'waiting';
  ui.status.textContent = fast ? 'Fast-forward' : game.done ? state.banner : `t = ${game.time.toFixed(2)} s · ${how}`;
  showProbabilities(last);
  ui.update.innerHTML = formula(state.update);
  const throttle = fast ? 400 : 100;
  if (state.dirty && now - state.drawn.chart > (fast ? 250 : 0)) { showStats(); state.dirty = false; state.drawn.chart = now; }
  if (!game.done && !ui.policyPanel.hidden && now - state.drawn.policy > throttle) { drawPolicy(); state.drawn.policy = now; }
  if (now - state.drawn.weights > (state.running ? throttle * 2.5 : 100)) { drawNetworkWeights(diagram, state.policy.net); state.drawn.weights = now; }
  const acts = forwardPass(state.policy.net, features(game));
  drawNetworkActivations(diagram, acts, { chosen: last ? last.action : null, probabilities: softmax(Array.from(acts[acts.length - 1])) });
}

function showProbabilities(last) {
  const p = last ? last.p : [1 / 3, 1 / 3, 1 / 3];
  ui.pRows.forEach((item, i) => {
    item.bar.style.left = '0';
    item.bar.style.width = `${(100 * p[i]).toFixed(1)}%`;
    item.value.textContent = `${Math.round(100 * p[i])}%`;
    item.row.classList.toggle('chosen', !!last && last.action === i);
  });
}

// The last episode's update, with its numbers.
function formula(u) {
  if (!u) return '<span class="hint">The policy updates at the end of each episode.</span>';
  const plain = (v) => `${v < 0 ? '−' : ''}${Math.abs(v).toFixed(2)}`;
  const advantage = u.reward - u.b;
  const direction = advantage > 0 ? 'more likely' : advantage < 0 ? 'less likely' : 'no more or less likely';
  return `last episode: R = ${plain(u.reward)}, b = ${u.baseline ? plain(u.b) : '0 (off)'}, so R − b = <strong>${signedValue(advantage)}</strong>` +
    `<br>its ${u.T} moves were made ${direction}: avg π(move taken) ${(100 * u.before).toFixed(1)}% → <strong>${(100 * u.after).toFixed(1)}%</strong>`;
}

function showStats() {
  const stats = episodeStats(state.episodes, 50);
  const recent = state.episodes.slice(-50);
  ui.caught.textContent = String(stats.caught);
  ui.missed.textContent = String(stats.missed);
  ui.rate50.textContent = stats.total ? `${Math.round(100 * stats.rate)}%` : '–';
  ui.reward50.textContent = recent.length ? signedValue(recent.reduce((sum, e) => sum + e.reward, 0) / recent.length) : '–';
  ui.count.textContent = `${stats.total} ${stats.total === 1 ? 'episode' : 'episodes'}`;
  drawRewardChart(ui.chart, state.episodes, { window: 50, minSpan: 50, previous: state.previous.map((run, i) => ({ episodes: run.episodes, color: RUN_COLORS[i] })) });
  showRunLegend();
}

const SLICE = { L: 50, R: 486, T: 10, B: 200, cols: 72, rows: 36, vmax: 12 };
const ACTION_FILLS = ['#e2d9f3', '#f1f3f5', '#cfe9e5'];
function sliceX(x) { return SLICE.L + x / CART.TRACK * (SLICE.R - SLICE.L); }
function sliceY(v) { return SLICE.B - (v + SLICE.vmax) / (2 * SLICE.vmax) * (SLICE.B - SLICE.T); }
// Most likely move for the current ball at every cart position and velocity.
function drawPolicy() {
  const root = ui.policy, game = state.game, net = state.policy.net;
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
        action = argmax(outputs(net, features(probe)));
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

const diagram = createNetworkDiagram(ui.network, SIZES, { outputCaption: 'π(a | s)' });

// Settings that shape learning, recorded with each episode so finished runs can be labelled.
function settingsLabel() {
  const parts = [ui.baseline.checked ? 'baseline' : 'no baseline', `entropy ${Number(ui.entropy.value).toFixed(3)}`, `lr ${Number(ui.rate.value).toFixed(4)}`, `γ ${Number(ui.gamma.value).toFixed(3)}`];
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
  state.policy = createPolicy({ sizes: SIZES });
  state.baseline = 0;
  state.episodes = [];
  state.update = null;
  startEpisode();
  state.dirty = true;
  state.drawn.weights = 0;
});
ui.rate.addEventListener('input', () => { ui.rateValue.textContent = Number(ui.rate.value).toFixed(4); });
ui.gamma.addEventListener('input', () => { ui.gammaValue.textContent = Number(ui.gamma.value).toFixed(3); });
ui.entropy.addEventListener('input', () => { ui.entropyValue.textContent = Number(ui.entropy.value).toFixed(3); });
ui.speed.addEventListener('input', () => { ui.speedValue.textContent = `${Number(ui.speed.value).toFixed(1)}×`; });
ui.fast.addEventListener('change', () => { state.drawn.policy = 0; if (ui.fast.checked && state.game.done) startEpisode(); });
bindGameScale(ui.scale, ui.scaleValue, ui.page);
bindPanelToggle(ui.policyToggle, ui.policyPanel, drawPolicy);

startEpisode();
requestAnimationFrame(frame);
