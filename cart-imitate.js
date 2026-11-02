const $ = (id) => document.getElementById(id);
const ui = {
  page: $('page'), game: $('game'), status: $('play-status'), episode: $('episode-label'), hint: $('hint'),
  human: $('driver-human'), network: $('driver-network'),
  launch: $('launch'), left: $('drive-left'), right: $('drive-right'),
  speed: $('speed'), speedValue: $('speed-value'), scale: $('scale'), scaleValue: $('scale-value'),
  record: $('record'), autoLaunch: $('auto-launch'), showLanding: $('show-landing'),
  games: $('stat-games'), humanRate: $('stat-human'), split: $('stat-split'), dataHint: $('data-hint'),
  train: $('train'), clear: $('clear'), epochs: $('epochs'), epochsValue: $('epochs-value'), epoch: $('epoch-label'),
  live: $('live'), trainAcc: $('stat-train'), valAcc: $('stat-val'), networkRate: $('stat-network'), chart: $('accuracy-chart'),
  decisions: $('decisions'), decisionsPanel: $('decisions-panel'), decisionsToggle: $('decisions-toggle'), netSvg: $('network')
};
const SIZES = [6, 24, 24, 3];
const STORAGE_KEY = 'cart-imitation-games-v1';
const ACTION_DOTS = ['#6f42c1', '#868e96', '#0f766e'];
const ACTION_FILLS = ['#e2d9f3', '#f1f3f5', '#cfe9e5'];
const scene = createScene(ui.game);
const diagram = createNetworkDiagram(ui.netSvg, SIZES, { outputCaption: 'p(a | s)' });
const state = {
  game: newEpisode(), phase: 'ready', driver: 'human', number: 1,
  keys: { left: false, right: false }, held: { left: false, right: false },
  games: loadGames(), current: [], path: [], model: createClassifier({ sizes: SIZES }), trained: false,
  training: null, history: [], tests: [], networkGames: { caught: 0, total: 0 }, split: null, logged: 0, tested: 0,
  banner: '', hold: 0, carry: 0, last: performance.now(), drawn: { background: 0, weights: 0 }, dataDirty: true
};

// ---------- Recorded data ----------
// Each example: s (6 features), a (action index), d (true landing − cart x, for the plot only), v (cart velocity).
function loadGames() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    return saved.map(game => ({ caught: game.caught, examples: game.examples.map(e => ({ s: e.slice(0, 6), a: e[6], d: e[7], v: e[8] })) }));
  } catch { return []; }
}
function saveGames() {
  try {
    const compact = state.games.map(game => ({ caught: game.caught, examples: game.examples.map(e => [...e.s, e.a, e.d, e.v].map(n => Math.round(n * 1e4) / 1e4)) }));
    localStorage.setItem(STORAGE_KEY, JSON.stringify(compact));
  } catch { /* storage unavailable: data lasts for this visit only */ }
}
function isValidation(index) { return index % 5 === 4; }
function splitData() {
  if (!state.split) {
    const train = [], validation = [];
    state.games.forEach((game, i) => (isValidation(i) ? validation : train).push(...game.examples));
    state.split = { train, validation };
  }
  return state.split;
}
function dataChanged() { state.split = null; state.dataDirty = true; }

// ---------- Game loop ----------
function motorInput() {
  const left = state.keys.left || state.held.left, right = state.keys.right || state.held.right;
  return (right ? 1 : 0) - (left ? 1 : 0);
}
function networkProbabilities(game) { return actionProbabilities(state.model.net, features(game)); }

function prepare() {
  state.game = newEpisode();
  state.phase = 'ready';
  state.current = [];
  state.path = [];
  state.number++;
  state.drawn.background = 0;
  ui.launch.disabled = false;
}

function launch() {
  if (state.phase !== 'ready') return;
  if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur();
  state.phase = 'flying';
  state.banner = '';
  state.carry = 0;
  ui.launch.disabled = true;
}

function finish() {
  const game = state.game;
  state.path.push({ d: game.landing - game.cart.x, v: game.cart.v });
  state.bannerCaught = game.caught;
  state.banner = `${state.driver === 'network' ? 'Network: ' : ''}${game.caught ? 'Caught' : `Missed by ${game.miss.toFixed(1)} m`}`;
  if (state.driver === 'human' && ui.record.checked) {
    state.games.push({ caught: game.caught, examples: state.current });
    saveGames();
    dataChanged();
    if (ui.live.checked && state.trained) { logAccuracy(); runTest(); }
  } else if (state.driver === 'network') {
    state.networkGames.total++;
    if (game.caught) state.networkGames.caught++;
    showTraining();
  }
  state.phase = 'done';
  state.hold = 50;
}

function tick() {
  const game = state.game;
  if (state.phase === 'flying') {
    if (isDecisionTick(game)) {
      let action;
      if (state.driver === 'human') {
        action = motorInput() + 1;
        if (ui.record.checked) {
          const example = { s: features(game), a: action, d: game.landing - game.cart.x, v: game.cart.v };
          state.current.push(example);
          if (ui.live.checked) onlineUpdate(example);
        }
      } else action = argmax(networkProbabilities(game));
      state.path.push({ d: game.landing - game.cart.x, v: game.cart.v });
      game.motor = CART.ACTIONS[action];
    }
    physicsTick(game);
    if (game.done) finish();
  } else if (state.phase === 'done') {
    physicsTick(game);
    if (--state.hold <= 0) {
      prepare();
      if (state.driver === 'network' && ui.autoLaunch.checked) launch();
    }
  }
}

function setDriver(driver) {
  state.driver = driver;
  ui.human.setAttribute('aria-pressed', String(driver === 'human'));
  ui.network.setAttribute('aria-pressed', String(driver === 'network'));
  ui.left.disabled = ui.right.disabled = driver !== 'human';
  ui.hint.textContent = driver === 'human'
    ? 'Space launches · hold ← / → (or A / D) to run the motor · let go to coast'
    : 'Space launches · the network picks the action with the highest probability every 0.05 s';
  state.banner = '';
  state.number--;
  prepare();
}

// ---------- Training ----------
// "Train network" starts a fresh network and trains it for a fixed number of epochs.
function startTraining() {
  const { train } = splitData();
  if (!train.length) return;
  state.model = createClassifier({ sizes: SIZES });
  state.history = [];
  state.tests = [];
  state.trained = true;
  state.networkGames = { caught: 0, total: 0 };
  const epochs = Number(ui.epochs.value);
  state.training = { total: epochs, updates: epochs * Math.ceil(train.length / 32) };
  syncButtons();
}

function logAccuracy() {
  const { train, validation } = splitData();
  state.history.push({ updates: state.model.updates, train: evaluateClassifier(state.model, train).accuracy, validation: evaluateClassifier(state.model, validation).accuracy });
  state.logged = state.model.updates;
}

function trainSome() {
  const { train } = splitData();
  const end = performance.now() + 12;
  let epochs = 0;
  while (state.training && performance.now() < end && epochs < 3) {
    trainEpoch(state.model, train, { rate: 0.003 });
    logAccuracy();
    epochs++;
    if (state.model.epochs >= state.training.total) {
      state.training = null;
      runTest();
      syncButtons();
    }
  }
  showTraining();
}

// Live training: one update per recorded step, on that example plus 31 random earlier training examples.
// (Training on the newest example alone just chases whatever key you are holding right now.)
function onlineUpdate(example) {
  if (state.training || isValidation(state.games.length)) return;
  const { train } = splitData();
  const earlier = train.length + state.current.length - 1;
  const batch = [example];
  for (let i = 0; i < Math.min(31, earlier); i++) {
    const k = Math.floor(Math.random() * earlier);
    batch.push(k < train.length ? train[k] : state.current[k - train.length]);
  }
  trainBatch(state.model, batch, 0.003);
  if (!state.trained) { state.trained = true; syncButtons(); }
  if (state.model.updates - state.logged >= 30) { logAccuracy(); showTraining(); }
}

function runTest(count = 200) {
  let caught = 0;
  for (let i = 0; i < count; i++) {
    const game = newEpisode();
    while (!game.done) controlStep(game, argmax(networkProbabilities(game)));
    if (game.caught) caught++;
  }
  state.tests.push({ updates: state.model.updates, caught, count });
  state.tested = state.model.updates;
  showTraining();
}

// ---------- Rendering ----------
const percent = (x) => Number.isFinite(x) ? `${Math.round(100 * x)}%` : '–';

function showData() {
  const { train, validation } = splitData();
  const all = train.concat(validation);
  const caught = state.games.filter(game => game.caught).length;
  const trainGames = state.games.filter((_, i) => !isValidation(i)).length;
  ui.games.textContent = String(state.games.length);
  ui.humanRate.textContent = state.games.length ? percent(caught / state.games.length) : '–';
  ui.split.textContent = `${trainGames} / ${state.games.length - trainGames}`;
  ui.dataHint.hidden = all.length > 0;
  syncButtons();
  if (!ui.decisionsPanel.hidden) drawDots();
}

function syncButtons() {
  ui.train.disabled = !splitData().train.length || !!state.training;
}

function showTraining() {
  const last = state.history[state.history.length - 1];
  ui.epoch.textContent = state.training ? `epoch ${state.model.epochs} / ${state.training.total}` : '';
  ui.trainAcc.textContent = last ? percent(last.train) : '–';
  ui.valAcc.textContent = last ? percent(last.validation) : '–';
  const watched = state.networkGames;
  const test = state.tests[state.tests.length - 1];
  ui.networkRate.textContent = test ? percent(test.caught / test.count) : '–';
  ui.networkRate.title = [test && `${test.caught} of ${test.count} test balls`, watched.total && `${watched.caught} of ${watched.total} games you watched`].filter(Boolean).join(' · ');
  drawAccuracyChart();
}

function drawAccuracyChart() {
  const root = ui.chart, L = 46, R = 486, T = 12, B = 166;
  root.replaceChildren();
  const lastLogged = state.history.length ? state.history[state.history.length - 1].updates : 0;
  const total = Math.max(100, state.training ? state.training.updates : 0, lastLogged, state.model.updates);
  const x = updates => L + updates / total * (R - L);
  const y = acc => B - acc * (B - T);
  svgNode(root, 'rect', { x: L, y: T, width: R - L, height: B - T, fill: '#fff', stroke: '#adb5bd' });
  for (const acc of [0, 0.25, 0.5, 0.75, 1]) {
    svgNode(root, 'line', { x1: L, x2: R, y1: y(acc), y2: y(acc), stroke: '#e9ecef' });
    svgNode(root, 'text', { x: L - 8, y: y(acc) + 4, 'text-anchor': 'end', class: 'chart-label' }, `${acc * 100}%`);
  }
  for (const tick of [0, Math.round(total / 2), total]) svgNode(root, 'text', { x: x(tick), y: B + 16, 'text-anchor': 'middle', class: 'chart-label' }, tick.toLocaleString());
  svgNode(root, 'text', { x: (L + R) / 2, y: 196, 'text-anchor': 'middle', class: 'chart-label' }, 'training updates (minibatches)');
  const step = Math.max(1, Math.ceil(state.history.length / 500));
  const line = (key, attributes) => {
    const points = state.history.filter((h, i) => (i % step === 0 || i === state.history.length - 1) && Number.isFinite(h[key])).map(h => `${x(h.updates).toFixed(1)} ${y(h[key]).toFixed(1)}`);
    if (points.length > 1) svgNode(root, 'path', { d: `M ${points.join(' L ')}`, fill: 'none', 'stroke-width': 2, ...attributes });
    else if (points.length) { const [cx, cy] = points[0].split(' '); svgNode(root, 'circle', { cx, cy, r: 2.5, fill: attributes.stroke }); }
  };
  line('validation', { stroke: '#6c757d', 'stroke-dasharray': '5 4' });
  line('train', { stroke: '#0d6efd' });
  for (const test of state.tests) svgNode(root, 'circle', { cx: x(test.updates), cy: y(test.caught / test.count), r: 3.5, fill: 'white', stroke: '#212529', 'stroke-width': 1.5 });
}

const PLOT = { L: 50, R: 486, T: 10, B: 196, cols: 72, rows: 32, dmax: 15, vmax: 12 };
function plotX(d) { return PLOT.L + (Math.max(-PLOT.dmax, Math.min(PLOT.dmax, d)) + PLOT.dmax) / (2 * PLOT.dmax) * (PLOT.R - PLOT.L); }
function plotY(v) { return PLOT.B - (Math.max(-PLOT.vmax, Math.min(PLOT.vmax, v)) + PLOT.vmax) / (2 * PLOT.vmax) * (PLOT.B - PLOT.T); }
const plotLayers = (() => {
  const root = ui.decisions;
  const background = svgNode(root, 'g');
  svgNode(root, 'rect', { x: PLOT.L, y: PLOT.T, width: PLOT.R - PLOT.L, height: PLOT.B - PLOT.T, fill: 'none', stroke: '#adb5bd' });
  svgNode(root, 'line', { x1: plotX(0), x2: plotX(0), y1: PLOT.T, y2: PLOT.B, stroke: '#adb5bd', 'stroke-opacity': 0.7 });
  svgNode(root, 'line', { x1: PLOT.L, x2: PLOT.R, y1: plotY(0), y2: plotY(0), stroke: '#adb5bd', 'stroke-opacity': 0.7 });
  for (const d of [-15, -10, -5, 0, 5, 10, 15]) svgNode(root, 'text', { x: plotX(d), y: PLOT.B + 15, 'text-anchor': 'middle', class: 'chart-label' }, d > 0 ? `+${d}` : d < 0 ? `−${-d}` : '0');
  for (const v of [-12, 0, 12]) svgNode(root, 'text', { x: PLOT.L - 7, y: plotY(v) + 4, 'text-anchor': 'end', class: 'chart-label' }, v > 0 ? `+${v}` : v < 0 ? `−${-v}` : '0');
  svgNode(root, 'text', { x: (PLOT.L + PLOT.R) / 2, y: PLOT.B + 32, 'text-anchor': 'middle', class: 'chart-label' }, 'landing spot − cart x (m) · ball is to the right →');
  svgNode(root, 'text', { x: 12, y: (PLOT.T + PLOT.B) / 2, 'text-anchor': 'middle', class: 'chart-label', transform: `rotate(-90 12 ${(PLOT.T + PLOT.B) / 2})` }, 'cart velocity cv (m/s)');
  const dots = svgNode(root, 'g');
  const current = svgNode(root, 'g');
  return { background, dots, current };
})();

// Background: the network's choice for the current ball at every (landing − cart x, cart v); blank where the cart would be off the track.
function drawBackground() {
  const root = plotLayers.background, game = state.game;
  root.replaceChildren();
  const probe = { ball: game.ball, cart: { x: 0, v: 0 } };
  const cw = (PLOT.R - PLOT.L) / PLOT.cols, rh = (PLOT.B - PLOT.T) / PLOT.rows;
  const paths = ['', '', ''];
  for (let row = 0; row < PLOT.rows; row++) {
    probe.cart.v = PLOT.vmax - (row + 0.5) * 2 * PLOT.vmax / PLOT.rows;
    const y = PLOT.T + row * rh;
    let runAction = -1, runStart = 0;
    for (let col = 0; col <= PLOT.cols; col++) {
      let action = -1;
      if (col < PLOT.cols) {
        probe.cart.x = game.landing - (-PLOT.dmax + (col + 0.5) * 2 * PLOT.dmax / PLOT.cols);
        if (probe.cart.x >= CART.LIMIT && probe.cart.x <= CART.TRACK - CART.LIMIT) action = argmax(outputs(state.model.net, features(probe)));
      }
      if (action !== runAction) {
        if (runAction >= 0) paths[runAction] += `M ${(PLOT.L + runStart * cw).toFixed(2)} ${y.toFixed(2)} h ${((col - runStart) * cw).toFixed(2)} v ${rh.toFixed(2)} h ${(-(col - runStart) * cw).toFixed(2)} Z `;
        runAction = action;
        runStart = col;
      }
    }
  }
  if (state.trained) paths.forEach((d, i) => svgNode(root, 'path', { d, fill: ACTION_FILLS[i] }));
}

function drawDots() {
  const root = plotLayers.dots;
  root.replaceChildren();
  const paths = ['', '', ''], r = 1.8;
  for (const game of state.games) for (const e of game.examples) {
    paths[e.a] += `M ${(plotX(e.d) - r).toFixed(1)} ${plotY(e.v).toFixed(1)} a ${r} ${r} 0 1 0 ${2 * r} 0 a ${r} ${r} 0 1 0 ${-2 * r} 0 `;
  }
  paths.forEach((d, i) => svgNode(root, 'path', { d, fill: ACTION_DOTS[i], 'fill-opacity': 0.55 }));
}

function drawCurrent() {
  const root = plotLayers.current, game = state.game;
  root.replaceChildren();
  const points = state.path.concat(game.done ? [] : [{ d: game.landing - game.cart.x, v: game.cart.v }]);
  if (points.length > 1) svgNode(root, 'path', { d: `M ${points.map(p => `${plotX(p.d).toFixed(1)} ${plotY(p.v).toFixed(1)}`).join(' L ')}`, fill: 'none', stroke: '#212529', 'stroke-width': 1.5 });
  const now = points[points.length - 1];
  svgNode(root, 'circle', { cx: plotX(now.d), cy: plotY(now.v), r: 5, fill: 'white', stroke: '#212529', 'stroke-width': 2 });
}

function render(now) {
  const game = state.game;
  drawScene(scene, game, { showLanding: ui.showLanding.checked, banner: state.banner, bannerCaught: state.bannerCaught });
  ui.episode.textContent = `Game ${state.number}`;
  const p = networkProbabilities(game);
  const network = `network: ${state.trained ? CART.ACTION_NAMES[argmax(p)].toLowerCase() : 'untrained'}`;
  if (state.phase === 'ready') ui.status.textContent = state.driver === 'network' && !state.trained ? 'Train the network first, or watch an untrained one flail' : 'Ready · press Space or Launch ball';
  else if (state.phase === 'flying') ui.status.textContent = state.driver === 'human' ? `you: ${CART.ACTION_NAMES[game.motor + 1].toLowerCase()} · ${network}${ui.record.checked && isValidation(state.games.length) ? ' · validation game' : ''}` : `t = ${game.time.toFixed(2)} s · ${network}`;
  else ui.status.textContent = state.banner;
  if (state.dataDirty) { showData(); state.dataDirty = false; }
  const showPlot = !ui.decisionsPanel.hidden;
  if (showPlot && !game.done && now - state.drawn.background > 150) { drawBackground(); state.drawn.background = now; }
  if (now - state.drawn.weights > 200) { drawNetworkWeights(diagram, state.model.net); state.drawn.weights = now; }
  drawNetworkActivations(diagram, forwardPass(state.model.net, features(game)), { chosen: state.trained ? argmax(p) : null, probabilities: p });
  if (showPlot) drawCurrent();
}

function frame(now) {
  const elapsed = Math.min(100, now - state.last);
  state.last = now;
  if (state.phase !== 'ready') {
    state.carry += elapsed / 1000 * Number(ui.speed.value) / CART.DT;
    while (state.carry >= 1 && state.phase !== 'ready') { state.carry--; tick(); }
  }
  if (state.training) trainSome();
  render(now);
  requestAnimationFrame(frame);
}

// ---------- Controls ----------
const KEYS = { ArrowLeft: 'left', a: 'left', A: 'left', ArrowRight: 'right', d: 'right', D: 'right' };
document.addEventListener('keydown', (event) => {
  if (event.target.matches('select, input:not([type=range])')) return;
  if (event.target.matches('input[type=range]') && KEYS[event.key] && state.phase !== 'flying') return;
  if (KEYS[event.key]) { event.preventDefault(); state.keys[KEYS[event.key]] = true; }
  else if (event.key === ' ') { event.preventDefault(); launch(); }
});
// After a mouse click on a checkbox, drop its focus so Space launches the ball instead of toggling it again.
let usedPointer = false;
document.addEventListener('pointerdown', () => { usedPointer = true; }, true);
document.addEventListener('keydown', () => { usedPointer = false; }, true);
document.addEventListener('change', (event) => { if (usedPointer && event.target.matches('input[type=checkbox]')) event.target.blur(); });
document.addEventListener('keyup', (event) => { if (KEYS[event.key]) state.keys[KEYS[event.key]] = false; });
window.addEventListener('blur', () => { state.keys.left = state.keys.right = state.held.left = state.held.right = false; });
for (const [button, side] of [[ui.left, 'left'], [ui.right, 'right']]) {
  button.addEventListener('pointerdown', (event) => { state.held[side] = true; button.setPointerCapture(event.pointerId); });
  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) button.addEventListener(type, () => { state.held[side] = false; });
}
ui.launch.addEventListener('click', launch);
ui.human.addEventListener('click', () => setDriver('human'));
ui.network.addEventListener('click', () => setDriver('network'));
ui.train.addEventListener('click', startTraining);
ui.clear.addEventListener('click', () => {
  if (state.games.length && !confirm(`Delete all ${state.games.length} recorded games?`)) return;
  state.games = [];
  saveGames();
  dataChanged();
});
ui.live.addEventListener('change', () => {
  state.logged = state.model.updates;
  if (!ui.live.checked && state.trained) runTest();
  showTraining();
});
ui.epochs.addEventListener('input', () => { ui.epochsValue.textContent = ui.epochs.value; });
ui.speed.addEventListener('input', () => { ui.speedValue.textContent = `${Number(ui.speed.value).toFixed(2)}×`; });
bindGameScale(ui.scale, ui.scaleValue, ui.page);
bindPanelToggle(ui.decisionsToggle, ui.decisionsPanel, () => { drawDots(); drawBackground(); drawCurrent(); });

setDriver('human');
showTraining();
requestAnimationFrame(frame);
