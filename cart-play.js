const $ = (id) => document.getElementById(id);
const ui = {
  page: $('page'), game: $('game'), status: $('play-status'), episode: $('episode-label'), count: $('count-label'),
  launch: $('launch'), left: $('drive-left'), right: $('drive-right'), reset: $('reset'),
  speed: $('speed'), speedValue: $('speed-value'), scale: $('scale'), scaleValue: $('scale-value'), showLanding: $('show-landing'),
  rows: $('state-rows'), chart: $('reward-chart'), caught: $('stat-caught'), missed: $('stat-missed'), rate: $('stat-rate')
};
const scene = createScene(ui.game);
const state = { game: newEpisode(), phase: 'ready', number: 1, keys: { left: false, right: false }, held: { left: false, right: false }, episodes: [], banner: '', hold: 0, carry: 0, last: performance.now() };

const stateCells = CART.FEATURES.map(feature => {
  const row = document.createElement('tr');
  row.innerHTML = `<td class="mono">${feature.name}</td><td>${feature.label}</td><td class="mono raw"></td><td class="bar-cell"><span class="value-bar"><i></i></span></td><td class="mono scaled"></td>`;
  ui.rows.appendChild(row);
  return { raw: row.querySelector('.raw'), bar: row.querySelector('i'), scaled: row.querySelector('.scaled'), unit: feature.unit };
});

function motorInput() {
  const left = state.keys.left || state.held.left, right = state.keys.right || state.held.right;
  return (right ? 1 : 0) - (left ? 1 : 0);
}

function prepare() {
  state.game = newEpisode();
  state.phase = 'ready';
  state.number = state.episodes.length + 1;
  ui.launch.disabled = false;
  ui.launch.textContent = state.episodes.length ? 'Next ball' : 'Launch ball';
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
  const reward = terminalReward(game, 'distance');
  state.episodes.push({ reward, caught: game.caught, miss: game.miss });
  state.banner = outcomeText(game, reward);
  state.bannerCaught = game.caught;
  state.phase = 'done';
  state.hold = 50;
  showStats();
}

function tick() {
  const game = state.game;
  if (state.phase === 'flying') {
    if (isDecisionTick(game)) game.motor = motorInput();
    physicsTick(game);
    if (game.done) finish();
  } else if (state.phase === 'done') {
    physicsTick(game);
    if (--state.hold <= 0) prepare();
  }
}

function showStats() {
  const stats = episodeStats(state.episodes, 20);
  ui.caught.textContent = String(stats.caught);
  ui.missed.textContent = String(stats.missed);
  ui.rate.textContent = stats.total ? `${Math.round(100 * stats.rate)}%` : '–';
  ui.count.textContent = `${stats.total} ${stats.total === 1 ? 'episode' : 'episodes'}`;
  drawRewardChart(ui.chart, state.episodes, { window: 20, minSpan: 20 });
}

function render() {
  const game = state.game;
  drawScene(scene, game, { showLanding: ui.showLanding.checked, banner: state.banner, bannerCaught: state.bannerCaught });
  ui.episode.textContent = `Episode ${state.number}`;
  if (state.phase === 'ready') ui.status.textContent = 'Ready · press Space or Launch ball';
  else if (state.phase === 'flying') ui.status.textContent = `t = ${game.time.toFixed(2)} s · motor ${CART.ACTION_NAMES[game.motor + 1].toLowerCase()}`;
  else ui.status.textContent = state.banner;
  const raw = rawState(game), scaled = features(game);
  stateCells.forEach((cell, i) => {
    cell.raw.textContent = `${raw[i].toFixed(2)} ${cell.unit}`;
    cell.scaled.textContent = signedValue(scaled[i]);
    const width = Math.min(1, Math.abs(scaled[i]) / 1.5) * 50;
    cell.bar.style.left = `${scaled[i] < 0 ? 50 - width : 50}%`;
    cell.bar.style.width = `${width}%`;
  });
}

function frame(now) {
  const elapsed = Math.min(100, now - state.last);
  state.last = now;
  if (state.phase === 'flying' || state.phase === 'done') {
    state.carry += elapsed / 1000 * Number(ui.speed.value) / CART.DT;
    while (state.carry >= 1 && state.phase !== 'ready') { state.carry--; tick(); }
  }
  render();
  requestAnimationFrame(frame);
}

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
ui.reset.addEventListener('click', () => {
  state.episodes = [];
  state.banner = '';
  prepare();
  showStats();
});
ui.speed.addEventListener('input', () => { ui.speedValue.textContent = `${Number(ui.speed.value).toFixed(2)}×`; });
bindGameScale(ui.scale, ui.scaleValue, ui.page);

showStats();
requestAnimationFrame(frame);
