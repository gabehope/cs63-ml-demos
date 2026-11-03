// Shared cart-control game: physics, state features, and SVG drawing helpers.
// No top-level DOM access, so the physics can also be loaded outside the browser.
const CART = {
  TRACK: 20, G: 9.8, DT: 1 / 60, SUBSTEPS: 3,
  MOTOR: 14, DRAG: 0.6, ROLL: 2,
  RIM: 1.0, CATCH: 0.9, LIMIT: 1.1, LAUNCH_X: 0.4, LAUNCH_Y: 0.6, BALL_R: 0.25,
  ACTIONS: [-1, 0, 1], ACTION_NAMES: ['Left', 'Coast', 'Right'],
  FEATURES: [
    { name: 'bx', label: 'ball x', unit: 'm' }, { name: 'by', label: 'ball y', unit: 'm' },
    { name: 'bvx', label: 'ball vx', unit: 'm/s' }, { name: 'bvy', label: 'ball vy', unit: 'm/s' },
    { name: 'cx', label: 'cart x', unit: 'm' }, { name: 'cv', label: 'cart v', unit: 'm/s' }
  ]
};

// Each episode picks a landing spot and flight time, then solves for the launch velocity.
function newEpisode(random = Math.random) {
  const landing = 5 + random() * 14;
  const flight = 1.7 + random() * 0.8;
  const vx = (landing - CART.LAUNCH_X) / flight;
  const vy = (CART.RIM - CART.LAUNCH_Y + CART.G * flight * flight / 2) / flight;
  return {
    ball: { x: CART.LAUNCH_X, y: CART.LAUNCH_Y, vx, vy },
    cart: { x: 2 + random() * 16, v: 0, travel: 0 },
    launch: { vx, vy, angle: Math.atan2(vy, vx) },
    motor: 0, time: 0, ticks: 0, landing, flight,
    done: false, caught: false, miss: 0, endTime: 0, offset: 0
  };
}

// One physics tick of length DT using the motor setting in game.motor (−1, 0, +1).
// After the episode ends the cart coasts and the ball either rides in the basket or drops to the ground.
function physicsTick(game) {
  const { ball, cart } = game;
  const dt = CART.DT;
  if (game.done) game.motor = 0;
  const before = { x: ball.x, y: ball.y, cart: cart.x };

  let v = cart.v + (CART.MOTOR * game.motor - CART.DRAG * cart.v) * dt;
  if (game.motor === 0) {
    const roll = CART.ROLL * dt;
    v = Math.abs(v) <= roll ? 0 : v - Math.sign(v) * roll;
  }
  let x = cart.x + v * dt;
  if (x < CART.LIMIT || x > CART.TRACK - CART.LIMIT) {
    x = Math.min(CART.TRACK - CART.LIMIT, Math.max(CART.LIMIT, x));
    v = 0;
  }
  cart.travel += x - cart.x;
  cart.x = x;
  cart.v = v;
  game.time += dt;
  game.ticks++;

  if (game.done && game.caught) {
    ball.x = cart.x + game.offset;
    ball.y = CART.RIM - 0.22;
    return;
  }
  if (game.done && ball.y <= CART.BALL_R) return;
  ball.vy -= CART.G * dt;
  ball.x += ball.vx * dt;
  ball.y = Math.max(CART.BALL_R, ball.y + ball.vy * dt);
  if (game.done && ball.y <= CART.BALL_R) ball.vx = ball.vy = 0;

  if (!game.done && ball.vy < 0 && before.y > CART.RIM && ball.y <= CART.RIM) {
    const f = (before.y - CART.RIM) / (before.y - ball.y);
    const ballX = before.x + f * (ball.x - before.x);
    const cartX = before.cart + f * (cart.x - before.cart);
    game.miss = Math.abs(ballX - cartX);
    game.caught = game.miss <= CART.CATCH;
    game.offset = Math.max(-0.5, Math.min(0.5, ballX - cartX));
    game.endTime = game.time - (1 - f) * dt;
    game.done = true;
  }
}

// Advance one decision step: hold the chosen action for SUBSTEPS ticks (or until the ball arrives).
function controlStep(game, action, rewardMode = 'distance') {
  game.motor = CART.ACTIONS[action];
  for (let i = 0; i < CART.SUBSTEPS && !game.done; i++) physicsTick(game);
  return game.done ? terminalReward(game, rewardMode) : 0;
}
function isDecisionTick(game) { return !game.done && game.ticks % CART.SUBSTEPS === 0; }

function terminalReward(game, rewardMode = 'distance') {
  if (game.caught) return 1;
  return rewardMode === 'distance' ? -Math.min(1, game.miss / 10) : -1;
}

function rawState(game) {
  const { ball, cart } = game;
  return [ball.x, ball.y, ball.vx, ball.vy, cart.x, cart.v];
}
// The six numbers the player (or agent) sees, scaled to roughly −1…1.
function features(game) {
  const { ball, cart } = game;
  return [
    ball.x / 10 - 1, ball.y / 4 - 1, ball.vx / 5 - 1, ball.vy / 10,
    cart.x / 10 - 1, cart.v / 10
  ];
}

function predictedLanding(game) {
  const { ball } = game;
  const drop = ball.y - CART.RIM;
  const t = (ball.vy + Math.sqrt(Math.max(0, ball.vy * ball.vy + 2 * CART.G * drop))) / CART.G;
  return ball.x + ball.vx * t;
}

// ---------- Drawing ----------
const CART_SVG = 'http://www.w3.org/2000/svg';
function svgNode(parent, name, attributes = {}, label) {
  const node = document.createElementNS(CART_SVG, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (label !== undefined) node.textContent = label;
  parent.appendChild(node);
  return node;
}
function signedValue(n, digits = 2) { return `${n < 0 ? '−' : '+'}${Math.abs(n).toFixed(digits)}`; }

const VIEW = { left: 20, ground: 268, scale: 28 };
function screenX(x) { return VIEW.left + x * VIEW.scale; }
function screenY(y) { return VIEW.ground - y * VIEW.scale; }

// Builds the static scene inside an <svg viewBox="0 0 600 300"> and returns handles for animation.
function createScene(root) {
  root.replaceChildren();
  svgNode(root, 'line', { x1: screenX(0), y1: VIEW.ground, x2: screenX(CART.TRACK), y2: VIEW.ground, stroke: '#adb5bd' });
  for (let m = 0; m <= CART.TRACK; m += 5) {
    svgNode(root, 'line', { x1: screenX(m), y1: VIEW.ground, x2: screenX(m), y2: VIEW.ground + 5, stroke: '#adb5bd' });
    svgNode(root, 'text', { x: screenX(m), y: VIEW.ground + 18, 'text-anchor': 'middle' }, `${m} m`);
  }
  for (const end of [CART.LIMIT - CART.CATCH, CART.TRACK - CART.LIMIT + CART.CATCH]) {
    svgNode(root, 'rect', { x: screenX(end) - 2, y: VIEW.ground - 18, width: 4, height: 18, fill: '#ced4da' });
  }
  const marker = svgNode(root, 'g', { opacity: 0 });
  svgNode(marker, 'line', { y1: screenY(CART.RIM), y2: VIEW.ground, stroke: '#0d6efd', 'stroke-dasharray': '3 3' });
  svgNode(marker, 'path', { d: `M -5 ${screenY(CART.RIM) - 7} L 5 ${screenY(CART.RIM) - 7} L 0 ${screenY(CART.RIM)} Z`, fill: '#0d6efd' });
  const trail = svgNode(root, 'path', { fill: 'none', stroke: '#adb5bd', 'stroke-opacity': 0.6, 'stroke-width': 1.5 });
  const barrel = svgNode(root, 'line', { x1: screenX(CART.LAUNCH_X), y1: screenY(CART.LAUNCH_Y), stroke: '#6c757d', 'stroke-width': 6, 'stroke-linecap': 'round' });
  svgNode(root, 'circle', { cx: screenX(CART.LAUNCH_X), cy: screenY(CART.LAUNCH_Y), r: 10, fill: '#e9ecef', stroke: '#6c757d', 'stroke-width': 2 });

  const cart = svgNode(root, 'g');
  const half = CART.CATCH * VIEW.scale;
  const rim = screenY(CART.RIM) - VIEW.ground;
  const thrust = svgNode(cart, 'path', { fill: '#fd7e14', opacity: 0 });
  svgNode(cart, 'path', { d: `M ${-half} ${rim} L ${-half + 6} -14 L ${half - 6} -14 L ${half} ${rim}`, fill: '#e9ecef', stroke: '#495057', 'stroke-width': 3, 'stroke-linejoin': 'round' });
  const wheels = [-half + 10, half - 10].map(cx => {
    const wheel = svgNode(cart, 'g', { transform: `translate(${cx} -7)` });
    svgNode(wheel, 'circle', { r: 7, fill: '#495057' });
    svgNode(wheel, 'path', { d: 'M -5 0 H 5 M 0 -5 V 5', stroke: '#ced4da', 'stroke-width': 1.5 });
    return { node: wheel, cx };
  });
  const ball = svgNode(root, 'circle', { r: CART.BALL_R * VIEW.scale, fill: '#212529' });
  const banner = svgNode(root, 'text', { x: 300, y: 26, 'text-anchor': 'middle', class: 'scene-banner' });
  return { root, trail, barrel, cart, wheels, thrust, ball, marker, banner };
}

function ballAt(game, t) {
  return { x: CART.LAUNCH_X + game.launch.vx * t, y: CART.LAUNCH_Y + game.launch.vy * t - CART.G * t * t / 2 };
}

function drawScene(scene, game, options = {}) {
  const { ball, cart } = game;
  const end = game.done && game.caught ? game.endTime : game.time;
  const points = [];
  const segments = Math.max(1, Math.ceil(end / 0.04));
  for (let i = 0; i <= segments; i++) {
    const p = ballAt(game, end * i / segments);
    if (p.y < CART.BALL_R) break;
    points.push(`${screenX(p.x).toFixed(1)} ${screenY(p.y).toFixed(1)}`);
  }
  scene.trail.setAttribute('d', points.length > 1 ? `M ${points.join(' L ')}` : '');
  scene.barrel.setAttribute('x2', screenX(CART.LAUNCH_X) + 22 * Math.cos(game.launch.angle));
  scene.barrel.setAttribute('y2', screenY(CART.LAUNCH_Y) - 22 * Math.sin(game.launch.angle));

  scene.ball.setAttribute('cx', screenX(ball.x).toFixed(2));
  scene.ball.setAttribute('cy', screenY(ball.y).toFixed(2));
  scene.cart.setAttribute('transform', `translate(${screenX(cart.x).toFixed(2)} ${VIEW.ground})`);
  const spin = cart.travel / (7 / VIEW.scale) * 180 / Math.PI;
  for (const wheel of scene.wheels) wheel.node.setAttribute('transform', `translate(${wheel.cx} -7) rotate(${spin.toFixed(1)})`);
  if (game.motor && !game.done) {
    const back = -game.motor * (CART.CATCH * VIEW.scale - 4);
    scene.thrust.setAttribute('d', `M ${back} -21 L ${back - game.motor * 15} -17 L ${back} -13 Z`);
    scene.thrust.setAttribute('opacity', 0.9);
  } else scene.thrust.setAttribute('opacity', 0);

  const showLanding = options.showLanding && !game.done;
  scene.marker.setAttribute('opacity', showLanding ? 1 : 0);
  if (showLanding) scene.marker.setAttribute('transform', `translate(${screenX(predictedLanding(game)).toFixed(2)} 0)`);
  scene.banner.textContent = options.banner || '';
  const caught = options.bannerCaught ?? game.caught;
  scene.banner.setAttribute('class', `scene-banner ${caught ? 'caught' : 'missed'}`);
}

function outcomeText(game, reward) {
  return game.caught ? `Caught · reward ${signedValue(reward)}` : `Missed by ${game.miss.toFixed(1)} m · reward ${signedValue(reward)}`;
}

// Moving average of episode reward as an SVG path (sampled every `step` episodes).
function movingAveragePath(episodes, window, x, y, step = 1) {
  let d = '', sum = 0;
  for (let i = 0; i < episodes.length; i++) {
    sum += episodes[i].reward;
    if (i >= window) sum -= episodes[i - window].reward;
    if (i % step && i !== episodes.length - 1) continue;
    d += `${d ? 'L' : 'M'} ${x(i).toFixed(1)} ${y(sum / Math.min(window, i + 1)).toFixed(1)} `;
  }
  return d;
}

// Reward per episode: one dot per episode plus a moving average.
// options.previous: earlier runs ({ episodes, color }) drawn as faded average lines for comparison.
function drawRewardChart(root, episodes, options = {}) {
  root.replaceChildren();
  const H = 220, L = 46, R = 486, T = 12, B = 182;
  const window = options.window || 20;
  const count = episodes.length;
  const previous = options.previous || [];
  const span = Math.max(options.minSpan || 20, count, ...previous.map(run => run.episodes.length));
  const x = i => L + i / (span - 1) * (R - L);
  const y = r => B - (r + 1) / 2 * (B - T);
  svgNode(root, 'rect', { x: L, y: T, width: R - L, height: B - T, fill: '#fff', stroke: '#adb5bd' });
  for (const r of [-1, -0.5, 0, 0.5, 1]) {
    svgNode(root, 'line', { x1: L, x2: R, y1: y(r), y2: y(r), stroke: '#e9ecef' });
    if (Number.isInteger(r)) svgNode(root, 'text', { x: L - 8, y: y(r) + 4, 'text-anchor': 'end', class: 'chart-label' }, r > 0 ? '+1' : r < 0 ? '−1' : '0');
  }
  for (const tick of [1, Math.round(span / 2), span]) svgNode(root, 'text', { x: x(tick - 1), y: B + 16, 'text-anchor': 'middle', class: 'chart-label' }, String(tick));
  svgNode(root, 'text', { x: (L + R) / 2, y: H - 4, 'text-anchor': 'middle', class: 'chart-label' }, 'episode');
  svgNode(root, 'text', { x: 12, y: (T + B) / 2, 'text-anchor': 'middle', class: 'chart-label', transform: `rotate(-90 12 ${(T + B) / 2})` }, 'reward');

  for (const run of previous) {
    if (run.episodes.length < 2) continue;
    const d = movingAveragePath(run.episodes, window, x, y, Math.max(1, Math.ceil(run.episodes.length / 700)));
    svgNode(root, 'path', { d, fill: 'none', stroke: run.color, 'stroke-width': 1.5, 'stroke-opacity': 0.75 });
  }
  const step = Math.max(1, Math.ceil(count / 700));
  const radius = count > 300 ? 1.6 : count > 80 ? 2.4 : 3.5;
  const dots = { caught: '', missed: '' };
  for (let i = 0; i < count; i += step) {
    const e = episodes[i];
    dots[e.caught ? 'caught' : 'missed'] += `M ${(x(i) - radius).toFixed(1)} ${y(e.reward).toFixed(1)} a ${radius} ${radius} 0 1 0 ${2 * radius} 0 a ${radius} ${radius} 0 1 0 ${-2 * radius} 0 `;
  }
  svgNode(root, 'path', { d: dots.missed, fill: '#dc3545', 'fill-opacity': 0.55 });
  svgNode(root, 'path', { d: dots.caught, fill: '#0f766e', 'fill-opacity': 0.55 });
  if (count >= 2) svgNode(root, 'path', { d: movingAveragePath(episodes, window, x, y, step), fill: 'none', stroke: '#0d6efd', 'stroke-width': 2 });
}

function episodeStats(episodes, window = 50) {
  const caught = episodes.filter(e => e.caught).length;
  const recent = episodes.slice(-window);
  const rate = recent.length ? recent.filter(e => e.caught).length / recent.length : 0;
  return { total: episodes.length, caught, missed: episodes.length - caught, rate, recent: recent.length };
}

// Network diagram inside an <svg viewBox="0 0 500 270">: weights drawn as bundled paths (sign × strength),
// nodes shaded by activation. options: outputLabels, outputCaption.
function createNetworkDiagram(root, sizes, options = {}) {
  const last = sizes.length - 1;
  const xs = sizes.map((_, l) => 84 + l * (430 - 84) / last);
  const spacing = n => n <= 3 ? 52 : n <= 8 ? 40 : 10.2;
  const layout = sizes.map((n, l) => Array.from({ length: n }, (_, i) => ({ x: xs[l], y: 135 + (i - (n - 1) / 2) * spacing(n) })));
  const weightPaths = [0, 1].map(sign => [0.12, 0.25, 0.45, 0.75].map(opacity => svgNode(root, 'path', { fill: 'none', stroke: sign ? '#0d6efd' : '#e8590c', 'stroke-opacity': opacity, 'stroke-width': 1 })));
  const radius = l => l === 0 ? 14 : l === last ? 15 : 4;
  layout.forEach((layer, l) => layer.forEach(p => svgNode(root, 'circle', { cx: p.x, cy: p.y, r: radius(l), class: 'q-node-bg' })));
  const nodes = layout.map((layer, l) => layer.map(p => svgNode(root, 'circle', { cx: p.x, cy: p.y, r: radius(l), class: 'q-node' })));
  CART.FEATURES.forEach((feature, i) => svgNode(root, 'text', { x: xs[0], y: layout[0][i].y + 4, 'text-anchor': 'middle' }, feature.name));
  const inputValues = layout[0].map(p => svgNode(root, 'text', { x: xs[0] - 22, y: p.y + 4, 'text-anchor': 'end', class: 'q-net-value' }));
  (options.outputLabels || ['←', '0', '→']).forEach((label, i) => svgNode(root, 'text', { x: xs[last], y: layout[last][i].y + 4, 'text-anchor': 'middle' }, label));
  const outputValues = layout[last].map(p => svgNode(root, 'text', { x: xs[last] + 22, y: p.y + 4, class: 'q-net-value' }));
  for (let l = 1; l < last; l++) svgNode(root, 'text', { x: xs[l], y: 266, 'text-anchor': 'middle', class: 'network-caption' }, `hidden ${l}`);
  svgNode(root, 'text', { x: xs[last], y: 266, 'text-anchor': 'middle', class: 'network-caption' }, options.outputCaption || 'Q(s, a)');
  return { layout, weightPaths, nodes, inputValues, outputValues };
}

function drawNetworkWeights(diagram, net) {
  const d = [['', '', '', ''], ['', '', '', '']];
  net.layers.forEach((layer, l) => {
    let max = 1e-9;
    for (const w of layer.W) max = Math.max(max, Math.abs(w));
    for (let j = 0; j < layer.outputs; j++) for (let k = 0; k < layer.inputs; k++) {
      const w = layer.W[j * layer.inputs + k], strength = Math.abs(w) / max;
      if (strength < 0.15) continue;
      const a = diagram.layout[l][k], b = diagram.layout[l + 1][j];
      d[w > 0 ? 1 : 0][Math.min(3, Math.floor(strength * 4))] += `M ${a.x} ${a.y.toFixed(1)} L ${b.x} ${b.y.toFixed(1)} `;
    }
  });
  diagram.weightPaths.forEach((group, sign) => group.forEach((path, level) => path.setAttribute('d', d[sign][level])));
}

// acts: every layer's activations. chosen: output index to outline. probabilities: show and shade outputs as p(a | s) instead of raw values.
function drawNetworkActivations(diagram, acts, { chosen = null, probabilities = null } = {}) {
  const last = acts.length - 1;
  acts.forEach((raw, l) => {
    const values = l === last && probabilities ? probabilities : raw;
    let max = 1e-9;
    for (const v of values) max = Math.max(max, Math.abs(v));
    if (l === 0 || l === last) max = Math.max(max, 1);
    diagram.nodes[l].forEach((node, i) => {
      const v = values[i], alpha = Math.min(1, Math.abs(v) / max) * (l === 0 || l === last ? 0.45 : 0.9);
      node.style.fill = v >= 0 ? `rgba(13, 110, 253, ${alpha.toFixed(3)})` : `rgba(232, 89, 12, ${alpha.toFixed(3)})`;
      node.style.strokeWidth = l === last && chosen === i ? '3' : '';
      node.style.stroke = l === last && chosen === i ? '#0d6efd' : '';
    });
  });
  acts[0].forEach((v, i) => { diagram.inputValues[i].textContent = signedValue(v); });
  acts[last].forEach((v, i) => { diagram.outputValues[i].textContent = probabilities ? `${Math.round(100 * probabilities[i])}%` : signedValue(v, 3); });
}

// Show/hide button for an optional panel; onShow redraws it when it opens.
function bindPanelToggle(button, panel, onShow) {
  button.addEventListener('click', () => {
    const show = panel.hidden;
    panel.hidden = !show;
    button.textContent = show ? 'Hide' : 'Show';
    button.setAttribute('aria-expanded', String(show));
    if (show && onShow) onShow();
  });
}

// Game size slider: scales the game panel's width (the side panel wraps below when it no longer fits).
function bindGameScale(input, output, layout) {
  const apply = () => {
    const scale = Number(input.value);
    layout.style.setProperty('--game-scale', scale);
    output.textContent = `${Math.round(scale * 100)}%`;
  };
  input.addEventListener('input', apply);
  apply();
}

if (typeof module !== 'undefined') module.exports = { CART, newEpisode, physicsTick, controlStep, isDecisionTick, terminalReward, rawState, features, predictedLanding };
