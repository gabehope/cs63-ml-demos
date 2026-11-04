// Continuous gridworld: the agent's position is any point (x, y); each move goes roughly the chosen way, with random
// variation in distance and angle. A small network learns either Q(x, y, a) for the four moves (Q-learning), or,
// on the page with data-method="reinforce", the move probabilities π(a | x, y) (REINFORCE).
const $ = (id) => document.getElementById(id);
const ui = {
  page: $('page'), world: $('world'), layers: $('layers'), main: $('main-plot'), mainTitle: $('main-title'), episode: $('episode-label'),
  small: [0, 1, 2, 3].map(a => $(`q-plot-${a}`)), arrows: $('show-arrows'),
  status: $('status'), update: $('update'),
  step: $('step'), continuous: $('continuous'), episodes: $('episodes'), reset: $('reset'),
  alpha: $('alpha'), gamma: $('gamma'), epsilon: $('epsilon'), length: $('length'), lengthNoise: $('length-noise'), angleNoise: $('angle-noise'),
  entropy: $('entropy'), baseline: $('baseline'), living: $('living'), speed: $('speed'), scale: $('scale'), scaleValue: $('scale-value'),
  netSvg: $('network'), log: $('update-log'), chart: $('reward-chart'), count: $('count-label'), statWin: $('stat-win'), statLose: $('stat-lose'), statReward: $('stat-reward')
};
const DIRECTIONS = [Math.PI / 2, 0, -Math.PI / 2, Math.PI];
const PG = document.body.dataset.method === 'reinforce';
const BASELINE_RATE = 0.02;
const HEAT_PER_TILE = 10, ARROWS_PER_TILE = 2;
const state = {
  net: null, policy: null, b: 0, traj: [], update: null, agent: null, trail: [], done: false, steps: 0, total: 0, returns: [], log: [], last: null,
  focus: null, inspect: null, heat: null, heatTime: 0, chartDirty: true, running: false, carry: 0, lastFrame: performance.now(), anim: null
};

// ---------- World ----------
// What is at a point: 'wall' (or outside), an exit's reward, or null for open floor.
function tileAt(x, y) {
  const tx = Math.floor(x), ty = Math.floor(y);
  if (tx < 0 || ty < 0 || tx >= GRID.width || ty >= GRID.height || GRID.walls.has(`${tx},${ty}`)) return 'wall';
  const exit = GRID.exits[`${tx},${ty}`];
  return exit === undefined ? null : exit;
}
const startPoint = () => ({ x: GRID.start.x + 0.5, y: GRID.start.y + 0.5 });
// One move: distance ≈ step length (± length noise) at angle ≈ the chosen direction (± angle noise).
// The agent stops at a wall; reaching an exit tile ends the episode. r is R(s), the living reward of where the move
// started; exit is the exit's reward, V(exit) = R(exit), when the move reaches one.
function contMove(s, a) {
  const distance = Math.max(0.05, Number(ui.length.value) + Number(ui.lengthNoise.value) * gaussian(Math.random));
  const angle = DIRECTIONS[a] + Number(ui.angleNoise.value) * Math.PI / 180 * gaussian(Math.random);
  const pieces = 12;
  let p = { ...s }, blocked = false;
  for (let i = 0; i < pieces; i++) {
    const q = { x: p.x + Math.cos(angle) * distance / pieces, y: p.y + Math.sin(angle) * distance / pieces };
    const at = tileAt(q.x, q.y);
    if (at === 'wall') { blocked = true; break; }
    p = q;
    if (at !== null) return { next: p, angle, distance, blocked, terminal: true, r: Number(ui.living.value), exit: at };
  }
  return { next: p, angle, distance, blocked, terminal: false, r: Number(ui.living.value) };
}

// ---------- Q-network ----------
const encode = (s) => [s.x / GRID.width * 2 - 1, s.y / GRID.height * 2 - 1];
const qAt = (s) => outputs(state.net, encode(s));
function resetNetwork() {
  const sizes = [2, ...ui.layers.value.split(',').map(Number), 4];
  if (PG) { state.policy = createPolicy({ sizes }); state.net = state.policy.net; state.b = 0; }
  else state.net = createNetwork(sizes);
}
// One gradient step: w ← w + α[target − Q(s, a; w)] ∇Q(s, a; w).
function sgdStep(s, a, target, rate) {
  const acts = forwardPass(state.net, encode(s)), out = acts[acts.length - 1];
  const delta = new Float64Array(out.length);
  delta[a] = out[a] - target;
  const grads = zeroGrads(state.net);
  backprop(state.net, acts, delta, grads);
  state.net.layers.forEach((layer, l) => {
    for (let k = 0; k < layer.W.length; k++) layer.W[k] -= rate * grads[l].W[k];
    for (let j = 0; j < layer.outputs; j++) layer.b[j] -= rate * grads[l].b[j];
  });
}

// ---------- Q-learning ----------
function startEpisode() {
  state.agent = startPoint();
  state.trail = [state.agent];
  state.done = false;
  state.steps = 0;
  state.total = 0;
  state.traj = [];
}
function step(manual = null) {
  if (PG) return pgStep(manual);
  if (state.done) { startEpisode(); state.last = null; return null; }
  const alpha = Number(ui.alpha.value), gamma = Number(ui.gamma.value);
  const s = state.agent, q = qAt(s);
  const explored = manual === null && Math.random() < Number(ui.epsilon.value);
  const a = manual !== null ? manual : explored ? Math.floor(Math.random() * 4) : argmax(q);
  const move = contMove(s, a);
  const nextQ = move.terminal ? null : qAt(move.next);
  const nextMax = move.terminal ? move.exit : Math.max(...nextQ);
  const target = move.r + gamma * nextMax;
  sgdStep(s, a, target, alpha);
  const value = qAt(s)[a];
  state.agent = move.next;
  state.trail.push(move.next);
  state.total += move.r + (move.terminal ? move.exit : 0);
  state.steps++;
  if (move.terminal) endEpisode();
  else if (state.steps >= GRID.maxSteps) endEpisode(true);
  state.heatTime = 0;
  return logUpdate({ s, a, ...move, old: q[a], nextMax, target, value, alpha, gamma, explored, manual: manual !== null });
}
function logUpdate(record) {
  state.log.unshift(record);
  state.log.length = Math.min(state.log.length, 8);
  return record;
}
function endEpisode(timedOut = false) {
  state.done = true;
  state.timedOut = timedOut;
  const exit = timedOut ? 0 : tileAt(state.agent.x, state.agent.y);
  state.returns.push({ reward: state.total, caught: exit > 0, lost: exit < 0 });
  if (PG) learnEpisode();
  state.chartDirty = true;
}

// ---------- REINFORCE ----------
// One move sampled from π(· | s) (or chosen with the arrow keys); nothing is learned until the episode ends.
function pgStep(manual) {
  if (state.done) { startEpisode(); state.last = null; return null; }
  const s = state.agent, x = encode(s), p = actionProbabilities(state.net, x);
  const a = manual !== null ? manual : sampleAction(state.policy, x).action;
  const move = contMove(s, a);
  state.traj.push({ s: x, pos: s, a, r: move.r, p: p[a] });
  state.final = move.terminal ? move.exit : 0;
  state.agent = move.next;
  state.trail.push(move.next);
  state.total += move.r + state.final;
  state.steps++;
  const record = { s, a, ...move, p, manual: manual !== null };
  if (move.terminal) endEpisode();
  else if (state.steps >= GRID.maxSteps) endEpisode(true);
  return record;
}
// G_t = R(s_t) + γ G_t+1 (ending with R(exit)), then one gradient step on Σ_t (G_t − b) log π(a_t | s_t) (plus the entropy bonus).
function learnEpisode() {
  const gamma = Number(ui.gamma.value), useBaseline = ui.baseline.checked, b = useBaseline ? state.b : 0, traj = state.traj;
  let G = state.final || 0;
  for (let t = traj.length - 1; t >= 0; t--) { G = traj[t].r + gamma * G; traj[t].G = G; }
  for (const move of traj) move.advantage = move.G - b;
  reinforce(state.policy, traj, { rate: Number(ui.alpha.value), entropy: Number(ui.entropy.value) });
  for (const move of traj) {
    move.after = actionProbabilities(state.net, move.s)[move.a];
    if (useBaseline) state.b += BASELINE_RATE * (move.G - state.b);
  }
  state.update = { steps: traj.slice(), b, baseline: useBaseline };
  state.heatTime = 0;
}
const probsOf = (acts) => softmax(Array.from(acts[acts.length - 1]));

// ---------- Drawing ----------
// Every plot is a canvas covering the world; heat holds the network's activations (every layer) at a grid of points.
function computeHeat() {
  const cols = GRID.width * HEAT_PER_TILE, rows = GRID.height * HEAT_PER_TILE, values = [];
  for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
    const x = (i + 0.5) / HEAT_PER_TILE, y = (j + 0.5) / HEAT_PER_TILE;
    values.push(tileAt(x, y) === null ? forwardPass(state.net, encode({ x, y })) : null);
  }
  state.heat = { cols, rows, values };
}
const qOf = (acts) => acts[acts.length - 1];
function prepareCanvas(canvas) {
  const width = canvas.clientWidth, height = width * GRID.height / GRID.width, dpr = window.devicePixelRatio || 1;
  canvas.style.height = `${height}px`;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const unit = width / GRID.width;
  return { ctx, width, height, unit, px: (x) => x * unit, py: (y) => height - y * unit };
}
// Heatmap colours as RGB, blended onto white: teal for positive values, red for negative (the shade() scale),
// and blue for a hidden unit's output (0–1 of its largest value).
function tint(v) {
  const alpha = Math.min(1, Math.abs(v)) * 0.55, base = v >= 0 ? [15, 118, 110] : [220, 53, 69];
  return base.map(c => 255 + (c - 255) * alpha);
}
function blueTint(fraction) { const alpha = 0.06 + 0.6 * fraction; return [13, 110, 253].map(c => 255 + (c - 255) * alpha); }
// Draws the heatmap (color(acts) → RGB for each point) as one smoothly scaled image, then the walls and exits on top.
const heatImage = document.createElement('canvas');
function drawWorld(canvas, color, { labels = true } = {}) {
  const { ctx, width, height, unit, px, py } = prepareCanvas(canvas);
  const { cols, rows, values } = state.heat;
  heatImage.width = cols;
  heatImage.height = rows;
  const hctx = heatImage.getContext('2d'), image = hctx.createImageData(cols, rows);
  for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
    const acts = values[j * cols + i], rgb = acts ? color(acts) : [255, 255, 255], o = ((rows - 1 - j) * cols + i) * 4;
    image.data[o] = rgb[0]; image.data[o + 1] = rgb[1]; image.data[o + 2] = rgb[2]; image.data[o + 3] = 255;
  }
  hctx.putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(heatImage, 0, 0, width, height);
  for (let ty = 0; ty < GRID.height; ty++) for (let tx = 0; tx < GRID.width; tx++) {
    const k = `${tx},${ty}`, x = px(tx), y = py(ty + 1);
    if (GRID.walls.has(k)) { ctx.fillStyle = '#868e96'; ctx.fillRect(x, y, unit, unit); }
    else if (GRID.exits[k] !== undefined) {
      const v = GRID.exits[k], inset = unit * 0.14;
      ctx.fillStyle = 'white'; ctx.fillRect(x, y, unit, unit);
      ctx.fillStyle = shade(v, 0.22); ctx.fillRect(x, y, unit, unit);
      ctx.fillStyle = 'white'; ctx.fillRect(x + inset, y + inset, unit - 2 * inset, unit - 2 * inset);
      ctx.fillStyle = shade(v); ctx.fillRect(x + inset, y + inset, unit - 2 * inset, unit - 2 * inset);
      ctx.strokeStyle = v > 0 ? '#0f766e' : '#dc3545'; ctx.lineWidth = Math.max(1, unit * 0.02);
      ctx.strokeRect(x + inset, y + inset, unit - 2 * inset, unit - 2 * inset);
      if (labels) {
        ctx.fillStyle = '#212529'; ctx.font = `${Math.round(unit * 0.2)}px ui-monospace, monospace`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(rewardText(v), x + unit / 2, y + unit / 2);
      }
    }
  }
  ctx.strokeStyle = 'rgba(73, 80, 87, .35)'; ctx.lineWidth = 1;
  for (let tx = 1; tx < GRID.width; tx++) { ctx.beginPath(); ctx.moveTo(px(tx), 0); ctx.lineTo(px(tx), height); ctx.stroke(); }
  for (let ty = 1; ty < GRID.height; ty++) { ctx.beginPath(); ctx.moveTo(0, py(ty)); ctx.lineTo(width, py(ty)); ctx.stroke(); }
  ctx.strokeStyle = '#495057'; ctx.lineWidth = 2; ctx.strokeRect(1, 1, width - 2, height - 2);
  return { ctx, unit, px, py };
}
function arrowGlyph(ctx, x, y, angle, size) {
  const ux = Math.cos(angle), uy = -Math.sin(angle);
  ctx.beginPath();
  ctx.moveTo(x - ux * size, y - uy * size);
  ctx.lineTo(x + ux * size * 0.45, y + uy * size * 0.45);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x + ux * size, y + uy * size);
  ctx.lineTo(x + ux * size * 0.2 - uy * size * 0.55, y + uy * size * 0.2 + ux * size * 0.55);
  ctx.lineTo(x + ux * size * 0.2 + uy * size * 0.55, y + uy * size * 0.2 - ux * size * 0.55);
  ctx.closePath();
  ctx.fill();
}

function render(now = performance.now()) {
  if (!state.heat || now - state.heatTime > (state.running ? 120 : 0)) { computeHeat(); state.heatTime = now; }
  const focus = state.focus, inspect = state.inspect;
  // Main plot: max Q, or one move's Q (small heatmap hovered), or one hidden unit's output (diagram hovered).
  let color;
  if (inspect) {
    let max = 1e-6;
    for (const acts of state.heat.values) if (acts) max = Math.max(max, acts[inspect.layer][inspect.unit]);
    color = (acts) => blueTint(acts[inspect.layer][inspect.unit] / max);
    ui.mainTitle.textContent = `Hidden unit ${inspect.unit + 1}${state.net.layers.length > 2 ? ` of layer ${inspect.layer}` : ''}: its output at each point`;
  } else if (focus !== null) {
    color = PG ? (acts) => blueTint(probsOf(acts)[focus]) : (acts) => tint(qOf(acts)[focus]);
    ui.mainTitle.textContent = PG ? `π(${ACTIONS[focus].name} | s) at each point` : `Q(s, ${ACTIONS[focus].name}) at each point`;
  } else {
    // REINFORCE: shaded by how sure the policy is of its most likely move (max π, from 25% to 100%).
    color = PG ? (acts) => blueTint((Math.max(...probsOf(acts)) - 0.25) / 0.75) : (acts) => tint(Math.max(...qOf(acts)));
    ui.mainTitle.textContent = PG ? 'Most likely move, shaded by how likely (max π)' : 'max Q(s, a) at each point';
  }
  const { ctx, unit, px, py } = drawWorld(ui.main, color);

  // Best move on a coarse grid of points.
  if (ui.arrows.checked && focus === null && !inspect) {
    ctx.strokeStyle = 'rgba(33, 37, 41, .55)'; ctx.fillStyle = 'rgba(33, 37, 41, .55)'; ctx.lineWidth = Math.max(1, unit * 0.018);
    for (let ty = 0; ty < GRID.height; ty++) for (let tx = 0; tx < GRID.width; tx++) for (let j = 0; j < ARROWS_PER_TILE; j++) for (let i = 0; i < ARROWS_PER_TILE; i++) {
      const x = tx + (i + 0.5) / ARROWS_PER_TILE, y = ty + (j + 0.5) / ARROWS_PER_TILE;
      if (tileAt(x, y) !== null) continue;
      arrowGlyph(ctx, px(x), py(y), DIRECTIONS[argmax(qAt({ x, y }))], unit * 0.11);
    }
  }

  // This episode's path, the last move, and the agent (sliding along its move while animating).
  const anim = state.anim, t = anim ? Math.min(1, (now - anim.start) / anim.duration) : 1;
  if (anim && t >= 1) state.anim = null;
  const trail = state.trail.slice();
  let agent = state.agent;
  if (anim && t < 1 && trail.length >= 2) {
    const from = trail[trail.length - 2], to = trail[trail.length - 1];
    agent = { x: from.x + (to.x - from.x) * t, y: from.y + (to.y - from.y) * t };
    trail[trail.length - 1] = agent;
  }
  ctx.strokeStyle = 'rgba(73, 80, 87, .5)'; ctx.lineWidth = Math.max(1, unit * 0.015); ctx.setLineDash([]);
  ctx.beginPath();
  trail.forEach((p, i) => (i ? ctx.lineTo(px(p.x), py(p.y)) : ctx.moveTo(px(p.x), py(p.y))));
  ctx.stroke();
  const last = state.last;
  ctx.fillStyle = '#0d6efd'; ctx.strokeStyle = 'white'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(px(agent.x), py(agent.y), Math.max(5, unit * 0.08), 0, 2 * Math.PI); ctx.fill(); ctx.stroke();

  // One small heatmap per move.
  ui.small.forEach((canvas, a) => drawWorld(canvas, PG ? (acts) => blueTint(probsOf(acts)[a]) : (acts) => tint(qOf(acts)[a]), { labels: false }));

  // Network diagram: weights, and activations at the agent's position.
  drawNetworkWeights(diagram, state.net);
  const acts = forwardPass(state.net, encode(state.agent));
  drawNetworkActivations(diagram, acts, { chosen: last ? last.a : null, probabilities: PG ? probsOf(acts) : null });

  ui.episode.textContent = `Episode ${state.returns.length + (state.done ? 0 : 1)} · step ${state.steps}`;
  ui.status.textContent = describe(last);
  ui.update.innerHTML = formula(last);
  showLog();
  if (state.chartDirty) showStats();
}

const at = (p) => `(${p.x.toFixed(2)}, ${p.y.toFixed(2)})`;
function describe(last) {
  if (state.done) return state.timedOut ? `Stopped after ${GRID.maxSteps} steps · press Step to start the next episode` : `Reached the ${rewardText(tileAt(state.agent.x, state.agent.y))} exit · episode reward ${signedValue(state.total)} · press Step to start the next episode`;
  if (!last) return `At ${at(state.agent)} · press Step or an arrow key`;
  const who = last.manual ? 'you chose' : PG ? `sampled (π = ${Math.round(100 * last.p[last.a])}%)` : last.explored ? 'chose at random' : 'chose greedily';
  const off = Math.round(((last.angle - DIRECTIONS[last.a]) * 180 / Math.PI + 540) % 360 - 180);
  return `At ${at(last.s)} · ${who}: ${ACTIONS[last.a].name} · moved ${last.distance.toFixed(2)} at ${off >= 0 ? '+' : '−'}${Math.abs(off)}°${last.blocked ? ' · hit a wall' : ''} · R(s) = ${signedValue(last.r)}`;
}
const label = (u) => `Q(${at(u.s)}, ${ACTIONS[u.a].name})`;
const plain = (v) => `${v < 0 ? '−' : ''}${Math.abs(v).toFixed(2)}`;
// REINFORCE: during an episode, how many moves so far; after it, the first move's advantage and the average change.
function pgFormula() {
  if (!state.done || !state.update) return `<span class="hint">${state.traj.length} ${state.traj.length === 1 ? 'move' : 'moves'} so far this episode · the policy updates when the episode ends</span>`;
  const u = state.update, first = u.steps[0], advantage = first.G - u.b;
  const before = u.steps.reduce((sum, m) => sum + m.p, 0) / u.steps.length, after = u.steps.reduce((sum, m) => sum + m.after, 0) / u.steps.length;
  return `first move: G<sub>0</sub> − b = ${plain(first.G)} − ${u.baseline ? plain(u.b) : '0'} = <strong>${signedValue(advantage)}</strong>` +
    `<br>its ${u.steps.length} moves: avg π(move taken) ${(100 * before).toFixed(1)}% → <strong>${(100 * after).toFixed(1)}%</strong>`;
}
function formula(last) {
  if (PG) return pgFormula();
  if (!last) return '';
  const target = `${num(last.r)} + ${last.gamma.toFixed(2)}·${num(last.nextMax)}`;
  const bracket = `[<span class="target" data-label="target = ${num(last.target)}${last.terminal ? ' (episode ends)' : ''}">${target}</span> − ${num(last.old)}]`;
  return `${label(last)}: ${num(last.old)} → <strong>${num(last.value)}</strong><br>error = ${bracket} = ${signedValue(last.target - last.old)}`;
}
function showLog() {
  if (PG) {
    const moves = state.done && state.update ? state.update.steps.slice().reverse().slice(0, 10) : [];
    ui.log.replaceChildren(...moves.map(m => {
      const row = document.createElement('tr');
      for (const text of [`${at(m.pos)} ${ACTIONS[m.a].name}`, plain(m.G), signedValue(m.advantage), `${Math.round(100 * m.p)}% → ${Math.round(100 * m.after)}%`]) {
        const cell = document.createElement('td');
        cell.textContent = text;
        row.appendChild(cell);
      }
      return row;
    }));
    return;
  }
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
function showStats() {
  const episodes = state.returns, recent = episodes.slice(-20);
  ui.count.textContent = `${episodes.length} ${episodes.length === 1 ? 'episode' : 'episodes'}`;
  ui.statWin.textContent = String(episodes.filter(e => e.caught).length);
  ui.statLose.textContent = String(episodes.filter(e => e.lost).length);
  ui.statReward.textContent = recent.length ? signedValue(recent.reduce((a, e) => a + e.reward, 0) / recent.length) : '–';
  drawRewardChart(ui.chart, episodes, { window: 20, minSpan: 20 });
  state.chartDirty = false;
}

// ---------- Controls ----------
const STEP_ANIMATION = 350;
function animate(duration) {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  state.anim = duration >= 60 && !reduced ? { start: performance.now(), duration } : null;
}
function doStep(manual = null) {
  const record = step(manual);
  if (record) { state.last = record; animate(STEP_ANIMATION); }
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
    let steps = 0, record = null;
    while (state.carry >= 1) { state.carry--; record = step(); if (record) state.last = record; steps++; }
    if (steps === 1 && record) animate(Math.min(STEP_ANIMATION, 800 / Number(ui.speed.value)));
    if (steps || state.anim) render(now);
  } else if (state.anim) render(now);
  requestAnimationFrame(frame);
}
// Network diagram; hovering (or focusing) a hidden unit shows its output at each point in the main plot.
let diagram = null;
function buildDiagram() {
  state.inspect = null;
  ui.netSvg.replaceChildren();
  diagram = createNetworkDiagram(ui.netSvg, [2, ...ui.layers.value.split(',').map(Number), 4], { inputLabels: ['x', 'y'], outputLabels: ['↑', '→', '↓', '←'], outputCaption: PG ? 'π(a | s)' : 'Q(s, ·)' });
  diagram.nodes.slice(1, -1).forEach((layer, l) => layer.forEach((node, unit) => {
    const hit = svgNode(ui.netSvg, 'circle', { cx: node.getAttribute('cx'), cy: node.getAttribute('cy'), r: 13, class: 'unit-hit', tabindex: 0, role: 'button', 'aria-label': `Show hidden unit ${unit + 1}'s output at each point` });
    const show = () => { state.inspect = { layer: l + 1, unit }; render(); };
    const hide = () => { if (state.inspect && state.inspect.layer === l + 1 && state.inspect.unit === unit) { state.inspect = null; render(); } };
    hit.addEventListener('pointerenter', show);
    hit.addEventListener('pointerleave', hide);
    hit.addEventListener('focus', show);
    hit.addEventListener('blur', hide);
  }));
}
function resetAll() {
  setRunning(false);
  state.update = null;
  resetNetwork();
  buildDiagram();
  startEpisode();
  state.returns = [];
  state.log = [];
  state.last = null;
  state.anim = null;
  state.heat = null;
  state.chartDirty = true;
  render();
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
  state.heat = null;
  render();
});
ui.reset.addEventListener('click', resetAll);
ui.world.addEventListener('change', () => { setWorld(ui.world.value); resetAll(); });
ui.layers.addEventListener('change', resetAll);
ui.arrows.addEventListener('change', () => render());
// Hovering (or focusing) a small heatmap shows that move's Q-values in the main plot.
ui.small.forEach((canvas, a) => {
  const show = () => { state.focus = a; render(); };
  const hide = () => { if (state.focus === a) { state.focus = null; render(); } };
  const card = canvas.closest('.q-card');
  card.addEventListener('pointerenter', show);
  card.addEventListener('pointerleave', hide);
  card.addEventListener('focus', show);
  card.addEventListener('blur', hide);
});
for (const [input, format] of [
  [ui.alpha, v => v.toFixed(3)], [ui.gamma, v => v.toFixed(2)], [ui.epsilon, v => v.toFixed(2)], [ui.entropy, v => v.toFixed(3)],
  [ui.length, v => v.toFixed(2)], [ui.lengthNoise, v => v.toFixed(2)], [ui.angleNoise, v => `${v}°`],
  [ui.living, v => v.toFixed(2)], [ui.speed, v => `${v} steps/s`]
].filter(([input]) => input)) {
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
new ResizeObserver(() => render()).observe(ui.main);

fillWorldMenu(ui.world);
resetAll();
requestAnimationFrame(frame);
