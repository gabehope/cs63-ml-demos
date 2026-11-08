// A genetic algorithm on the cart game: each individual is a small network's weights, written out as one list of genes.
// Every individual plays the same K balls; fitter ones are more likely to be picked as parents, crossed over and mutated.
const $ = (id) => document.getElementById(id);
const ui = {
  page: $('page'), game: $('game'), status: $('ga-status'), champion: $('champion-label'), count: $('count-label'),
  trace: $('trace'), traceText: $('trace-text'),
  population: $('population'), populationValue: $('population-value'), balls: $('balls'), ballsValue: $('balls-value'),
  cross: $('cross'), crossValue: $('cross-value'), mutate: $('mutate'), mutateValue: $('mutate-value'),
  sigma: $('sigma'), sigmaValue: $('sigma-value'), selection: $('selection'), elitism: $('elitism'),
  speed: $('speed'), speedValue: $('speed-value'), scale: $('scale'), scaleValue: $('scale-value'), showLanding: $('show-landing'),
  step: $('step'), continuous: $('continuous'), reset: $('reset'),
  chart: $('fitness-chart'), generation: $('stat-generation'), best: $('stat-best'), avg: $('stat-avg'), caught: $('stat-caught'), runLegend: $('run-legend'),
  rows: $('population-rows'), populationLabel: $('population-label'), network: $('network')
};
const SIZES = [6, 8, 3];
const RUN_COLORS = ['#e8590c', '#6f42c1', '#868e96'];
const scene = createScene(ui.game);
const shown = createNetwork(SIZES);
const state = {
  population: [], generation: 0, history: [], episodes: 0, trace: null, champion: null, previous: [], running: false,
  game: newEpisode(), hold: 0, banner: '', action: null, carry: 0, gameCarry: 0, lastFrame: performance.now(), dirty: true
};

// ---------- Chromosomes ----------
// Genes are the network's parameters layer by layer: each layer's weights, then its biases.
function genesOf(net) {
  const genes = [];
  net.layers.forEach(layer => genes.push(...layer.W, ...layer.b));
  return Float64Array.from(genes);
}
function loadGenes(net, genes) {
  let i = 0;
  for (const layer of net.layers) {
    for (let k = 0; k < layer.W.length; k++) layer.W[k] = genes[i++];
    for (let k = 0; k < layer.b.length; k++) layer.b[k] = genes[i++];
  }
}
const GENE_COUNT = genesOf(shown).length;
const SEGMENTS = (() => {
  const parts = [];
  let start = 0;
  shown.layers.forEach((layer, l) => {
    const name = l === shown.layers.length - 1 ? 'output' : 'hidden';
    parts.push({ start, end: start + layer.W.length, label: `${name} weights` });
    start += layer.W.length;
    parts.push({ start, end: start + layer.b.length, label: 'b' });
    start += layer.b.length;
  });
  return parts;
})();

// ---------- Fitness ----------
// One ball with greedy play: the move with the largest output, no randomness.
function play(net, ball) {
  const game = structuredClone(ball);
  while (!game.done) controlStep(game, argmax(outputs(net, features(game))));
  return { reward: terminalReward(game), caught: game.caught };
}
function evaluate(population) {
  const balls = Array.from({ length: Number(ui.balls.value) }, () => newEpisode());
  for (const individual of population) {
    loadGenes(shown, individual.genes);
    let total = 0, caught = 0;
    for (const ball of balls) { const result = play(shown, ball); total += result.reward; caught += result.caught; }
    individual.fitness = total / balls.length;
    individual.caught = caught;
    individual.balls = balls.length;
  }
  state.episodes += population.length * balls.length;
  population.forEach((individual, i) => { individual.id = i + 1; });
  const fitness = population.map(individual => individual.fitness);
  const best = population.reduce((a, b) => (b.fitness > a.fitness ? b : a));
  state.history.push({ best: best.fitness, avg: fitness.reduce((a, b) => a + b, 0) / fitness.length, all: fitness });
  state.champion = { genes: best.genes, fitness: best.fitness, caught: best.caught, balls: best.balls, generation: state.generation, id: best.id };
  selectionRates(population);
}

// The chance each individual has of being picked for one parent slot.
function selectionRates(population) {
  const n = population.length;
  if (ui.selection.value === 'tournament') {
    // Better of two picked at random (with replacement): P = (2r − 1) / n², r = rank from the bottom.
    const order = population.slice().sort((a, b) => a.fitness - b.fitness);
    order.forEach((individual, i) => { individual.rate = (2 * (i + 1) - 1) / (n * n); });
    return;
  }
  const worst = Math.min(...population.map(individual => individual.fitness));
  const weights = population.map(individual => individual.fitness - worst);
  const total = weights.reduce((a, b) => a + b, 0);
  population.forEach((individual, i) => { individual.rate = total > 0 ? weights[i] / total : 1 / n; });
}
function pick(population) {
  let u = Math.random();
  for (const individual of population) { u -= individual.rate; if (u <= 0) return individual; }
  return population[population.length - 1];
}

// ---------- One generation ----------
function nextGeneration() {
  const parents = state.population, n = Number(ui.population.value);
  const pCross = Number(ui.cross.value), pMutate = Number(ui.mutate.value), sigma = Number(ui.sigma.value);
  selectionRates(parents);
  const next = [];
  const elite = ui.elitism.checked ? parents.reduce((a, b) => (b.fitness > a.fitness ? b : a)) : null;
  if (elite) next.push({ genes: Float64Array.from(elite.genes), kept: elite.id });
  let trace = null;
  while (next.length < n) {
    const a = pick(parents), b = pick(parents);
    const children = [Float64Array.from(a.genes), Float64Array.from(b.genes)];
    let cut = null;
    if (Math.random() < pCross) {
      cut = 1 + Math.floor(Math.random() * (GENE_COUNT - 1));
      for (let i = cut; i < GENE_COUNT; i++) [children[0][i], children[1][i]] = [children[1][i], children[0][i]];
    }
    children.forEach((child, c) => {
      const mutated = [];
      for (let i = 0; i < GENE_COUNT; i++) if (Math.random() < pMutate) { child[i] += gaussian(Math.random) * sigma; mutated.push(i); }
      if (next.length >= n) return;
      if (!trace && c === 0) trace = { a, b, cut, mutated, child, aGenes: a.genes, bGenes: b.genes, index: next.length };
      next.push({ genes: child });
    });
  }
  state.generation++;
  state.population = next;
  evaluate(next);
  trace.childFitness = next[trace.index].fitness;
  trace.childId = next[trace.index].id;
  trace.generation = state.generation;
  state.trace = trace;
  state.dirty = true;
}

function resetRun() {
  // Generation 0: every gene drawn at random from N(0, 0.5²).
  state.population = Array.from({ length: Number(ui.population.value) }, () => ({ genes: Float64Array.from({ length: GENE_COUNT }, () => 0.5 * gaussian(Math.random)) }));
  state.generation = 0;
  state.history = [];
  state.episodes = 0;
  state.trace = null;
  evaluate(state.population);
  startBall();
  state.dirty = true;
}

// ---------- The best network plays fresh balls ----------
function startBall() {
  state.game = newEpisode();
  state.hold = 0;
  state.banner = '';
  state.action = null;
  loadGenes(shown, state.champion.genes);
  state.playing = state.champion;
  drawNetworkWeights(diagram, shown);
}
function tick() {
  const game = state.game;
  if (game.done) {
    physicsTick(game);
    if (--state.hold <= 0) startBall();
    return;
  }
  if (isDecisionTick(game)) {
    state.action = argmax(outputs(shown, features(game)));
    game.motor = CART.ACTIONS[state.action];
  }
  physicsTick(game);
  if (game.done) {
    state.banner = outcomeText(game, terminalReward(game));
    state.hold = 50;
  }
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
    state.carry += elapsed / 1000 * Number(ui.speed.value);
    const end = performance.now() + 30;
    while (state.carry >= 1 && performance.now() < end) { state.carry--; nextGeneration(); }
    state.carry = Math.min(state.carry, 1);
  }
  state.gameCarry += elapsed / 1000 / CART.DT;
  while (state.gameCarry >= 1) { state.gameCarry--; tick(); }
  render();
  requestAnimationFrame(frame);
}

// ---------- Rendering ----------
function render() {
  const game = state.game, playing = state.playing;
  drawScene(scene, game, { showLanding: ui.showLanding.checked, banner: state.banner });
  ui.champion.textContent = `Generation ${playing.generation} best`;
  const move = state.action === null ? 'waiting' : `chooses ${CART.ACTION_NAMES[state.action]}`;
  ui.status.textContent = game.done ? state.banner : `Best of generation ${playing.generation} (fitness ${signedValue(playing.fitness)}) on a new ball · ${move}`;
  const acts = forwardPass(shown, features(game));
  drawNetworkActivations(diagram, acts, { chosen: game.done ? null : state.action });
  if (state.dirty) { showStats(); drawTrace(); showPopulation(); state.dirty = false; }
}

function showStats() {
  const champion = state.champion, last = state.history[state.history.length - 1];
  ui.generation.textContent = String(state.generation);
  ui.best.textContent = signedValue(champion.fitness);
  ui.avg.textContent = signedValue(last.avg);
  ui.caught.textContent = `${champion.caught} / ${champion.balls}`;
  ui.count.textContent = `${state.episodes.toLocaleString()} episodes played`;
  drawFitnessChart();
  showRunLegend();
}

function drawFitnessChart() {
  const root = ui.chart, history = state.history;
  root.replaceChildren();
  const H = 220, L = 46, R = 486, T = 12, B = 182;
  const span = Math.max(20, history.length, ...state.previous.map(run => run.history.length));
  const x = g => L + g / (span - 1) * (R - L);
  const y = f => B - (Math.max(-1, Math.min(1, f)) + 1) / 2 * (B - T);
  svgNode(root, 'rect', { x: L, y: T, width: R - L, height: B - T, fill: '#fff', stroke: '#adb5bd' });
  for (const f of [-1, -0.5, 0, 0.5, 1]) {
    svgNode(root, 'line', { x1: L, x2: R, y1: y(f), y2: y(f), stroke: '#e9ecef' });
    if (Number.isInteger(f)) svgNode(root, 'text', { x: L - 8, y: y(f) + 4, 'text-anchor': 'end', class: 'chart-label' }, f > 0 ? '+1' : f < 0 ? '−1' : '0');
  }
  for (const tick of [0, Math.round((span - 1) / 2), span - 1]) svgNode(root, 'text', { x: x(tick), y: B + 16, 'text-anchor': 'middle', class: 'chart-label' }, String(tick));
  svgNode(root, 'text', { x: (L + R) / 2, y: H - 4, 'text-anchor': 'middle', class: 'chart-label' }, 'generation');
  svgNode(root, 'text', { x: 12, y: (T + B) / 2, 'text-anchor': 'middle', class: 'chart-label', transform: `rotate(-90 12 ${(T + B) / 2})` }, 'fitness');
  const line = (runs, key) => runs.map((h, g) => `${g ? 'L' : 'M'} ${x(g).toFixed(1)} ${y(h[key]).toFixed(1)}`).join(' ');
  for (const [i, run] of state.previous.entries()) svgNode(root, 'path', { d: line(run.history, 'best'), fill: 'none', stroke: RUN_COLORS[i], 'stroke-width': 1.5, 'stroke-opacity': 0.75 });
  let dots = '';
  const step = Math.max(1, Math.ceil(history.length / 250)), r = history.length > 100 ? 1.2 : 1.8;
  for (let g = 0; g < history.length; g += step) for (const f of history[g].all) dots += `M ${(x(g) - r).toFixed(1)} ${y(f).toFixed(1)} a ${r} ${r} 0 1 0 ${2 * r} 0 a ${r} ${r} 0 1 0 ${-2 * r} 0 `;
  svgNode(root, 'path', { d: dots, fill: '#6c757d', 'fill-opacity': 0.22 });
  if (history.length >= 2) {
    svgNode(root, 'path', { d: line(history, 'avg'), fill: 'none', stroke: '#6c757d', 'stroke-width': 1.5, 'stroke-dasharray': '4 3' });
    svgNode(root, 'path', { d: line(history, 'best'), fill: 'none', stroke: '#0d6efd', 'stroke-width': 2 });
  }
}

// Gene colour: blue for positive, orange for negative, stronger for larger magnitude.
function geneRGB(w) {
  const t = Math.tanh(Math.abs(w) / 1.2), c = w >= 0 ? [13, 110, 253] : [232, 89, 12];
  return c.map(v => Math.round(255 + (v - 255) * t));
}
function geneColor(w) { return `rgb(${geneRGB(w).join(',')})`; }

function chromosomeCanvas(genes) {
  const canvas = document.createElement('canvas');
  canvas.width = genes.length;
  canvas.height = 1;
  canvas.className = 'ga-genes';
  const ctx = canvas.getContext('2d'), image = ctx.createImageData(genes.length, 1);
  genes.forEach((w, i) => { image.data.set([...geneRGB(w), 255], i * 4); });
  ctx.putImageData(image, 0, 0);
  return canvas;
}

// The population as a table like the lecture's: sorted by fitness, with each one's chance of being picked as a parent.
function showPopulation() {
  const trace = state.trace, population = state.population.slice().sort((a, b) => b.fitness - a.fitness);
  ui.rows.replaceChildren();
  ui.populationLabel.textContent = `generation ${state.generation} · n = ${population.length}`;
  for (const individual of population) {
    const row = document.createElement('tr');
    const name = document.createElement('td');
    name.className = 'mono';
    name.textContent = `#${individual.id}`;
    if (individual.kept) name.append(Object.assign(document.createElement('span'), { className: 'ga-tag kept', textContent: 'kept', title: `copy of the best of generation ${state.generation - 1}` }));
    if (trace && trace.childId === individual.id) name.append(Object.assign(document.createElement('span'), { className: 'ga-tag child', textContent: 'child', title: 'the offspring traced under the game' }));
    const strip = document.createElement('td');
    strip.className = 'bar-cell';
    strip.appendChild(chromosomeCanvas(individual.genes));
    const fitness = document.createElement('td');
    fitness.className = 'mono raw';
    fitness.textContent = signedValue(individual.fitness);
    const rate = document.createElement('td');
    rate.className = 'mono raw';
    rate.textContent = `${(100 * individual.rate).toFixed(1)}%`;
    row.append(name, strip, fitness, rate);
    ui.rows.appendChild(row);
  }
}

// Parent A, parent B and the child gene by gene: the cut point, which part came from whom, and the mutated genes.
function drawTrace() {
  const root = ui.trace, trace = state.trace;
  root.replaceChildren();
  const L = 56, R = 592, w = (R - L) / GENE_COUNT, rowH = 16, rows = { a: 6, b: 30, child: 66 };
  const label = (text, y, cls = '') => svgNode(root, 'text', { x: L - 8, y: y + 12, 'text-anchor': 'end', class: `ga-row-label ${cls}` }, text);
  if (!trace) {
    svgNode(root, 'text', { x: 300, y: 56, 'text-anchor': 'middle', class: 'ga-row-label' }, 'Press Next generation to breed the first offspring.');
    ui.traceText.innerHTML = '<span class="hint">Generation 0 is random networks; each has played the same balls to get its fitness.</span>';
    return;
  }
  const strip = (genes, y, from) => {
    genes.forEach((g, i) => {
      const attrs = { x: (L + i * w).toFixed(2), y, width: (w + 0.3).toFixed(2), height: rowH, fill: geneColor(g) };
      if (from && from(i) === false) attrs.opacity = 0.25;
      svgNode(root, 'rect', attrs);
    });
    svgNode(root, 'rect', { x: L, y, width: R - L, height: rowH, fill: 'none', stroke: '#ced4da' });
  };
  const cut = trace.cut;
  // Faded genes are the ones that did not go into the child.
  strip(trace.aGenes, rows.a, i => cut === null || i < cut);
  strip(trace.bGenes, rows.b, i => cut !== null && i >= cut);
  strip(trace.child, rows.child);
  label('A', rows.a, 'parent');
  label('B', rows.b, 'parent');
  label('child', rows.child);
  for (const i of trace.mutated) svgNode(root, 'rect', { x: (L + i * w - 1).toFixed(2), y: rows.child - 2, width: (w + 2).toFixed(2), height: rowH + 4, fill: 'none', stroke: '#212529', 'stroke-width': 1.5 });
  if (cut !== null) {
    const cx = L + cut * w;
    svgNode(root, 'line', { x1: cx, x2: cx, y1: rows.a - 4, y2: rows.child + rowH + 4, stroke: '#212529', 'stroke-width': 1.5, 'stroke-dasharray': '3 2' });
  }
  // Brackets under the child: which parent each part came from.
  const bracket = (from, to, text) => {
    if (to - from < 1) return;
    const x1 = L + from * w + 1, x2 = L + to * w - 1, yb = rows.child + rowH + 5;
    svgNode(root, 'path', { d: `M ${x1} ${yb} v 4 H ${x2} v -4`, fill: 'none', stroke: '#6c757d' });
    if (x2 - x1 > 30) svgNode(root, 'text', { x: (x1 + x2) / 2, y: yb + 17, 'text-anchor': 'middle', class: 'ga-row-label' }, text);
  };
  if (cut === null) bracket(0, GENE_COUNT, 'copy of A');
  else { bracket(0, cut, 'from A'); bracket(cut, GENE_COUNT, 'from B'); }
  // Gene groups along the top of parent A.
  for (const part of SEGMENTS) if (part.start) svgNode(root, 'line', { x1: L + part.start * w, x2: L + part.start * w, y1: rows.a, y2: rows.b + rowH, stroke: '#fff', 'stroke-width': 1 });
  const pct = r => `${(100 * r).toFixed(1)}%`;
  const crossText = cut === null ? `no crossover (probability 1 − pCross), so the child starts as a copy of A` : `crossover after gene ${cut} of ${GENE_COUNT}: genes 1–${cut} from A, the rest from B`;
  const mutateText = trace.mutated.length ? `${trace.mutated.length} gene${trace.mutated.length === 1 ? '' : 's'} mutated (outlined)` : 'no genes mutated';
  ui.traceText.innerHTML = `selected A = #${trace.a.id} (fitness ${signedValue(trace.a.fitness)}, ${pct(trace.a.rate)} per pick) and B = #${trace.b.id} (${signedValue(trace.b.fitness)}, ${pct(trace.b.rate)})` +
    `<br>${crossText} · ${mutateText}<br>the child is #${trace.childId} in generation ${trace.generation}: fitness <strong>${signedValue(trace.childFitness)}</strong>`;
}

const diagram = createNetworkDiagram(ui.network, SIZES, { outputCaption: 'network(s)' });

// Settings that shape the search, recorded with each run so finished runs can be labelled.
function settingsLabel() {
  const parts = [`n ${ui.population.value}`, `K ${ui.balls.value}`, `pCross ${Number(ui.cross.value).toFixed(2)}`, `pMutate ${Number(ui.mutate.value).toFixed(2)}`, `σ ${Number(ui.sigma.value).toFixed(2)}`];
  if (ui.selection.value === 'tournament') parts.push('tournament');
  if (!ui.elitism.checked) parts.push('no elitism');
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
    item.append(swatch, ` ${i ? `${i + 1} runs ago` : 'previous run'} (best): ${run.label}`);
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
ui.step.addEventListener('click', () => { setRunning(false); nextGeneration(); });
ui.continuous.addEventListener('click', () => { setRunning(!state.running); state.carry = 1; });
ui.reset.addEventListener('click', () => {
  setRunning(false);
  if (state.generation >= 5) {
    state.previous.unshift({ history: state.history, label: settingsLabel() });
    state.previous.length = Math.min(state.previous.length, RUN_COLORS.length);
  }
  resetRun();
});
const show = (input, output, digits) => input.addEventListener('input', () => { output.textContent = Number(input.value).toFixed(digits); });
show(ui.population, ui.populationValue, 0);
show(ui.balls, ui.ballsValue, 0);
show(ui.cross, ui.crossValue, 2);
show(ui.mutate, ui.mutateValue, 2);
show(ui.sigma, ui.sigmaValue, 2);
ui.speed.addEventListener('input', () => { ui.speedValue.textContent = String(Number(ui.speed.value)); });
ui.selection.addEventListener('change', () => { selectionRates(state.population); state.dirty = true; });
bindGameScale(ui.scale, ui.scaleValue, ui.page);

resetRun();
requestAnimationFrame(frame);
