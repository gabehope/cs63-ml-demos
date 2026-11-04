// Shared gridworld: the world's rules and the grid drawing, used by the Q-table, Q-network and value-iteration pages.
// Needs cart-game.js (svgNode, signedValue).
// Worlds, top row first: '.' open, '#' wall, 'S' start, a number = an exit paying that reward.
// Layouts follow the Berkeley CS188 gridworlds, with rewards scaled to lie between −1 and +1.
const WORLDS = {
  classic: { label: 'Classic (4 × 3)', rows: [['.', '.', '.', 1], ['.', '#', '.', -1], ['S', '.', '.', '.']] },
  bridge: { label: 'Bridge (7 × 3)', rows: [['#', -1, -1, -1, -1, -1, '#'], [0.1, 'S', '.', '.', '.', '.', 1], ['#', -1, -1, -1, -1, -1, '#']] },
  cliff: { label: 'Cliff (5 × 3)', rows: [['.', '.', '.', '.', '.'], [0.8, 'S', '.', '.', 1], [-1, -1, -1, -1, -1]] },
  discount: { label: 'Discount (5 × 5)', rows: [['.', '.', '.', '.', '.'], ['.', '#', '.', '.', '.'], ['.', '#', 0.1, '#', 1], ['S', '.', '.', '.', '.'], [-1, -1, -1, -1, -1]] },
  maze: { label: 'Maze (5 × 5)', rows: [['.', '.', '.', '#', 1], ['.', '#', '.', '#', '.'], ['.', '#', '.', '.', '.'], ['.', '#', '#', '#', '.'], ['S', '.', '.', '.', '.']] }
};
let GRID, cells, states;
// Switches to a world: rebuilds GRID, the open squares (cells) and the squares the agent acts from (states).
function setWorld(name) {
  const rows = WORLDS[name].rows, height = rows.length, width = rows[0].length;
  GRID = { name, width, height, walls: new Set(), exits: {}, start: { x: 0, y: 0 }, maxSteps: 200 };
  rows.forEach((row, r) => row.forEach((token, x) => {
    const k = `${x},${height - 1 - r}`;
    if (token === '#') GRID.walls.add(k);
    else if (typeof token === 'number') GRID.exits[k] = token;
    else if (token === 'S') GRID.start = { x, y: height - 1 - r };
  }));
  cells = [];
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) if (!GRID.walls.has(`${x},${y}`)) cells.push({ x, y });
  // Landing on an exit square ends the episode, so exits have no values of their own.
  states = cells.filter(c => !isExit(c));
}
// Fills a <select> with the worlds.
function fillWorldMenu(select) {
  for (const [name, world] of Object.entries(WORLDS)) {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = world.label;
    select.appendChild(option);
  }
}
const rewardText = (v) => `${v > 0 ? '+' : '−'}${Math.abs(v)}`;
const ACTIONS = [
  { name: 'north', dx: 0, dy: 1, arrow: '↑' }, { name: 'east', dx: 1, dy: 0, arrow: '→' },
  { name: 'south', dx: 0, dy: -1, arrow: '↓' }, { name: 'west', dx: -1, dy: 0, arrow: '←' }
];
const key = (s) => `${s.x},${s.y}`;
const where = (s) => `(${s.x}, ${s.y})`;
const isExit = (s) => GRID.exits[key(s)] !== undefined;
setWorld('classic');

function moveFrom(s, a) {
  const x = s.x + ACTIONS[a].dx, y = s.y + ACTIONS[a].dy;
  if (x < 0 || y < 0 || x >= GRID.width || y >= GRID.height || GRID.walls.has(`${x},${y}`)) return { ...s };
  return { x, y };
}
// Where move a can end up: the intended way with prob. 1 − noise, otherwise 90° to the left or right.
function outcomes(s, a, noise) {
  return [{ actual: a, p: 1 - noise }, { actual: (a + 1) % 4, p: noise / 2 }, { actual: (a + 3) % 4, p: noise / 2 }]
    .filter(o => o.p > 0).map(o => ({ ...o, next: moveFrom(s, o.actual) }));
}
// R(s): the reward of being in square s. A plain square pays the living reward; an exit pays its own reward and
// ends the episode, so an exit's value is just V(exit) = R(exit).
function stateReward(s, living) { return isExit(s) ? GRID.exits[key(s)] : living; }

// ---------- Drawing ----------
const S = 145, X0 = 10, Y0 = 8;
const cellX = (s) => X0 + s.x * S, cellY = (s) => Y0 + (GRID.height - 1 - s.y) * S;
const center = (s) => ({ x: cellX(s) + S / 2, y: cellY(s) + S / 2 });
const num = (v) => (Object.is(v, -0) ? 0 : v).toFixed(2);
function shade(v, strength = 0.55) {
  const alpha = Math.min(1, Math.abs(v)) * strength;
  return v >= 0 ? `rgba(15, 118, 110, ${alpha.toFixed(3)})` : `rgba(220, 53, 69, ${alpha.toFixed(3)})`;
}

// Arrowhead pointing in move a's direction from a square's centre: base POINTER.base out, tip POINTER.tip out.
// The faint arrow in every triangle and the black pointer for the chosen move share this shape and place.
const POINTER = { base: 17, tip: 26, half: 6 };
function arrowHead(c, a) {
  const ux = ACTIONS[a].dx, uy = -ACTIONS[a].dy, px = -uy, py = ux, { base, tip, half } = POINTER;
  return [[c.x + ux * tip, c.y + uy * tip], [c.x + ux * base + px * half, c.y + uy * base + py * half], [c.x + ux * base - px * half, c.y + uy * base - py * half]].map(p => p.join(',')).join(' ');
}
function directionArrow(s, a) { return arrowHead(center(s), a); }

// Triangle corners and label positions for north, east, south, west within a cell.
function triangles(s) {
  const x = cellX(s), y = cellY(s), c = center(s);
  const tl = [x, y], tr = [x + S, y], br = [x + S, y + S], bl = [x, y + S];
  return [
    { points: [tl, tr, [c.x, c.y]], label: { x: c.x, y: y + 26, anchor: 'middle' } },
    { points: [tr, br, [c.x, c.y]], label: { x: x + S - 8, y: c.y + 5, anchor: 'end' } },
    { points: [br, bl, [c.x, c.y]], label: { x: c.x, y: y + S - 14, anchor: 'middle' } },
    { points: [bl, tl, [c.x, c.y]], label: { x: x + 8, y: c.y + 5, anchor: 'start' } }
  ];
}

// Builds the grid inside an <svg>, sized to the current world: walls, exits, four triangles per square (Q-values view),
// one value and policy arrow per square (values view), and the marks and agent the pages animate.
function createGridView(root) {
  root.replaceChildren();
  root.setAttribute('viewBox', `0 0 ${2 * X0 + GRID.width * S} ${2 * Y0 + GRID.height * S + 3}`);
  const qLayer = svgNode(root, 'g'), vLayer = svgNode(root, 'g'), marks = svgNode(root, 'g');
  // Values view in three layers: backgrounds, then anything a page draws between (vMid), then the numbers and arrows.
  const vBg = svgNode(vLayer, 'g'), vMid = svgNode(vLayer, 'g'), vText = svgNode(vLayer, 'g');
  // Hidden-unit view (network page): one shaded value per square, shown while a hidden unit is hovered.
  const unitLayer = svgNode(root, 'g', { display: 'none' });
  const unitParts = {};
  const parts = {};
  for (let y = 0; y < GRID.height; y++) for (let x = 0; x < GRID.width; x++) {
    if (GRID.walls.has(`${x},${y}`)) svgNode(root, 'rect', { x: cellX({ x, y }), y: cellY({ x, y }), width: S, height: S, class: 'grid-wall' });
  }
  for (const s of cells) {
    const x = cellX(s), y = cellY(s), k = key(s);
    if (isExit(s)) {
      const inset = 20, tone = GRID.exits[k] > 0 ? 'good' : 'bad';
      // Exit squares just show their reward: a light band around a box in the full reward colour (same scale as the values).
      const band = svgNode(qLayer, 'path', { d: `M ${x} ${y} h ${S} v ${S} h ${-S} Z M ${x + inset} ${y + inset} v ${S - 2 * inset} h ${S - 2 * inset} v ${-(S - 2 * inset)} Z` });
      band.style.fill = shade(GRID.exits[k], 0.22);
      const box = svgNode(qLayer, 'rect', { x: x + inset, y: y + inset, width: S - 2 * inset, height: S - 2 * inset, class: `grid-exit ${tone}` });
      box.style.fill = shade(GRID.exits[k]);
      const reward = rewardText(GRID.exits[k]);
      svgNode(box, 'title', {}, `Exit square: landing here earns ${reward} and ends the episode`);
      // The reward sits high in the box, leaving room for the agent below it.
      svgNode(qLayer, 'text', { x: x + S / 2, y: y + 54, 'text-anchor': 'middle', class: 'grid-exit-value' }, reward);
      svgNode(qLayer, 'text', { x: x + S / 2, y: y + S - 6, 'text-anchor': 'middle', class: `grid-caption ${tone}` }, 'episode ends');
      parts[k] = { box };
      continue;
    }
    const tris = triangles(s).map((t, a) => {
      const poly = svgNode(qLayer, 'polygon', { points: t.points.map(p => p.join(',')).join(' '), class: 'grid-tri' });
      svgNode(poly, 'title', {}, `Q(${where(s)}, ${ACTIONS[a].name}): value of moving ${ACTIONS[a].name} from here`);
      const text = svgNode(qLayer, 'text', { x: t.label.x, y: t.label.y, 'text-anchor': t.label.anchor, class: 'grid-q' });
      return { poly, text, arrow: svgNode(qLayer, 'polygon', { points: directionArrow(s, a), class: 'grid-direction' }) };
    });
    const bg = svgNode(vBg, 'rect', { x, y, width: S, height: S });
    // Values view: the value at the top and the policy arrow at the bottom, leaving the centre for the agent.
    const value = svgNode(vText, 'text', { x: x + S / 2, y: y + 39, 'text-anchor': 'middle', class: 'grid-exit-value grid-v-value' });
    const arrow = svgNode(vText, 'text', { x: x + S / 2, y: y + S - 18, 'text-anchor': 'middle', class: 'grid-arrow' });
    parts[k] = { tris, bg, value, arrow };
  }
  for (let i = 0; i <= GRID.width; i++) svgNode(root, 'line', { x1: X0 + i * S, x2: X0 + i * S, y1: Y0, y2: Y0 + GRID.height * S, class: 'grid-line' });
  for (let j = 0; j <= GRID.height; j++) svgNode(root, 'line', { x1: X0, x2: X0 + GRID.width * S, y1: Y0 + j * S, y2: Y0 + j * S, class: 'grid-line' });
  const updated = svgNode(marks, 'polygon', { class: 'grid-updated' });
  const used = svgNode(marks, 'polygon', { class: 'grid-used' });
  const arrow = (cls) => { const g = svgNode(root, 'g', { class: cls }); return { g, line: svgNode(g, 'path'), head: svgNode(g, 'polygon') }; };
  const chosen = arrow('grid-move'), slipped = arrow('grid-move grid-slip');
  const overlay = svgNode(root, 'g');
  const agent = svgNode(root, 'circle', { r: 17, class: 'grid-agent' });
  const maxTag = svgNode(root, 'text', { class: 'grid-max-tag', 'text-anchor': 'middle' }, 'max');
  for (const s of states) {
    const x = cellX(s), y = cellY(s);
    svgNode(unitLayer, 'rect', { x, y, width: S, height: S, fill: 'white' });
    unitParts[key(s)] = { bg: svgNode(unitLayer, 'rect', { x, y, width: S, height: S, class: 'grid-unit-bg' }), text: svgNode(unitLayer, 'text', { x: x + S / 2, y: y + S / 2 + 9, 'text-anchor': 'middle', class: 'grid-exit-value' }) };
  }
  return { root, qLayer, vLayer, vMid, marks, overlay, unitLayer, unitParts, parts, updated, used, maxTag, chosen, slipped, agent };
}

// Paints every square. q: per-square arrays of four values for the triangles. v: per-square value for the values view
// (by default the largest Q). decorate(k, a, polygon) can add extra styling to a triangle.
function paintGrid(view, q, showQ, { v = null, decorate = null, showPolicy = true } = {}) {
  view.vLayer.setAttribute('display', showQ ? 'none' : 'inline');
  for (const s of states) {
    const k = key(s), part = view.parts[k], values = q[k];
    part.tris.forEach((t, a) => {
      t.text.textContent = num(values[a]);
      t.poly.style.fill = shade(values[a]);
      if (decorate) decorate(k, a, t.poly);
      for (const node of [t.text, t.poly, t.arrow]) node.setAttribute('display', showQ ? 'inline' : 'none');
    });
    const top = Math.max(...values), value = v ? v[k] : top;
    part.value.textContent = num(value);
    part.bg.style.fill = shade(value);
    part.arrow.textContent = !showPolicy || values.every(x => x === top) ? '' : values.map((x, a) => x === top ? ACTIONS[a].arrow : '').join(' ');
  }
}

// Outline of the triangle holding Q(s, a), or of the whole square when a is null.
function outline(node, s, a = null) {
  if (!s) { node.setAttribute('points', ''); return; }
  if (a === null) { const x = cellX(s), y = cellY(s); node.setAttribute('points', `${x},${y} ${x + S},${y} ${x + S},${y + S} ${x},${y + S}`); return; }
  node.setAttribute('points', triangles(s)[a].points.map(p => p.join(',')).join(' '));
}
// Where the "max" tag goes: just inside the value it labels.
function maxTagPoint(s, a) {
  const x = cellX(s), y = cellY(s);
  return [{ x: x + S / 2, y: y + 40 }, { x: x + S - 26, y: y + S / 2 + 19 }, { x: x + S / 2, y: y + S - 30 }, { x: x + 26, y: y + S / 2 + 19 }][a];
}
function drawArrow(arrow, from, a) {
  if (a === null) { arrow.g.setAttribute('display', 'none'); return; }
  const dx = ACTIONS[a].dx, dy = -ACTIONS[a].dy;
  arrow.g.setAttribute('display', 'inline');
  arrow.line.setAttribute('d', `M ${from.x + dx * 15} ${from.y + dy * 15} L ${from.x + dx * POINTER.base} ${from.y + dy * POINTER.base}`);
  arrow.head.setAttribute('points', arrowHead(from, a));
}
