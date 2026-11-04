// Small multilayer perceptrons for the cart game: a deep Q-network (reinforcement learning)
// and a softmax action classifier (imitation learning). No DOM access, so it can also run outside the browser.
function createNetwork(sizes, random = Math.random) {
  const layers = [];
  for (let l = 1; l < sizes.length; l++) {
    const inputs = sizes[l - 1], outputs = sizes[l];
    const scale = Math.sqrt(2 / inputs) * (l === sizes.length - 1 ? 0.1 : 1);
    const W = new Float64Array(outputs * inputs);
    for (let i = 0; i < W.length; i++) W[i] = gaussian(random) * scale;
    layers.push({ inputs, outputs, W, b: new Float64Array(outputs) });
  }
  return { sizes: sizes.slice(), layers };
}
function gaussian(random) {
  const u = 1 - random(), v = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function copyNetwork(from, to) {
  from.layers.forEach((layer, l) => { to.layers[l].W.set(layer.W); to.layers[l].b.set(layer.b); });
}

// Returns every layer's activations (ReLU hidden layers, linear output).
function forwardPass(net, input) {
  const activations = [Float64Array.from(input)];
  net.layers.forEach((layer, l) => {
    const x = activations[l], out = new Float64Array(layer.outputs);
    const last = l === net.layers.length - 1;
    for (let j = 0; j < layer.outputs; j++) {
      let z = layer.b[j];
      const row = j * layer.inputs;
      for (let k = 0; k < layer.inputs; k++) z += layer.W[row + k] * x[k];
      out[j] = last ? z : Math.max(0, z);
    }
    activations.push(out);
  });
  return activations;
}
function outputs(net, input) { const a = forwardPass(net, input); return Array.from(a[a.length - 1]); }
function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

// ---------- Gradients and Adam ----------
function zeroGrads(net) { return net.layers.map(layer => ({ W: new Float64Array(layer.W.length), b: new Float64Array(layer.outputs) })); }
// Adds the gradient for one example to grads, given dLoss/dOutput in delta.
function backprop(net, activations, delta, grads) {
  for (let l = net.layers.length - 1; l >= 0; l--) {
    const layer = net.layers[l], x = activations[l], g = grads[l];
    const back = new Float64Array(layer.inputs);
    for (let j = 0; j < layer.outputs; j++) {
      const d = delta[j];
      if (d === 0) continue;
      const row = j * layer.inputs;
      g.b[j] += d;
      for (let k = 0; k < layer.inputs; k++) { g.W[row + k] += d * x[k]; back[k] += d * layer.W[row + k]; }
    }
    if (l > 0) for (let k = 0; k < back.length; k++) if (x[k] <= 0) back[k] = 0;
    delta = back;
  }
}
function createAdam(net) {
  return { t: 0, layers: net.layers.map(layer => ({ mW: new Float64Array(layer.W.length), vW: new Float64Array(layer.W.length), mb: new Float64Array(layer.outputs), vb: new Float64Array(layer.outputs) })) };
}
function adamStep(net, adam, grads, count, rate) {
  adam.t++;
  const b1 = 0.9, b2 = 0.999, c1 = 1 - Math.pow(b1, adam.t), c2 = 1 - Math.pow(b2, adam.t);
  const step = (params, grad, mo, ve) => {
    for (let k = 0; k < params.length; k++) {
      const gk = grad[k] / count;
      mo[k] = b1 * mo[k] + (1 - b1) * gk;
      ve[k] = b2 * ve[k] + (1 - b2) * gk * gk;
      params[k] -= rate * (mo[k] / c1) / (Math.sqrt(ve[k] / c2) + 1e-8);
    }
  };
  net.layers.forEach((layer, l) => {
    const s = adam.layers[l];
    step(layer.W, grads[l].W, s.mW, s.vW);
    step(layer.b, grads[l].b, s.mb, s.vb);
  });
}

// ---------- Deep Q-network ----------
const qValues = outputs;

function createAgent(options = {}) {
  const random = options.random || Math.random;
  const sizes = options.sizes || [6, 24, 24, 3];
  const net = createNetwork(sizes, random);
  const target = createNetwork(sizes, random);
  copyNetwork(net, target);
  const capacity = options.capacity || 20000;
  const dims = sizes[0];
  return {
    net, target, random, steps: 0, updates: 0, adam: createAdam(net),
    memory: { capacity, size: 0, next: 0, s: new Float64Array(capacity * dims), s2: new Float64Array(capacity * dims), a: new Uint8Array(capacity), r: new Float64Array(capacity), done: new Uint8Array(capacity) },
    loss: 0
  };
}

function chooseAction(agent, input, epsilon) {
  const q = qValues(agent.net, input);
  if (agent.random() < epsilon) return { action: Math.floor(agent.random() * q.length), explored: true, q };
  return { action: argmax(q), explored: false, q };
}

function remember(agent, s, a, r, s2, done) {
  const m = agent.memory, dims = agent.net.sizes[0], i = m.next;
  m.s.set(s, i * dims);
  m.s2.set(s2, i * dims);
  m.a[i] = a; m.r[i] = r; m.done[i] = done ? 1 : 0;
  m.next = (i + 1) % m.capacity;
  m.size = Math.min(m.capacity, m.size + 1);
}

// One double-DQN update: target = r + γ·Q_target(s′, argmax Q(s′)), Huber loss, Adam.
// With replay, the update averages a random minibatch from memory; without it, only the step that just happened.
function learn(agent, { gamma = 0.98, rate = 0.001, batch = 32, sync = 300, replay = true } = {}) {
  const m = agent.memory, dims = agent.net.sizes[0];
  agent.steps++;
  if (!replay) batch = 1;
  if (m.size < (replay ? Math.max(batch, 500) : 1)) return null;
  const grads = zeroGrads(agent.net);
  let lossSum = 0;
  for (let n = 0; n < batch; n++) {
    const i = replay ? Math.floor(agent.random() * m.size) : (m.next - 1 + m.capacity) % m.capacity;
    const s = m.s.subarray(i * dims, (i + 1) * dims);
    let target = m.r[i];
    if (!m.done[i]) {
      const s2 = m.s2.subarray(i * dims, (i + 1) * dims);
      const next = argmax(qValues(agent.net, s2));
      target += gamma * qValues(agent.target, s2)[next];
    }
    const acts = forwardPass(agent.net, s);
    const out = acts[acts.length - 1];
    const error = out[m.a[i]] - target;
    lossSum += Math.abs(error) < 1 ? error * error / 2 : Math.abs(error) - 0.5;
    const delta = new Float64Array(out.length);
    delta[m.a[i]] = Math.max(-1, Math.min(1, error));
    backprop(agent.net, acts, delta, grads);
  }
  agent.updates++;
  adamStep(agent.net, agent.adam, grads, batch, rate);
  if (agent.updates % sync === 0) copyNetwork(agent.net, agent.target);
  agent.loss = lossSum / batch;
  return agent.loss;
}

// ---------- Action classifier (imitation learning) ----------
function softmax(z) {
  const max = Math.max(...z);
  const e = z.map(v => Math.exp(v - max));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / sum);
}
function actionProbabilities(net, input) { return softmax(outputs(net, input)); }

function createClassifier(options = {}) {
  const random = options.random || Math.random;
  const net = createNetwork(options.sizes || [6, 24, 24, 3], random);
  return { net, random, adam: createAdam(net), epochs: 0, updates: 0 };
}

// One Adam step on a minibatch of examples ({ s, a }) with cross-entropy loss.
function trainBatch(model, batch, rate = 0.003) {
  const grads = zeroGrads(model.net);
  for (const example of batch) {
    const acts = forwardPass(model.net, example.s);
    const p = softmax(Array.from(acts[acts.length - 1]));
    backprop(model.net, acts, p.map((value, k) => value - (k === example.a ? 1 : 0)), grads);
  }
  adamStep(model.net, model.adam, grads, batch.length, rate);
  model.updates++;
}

// One pass over the examples in shuffled minibatches.
function trainEpoch(model, examples, { rate = 0.003, batch = 32 } = {}) {
  const order = examples.slice();
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(model.random() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  for (let start = 0; start < order.length; start += batch) trainBatch(model, order.slice(start, start + batch), rate);
  model.epochs++;
}

// Average cross-entropy and accuracy of the classifier on a set of examples.
function evaluateClassifier(model, examples) {
  if (!examples.length) return { loss: NaN, accuracy: NaN };
  let loss = 0, correct = 0;
  for (const example of examples) {
    const p = actionProbabilities(model.net, example.s);
    loss -= Math.log(Math.max(1e-12, p[example.a]));
    if (argmax(p) === example.a) correct++;
  }
  return { loss: loss / examples.length, accuracy: correct / examples.length };
}

// ---------- Policy gradient (REINFORCE) ----------
// The network's outputs are action preferences; π(a | s) = softmax(outputs). Actions are sampled from π.
function createPolicy(options = {}) {
  const random = options.random || Math.random;
  const net = createNetwork(options.sizes || [6, 24, 24, 3], random);
  return { net, random, adam: createAdam(net), updates: 0 };
}
function sampleAction(policy, input) {
  const p = actionProbabilities(policy.net, input);
  let u = policy.random(), a = 0;
  while (a < p.length - 1 && u >= p[a]) { u -= p[a]; a++; }
  return { action: a, p };
}
// One update from a whole episode: θ ← θ + α Σ_t A_t ∇ log π(a_t | s_t), where A_t is how much better than the
// baseline the return from step t was. An optional entropy bonus (β) keeps the policy from becoming certain too early.
function reinforce(policy, steps, { rate = 0.001, entropy = 0 } = {}) {
  const grads = zeroGrads(policy.net);
  for (const { s, a, advantage } of steps) {
    const acts = forwardPass(policy.net, s);
    const p = softmax(Array.from(acts[acts.length - 1]));
    const h = -p.reduce((sum, pk) => sum + pk * Math.log(Math.max(1e-12, pk)), 0);
    // Gradient of the loss −A log π(a | s) − β H(π) with respect to the preferences.
    const delta = p.map((pk, k) => advantage * (pk - (k === a ? 1 : 0)) + entropy * pk * (Math.log(Math.max(1e-12, pk)) + h));
    backprop(policy.net, acts, delta, grads);
  }
  adamStep(policy.net, policy.adam, grads, steps.length, rate);
  policy.updates++;
}

if (typeof module !== 'undefined') module.exports = {
  createNetwork, forwardPass, outputs, qValues, argmax, softmax,
  createAgent, chooseAction, remember, learn,
  createClassifier, actionProbabilities, trainBatch, trainEpoch, evaluateClassifier,
  createPolicy, sampleAction, reinforce
};
