function mulberry32(seed) {
  let t = seed >>> 0;
  return function () {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

function dist2(ax, ay, bx, by) {
  let dx = ax - bx;
  let dy = ay - by;
  return dx * dx + dy * dy;
}

function randomSeed() {
  const buf = new Uint32Array(1);
  if (globalThis.crypto && crypto.getRandomValues) {
    crypto.getRandomValues(buf);
    return (buf[0] >>> 0) || 1;
  }
  return ((Math.random() * 2 ** 32) >>> 0) || 1;
}

function uuid() {
  if (globalThis.crypto && crypto.randomUUID) return crypto.randomUUID();
  return `${Date.now().toString(16)}-${Math.floor(Math.random() * 1e9).toString(16)}`;
}

function cartesianGrid(gridSpec) {
  const keys = Object.keys(gridSpec || {});
  for (const key of keys) {
    if (!Array.isArray(gridSpec[key])) {
      throw new Error(`grid_spec.${key} must be a list/array`);
    }
  }
  const points = [];
  const recur = (idx, acc) => {
    if (idx === keys.length) {
      points.push({ ...acc });
      return;
    }
    const key = keys[idx];
    const values = gridSpec[key];
    for (const v of values) {
      acc[key] = v;
      recur(idx + 1, acc);
    }
  };
  recur(0, {});
  return points;
}

function generateState(params, seedValue, model) {
  const rng = mulberry32(seedValue || 1);
  const width = model.get("arena_width");
  const height = model.get("arena_height");
  const radius = model.get("unit_radius");
  const spawnMode = String(model.get("spawn_mode") || "sides");

  const spawnSide = (team) => {
    const margin = 2 * radius + 4;
    const x = spawnMode === "mixed"
      ? margin + rng() * (width - 2 * margin)
      : team === "blue"
        ? margin + rng() * (width * 0.35 - margin)
        : width * 0.65 + rng() * (width * 0.35 - margin);
    const y = margin + rng() * (height - 2 * margin);
    return { x, y };
  };

  let nextId = 1;
  const mk = (team) => {
    const { x, y } = spawnSide(team);
    return {
      id: nextId++,
      team,
      x,
      y,
      hp: model.get("hp"),
      cooldown: 0,
    };
  };

  const nBlue = Number(params.n_blue ?? 50);
  const nRed = Number(params.n_red ?? 50);

  return {
    t: 0,
    blue: Array.from({ length: nBlue }, () => mk("blue")),
    red: Array.from({ length: nRed }, () => mk("red")),
    rng,
    done: false,
    winner: null,
  };
}

function step(state, model) {
  const dt = Number(model.get("step_dt")) || 0.02;
  const speed = Number(model.get("move_speed")) || 55.0;
  const radius = Number(model.get("unit_radius")) || 4.0;
  const width = Number(model.get("arena_width")) || 640;
  const height = Number(model.get("arena_height")) || 420;
  const range = Number(model.get("attack_range")) || 10.0;
  const cooldown = Number(model.get("attack_cooldown")) || 0.25;
  const hitChance = Number(model.get("hit_chance")) || 0.85;
  const damage = Number(model.get("damage")) || 1;

  if (!Number.isFinite(dt) || dt <= 0) return;

  const range2 = range * range;
  const minSep = 2 * radius;
  const minSep2 = minSep * minSep;

  const all = state.blue.concat(state.red);
  const aliveBlue = state.blue;
  const aliveRed = state.red;

  // Move toward nearest opponent.
  for (const u of all) {
    const opponents = u.team === "blue" ? aliveRed : aliveBlue;
    if (opponents.length === 0) continue;

    let best = null;
    let bestD2 = Infinity;
    for (const v of opponents) {
      const d2 = dist2(u.x, u.y, v.x, v.y);
      if (d2 < bestD2) {
        bestD2 = d2;
        best = v;
      }
    }
    if (!best) continue;

    const dx = best.x - u.x;
    const dy = best.y - u.y;
    const d = Math.sqrt(dx * dx + dy * dy) || 1;
    const vx = (dx / d) * speed;
    const vy = (dy / d) * speed;
    u.x = clamp(u.x + vx * dt, radius, width - radius);
    u.y = clamp(u.y + vy * dt, radius, height - radius);
  }

  // Simple collision separation.
  for (let pass = 0; pass < 2; pass++) {
    for (let i = 0; i < all.length; i++) {
      for (let j = i + 1; j < all.length; j++) {
        const a = all[i];
        const b = all[j];
        const d2 = dist2(a.x, a.y, b.x, b.y);
        if (d2 >= minSep2 || d2 === 0) continue;
        const d = Math.sqrt(d2);
        const overlap = (minSep - d) / 2;
        const nx = (a.x - b.x) / d;
        const ny = (a.y - b.y) / d;
        a.x = clamp(a.x + nx * overlap, radius, width - radius);
        a.y = clamp(a.y + ny * overlap, radius, height - radius);
        b.x = clamp(b.x - nx * overlap, radius, width - radius);
        b.y = clamp(b.y - ny * overlap, radius, height - radius);
      }
    }
  }

  // Attacks (will be replaced with melee soon; kept minimal for now).
  const toKill = new Set();
  for (const u of all) {
    u.cooldown = Math.max(0, u.cooldown - dt);
    if (u.cooldown > 0) continue;

    const opponents = u.team === "blue" ? aliveRed : aliveBlue;
    if (opponents.length === 0) continue;

    let best = null;
    let bestD2 = Infinity;
    for (const v of opponents) {
      const d2 = dist2(u.x, u.y, v.x, v.y);
      if (d2 < bestD2) {
        bestD2 = d2;
        best = v;
      }
    }

    if (!best || bestD2 > range2) continue;

    u.cooldown = cooldown;
    if (state.rng() <= hitChance) {
      best.hp -= damage;
      if (best.hp <= 0) toKill.add(best.id);
    }
  }

  if (toKill.size > 0) {
    state.blue = state.blue.filter((u) => !toKill.has(u.id));
    state.red = state.red.filter((u) => !toKill.has(u.id));
  }

  state.t += dt;
  if (state.blue.length === 0 || state.red.length === 0) {
    state.done = true;
    state.winner = state.blue.length > 0 ? "blue" : state.red.length > 0 ? "red" : "draw";
  }
}

function draw(ctx, state, model) {
  const width = model.get("arena_width");
  const height = model.get("arena_height");
  const radius = model.get("unit_radius");

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0b1220";
  ctx.fillRect(0, 0, width, height);

  const drawUnit = (u, fill) => {
    ctx.beginPath();
    ctx.arc(u.x, u.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
  };

  for (const u of state.blue) drawUnit(u, "#60a5fa");
  for (const u of state.red) drawUnit(u, "#fb7185");
}

function ensureCanvas(el, model) {
  const canvas = document.createElement("canvas");
  canvas.className = "battle-canvas";
  const resize = () => {
    canvas.width = model.get("arena_width");
    canvas.height = model.get("arena_height");
  };
  resize();
  model.on("change:arena_width", resize);
  model.on("change:arena_height", resize);
  el.appendChild(canvas);
  return canvas;
}

const raf = () => new Promise((resolve) => requestAnimationFrame(resolve));

export default {
  render({ model, el }) {
    const renderEnabled = Boolean(model.get("render"));
    const canvas = renderEnabled ? ensureCanvas(el, model) : null;
    const ctx = canvas ? canvas.getContext("2d") : null;

    model.set("done", false);
    model.set("results", []);
    model.set("results_len", 0);
    model.set("error", "");
    model.save_changes();

    let cancelled = false;

    (async () => {
      try {
        const gridPoints = cartesianGrid(model.get("grid_spec"));
        const runsPerPoint = Math.max(1, Number(model.get("runs_per_point") || 1));
        const seedMode = String(model.get("seed_mode") || "random");
        const baseSeed = Number(model.get("base_seed") || 1) >>> 0;
        const maxTime = Number(model.get("max_time") || 60.0);
        const recordDt = Math.max(0.0001, Number(model.get("record_dt") || 0.1));

        const results = [];
        let globalRunIndex = 0;

        for (const params of gridPoints) {
          for (let r = 0; r < runsPerPoint; r++) {
            if (cancelled) return;

            const runId = uuid();
            const seedUsed =
              seedMode === "base_plus_index" ? ((baseSeed + globalRunIndex) >>> 0) || 1 : randomSeed();
            globalRunIndex += 1;

            const state = generateState(params, seedUsed, model);

            let nextRecordT = 0;
            results.push({
              run_id: runId,
              seed: seedUsed,
              time: 0,
              n_blue: state.blue.length,
              n_red: state.red.length,
            });
            nextRecordT += recordDt;

            let steps = 0;
            while (!state.done && state.t < maxTime) {
              step(state, model);
              steps += 1;

              if (state.t + 1e-9 >= nextRecordT) {
                results.push({
                  run_id: runId,
                  seed: seedUsed,
                  time: Number(state.t.toFixed(6)),
                  n_blue: state.blue.length,
                  n_red: state.red.length,
                });
                nextRecordT += recordDt;
              }

              if (!Number.isFinite(state.t)) break;

              if (renderEnabled && ctx && steps % 5 === 0) {
                draw(ctx, state, model);
                await raf();
              } else if (steps % 500 === 0) {
                await raf();
              }
            }

            if (renderEnabled && ctx) {
              draw(ctx, state, model);
              await raf();
            }

            // Ensure the terminal state is recorded (the first row where one side hits 0).
            if (state.done) {
              const last = results[results.length - 1];
              const needsTerminalRow =
                !last ||
                last.run_id !== runId ||
                last.n_blue !== state.blue.length ||
                last.n_red !== state.red.length;
              if (needsTerminalRow) {
                results.push({
                  run_id: runId,
                  seed: seedUsed,
                  time: Number(state.t.toFixed(6)),
                  n_blue: state.blue.length,
                  n_red: state.red.length,
                });
              }
            }

            // Flush after each run so notebook cells can react incrementally.
            model.set("results", results.slice());
            model.set("results_len", results.length);
            model.save_changes();

            await raf();
          }
        }

        model.set("results", results.slice());
        model.set("results_len", results.length);
        model.set("done", true);
        model.save_changes();
      } catch (e) {
        model.set("error", String(e && e.message ? e.message : e));
        model.set("done", true);
        model.save_changes();
      }
    })();

    return () => {
      cancelled = true;
    };
  },
};
