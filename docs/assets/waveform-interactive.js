// Lightweight 2D waveform viewer with zoom/pan
// Active field components are derived from each config:
// - lici: nonzero x,z  -> Ex/Ez
// - others: xy         -> Ex/Ey
const WAVEFORM_2D = {
  lici: {
    name: 'Chirped-Rotating Pulse',
    color: '#60a5fa',
    colorRGB: [96, 165, 250],
    components: [
      { label: 'Ex', fn: (t, env) => env * 0.107159 * Math.cos(0.005*t) * Math.sin(0.057*t + 0.00005*t*t), color: 'rgba(239,68,68,0.85)' },
      { label: 'Ez', fn: (t, env) => env * 0.107159 * Math.sin(0.005*t) * Math.sin(0.057*t + 0.00005*t*t), color: 'rgba(59,130,246,0.85)' }
    ]
  },
  liss: {
    name: 'OTC Figure-8 Lissajous',
    color: '#34d399',
    colorRGB: [52, 211, 153],
    components: [
      { label: 'Ex', fn: (t, env) => env * 0.077369 * Math.sin(0.057*t), color: 'rgba(239,68,68,0.85)' },
      { label: 'Ey', fn: (t, env) => env * 0.077369 * Math.sin(2*0.057*t + Math.PI/2), color: 'rgba(34,197,94,0.85)' }
    ]
  },
  saw: {
    name: 'Sawtooth OTC Waveform',
    color: '#fbbf24',
    colorRGB: [251, 191, 36],
    components: [
      { label: 'Ex', fn: (t, env) => env * 0.08578 * Math.cos(0.057*t), color: 'rgba(239,68,68,0.85)' },
      { label: 'Ey', fn: (t, env) => env * 0.08578 * Math.cos(2*0.057*t + Math.PI/2), color: 'rgba(34,197,94,0.85)' }
    ]
  },
  tre: {
    name: 'Counter-Rotating Trefoil',
    color: '#fb7185',
    colorRGB: [251, 113, 133],
    components: [
      { label: 'Ex', fn: (t, env) => env * 0.053964 * (Math.cos(0.057*t) + Math.cos(2*0.057*t)), color: 'rgba(239,68,68,0.85)' },
      { label: 'Ey', fn: (t, env) => env * 0.053964 * (Math.sin(0.057*t) - Math.sin(2*0.057*t)), color: 'rgba(34,197,94,0.85)' }
    ]
  }
};

// Generate 2D samples (t, value) for a component
function generateSamples(comp, N = 600) {
  const samples = [];
  const t_max = 10 * Math.PI / 0.057;
  for (let i = 0; i <= N; i++) {
    const t = (i / N) * t_max;
    const env = Math.sin(t * 0.057 / 10) ** 2;
    samples.push({ t, v: comp.fn(t, env) });
  }
  return samples;
}

// Viewer state
const state = {
  zoom: 1,
  panX: 0,
  panY: 0,
  isDragging: false,
  lastMouse: [0, 0],
  current: null
};

let overlayEl = null;
let canvasEl = null;
let titleEl = null;
let samplesCache = {};

function initOverlay() {
  overlayEl = document.getElementById('wfOverlay');
  canvasEl = document.getElementById('wfCanvas');
  titleEl = document.getElementById('wfTitle');

  if (!overlayEl || !canvasEl) return;
  overlayEl.style.display = 'none';

  const closeBtn = document.getElementById('wfClose');
  if (closeBtn) {
    closeBtn.onclick = () => closeOverlay();
  }

  overlayEl.addEventListener('click', (e) => {
    if (e.target === overlayEl || e.target.classList.contains('wf-backdrop')) {
      closeOverlay();
    }
  });

  canvasEl.addEventListener('mousedown', (e) => {
    state.isDragging = true;
    state.lastMouse = [e.clientX, e.clientY];
    canvasEl.style.cursor = 'grabbing';
    e.preventDefault();
  });

  document.addEventListener('mousemove', (e) => {
    if (!state.isDragging) return;
    const dx = e.clientX - state.lastMouse[0];
    const dy = e.clientY - state.lastMouse[1];
    state.panX += dx;
    state.panY += dy;
    state.lastMouse = [e.clientX, e.clientY];
    draw();
  });

  document.addEventListener('mouseup', () => {
    if (state.isDragging) {
      state.isDragging = false;
      canvasEl.style.cursor = 'grab';
    }
  });

  overlayEl.addEventListener('wheel', (e) => {
    e.preventDefault();
    const factor = -e.deltaY * 0.001;
    const newZoom = Math.max(0.5, Math.min(5, state.zoom * (1 + factor)));
    state.zoom = newZoom;
    draw();
  }, { passive: false });

  canvasEl.addEventListener('touchstart', (e) => {
    if (e.touches.length === 1) {
      state.isDragging = true;
      state.lastMouse = [e.touches[0].clientX, e.touches[0].clientY];
    }
    e.preventDefault();
  }, { passive: false });

  canvasEl.addEventListener('touchmove', (e) => {
    if (!state.isDragging || e.touches.length !== 1) return;
    const dx = e.touches[0].clientX - state.lastMouse[0];
    const dy = e.touches[0].clientY - state.lastMouse[1];
    state.panX += dx;
    state.panY += dy;
    state.lastMouse = [e.touches[0].clientX, e.touches[0].clientY];
    draw();
    e.preventDefault();
  }, { passive: false });

  canvasEl.addEventListener('touchend', () => { state.isDragging = false; });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && state.current !== null) closeOverlay();
  });

  canvasEl.addEventListener('mousemove', (e) => {
    if (state.current == null) return;
    const rect = canvasEl.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    drawCrosshair(mx, my);
  });
}

function openOverlay(key) {
  const wf = WAVEFORM_2D[key];
  if (!wf) return;
  state.current = key;
  state.zoom = 1;
  state.panX = 0;
  state.panY = 0;

  if (!samplesCache[key]) {
    samplesCache[key] = wf.components.map(c => generateSamples(c, 700));
  }

  titleEl.textContent = wf.name;
  canvasEl.width = 900;
  canvasEl.height = 520;

  overlayEl.style.display = 'flex';
  requestAnimationFrame(() => overlayEl.classList.add('active'));

  draw();
}

function closeOverlay() {
  overlayEl.classList.remove('active');
  state.current = null;
  setTimeout(() => { overlayEl.style.display = 'none'; }, 300);
}

function draw() {
  if (!state.current) return;
  const wf = WAVEFORM_2D[state.current];
  const ctx = canvasEl.getContext('2d');
  const W = canvasEl.width;
  const H = canvasEl.height;
  const samples = samplesCache[state.current];
  if (!samples) return;

  ctx.clearRect(0, 0, W, H);
  const grad = ctx.createLinearGradient(0, 0, W, H);
  grad.addColorStop(0, '#0a0f14');
  grad.addColorStop(1, '#111827');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);

  const pad = { top: 50, right: 40, bottom: 50, left: 60 };
  const plotW = (W - pad.left - pad.right) * state.zoom;
  const plotH = (H - pad.top - pad.bottom) * state.zoom;
  const cx = W / 2 + state.panX;
  const cy = H / 2 + state.panY;

  const tMax = 10 * Math.PI / 0.057;
  const vMax = 0.12;
  const tx = (t) => cx - plotW / 2 + (t / tMax) * plotW;
  const ty = (v) => cy + (v / vMax) * (plotH / 2);

  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 6; i++) {
    const x = pad.left + (i / 6) * (W - pad.left - pad.right);
    ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, H - pad.bottom); ctx.stroke();
    const y = pad.top + (i / 6) * (H - pad.top - pad.bottom);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
  }

  ctx.beginPath();
  ctx.moveTo(tx(0), ty(0));
  for (let i = 1; i < samples[0].length; i++) {
    ctx.lineTo(tx(samples[0][i].t), ty(samples[0][i].v));
  }
  ctx.strokeStyle = wf.components[0].color;
  ctx.lineWidth = 2.5;
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(tx(0), ty(0));
  for (let i = 1; i < samples[1].length; i++) {
    ctx.lineTo(tx(samples[1][i].t), ty(samples[1][i].v));
  }
  ctx.strokeStyle = wf.components[1].color;
  ctx.lineWidth = 2.5;
  ctx.stroke();

  ctx.fillStyle = 'rgba(255,255,255,0.7)';
  ctx.font = '600 12px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(wf.name.toUpperCase(), W / 2, 28);

  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  ctx.font = '10px Inter, sans-serif';
  ctx.fillText(`zoom: ${state.zoom.toFixed(2)}x  |  scroll to zoom, drag to pan`, W / 2, H - 18);

  ctx.fillStyle = wf.components[0].color;
  ctx.fillText(wf.components[0].label, 30, 34);
  ctx.fillStyle = wf.components[1].color;
  ctx.textAlign = 'right';
  ctx.fillText(wf.components[1].label, W - 30, 34);
  ctx.textAlign = 'left';
}

function drawCrosshair(mx, my) {
  if (!state.current) return;
  draw();
  const ctx = canvasEl.getContext('2d');
  ctx.setLineDash([6, 6]);
  ctx.strokeStyle = 'rgba(255,255,255,0.4)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(mx, 0); ctx.lineTo(mx, canvasEl.height); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, my); ctx.lineTo(canvasEl.width, my); ctx.stroke();
  ctx.setLineDash([]);
}

document.addEventListener('DOMContentLoaded', () => {
  initOverlay();

  document.querySelectorAll('.waveform-card[data-waveform]').forEach(card => {
    const key = card.dataset.waveform;
    const svg = card.querySelector('svg');

    svg.addEventListener('click', () => openOverlay(key));
    svg.addEventListener('mouseenter', () => {
      svg.style.transform = 'scale(1.01)';
      svg.style.transition = 'transform 0.2s ease';
    });
    svg.addEventListener('mouseleave', () => {
      svg.style.transform = 'scale(1)';
    });
  });
});
