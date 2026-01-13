const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const clearBtn = document.getElementById('clear');
const brush = document.getElementById('brush');
const rt = document.getElementById('rt');
const predictBtn = document.getElementById('predictBtn');
const predEl = document.getElementById('pred');
const barsEl = document.getElementById('bars');
const fileInput = document.getElementById('file'); 

/**********************************************************************************
***     Canvas
***********************************************************************************/

ctx.fillStyle = '#fff';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = '#000';
ctx.lineWidth = Number(brush.value);

brush.addEventListener('input', () => {
  ctx.lineWidth = Number(brush.value);
});

let drawing = false;
let last = null;

function posFromEvent(e) {
  const rect = canvas.getBoundingClientRect();
  const touch = e.touches && e.touches[0];
  const clientX = touch ? touch.clientX : e.clientX;
  const clientY = touch ? touch.clientY : e.clientY;
  return {
    x: (clientX - rect.left) * (canvas.width / rect.width),
    y: (clientY - rect.top) * (canvas.height / rect.height)
  };
}

function drawLine(a, b) {
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}

function onDown(e) {
  e.preventDefault();
  drawing = true;
  last = posFromEvent(e);
}

function onMove(e) {
  if (!drawing) return;
  const p = posFromEvent(e);
  drawLine(last, p);
  last = p;
  debouncedPredict();
}

function onUp(e) {
  drawing = false;
  last = null;
  debouncedPredict();
}

/**********************************************************************************
***     Mouse & Touch Events 
***********************************************************************************/

canvas.addEventListener('mousedown', onDown);
canvas.addEventListener('mousemove', onMove);
window.addEventListener('mouseup', onUp);
canvas.addEventListener('touchstart', onDown, { passive: false });
canvas.addEventListener('touchmove', onMove, { passive: false });
canvas.addEventListener('touchend', onUp);

/**********************************************************************************
***     Buttons & Interactions
***********************************************************************************/

clearBtn.addEventListener('click', () => {
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#000';
  setPrediction(null);
  fileInput.value = ''; 
});

/**********************************************************************************
***     Prediction
***********************************************************************************/

predictBtn.addEventListener('click', () => predict());

function setPrediction(res) {
  if (!res) {
    predEl.textContent = '-';
    barsEl.innerHTML = '';
    for (let i = 0; i < 10; i++) addBar(i, 0);
    return;
  }
  predEl.textContent = String(res.prediction);
  barsEl.innerHTML = '';
  res.probabilities.forEach((p, i) => addBar(i, p));
}

function addBar(digit, p) {
  const label = document.createElement('div');
  label.className = 'label';
  label.textContent = digit;

  const bar = document.createElement('div');
  bar.className = 'bar';
  const fill = document.createElement('span');
  fill.style.width = Math.round(p * 100) + '%';
  bar.appendChild(fill);

  const value = document.createElement('div');
  value.className = 'value';
  value.textContent = (p * 100).toFixed(1) + '%';

  barsEl.appendChild(label);
  barsEl.appendChild(bar);
  barsEl.appendChild(value);
}

setPrediction(null);

async function predict() {
  try {
    const dataURL = canvas.toDataURL('image/png');
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataURL })
    });
    const json = await res.json();
    if (json.error) throw new Error(json.error);
    setPrediction(json);
  } catch (err) {
    console.error(err);
  }
}

let t = null;
function debouncedPredict() {
  if (!rt.checked) return;
  clearTimeout(t);
  t = setTimeout(predict, 250);
}

/**********************************************************************************
***     File Upload
***********************************************************************************/

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function(evt) {
    const img = new Image();
    img.onload = function() {
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
      const x = (canvas.width - img.width * scale) / 2;
      const y = (canvas.height - img.height * scale) / 2;
      ctx.drawImage(img, x, y, img.width * scale, img.height * scale);

      debouncedPredict();
    };
    img.src = evt.target.result;
  };
  reader.readAsDataURL(file);
});
