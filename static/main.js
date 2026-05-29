// ─── Utilities ───────────────────────────────────────────────────────────────
function ts() {
  const d = new Date();
  return [d.getHours(), d.getMinutes(), d.getSeconds()]
    .map(n => String(n).padStart(2, '0')).join(':');
}

const logBody = document.getElementById('log-body');
function log(msg, type = 'info') {
  // Remove stray cursor line first
  const cursors = logBody.querySelectorAll('.cursor');
  cursors.forEach(c => c.closest('.log-line')?.remove());

  const line = document.createElement('div');
  line.className = 'log-line';
  line.innerHTML = `<span class="log-ts">${ts()}</span><span class="log-${type}">${msg}</span>`;
  logBody.appendChild(line);

  // Re-add cursor
  const cur = document.createElement('div');
  cur.className = 'log-line';
  cur.style.alignItems = 'center';
  cur.innerHTML = `<span class="log-ts"></span><span class="log-info"><span class="cursor"></span></span>`;
  logBody.appendChild(cur);
  logBody.scrollTop = logBody.scrollHeight;
}

// ─── State ───────────────────────────────────────────────────────────────────
let selectedFile = null;

const dropZone    = document.getElementById('drop-zone');
const fileInput   = document.getElementById('file-input');
const previewWrap = document.getElementById('preview-wrap');
const previewImg  = document.getElementById('preview-img');
const fileNameLbl = document.getElementById('file-name-label');
const scanOverlay = document.getElementById('scan-overlay');
const btnClassify = document.getElementById('btn-classify');

const genderLabel   = document.getElementById('gender-label');
const confidencePct = document.getElementById('confidence-pct');
const confFill      = document.getElementById('conf-fill');
const mSigmoid      = document.getElementById('m-sigmoid');
const mDecision     = document.getElementById('m-decision');
const mConf         = document.getElementById('m-conf');

// Set init timestamp
document.getElementById('init-ts').textContent = ts();

// ─── File handling ────────────────────────────────────────────────────────────
function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    log('Invalid file type — please upload an image.', 'err');
    return;
  }
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewWrap.style.display = 'block';
  fileNameLbl.textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
  btnClassify.disabled = false;
  resetResult();
  log(`Image loaded: ${file.name}`);
}

fileInput.addEventListener('change', e => handleFile(e.target.files[0]));

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  handleFile(e.dataTransfer.files[0]);
});

// ─── Reset result ─────────────────────────────────────────────────────────────
function resetResult() {
  genderLabel.className = '';
  genderLabel.textContent = '———';
  confidencePct.className = '';
  confidencePct.textContent = 'AWAITING ANALYSIS';
  confFill.style.width = '0%';
  confFill.className = '';
  mSigmoid.textContent = '—';
  mDecision.textContent = '—';
  mConf.textContent = '—';
}

// ─── Classify ─────────────────────────────────────────────────────────────────
btnClassify.addEventListener('click', async () => {
  if (!selectedFile) return;

  btnClassify.disabled = true;
  scanOverlay.classList.add('active');
  log('Sending image to FastAPI /predict …');

  const form = new FormData();
  form.append('file', selectedFile);

  try {
    const res = await fetch('/predict', { method: 'POST', body: form });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    // data: { label, confidence, raw_probability }

    const isM = data.label === 'Male';
    const barPct = isM
      ? 50 + (data.confidence / 2)   // male → right half
      : 50 - (data.confidence / 2);  // female → left half

    // Animate label
    genderLabel.textContent = data.label.toUpperCase();
    genderLabel.className = isM ? 'male' : 'female';

    confidencePct.textContent = `${data.confidence.toFixed(1)}% CONFIDENCE`;
    confidencePct.className = isM ? 'male' : 'female';

    // Confidence bar
    confFill.className = isM ? '' : 'female';
    confFill.style.width = `${barPct}%`;

    // Metrics
    mSigmoid.textContent  = data.raw_probability.toFixed(6);
    mDecision.textContent = data.label;
    mConf.textContent     = `${data.confidence.toFixed(2)}%`;

    log(`Result: ${data.label} · ${data.confidence.toFixed(2)}% confidence`, 'ok');

  } catch (err) {
    log(`Error: ${err.message}`, 'err');
    resetResult();
  } finally {
    scanOverlay.classList.remove('active');
    btnClassify.disabled = false;
  }
});
