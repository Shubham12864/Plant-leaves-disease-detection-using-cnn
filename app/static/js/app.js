/* Landing page: the inline teaser scanner.
   The full product lives on scan.html and upload.html. This page shows one
   recorded case, so a visitor sees the result before reading anything.
   Shared nav, reveals, tagline and model card come from ui.js. */

(() => {
  const DEMO = window.DEMO;
  if (!DEMO) return;

  const { qs, qsa, pct, escape } = window.UI;
  const MAX_BYTES = 10 * 1024 * 1024;
  const state = { caseKey: 'confident', threshold: DEMO.model.confidence_threshold || 0.7, mode: 'empty' };

  const el = {
    frame: qs('#frame'),
    dropstate: qs('#dropstate'),
    file: qs('#file'),
    shot: qs('#shot'),
    shotBase: qs('#shotBase'),
    shotHeat: qs('#shotHeat'),
    compareRow: qs('#compareRow'),
    alpha: qs('#alpha'),
    notice: qs('#notice'),
    result: qs('#result'),
  };

  const metaRows = (caseData) => {
    const rows = [
      ['model', DEMO.model.architecture],
      ['grad cam', `${caseData.timing_ms.gradcam.toFixed(0)} ms on CPU`],
      ['source', 'locked test split'],
    ];
    if (!caseData.truth.match) {
      rows.push(['true label', `${caseData.truth.crop} ${caseData.truth.condition.toLowerCase()}`]);
    }
    return rows
      .map(([k, v]) => {
        const flag = k === 'true label' ? ' class="is-truth"' : '';
        return `<dt>${k}</dt><dd${flag}>${escape(v)}</dd>`;
      })
      .join('');
  };

  const readCopy = (caseData, confident) => {
    const runnerUp = caseData.top3[1];
    if (!caseData.truth.match) {
      return `<strong>A real misclassification.</strong> The true label, ${caseData.truth.crop.toLowerCase()} ${caseData.truth.condition.toLowerCase()}, is the ${pct(runnerUp.probability)} runner up, so this was close to a coin toss between two tomato diseases.`;
    }
    const tail = confident
      ? 'Warm colour sits on the leaf itself rather than the bench behind it, so the decision came from the symptom.'
      : 'The model is under the threshold you set, so treat this as inconclusive even though the top label is correct.';
    return `A correct call on a held out image. ${tail}`;
  };

  function renderResult() {
    const caseData = DEMO.cases[state.caseKey];
    if (!caseData) return;
    const confident = caseData.probability >= state.threshold;
    const stateText = confident
      ? `Confident at threshold ${state.threshold.toFixed(2)}`
      : `Below threshold ${state.threshold.toFixed(2)}`;

    el.result.innerHTML = `
      <p class="verdict-label mono">top 1 of ${DEMO.model.classes}</p>
      <p class="crop">${escape(caseData.crop)}<span class="condition">${escape(caseData.condition.toLowerCase())}</span></p>
      <p class="pct">${(caseData.probability * 100).toFixed(1)}<span>%</span></p>
      <p class="state ${confident ? 'ok' : 'unsure'}">${stateText}</p>
      <ul class="bars">${window.UI.bars(caseData.top3)}</ul>
      <dl class="meta mono">${metaRows(caseData)}</dl>
      <p class="read">${readCopy(caseData, confident)}</p>
      <div class="toolbar">
        <a class="btn-ghost btn-sm" href="scan.html">Open the scanner</a>
        <a class="btn-ghost btn-sm" href="upload.html">Batch upload</a>
      </div>
    `;
    window.UI.animateBars(el.result);
  }

  function renderSkeleton() {
    el.result.innerHTML = `<div class="skeleton" aria-label="Reading the leaf">
      <div class="sk sk-1"></div><div class="sk sk-2"></div><div class="sk sk-3"></div>
      <div class="sk sk-4"></div><div class="sk sk-5"></div><div class="sk sk-6"></div>
    </div>`;
  }

  function showDemoImage(caseData) {
    el.shotBase.src = caseData.image.photo;
    el.shotBase.alt = caseData.image.alt;
    el.shotHeat.hidden = false;
    el.shotHeat.src = caseData.image.gradcam;
    el.shot.hidden = false;
    el.dropstate.hidden = true;
    el.compareRow.hidden = false;
    el.alpha.value = 55;
    el.shotHeat.style.setProperty('--cut', '55%');
    el.notice.hidden = true;
  }

  function resetToEmpty() {
    state.mode = 'empty';
    el.shot.hidden = true;
    el.dropstate.hidden = false;
    el.compareRow.hidden = true;
    el.notice.hidden = true;
    el.result.innerHTML = '';
  }

  function runCase(key) {
    const caseData = DEMO.cases[key];
    if (!caseData) return;
    state.caseKey = key;
    state.mode = 'loading';
    showDemoImage(caseData);
    renderSkeleton();
    window.setTimeout(() => {
      state.mode = 'result';
      renderResult();
    }, 620);
  }

  function showOwnFile(file) {
    if (!file.type.startsWith('image/')) {
      el.notice.hidden = false;
      el.notice.textContent = 'That file is not an image. Use JPG, PNG or WEBP.';
      return;
    }
    if (file.size > MAX_BYTES) {
      el.notice.hidden = false;
      el.notice.textContent = `That file is ${(file.size / 1024 / 1024).toFixed(1)} MB. The limit is 10 MB.`;
      return;
    }
    el.shotBase.src = URL.createObjectURL(file);
    el.shotBase.alt = 'The photo you dropped, shown without any analysis.';
    el.shotHeat.hidden = true;
    el.shotHeat.removeAttribute('src');
    el.shot.hidden = false;
    el.dropstate.hidden = true;
    el.compareRow.hidden = true;
    el.notice.hidden = false;
    el.notice.textContent =
      'Demo mode: there is no backend on this page, so your photo is not classified. Open the scanner to run a recorded test image with the full result layout.';
    el.result.innerHTML = `
      <p class="verdict-label mono">not classified</p>
      <p class="crop">Your photo<span class="condition">stays in your browser</span></p>
      <p class="read">Nothing is uploaded and no number is invented. The scanner page runs the same checkpoint on a held out test image so the result layout is visible.</p>
      <div class="toolbar">
        <a class="btn-primary" href="scan.html">Open the scanner</a>
        <button class="btn-ghost btn-sm" data-clear type="button">Clear</button>
      </div>
    `;
    state.mode = 'own';
  }

  /* ── wiring ───────────────────────────────────────────────────────── */
  window.UI.chipGroup('[data-case]', (button) => runCase(button.dataset.case));

  window.UI.chipGroup('[data-threshold]', (button) => {
    state.threshold = Number(button.dataset.threshold);
    if (state.mode === 'result') renderResult();
  });

  const pickFile = () => el.file.click();
  el.dropstate.addEventListener('click', pickFile);
  el.dropstate.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      pickFile();
    }
  });
  el.file.addEventListener('change', () => {
    const [file] = el.file.files;
    if (file) showOwnFile(file);
    el.file.value = '';
  });

  ['dragenter', 'dragover'].forEach((type) =>
    el.frame.addEventListener(type, (event) => {
      event.preventDefault();
      el.dropstate.classList.add('is-over');
    })
  );
  ['dragleave', 'drop'].forEach((type) =>
    el.frame.addEventListener(type, (event) => {
      event.preventDefault();
      el.dropstate.classList.remove('is-over');
    })
  );
  el.frame.addEventListener('drop', (event) => {
    const [file] = event.dataTransfer.files;
    if (file) showOwnFile(file);
  });

  el.result.addEventListener('click', (event) => {
    if (event.target.closest('[data-clear]')) resetToEmpty();
  });

  el.alpha.addEventListener('input', () => {
    el.shotHeat.style.setProperty('--cut', `${el.alpha.value}%`);
  });

  window.UI.boot();
  runCase('confident');
})();
