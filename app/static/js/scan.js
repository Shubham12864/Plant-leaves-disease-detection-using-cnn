/* Scan page: one photo in, one verdict out.
   Real API when window.LEAFREAD_API is set, recorded checkpoint output otherwise.
   Nothing is invented: the demo path only ever shows responses that were
   generated from the checkpoint at build time. */

(() => {
  const { qs, qsa, pct, escape } = window.UI;
  const DEMO = window.DEMO;

  const MAX_BYTES = 10 * 1024 * 1024;
  const state = { caseKey: 'confident', threshold: 0.7, mode: 'idle', history: [] };

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
    stages: qs('#stages'),
    progress: qs('#progress'),
    progressBar: qs('#progress > i'),
    result: qs('#result'),
    camPane: qs('#camPane'),
    camVideo: qs('#camVideo'),
    recent: qs('#recent'),
    strip: qs('#strip'),
  };

  /* ── view models: one shape whether the answer came from demo or API ─ */
  const fromRecorded = (caseData, sourceLabel = 'recorded test image') => ({
    crop: caseData.crop,
    condition: caseData.condition,
    probability: caseData.probability,
    top3: caseData.top3,
    photo: caseData.image.photo,
    heat: caseData.image.gradcam,
    alt: caseData.image.alt,
    gradcamMs: caseData.timing_ms.gradcam,
    truth: caseData.truth,
    sourceLabel,
  });

  const fromApi = (payload, photoUrl) => {
    const prediction = payload.prediction;
    return {
      crop: prediction.crop,
      condition: prediction.condition,
      probability: prediction.probability,
      top3: payload.top3,
      photo: photoUrl,
      heat: `data:image/png;base64,${payload.gradcam.png_base64}`,
      alt: 'Your photo with the Grad-CAM heat map over it.',
      gradcamMs: payload.timing_ms && payload.timing_ms.gradcam,
      truth: null,
      sourceLabel: 'your photo',
    };
  };

  /* ── rendering ────────────────────────────────────────────────────── */
  function metaRows(vm) {
    const rows = [
      ['model', DEMO.model.architecture],
      ['source', vm.sourceLabel],
    ];
    if (vm.gradcamMs != null) rows.push(['grad cam', `${Math.round(vm.gradcamMs)} ms`]);
    if (vm.truth && !vm.truth.match) {
      rows.push(['true label', `${vm.truth.crop} ${vm.truth.condition.toLowerCase()}`]);
    }
    return rows
      .map(([k, v], index) => {
        const isTruth = k === 'true label';
        return `<dt>${k}</dt><dd${isTruth ? ' class="is-truth"' : ''}>${escape(v)}</dd>`;
      })
      .join('');
  }

  function readCopy(vm, confident) {
    const runnerUp = vm.top3[1];
    if (vm.truth && !vm.truth.match) {
      return `<strong>A real misclassification.</strong> The true label, ${vm.truth.crop.toLowerCase()} ${vm.truth.condition.toLowerCase()}, is the ${pct(runnerUp.probability)} runner up. Read the heat map before you trust the number.`;
    }
    if (!confident) {
      return `Under the threshold you set, so treat this as inconclusive even if the top label happens to be right. The three candidates above are the honest answer here.`;
    }
    return `Warm colour sits on the leaf rather than the background, so the decision came from the symptom. A first read, not a diagnosis.`;
  }

  function renderResult(vm) {
    const confident = vm.probability >= state.threshold;
    el.result.innerHTML = `
      <p class="verdict-label mono">top 1 of ${DEMO.model.classes}</p>
      <p class="crop">${escape(vm.crop)}<span class="condition">${escape(vm.condition.toLowerCase())}</span></p>
      <p class="pct">${(vm.probability * 100).toFixed(1)}<span>%</span></p>
      <p class="state ${confident ? 'ok' : 'unsure'}">${confident
        ? `Confident at threshold ${state.threshold.toFixed(2)}`
        : `Below threshold ${state.threshold.toFixed(2)}`}</p>
      <ul class="bars">${window.UI.bars(vm.top3)}</ul>
      <dl class="meta mono">${metaRows(vm)}</dl>
      <p class="read">${readCopy(vm, confident)}</p>
      <div class="toolbar">
        <button class="btn-ghost btn-sm" data-reset type="button">Scan another</button>
        ${vm.truth ? `<a class="btn-ghost btn-sm" href="inference.html?case=${state.caseKey}">Every number</a>` : ''}
        <a class="btn-ghost btn-sm" href="upload.html">Batch upload</a>
      </div>
    `;
    window.UI.animateBars(el.result);
    pushHistory(vm);
  }

  function renderSkeleton() {
    el.result.innerHTML = `<div class="skeleton" aria-label="Reading the leaf">
      <div class="sk sk-1"></div><div class="sk sk-2"></div><div class="sk sk-3"></div>
      <div class="sk sk-4"></div><div class="sk sk-5"></div><div class="sk sk-6"></div>
    </div>`;
  }

  function renderNotClassified(message) {
    el.result.innerHTML = `
      <p class="verdict-label mono">not classified</p>
      <p class="crop">Your photo<span class="condition">stays in this browser</span></p>
      <p class="read">${escape(message)}</p>
      <div class="toolbar">
        <button class="btn-primary" data-run-recorded type="button">Run a recorded test image</button>
        <button class="btn-ghost btn-sm" data-reset type="button">Clear</button>
      </div>
    `;
  }

  /* ── image surfaces ───────────────────────────────────────────────── */
  function showDemoImage(vm) {
    el.shotBase.src = vm.photo;
    el.shotBase.alt = vm.alt;
    el.shotHeat.hidden = false;
    el.shotHeat.src = vm.heat;
    el.shot.hidden = false;
    el.dropstate.hidden = true;
    el.camPane.hidden = true;
    el.compareRow.hidden = false;
    el.alpha.value = 55;
    el.shotHeat.style.setProperty('--cut', '55%');
  }

  function showOwnImage(url, alt) {
    el.shotBase.src = url;
    el.shotBase.alt = alt;
    el.shotHeat.hidden = true;      // a demo heat map must never sit on a user photo
    el.shotHeat.removeAttribute('src');
    el.shot.hidden = false;
    el.dropstate.hidden = true;
    el.camPane.hidden = true;
    el.compareRow.hidden = true;
  }

  function reset() {
    state.mode = 'idle';
    el.shot.hidden = true;
    el.dropstate.hidden = false;
    el.camPane.hidden = true;
    el.compareRow.hidden = true;
    el.notice.hidden = true;
    el.stages.hidden = true;
    el.progress.hidden = true;
    el.progressBar.style.width = '0%';
    el.result.innerHTML = '';
  }

  function notice(message) {
    el.notice.hidden = false;
    el.notice.textContent = message;
  }

  /* ── staged loading ───────────────────────────────────────────────── */
  function setStage(index, done) {
    qsa('.stage', el.stages).forEach((row, i) => {
      row.classList.toggle('active', i === index && !done);
      row.classList.toggle('done', i < index || (i === index && done));
    });
  }

  function startStages() {
    el.stages.hidden = false;
    el.progress.hidden = false;
    el.progressBar.style.width = '8%';
    setStage(0, false);
  }

  function finishStages() {
    qsa('.stage', el.stages).forEach((row) => {
      row.classList.remove('active');
      row.classList.add('done');
    });
    el.progressBar.style.width = '100%';
  }

  const wait = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

  /* ── the two run paths ────────────────────────────────────────────── */
  async function runRecorded(key) {
    const caseData = DEMO.cases[key];
    if (!caseData) return;
    state.caseKey = key;
    state.mode = 'busy';
    el.notice.hidden = true;

    const vm = fromRecorded(caseData);
    showDemoImage(vm);
    renderSkeleton();
    startStages();

    setStage(0, true);
    el.progressBar.style.width = '34%';
    await wait(320);
    setStage(1, true);
    el.progressBar.style.width = '68%';
    await wait(320);
    setStage(2, false);
    await wait(260);
    finishStages();

    state.mode = 'result';
    renderResult(vm);
  }

  async function runLive(file, photoUrl) {
    state.mode = 'busy';
    el.notice.hidden = true;
    showOwnImage(photoUrl, 'The photo you chose.');
    renderSkeleton();
    startStages();
    setStage(0, true);
    el.progressBar.style.width = '30%';

    try {
      setStage(1, false);
      el.progressBar.style.width = '62%';
      const payload = await window.API.predict(file);
      el.progressBar.style.width = '86%';
      setStage(2, false);
      await wait(180);
      finishStages();
      state.mode = 'result';
      renderResult(fromApi(payload, photoUrl));
    } catch (error) {
      el.stages.hidden = true;
      el.progress.hidden = true;
      notice(`${error.code === 'demo'
        ? 'No backend is configured, so this photo was not classified.'
        : error.message} Nothing was guessed.`);
      renderNotClassified(
        'The page is ready to send this photo to POST /api/predict, but no service is running yet. You can still run a recorded test image to see the full result layout.'
      );
      state.mode = 'own';
    }
  }

  /* ── file handling ────────────────────────────────────────────────── */
  function validate(file) {
    if (!file.type.startsWith('image/')) return 'That file is not an image. Use JPG, PNG or WEBP.';
    if (file.size > MAX_BYTES) return `That file is ${(file.size / 1024 / 1024).toFixed(1)} MB. The limit is 10 MB.`;
    return null;
  }

  function handleFile(file) {
    const problem = validate(file);
    if (problem) {
      notice(problem);
      return;
    }
    const url = URL.createObjectURL(file);
    if (window.API.isLive()) {
      runLive(file, url);
    } else {
      showOwnImage(url, 'The photo you chose, shown without any analysis.');
      notice('Demo mode: there is no backend yet, so this photo is not classified.');
      renderNotClassified(
        'The upload and preview work. The model needs the FastAPI service from the plan, so this photo is not sent anywhere and no number is invented for it.'
      );
      state.mode = 'own';
    }
  }

  /* ── camera ───────────────────────────────────────────────────────── */
  let stream = null;

  async function openCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      notice('This browser exposes no camera API, so use a photo file instead.');
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 } },
        audio: false,
      });
      el.camVideo.srcObject = stream;
      el.camPane.hidden = false;
      el.dropstate.hidden = true;
      el.shot.hidden = true;
      el.compareRow.hidden = true;
      el.notice.hidden = true;
      state.mode = 'camera';
    } catch (error) {
      notice(`Camera unavailable (${error.name.toLowerCase()}). Browsers only grant it on localhost or https.`);
    }
  }

  function closeCamera() {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
    el.camVideo.srcObject = null;
    el.camPane.hidden = true;
  }

  function capture() {
    const video = el.camVideo;
    if (!video.videoWidth) return;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      closeCamera();
      if (!blob) {
        notice('The capture failed. Try again.');
        return;
      }
      handleFile(new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' }));
    }, 'image/jpeg', 0.92);
  }

  /* ── history strip ────────────────────────────────────────────────── */
  function pushHistory(vm) {
    state.history.push({ photo: vm.photo, crop: vm.crop, probability: vm.probability });
    el.recent.hidden = false;
    el.strip.innerHTML = state.history
      .map(
        (item, index) => `<button type="button" data-history="${index}" title="${escape(item.crop)} ${pct(item.probability)}">
          <img src="${item.photo}" alt="${escape(item.crop)} at ${pct(item.probability)}">
        </button>`
      )
      .join('');
  }

  /* ── wiring ───────────────────────────────────────────────────────── */
  qs('#modeBadge').textContent = window.API.isLive()
    ? 'live mode · connected to the inference service'
    : 'demo mode · recorded checkpoint output';
  qs('#modeBadge').className = `badge ${window.API.isLive() ? 'live' : 'demo'}`;

  window.UI.chipGroup('[data-threshold]', (button) => {
    state.threshold = Number(button.dataset.threshold);
    if (state.mode === 'result') runRecorded(state.caseKey);
  });

  window.UI.chipGroup('[data-case]', (button) => runRecorded(button.dataset.case));

  qs('#pickBtn').addEventListener('click', () => el.file.click());
  el.dropstate.addEventListener('click', (event) => {
    if (event.target.closest('[data-case]')) return; // the chips handled it
    el.file.click();
  });
  el.dropstate.addEventListener('keydown', (event) => {
    if ((event.key === 'Enter' || event.key === ' ') && !event.target.closest('[data-case]')) {
      event.preventDefault();
      el.file.click();
    }
  });
  el.file.addEventListener('change', () => {
    const [file] = el.file.files;
    if (file) handleFile(file);
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
    if (file) handleFile(file);
  });

  qs('#camBtn').addEventListener('click', openCamera);
  qs('#camCancel').addEventListener('click', () => {
    closeCamera();
    reset();
  });
  qs('#snapBtn').addEventListener('click', capture);

  el.alpha.addEventListener('input', () => {
    el.shotHeat.style.setProperty('--cut', `${el.alpha.value}%`);
  });

  el.result.addEventListener('click', (event) => {
    if (event.target.closest('[data-reset]')) reset();
    if (event.target.closest('[data-run-recorded]')) runRecorded(state.caseKey);
  });

  el.strip.addEventListener('click', (event) => {
    const button = event.target.closest('[data-history]');
    if (!button) return;
    const item = state.history[Number(button.dataset.history)];
    if (item) runRecorded(state.caseKey);
  });

  window.UI.boot();
})();
