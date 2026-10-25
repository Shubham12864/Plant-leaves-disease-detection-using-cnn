/* Inference page: one recorded case, every number exposed.
   Reads ?case=confident|hard from the URL so a result can be linked to. */

(() => {
  const { qs, qsa, pct, escape } = window.UI;
  const DEMO = window.DEMO;
  const THRESHOLD = DEMO.model.confidence_threshold;
  const VISIBLE_ROWS = 10;
  const TARGET_LAYER = 'features.8'; // last Conv2dNormActivation block of EfficientNet-B0

  const el = {
    verdict: qs('#verdict'),
    shotBase: qs('#shotBase'),
    shotHeat: qs('#shotHeat'),
    alpha: qs('#alpha'),
    timings: qs('#timings'),
    ranking: qs('#ranking'),
    rankingNote: qs('#rankingNote'),
    toggleAll: qs('#toggleAll'),
    ckpt: qs('#ckpt'),
    json: qs('#json'),
    caseLabel: qs('#caseLabel'),
  };

  const params = new URLSearchParams(window.location.search);
  let caseKey = params.get('case') === 'hard' ? 'hard' : 'confident';
  let expanded = false;

  /* ── verdict ──────────────────────────────────────────────────────── */
  function renderVerdict(caseData) {
    const confident = caseData.probability >= THRESHOLD;
    const margin = caseData.probability - (caseData.top3[1] ? caseData.top3[1].probability : 0);
    const truthLine = caseData.truth.match
      ? '<dt>true label</dt><dd>' + escape(caseData.truth.crop + ' ' + caseData.truth.condition.toLowerCase()) + '</dd>'
      : '<dt>true label</dt><dd class="is-truth">' + escape(caseData.truth.crop + ' ' + caseData.truth.condition.toLowerCase()) + '</dd>';

    el.verdict.innerHTML = `
      <p class="verdict-label mono">top 1 of ${DEMO.model.classes}</p>
      <p class="crop">${escape(caseData.crop)}<span class="condition">${escape(caseData.condition.toLowerCase())}</span></p>
      <p class="pct">${(caseData.probability * 100).toFixed(1)}<span>%</span></p>
      <p class="state ${confident ? 'ok' : 'unsure'}">${confident
        ? `Confident at threshold ${THRESHOLD.toFixed(2)}`
        : `Below threshold ${THRESHOLD.toFixed(2)}`}</p>
      <ul class="bars">${window.UI.bars(caseData.top3)}</ul>
      <dl class="meta mono">
        ${truthLine}
        <dt>margin</dt><dd>${(margin * 100).toFixed(1)} points over the runner up</dd>
        <dt>source</dt><dd>${escape(caseData.file || 'held out test image')}</dd>
      </dl>
      <p class="read">${caseData.truth.match
        ? 'The top label matches the recorded truth. The heat map is the part worth reading: it says which pixels carried the decision.'
        : `<strong>A real misclassification.</strong> The model put ${pct(caseData.probability)} on ${escape(caseData.crop.toLowerCase())}, while the truth sits at ${pct(caseData.top3[1].probability)}. A margin this thin is a coin toss, and the page says so instead of hiding it.`}</p>
      <div class="row-actions">
        <a class="btn-ghost btn-sm" href="scan.html">Back to the scanner</a>
        <a class="btn-ghost btn-sm" href="upload.html">Batch upload</a>
      </div>
    `;
    window.UI.animateBars(el.verdict);
  }

  /* ── timing tiles ─────────────────────────────────────────────────── */
  function renderTimings(caseData) {
    const t = caseData.timing_ms;
    const total = t.preprocess + t.forward + t.gradcam;
    const tiles = [
      ['preprocess', `${Math.round(t.preprocess)} ms`],
      ['forward pass', `${Math.round(t.forward)} ms`],
      ['grad cam', `${Math.round(t.gradcam)} ms`],
      ['total', `${Math.round(total)} ms`],
    ];
    el.timings.innerHTML = tiles
      .map(([term, value]) => `<div class="tile"><dt>${term}</dt><dd>${value}</dd></div>`)
      .join('');
  }

  /* ── all 38 classes ───────────────────────────────────────────────── */
  function renderRanking(caseData) {
    const ranking = caseData.ranking || caseData.top3;
    const negligible = ranking.filter((row) => row.probability < 0.0001).length;

    el.rankingNote.textContent = negligible
      ? `${negligible} of ${ranking.length} classes scored under 0.01 percent and are dimmed, but every one of them is listed.`
      : `All ${ranking.length} classes are listed.`;

    el.ranking.innerHTML = ranking
      .map((row, index) => {
        const hidden = index >= VISIBLE_ROWS && !expanded;
        const isTop = index === 0;
        const isTrue = caseData.truth && row.label === caseData.truth.label && !isTop;
        const isEmpty = row.probability < 0.0001;
        const classes = ['rankrow'];
        if (isTop) classes.push('is-top');
        if (isTrue) classes.push('is-true');
        if (isEmpty) classes.push('is-empty');
        return `<li class="${classes.join(' ')}"${hidden ? ' hidden' : ''}>
          <span class="rk-rank">${index + 1}</span>
          <span class="rk-name">${escape(row.crop)}, ${escape(row.condition.toLowerCase())}</span>
          <span class="rk-track"><span class="rk-fill" style="width:${Math.max(row.probability * 100, 0.4).toFixed(2)}%"></span></span>
          <span class="rk-pct">${pct(row.probability, 2)}</span>
        </li>`;
      })
      .join('');

    el.toggleAll.textContent = expanded ? 'Show the first 10' : `Show all ${ranking.length}`;
    el.toggleAll.setAttribute('aria-expanded', String(expanded));
  }

  /* ── checkpoint ───────────────────────────────────────────────────── */
  function renderCheckpoint() {
    const m = DEMO.model;
    const rows = [
      ['file', m.checkpoint],
      ['sha256', m.checkpoint_sha256],
      ['architecture', m.architecture],
      ['parameters', m.parameters.toLocaleString('en-US')],
      ['classes', String(m.classes)],
      ['input', `${m.image_size} by ${m.image_size} px`],
      ['normalise mean', m.normalization.mean.map((v) => v.toFixed(3)).join(', ')],
      ['normalise std', m.normalization.std.map((v) => v.toFixed(3)).join(', ')],
      ['best epoch', String(m.epoch)],
      ['validation macro F1', m.val_macro_f1.toFixed(4)],
      ['confidence threshold', m.confidence_threshold.toFixed(2)],
      ['grad cam layer', TARGET_LAYER],
    ];
    el.ckpt.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${escape(v)}</dd>`).join('');
  }

  /* ── the response the backend must return ─────────────────────────── */
  function renderJson(caseData) {
    const t = caseData.timing_ms;
    const payload = {
      ok: true,
      prediction: {
        label: caseData.label,
        crop: caseData.crop,
        condition: caseData.condition,
        probability: caseData.probability,
        confident: caseData.probability >= THRESHOLD,
      },
      top3: caseData.top3.map((row) => ({
        label: row.label,
        crop: row.crop,
        condition: row.condition,
        probability: row.probability,
      })),
      gradcam: {
        png_base64: '<the 224 by 224 heat map, base64>',
        target_layer: TARGET_LAYER,
        alpha: 0.45,
      },
      timing_ms: {
        preprocess: Math.round(t.preprocess),
        forward: Math.round(t.forward),
        gradcam: Math.round(t.gradcam),
        total: Math.round(t.preprocess + t.forward + t.gradcam),
      },
      warnings: caseData.probability >= THRESHOLD
        ? []
        : [`probability ${caseData.probability.toFixed(4)} is under the ${THRESHOLD.toFixed(2)} threshold`],
      model: {
        architecture: DEMO.model.architecture,
        image_size: DEMO.model.image_size,
        classes: DEMO.model.classes,
        val_macro_f1: DEMO.model.val_macro_f1,
        checkpoint_sha256: DEMO.model.checkpoint_sha256,
      },
    };
    el.json.textContent = JSON.stringify(payload, null, 2);
  }

  /* ── render one case ──────────────────────────────────────────────── */
  function render() {
    const caseData = DEMO.cases[caseKey];
    if (!caseData) return;

    el.shotBase.src = caseData.image.photo;
    el.shotBase.alt = caseData.image.alt;
    el.shotHeat.src = caseData.image.gradcam;
    el.alpha.value = 55;
    el.shotHeat.style.setProperty('--cut', '55%');
    el.caseLabel.textContent = caseKey === 'hard' ? 'hard case' : 'confident case';

    renderVerdict(caseData);
    renderTimings(caseData);
    renderRanking(caseData);
    renderJson(caseData);

    qsa('[data-case]').forEach((button) => {
      const on = button.dataset.case === caseKey;
      button.classList.toggle('is-on', on);
      button.setAttribute('aria-pressed', String(on));
    });
  }

  qsa('[data-case]').forEach((button) => {
    button.addEventListener('click', () => {
      caseKey = button.dataset.case;
      expanded = false;
      const url = new URL(window.location.href);
      url.searchParams.set('case', caseKey);
      window.history.replaceState({}, '', url);
      render();
    });
  });

  el.toggleAll.addEventListener('click', () => {
    expanded = !expanded;
    renderRanking(DEMO.cases[caseKey]);
    if (!expanded) el.ranking.scrollIntoView({ block: 'start', behavior: 'smooth' });
  });

  el.alpha.addEventListener('input', () => {
    el.shotHeat.style.setProperty('--cut', `${el.alpha.value}%`);
  });

  qs('#modeBadge').textContent = window.API.isLive()
    ? 'live mode · connected to the inference service'
    : 'demo mode · recorded checkpoint output';
  qs('#modeBadge').className = `badge ${window.API.isLive() ? 'live' : 'demo'}`;

  renderCheckpoint();
  window.UI.boot();
  render();
})();
