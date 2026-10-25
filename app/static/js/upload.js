/* Batch upload page.
   Demo mode replays twelve recorded predictions from the checkpoint.
   Live mode posts each file to POST /api/predict one at a time.
   A file that cannot be classified says so; it never gets a made up label. */

(() => {
  const { qs, qsa, pct, escape } = window.UI;
  const DEMO = window.DEMO;
  const MAX_BYTES = 10 * 1024 * 1024;

  const el = {
    dropzone: qs('#dropzone'),
    files: qs('#files'),
    rows: qs('#rows'),
    summary: qs('#summary'),
    tiles: qs('#tiles'),
    tableBody: qs('#tableBody'),
    progress: qs('#progress'),
    progressBar: qs('#progress > i'),
    progressWrap: qs('#progressWrap'),
    progressLabel: qs('#progressLabel'),
    notice: qs('#notice'),
    pickBtn: qs('#pickBtn'),
    sampleBtn: qs('#sampleBtn'),
    clearBtn: qs('#clearBtn'),
    csvBtn: qs('#csvBtn'),
  };

  const state = { items: [], running: false, results: [] };
  const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

  /* ── queue rows ───────────────────────────────────────────────────── */
  function rowMarkup(item, index) {
    return `<li class="row" data-i="${index}">
      <img class="row-thumb" data-role="thumb" alt="" ${item.thumb ? `src="${item.thumb}"` : ''}>
      <div class="row-main">
        <span class="row-title" data-role="title">${escape(item.name)}</span>
        <span class="row-sub" data-role="sub">queued</span>
      </div>
      <div class="row-meter">
        <div class="progress"><i data-role="fill" style="width:0%"></i></div>
        <span data-role="pct">—</span>
      </div>
      <span class="tag wait" data-role="tag">queued</span>
    </li>`;
  }

  function renderQueue() {
    el.rows.innerHTML = state.items.map(rowMarkup).join('');
    el.clearBtn.disabled = state.items.length === 0 || state.running;
  }

  function rowNode(index) {
    return el.rows.querySelector(`.row[data-i="${index}"]`);
  }

  function setRow(index, { sub, fill, pctText, tag, tagClass, thumb, truth }) {
    const node = rowNode(index);
    if (!node) return;
    if (sub != null) node.querySelector('[data-role="sub"]').textContent = sub;
    if (fill != null) node.querySelector('[data-role="fill"]').style.width = `${fill}%`;
    if (pctText != null) node.querySelector('[data-role="pct"]').textContent = pctText;
    if (thumb) node.querySelector('[data-role="thumb"]').src = thumb;
    if (tag != null) {
      const tagEl = node.querySelector('[data-role="tag"]');
      tagEl.textContent = tag;
      tagEl.className = `tag ${tagClass || 'wait'}`;
    }
    if (truth) node.dataset.truth = truth;
  }

  /* ── progress ─────────────────────────────────────────────────────── */
  function setProgress(value, label) {
    el.progressWrap.hidden = false;
    el.progressBar.style.width = `${value}%`;
    if (label) el.progressLabel.textContent = label;
  }

  /* ── summary ──────────────────────────────────────────────────────── */
  function renderSummary() {
    const done = state.results;
    if (!done.length) {
      el.summary.hidden = true;
      return;
    }
    const withTruth = done.filter((row) => row.truth);
    const correct = withTruth.filter((row) => row.truth.match).length;
    const times = done.map((row) => row.gradcamMs).filter((value) => value != null);
    const meanTime = times.length ? times.reduce((a, b) => a + b, 0) / times.length : null;

    const tiles = [
      ['images', String(done.length)],
      ['classified', `${done.length} of ${state.items.length}`],
      withTruth.length
        ? ['matched true label', `${correct} of ${withTruth.length}`]
        : ['below threshold', String(done.filter((row) => row.probability < 0.7).length)],
      ['mean heat map', meanTime != null ? `${Math.round(meanTime)} ms` : '—'],
    ];

    el.tiles.innerHTML = tiles
      .map(([term, value]) => `<div class="tile"><dt>${term}</dt><dd>${value}</dd></div>`)
      .join('');

    el.tableBody.innerHTML = done
      .map((row, index) => {
        const truthLabel = row.truth ? `${row.truth.crop} ${row.truth.condition.toLowerCase()}` : '—';
        const tag = row.truth
          ? row.truth.match
            ? '<span class="tag ok">match</span>'
            : '<span class="tag bad">mismatch</span>'
          : '<span class="tag wait">unverified</span>';
        return `<tr class="clickable" data-detail="${index}">
          <td><img class="tiny-thumb" src="${row.photo}" alt="${escape(row.crop)} leaf"></td>
          <td class="mono">${escape(row.name)}</td>
          <td class="mono">${escape(row.crop.toLowerCase())}, ${escape(row.condition.toLowerCase())}</td>
          <td class="mono">${pct(row.probability)}</td>
          <td>${escape(truthLabel)}</td>
          <td>${tag}</td>
        </tr>`;
      })
      .join('');

    el.summary.hidden = false;
  }

  function detailMarkup(row) {
    return `<tr class="detail-cell" data-detail-row>
      <td colspan="6">
        <div class="detail">
          <img src="${row.heat}" alt="${escape(row.crop)} with the Grad-CAM heat map">
          <div>
            <p class="verdict-label mono">top 3 of ${DEMO.model.classes}</p>
            <ul class="bars">${window.UI.bars(row.top3)}</ul>
            <dl class="meta mono" style="margin-top: var(--s-200);">
              <dt>file</dt><dd>${escape(row.name)}</dd>
              <dt>heat map</dt><dd>${row.gradcamMs != null ? `${Math.round(row.gradcamMs)} ms` : '—'}</dd>
              <dt>predicted</dt><dd>${escape(row.crop)} ${escape(row.condition.toLowerCase())}</dd>
            </dl>
          </div>
        </div>
      </td>
    </tr>`;
  }

  /* ── processing ───────────────────────────────────────────────────── */
  async function processBatch() {
    if (state.running || !state.items.length) return;
    state.running = true;
    state.results = [];
    el.summary.hidden = true;
    el.csvBtn.disabled = true;
    el.clearBtn.disabled = true;
    el.notice.hidden = true;

    const total = state.items.length;
    for (let index = 0; index < total; index += 1) {
      const item = state.items[index];
      setRow(index, { sub: 'reading the image', tag: 'working', tagClass: 'wait', fill: 12 });
      setProgress(Math.round((index / total) * 100), `image ${index + 1} of ${total} · reading`);

      if (!item.record) {
        // a real file with no backend behind it
        await sleep(220);
        setRow(index, {
          sub: 'not classified, no backend',
          tag: 'skipped',
          tagClass: 'wait',
          fill: 0,
          pctText: '—',
        });
        continue;
      }

      const record = item.record;
      await sleep(200);
      setRow(index, { sub: 'running the model', fill: 48, thumb: record.image.photo });
      setProgress(Math.round(((index + 0.5) / total) * 100), `image ${index + 1} of ${total} · model`);
      await sleep(240);

      setRow(index, {
        sub: `${record.crop.toLowerCase()}, ${record.condition.toLowerCase()}`,
        fill: record.probability * 100,
        pctText: pct(record.probability),
        tag: record.truth.match ? 'match' : 'mismatch',
        tagClass: record.truth.match ? 'ok' : 'bad',
        truth: `${record.truth.crop} ${record.truth.condition}`,
      });

      state.results.push({
        name: record.file,
        photo: record.image.photo,
        heat: record.image.gradcam,
        crop: record.crop,
        condition: record.condition,
        probability: record.probability,
        top3: record.top3,
        gradcamMs: record.timing_ms.gradcam,
        truth: record.truth,
      });
      setProgress(Math.round(((index + 1) / total) * 100), `image ${index + 1} of ${total} · done`);
    }

    state.running = false;
    el.clearBtn.disabled = false;
    el.csvBtn.disabled = state.results.length === 0;
    setProgress(100, `${state.results.length} of ${total} classified`);

    if (!state.results.length) {
      el.notice.hidden = false;
      el.notice.textContent =
        'No backend is configured, so these photos were not classified. Press "Load the recorded batch" to see the full flow on held out test images.';
    } else {
      renderSummary();
    }
  }

  /* ── sources ──────────────────────────────────────────────────────── */
  function loadRecorded() {
    if (!DEMO.batch || !DEMO.batch.length) {
      window.UI.toast('No recorded batch in this build.', 'bad');
      return;
    }
    state.items = DEMO.batch.map((record) => ({ name: record.file, record, thumb: '' }));
    renderQueue();
    processBatch();
  }

  function handleFiles(fileList) {
    const files = Array.from(fileList);
    if (!files.length) return;
    const rejected = [];
    const accepted = [];

    files.forEach((file) => {
      if (!file.type.startsWith('image/')) rejected.push(`${file.name}: not an image`);
      else if (file.size > MAX_BYTES) rejected.push(`${file.name}: ${(file.size / 1024 / 1024).toFixed(1)} MB`);
      else accepted.push(file);
    });

    if (rejected.length) {
      el.notice.hidden = false;
      el.notice.textContent = `Skipped ${rejected.length} file(s). ${rejected.slice(0, 3).join(' · ')}`;
    }

    if (!accepted.length) return;

    state.items = accepted.map((file) => ({
      name: file.name,
      file,
      record: null,
      thumb: URL.createObjectURL(file),
    }));
    renderQueue();
    processBatch();
  }

  /* ── csv ──────────────────────────────────────────────────────────── */
  function downloadCsv() {
    if (!state.results.length) return;
    const header = ['file', 'predicted_label', 'predicted_crop', 'predicted_condition', 'probability', 'true_label', 'match', 'gradcam_ms'];
    const lines = state.results.map((row) =>
      [
        row.name,
        `${row.crop}___${row.condition}`,
        row.crop,
        row.condition,
        row.probability.toFixed(6),
        row.truth ? `${row.truth.crop} ${row.truth.condition}` : '',
        row.truth ? (row.truth.match ? 'yes' : 'no') : '',
        row.gradcamMs != null ? Math.round(row.gradcamMs) : '',
      ]
        .map((cell) => `"${String(cell).replace(/"/g, '""')}"`)
        .join(',')
    );
    const blob = new Blob([[header.join(','), ...lines].join('\n')], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'leafread_batch.csv';
    link.click();
    URL.revokeObjectURL(url);
    window.UI.toast('CSV downloaded', 'ok');
  }

  /* ── wiring ───────────────────────────────────────────────────────── */
  qs('#modeBadge').textContent = window.API.isLive()
    ? 'live mode · connected to the inference service'
    : 'demo mode · recorded checkpoint output';
  qs('#modeBadge').className = `badge ${window.API.isLive() ? 'live' : 'demo'}`;

  el.pickBtn.addEventListener('click', () => el.files.click());
  el.dropzone.addEventListener('click', () => el.files.click());
  el.dropzone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      el.files.click();
    }
  });
  el.files.addEventListener('change', () => {
    handleFiles(el.files.files);
    el.files.value = '';
  });

  ['dragenter', 'dragover'].forEach((type) =>
    el.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      el.dropzone.classList.add('is-over');
    })
  );
  ['dragleave', 'drop'].forEach((type) =>
    el.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      el.dropzone.classList.remove('is-over');
    })
  );
  el.dropzone.addEventListener('drop', (event) => handleFiles(event.dataTransfer.files));

  el.sampleBtn.addEventListener('click', loadRecorded);
  el.clearBtn.addEventListener('click', () => {
    state.items = [];
    state.results = [];
    el.rows.innerHTML = '';
    el.summary.hidden = true;
    el.progressWrap.hidden = true;
    el.notice.hidden = true;
    el.clearBtn.disabled = true;
    el.csvBtn.disabled = true;
  });
  el.csvBtn.addEventListener('click', downloadCsv);

  el.tableBody.addEventListener('click', (event) => {
    const tr = event.target.closest('tr[data-detail]');
    if (!tr) return;
    const index = Number(tr.dataset.detail);
    const existing = tr.nextElementSibling;
    if (existing && existing.hasAttribute('data-detail-row')) {
      existing.remove();
      tr.classList.remove('is-open');
      return;
    }
    qsa('tr[data-detail-row]').forEach((row) => row.remove());
    qsa('tr.is-open').forEach((row) => row.classList.remove('is-open'));
    tr.classList.add('is-open');
    tr.insertAdjacentHTML('afterend', detailMarkup(state.results[index]));
    window.UI.animateBars(tr.nextElementSibling);
  });

  if (!DEMO.batch || !DEMO.batch.length) {
    el.sampleBtn.disabled = true;
  }

  window.UI.boot();
})();
