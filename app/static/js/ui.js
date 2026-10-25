/* Shared frontend behaviour for every Leafread page.
   No dependencies, no build step. Each page calls the pieces it needs. */

(() => {
  const UI = {};

  UI.qs = (sel, root = document) => root.querySelector(sel);
  UI.qsa = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  UI.pct = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;

  UI.escape = (text) =>
    String(text).replace(/[&<>"']/g, (ch) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[ch]));

  /* 'Tomato___Late_blight' -> { crop: 'Tomato', condition: 'Late blight' } */
  UI.split = (label) => {
    const [crop, , rest = ''] = String(label).split('___');
    const condition = rest.replace(/_/g, ' ').trim();
    return {
      crop: crop.replace(/_/g, ' ').trim(),
      condition: condition ? condition[0].toUpperCase() + condition.slice(1) : '',
    };
  };

  /* ── toast ────────────────────────────────────────────────────────── */
  UI.toast = (message, tone = 'neutral') => {
    let host = UI.qs('#toast');
    if (!host) {
      host = document.createElement('div');
      host.id = 'toast';
      host.className = 'toast mono';
      host.setAttribute('role', 'status');
      document.body.appendChild(host);
    }
    host.textContent = message;
    host.dataset.tone = tone;
    host.classList.add('on');
    window.clearTimeout(UI._toastTimer);
    UI._toastTimer = window.setTimeout(() => host.classList.remove('on'), 3800);
  };

  /* ── nav + mobile menu ────────────────────────────────────────────── */
  UI.initNav = () => {
    const button = UI.qs('#menuBtn');
    const menu = UI.qs('#menu');
    if (!button || !menu) return;

    const close = () => {
      button.setAttribute('aria-expanded', 'false');
      menu.classList.remove('is-open');
      menu.hidden = true;
    };
    const open = () => {
      button.setAttribute('aria-expanded', 'true');
      menu.hidden = false;
      requestAnimationFrame(() => menu.classList.add('is-open'));
    };

    button.addEventListener('click', () => {
      if (button.getAttribute('aria-expanded') === 'true') close();
      else open();
    });
    menu.addEventListener('click', (event) => {
      if (event.target.closest('a') || event.target === menu) close();
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') close();
    });
  };

  /* ── scroll reveals ───────────────────────────────────────────────── */
  UI.initReveals = () => {
    const targets = UI.qsa('.reveal');
    if (!targets.length) return;
    if (!('IntersectionObserver' in window)) {
      targets.forEach((el) => el.classList.add('in'));
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('in');
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.15, rootMargin: '0px 0px -80px 0px' }
    );
    targets.forEach((el) => observer.observe(el));
  };

  /* ── per word tagline reveal ──────────────────────────────────────── */
  UI.initTagline = () => {
    const host = UI.qs('[data-words]');
    if (!host) return;
    const words = host.textContent.trim().split(/\s+/);
    host.textContent = '';
    words.forEach((word) => {
      const span = document.createElement('span');
      span.className = 'word';
      span.textContent = word;
      host.append(span, document.createTextNode(' '));
    });

    if (!('IntersectionObserver' in window)) {
      UI.qsa('.word', host).forEach((word) => word.classList.add('on'));
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('on');
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0, rootMargin: '-30% 0px -35% 0px' }
    );
    UI.qsa('.word', host).forEach((word) => observer.observe(word));
  };

  /* ── model card from the generated payload ────────────────────────── */
  UI.renderModelCard = (target) => {
    const dl = typeof target === 'string' ? UI.qs(target) : target;
    if (!dl || !window.DEMO) return;
    const m = window.DEMO.model;
    const rows = [
      ['architecture', m.architecture],
      ['parameters', m.parameters.toLocaleString('en-US')],
      ['classes', String(m.classes)],
      ['input', `${m.image_size} by ${m.image_size} px`],
      ['validation macro F1', m.val_macro_f1.toFixed(4)],
      ['best epoch', String(m.epoch)],
      ['confidence threshold', m.confidence_threshold.toFixed(2)],
      ['checkpoint', m.checkpoint],
      ['assets generated', window.DEMO.generated],
    ];
    dl.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  };

  /* ── chip groups ──────────────────────────────────────────────────── */
  UI.chipGroup = (selector, onChange) => {
    UI.qsa(selector).forEach((button) => {
      button.addEventListener('click', () => {
        UI.qsa(selector).forEach((other) => {
          const on = other === button;
          other.classList.toggle('is-on', on);
          other.setAttribute('aria-pressed', String(on));
        });
        onChange(button, selector);
      });
    });
  };

  /* ── top three bars, shared by both pages ─────────────────────────── */
  UI.bars = (top3) =>
    top3
      .map((row, index) => {
        const width = Math.max(row.probability * 100, 0.6).toFixed(1);
        const name = `${row.crop.toLowerCase()}, ${row.condition.toLowerCase()}`;
        return `<li class="bar">
          <span class="bar-rank mono">${index + 1}</span>
          <span class="bar-track"><span class="bar-fill" data-w="${width}"></span></span>
          <span class="bar-pct mono">${UI.pct(row.probability)}</span>
          <span class="bar-name mono">${UI.escape(name)}</span>
        </li>`;
      })
      .join('');

  UI.animateBars = (root) => {
    requestAnimationFrame(() => {
      UI.qsa('.bar-fill', root).forEach((fill) => {
        fill.style.width = `${fill.dataset.w}%`;
      });
    });
  };

  /* ── one boot call per page ───────────────────────────────────────── */
  UI.boot = () => {
    UI.initNav();
    UI.initReveals();
    UI.initTagline();
    UI.renderModelCard('#modelcard');
  };

  window.UI = UI;
})();
