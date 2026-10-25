/* Inference client.
   The pages call API.predict(file). With no backend configured the call fails
   fast with code 'demo' and every page degrades to the recorded checkpoint
   output instead of inventing a number. Set window.LEAFREAD_API before this
   file loads (see app/static/js/config.example.js) to talk to the FastAPI
   service described in docs/FRONTEND_PLAN.md and nothing else changes. */

(() => {
  const API = {};

  API.base = (window.LEAFREAD_API || '').replace(/\/$/, '') || null;
  API.timeoutMs = 12000;

  API.isLive = () => Boolean(API.base);

  API.predict = async (file) => {
    if (!API.isLive()) {
      throw Object.assign(new Error('No backend is configured, so the photo was not classified.'), {
        code: 'demo',
      });
    }

    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), API.timeoutMs);
    const body = new FormData();
    body.append('file', file, file.name || 'leaf.jpg');

    try {
      const response = await fetch(`${API.base}/api/predict`, {
        method: 'POST',
        body,
        signal: controller.signal,
      });
      if (!response.ok) {
        const detail = await response.text().catch(() => '');
        throw Object.assign(new Error(detail || `Server returned ${response.status}.`), {
          code: 'http',
        });
      }
      return await response.json();
    } catch (error) {
      if (error.name === 'AbortError') {
        throw Object.assign(new Error('The model did not answer in time. Try a smaller photo.'), {
          code: 'timeout',
        });
      }
      throw error;
    } finally {
      window.clearTimeout(timer);
    }
  };

  API.model = async () => {
    if (!API.isLive()) return null;
    try {
      const response = await fetch(`${API.base}/api/model`);
      return response.ok ? await response.json() : null;
    } catch {
      return null;
    }
  };

  window.API = API;
})();
