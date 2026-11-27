// Runs a workbench page script against a page shell inside a story.
// The page scripts run their setup at import time, so each story loads a fresh copy
// (blob URL) and removes the window/document listeners the previous story added.

const tracked = [];
for (const target of [window, document]) {
  const add = target.addEventListener.bind(target);
  target.addEventListener = (type, listener, options) => {
    tracked.push([target, type, listener, options]);
    return add(type, listener, options);
  };
}
const realFetch = window.fetch.bind(window);

function reset() {
  for (const [target, type, listener, options] of tracked.splice(0)) {
    target.removeEventListener(type, listener, options);
  }
  window.fetch = realFetch;
  history.replaceState(null, '', location.pathname + location.search);
}

function respond(result) {
  const { status = 200, body = {} } = result;
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

/**
 * routes(path, init) → {status?, body} | Promise<…> | undefined (404).
 * `path` is the request path relative to the page, e.g. 'index.json', 'data/<key>.json', 'save'.
 */
function mockFetch(routes, log) {
  window.fetch = async (input, init = {}) => {
    const url = new URL(typeof input === 'string' ? input : input.url, document.baseURI);
    const base = new URL('.', document.baseURI).pathname;
    const path = decodeURIComponent(url.pathname.startsWith(base) ? url.pathname.slice(base.length) : url.pathname);
    log?.push({ method: init.method || 'GET', path, body: init.body ? JSON.parse(init.body) : undefined });
    const result = await routes(path, init);
    if (result === undefined) return respond({ status: 404, body: { error: `${path} not mocked` } });
    return respond(result);
  };
}

/**
 * @param {string} shell      page body HTML from storybook/generated/
 * @param {object} opts
 * @param {string} [opts.script]   page script source (imported with ?raw)
 * @param {Function} [opts.routes] fetch mock, see mockFetch
 * @param {string} [opts.hash]     location hash to start from
 * @param {Function} [opts.before] runs after the shell is in the document, before the script
 */
export function page(shell, opts = {}) {
  reset();
  const root = document.createElement('div');
  root.className = 'bx-story';
  root.innerHTML = shell;
  if (opts.routes) mockFetch(opts.routes, (window.__bxRequests = []));
  if (opts.hash) history.replaceState(null, '', '#' + encodeURIComponent(opts.hash));
  // Run once the story root is attached: the scripts look elements up by id.
  setTimeout(async () => {
    opts.before?.(root);
    if (opts.script) {
      const url = URL.createObjectURL(new Blob([opts.script], { type: 'text/javascript' }));
      try {
        await import(/* @vite-ignore */ url);
      } finally {
        URL.revokeObjectURL(url);
      }
    }
  });
  return root;
}
