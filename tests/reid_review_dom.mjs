import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

// Reviewer regression harness. Reads the actual product module; no browser,
// network, media, model, real localStorage, or submission directory is used.
const source = await readFile(new URL('../src/bxjudge/web/reid_review.js', import.meta.url), 'utf8');
const sha = 's'.repeat(64);
const parentSha = 'a'.repeat(64);
const fields = {
  relation: 'same', scope: 'same_round',
  side_before: 'unreviewed', side_after: 'unreviewed',
  visibility_before: 'unreviewed', visibility_after: 'unreviewed',
  collision: 'unreviewed'
};
const originalCase = () => ({ case_id: 'a', reviewed: true, ...fields, note: 'saved note' });
const endpoint = { frame: 1, box: [0, 0, 10, 10], image: 'synthetic.png' };
const manifest = { cases: [{
  case_id: 'a', video: 'synthetic.mp4', source_size: [20, 20],
  before: endpoint, after: endpoint, clip: { path: 'synthetic.mp4' },
  reasons: [], native_ids: [1, 2]
}] };
let importNumber = 0;

async function mount({ badDraft = false } = {}) {
  const initialCases = [originalCase()];
  if (badDraft) initialCases[0].unexpected = 'original raw must survive validation failure';
  const initialRaw = JSON.stringify({ manifest_sha256: sha, parent: null, cases: initialCases });
  let storedRaw = initialRaw;
  let failStorage = false;
  let failPost = false;
  let savedPayload;
  let pendingPost = null, releasePost = null, postCount = 0;
  const elements = new Map();
  const controls = Object.entries({ ...fields, note: '', reviewed: false }).map(([field]) => ({
    dataset: { i: '0', field }, type: field === 'reviewed' ? 'checkbox' : field === 'note' ? 'textarea' : 'select',
    value: '', checked: false, disabled: false
  }));
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      id, textContent: '', innerHTML: '', disabled: false, inert: false, listeners: {},
      addEventListener(event, handler) { this.listeners[event] = handler; },
      querySelectorAll() { return id === 'cases' ? controls : []; },
      setAttribute(name, value) { this[name] = value; },
      removeAttribute(name) { delete this[name]; },
      classList: { add() {}, remove() {} }
    });
    return elements.get(id);
  }
  element('data').textContent = JSON.stringify({ manifest, sha });
  globalThis.document = {
    getElementById: element,
    querySelectorAll: () => [...controls, ...elements.values()],
    createElement: () => ({ click() {}, remove() {} })
  };
  globalThis.localStorage = {
    getItem() { return storedRaw; },
    setItem(_key, value) {
      if (failStorage) throw new Error('SYNTHETIC quota exceeded');
      storedRaw = value;
    }
  };
  globalThis.confirm = () => false;
  globalThis.fetch = async (path, options) => {
    if (path === '/reid-review/save') {
      if (failPost) throw new Error('SYNTHETIC POST failure');
      postCount++;
      savedPayload = JSON.parse(options.body);
      if (pendingPost) await pendingPost;
      return { ok: true, async json() {
        return { saved: true, sha256: parentSha, filename: 'reid-' + parentSha + '.json', reviewed_cases: 1 };
      } };
    }
    assert.equal(path, '/reid-review/saved');
    return { ok: true, async json() { return { snapshots: [] }; } };
  };
  await import('data:text/javascript;base64,' +
    Buffer.from(source + '\n// isolated mount ' + (++importNumber)).toString('base64'));
  await new Promise(resolve => setImmediate(resolve));
  return {
    initialRaw,
    raw: () => storedRaw,
    saved: () => savedPayload,
    postCount: () => postCount,
    pausePost() { pendingPost = new Promise(resolve => { releasePost = resolve; }); },
    releasePost() { releasePost(); },
    element,
    failPersistence() { failStorage = true; failPost = true; },
    async save() {
      assert.equal(element('save').disabled, false, 'save must be available in this valid-draft scenario');
      await element('save').listeners.click();
    },
    edit(field, value, event = 'change') {
      const target = controls.find(c => c.dataset.field === field);
      // Honor a real disabled/inert form: a user cannot edit a locked control.
      if (target.disabled || element('cases').disabled || element('cases').inert) return;
      target.value = value;
      if (target.type === 'checkbox') target.checked = value;
      element('cases').listeners[event]({ target });
    }
  };
}

test('invalid existing draft is never overwritten by a subsequent edit', async () => {
  const ui = await mount({ badDraft: true });
  assert.match(ui.element('error').textContent, /草稿/);
  ui.edit('relation', 'uncertain');
  assert.equal(ui.raw(), ui.initialRaw, 'validation failure must preserve the exact original localStorage raw value');
});

test('note input after a receipt immediately marks newer content unsaved', async () => {
  const ui = await mount();
  await ui.save();
  assert.match(ui.element('status').textContent, /已保存/);
  ui.edit('note', 'new unsaved note', 'input');
  assert.equal(ui.saved().cases[0].note, 'saved note');
  assert.equal(JSON.parse(ui.raw()).cases[0].note, 'new unsaved note');
  assert.match(ui.element('status').textContent, /未存|未保存|尚未保存|草稿已變更/,
    'the actual input event must invalidate the durable-save status without waiting for blur/change');
});

test('localStorage plus POST failure does not promise a durable draft', async () => {
  const ui = await mount();
  ui.failPersistence();
  ui.edit('note', 'memory-only note', 'input');
  await ui.save();
  const message = ui.element('error').textContent + ' ' + ui.element('status').textContent;
  assert.doesNotMatch(message, /草稿保留|草稿仍保留|草稿已保存/,
    'neither persistence path succeeded, so the error must not promise durable preservation');
  assert.match(message, /記憶|勿關|不要關|不可關|尚未持久/,
    'explain memory-only content or explicitly warn against closing/reloading');
});

test('editing while a save is pending never enables parallel save or restore', async () => {
  const ui = await mount();
  ui.pausePost();
  const pending = ui.save();
  assert.equal(ui.postCount(), 1);
  ui.edit('relation', 'uncertain');
  assert.equal(ui.element('save').disabled, true);
  assert.equal(ui.element('restore').disabled, true);
  await ui.element('save').listeners.click();
  assert.equal(ui.postCount(), 1);
  ui.releasePost();
  await pending;
  assert.equal(ui.element('save').disabled, false);
  assert.equal(ui.saved().cases[0].relation, 'same');
  assert.match(ui.element('status').textContent, /較新的未存修改/);
});
