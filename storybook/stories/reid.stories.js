import shell from '../generated/reid.html?raw';
import script from '../../src/bxjudge/web/reid_review.js?raw';
import { page } from '../harness.js';
import fixture from '../fixtures/reid_manifest.json';

export default {
  title: '工作台/斷點核對',
  parameters: {
    docs: { description: { component: '`/reid-review/`：前後片段是否同一顆。草稿存在瀏覽器，按「保存到專案」才寫入。合成資料。' } },
  },
};

const KEY = 'bxjudge.reid-review.' + fixture.sha;
const ids = fixture.manifest.cases.map((c) => c.case_id);
const blank = (id) => ({ case_id: id, reviewed: false, relation: 'unreviewed', scope: 'unreviewed', side_before: 'unreviewed',
  side_after: 'unreviewed', visibility_before: 'unreviewed', visibility_after: 'unreviewed', collision: 'unreviewed', note: '' });
const reviewedCases = ids.map((id, i) => ({ ...blank(id), reviewed: i === 0, relation: i === 0 ? 'same' : 'unreviewed',
  scope: i === 0 ? 'same_round' : 'unreviewed' }));
const savedSha = 'a'.repeat(64);

function storage(value) {
  return () => {
    try {
      if (value === null) localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, value);
    } catch { /* storage blocked: the page shows its own warning */ }
  };
}

function routes({ snapshots = [], saveFails = false } = {}) {
  return (path, init) => {
    if (path === 'saved') return { body: { snapshots } };
    if (path === 'save' && init.method === 'POST') {
      if (saveFails) return { status: 500, body: { error: '模擬：專案資料夾無法寫入' } };
      const payload = JSON.parse(init.body);
      return { body: { saved: true, sha256: savedSha, filename: 'reid-labels-synthetic.json',
        reviewed_cases: payload.cases.filter((c) => c.reviewed).length } };
    }
    return undefined;
  };
}

const snapshot = { sha256: savedSha, reviewed_cases: 1,
  payload: { submitted_at: '2026-11-26T09:30:00.000Z', cases: reviewedCases } };

export const 空白草稿 = { render: () => page(shell, { script, routes: routes(), before: storage(null) }) };
export const 有已存版本 = { render: () => page(shell, { script, routes: routes({ snapshots: [snapshot] }), before: storage(null) }) };
export const 瀏覽器草稿進行中 = {
  render: () => page(shell, { script, routes: routes(),
    before: storage(JSON.stringify({ manifest_sha256: fixture.sha, parent: null, cases: reviewedCases })) }),
};
export const 草稿無法驗證 = {
  render: () => page(shell, { script, routes: routes({ snapshots: [snapshot] }),
    before: storage(JSON.stringify({ manifest_sha256: fixture.sha, parent: null, cases: [{ case_id: 'wrong' }] })) }),
};
export const 保存失敗 = {
  name: '保存失敗（先勾選一組再按保存）',
  render: () => page(shell, { script, routes: routes({ saveFails: true }),
    before: storage(JSON.stringify({ manifest_sha256: fixture.sha, parent: null, cases: reviewedCases })) }),
};
