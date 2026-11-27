import shell from '../generated/judge.html?raw';
import script from '../../src/bxjudge/web/judge_view.js?raw';
import { page } from '../harness.js';
import { rounds, index } from '../fixtures/judge.js';

export default {
  title: '工作台/回合裁判',
  parameters: {
    docs: { description: { component: '`/judge/`：原影片上的疊加框、逐項檢查、判定與人工覆核原因。合成資料，沒有影片。' } },
  },
};

function routes(idx, failRound = false) {
  return (path) => {
    if (path === 'index.json') return { body: idx };
    const m = path.match(/^data\/(.+)\.json$/);
    if (m && !failRound && rounds[m[1]]) return { body: rounds[m[1]] };
    return undefined;
  };
}

export const 待覆核 = { render: () => page(shell, { script, routes: routes(index()), hash: 'synthetic_match_a_r2' }) };
export const 已判定 = { render: () => page(shell, { script, routes: routes(index()), hash: 'synthetic_match_a_r1' }) };
export const 已人工覆核 = { render: () => page(shell, { script, routes: routes(index()), hash: 'synthetic_match_b_r2' }) };
export const 場地幾何不可用 = { render: () => page(shell, { script, routes: routes(index()), hash: 'synthetic_match_b_r1' }) };
export const 沒有回合 = { render: () => page(shell, { script, routes: routes(index([])) }) };
export const 讀不到資料 = { render: () => page(shell, { script, routes: () => undefined }) };
