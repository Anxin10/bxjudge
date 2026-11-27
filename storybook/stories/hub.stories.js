import withRun from '../generated/hub.html?raw';
import noRun from '../generated/hub-no-run.html?raw';
import { page } from '../harness.js';

export default {
  title: '工作台/總覽',
  parameters: {
    docs: { description: { component: '首頁：由 `workbench_theme.render_hub` 產生的靜態頁。合成快照。' } },
  },
};

export const 有判定_run = { name: '有判定 run', render: () => page(withRun) };
export const 尚無判定_run = { name: '尚無判定 run', render: () => page(noRun) };
