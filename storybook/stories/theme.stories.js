import { page } from '../harness.js';

export default { title: '主題/HUD 元件' };

const colours = ['cyan', 'orange', 'purple', 'gold', 'coral', 'grey'];

export const 色彩與標籤 = {
  render: () => page(`<main class="bx-shell"><section class="bx-block"><header><span class="block-label">Tokens · 色彩</span></header>
<div class="cards">${colours.map((c) => `<article class="panel"><h3>--${c}</h3><div class="big c-${c}">${c}</div><span class="tag c-${c}">tag c-${c}</span></article>`).join('')}</div></section></main>`),
};

export const 按鈕與表單 = {
  render: () => page(`<main class="bx-shell"><section class="bx-block"><header><span class="block-label live">Controls · 控制項</span></header>
<div class="panel"><div class="controls-row"><button type="button">一般</button><button type="button" class="primary">主要</button><button type="button" disabled>停用</button>
<select><option>選項</option></select><label><input type="checkbox" checked> 勾選</label></div>
<p class="notice">notice：提醒文字。</p><p class="muted">muted：次要說明。<code>mono</code></p></div></section></main>`),
};

export const 面板與狀態 = {
  render: () => page(`<main class="bx-shell"><section class="bx-block" data-state="review"><header><span class="block-label live">Live · 狀態</span><h2>review / judged</h2></header>
<div class="cards"><article class="panel focus"><h3>focus 面板</h3><ul class="checklist"><li>通過的項目</li><li class="flag">需要覆核 <small class="c-orange">2 項</small></li><li class="idle">未觸發</li></ul></article>
<article class="panel"><h3>清單</h3><ul class="store-list"><li><span>reid-review</span><b>1</b></li><li><span>identity-audit</span><b>2</b></li></ul></article></div></section>
<footer class="bx-status" role="status" data-state="review"><span>回合 <b>R1</b></span><span>信心 <b>0.41 &lt; 0.80</b></span><span>Fallback <b>ON · 3</b></span></footer></main>`),
};
