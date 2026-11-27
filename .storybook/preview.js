// Same assets the workbench serves: the HUD theme, page CSS and the motion layer (sets window.BXFX).
import '../src/bxjudge/web/bx_theme.css';
import '../src/bxjudge/web/reid_review.css';
import '../src/bxjudge/web/bx_fx.js';

document.body.classList.add('bx');
document.documentElement.lang = 'zh-Hant';

const preview = {
  parameters: {
    layout: 'fullscreen',
    backgrounds: { disable: true },
    controls: { disable: true },
  },
};
export default preview;
