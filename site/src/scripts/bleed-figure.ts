import { bleedSlider, previewFonts } from './preview-font';

// adds the Bleed slider under the large figure once the site's font has loaded, the figure itself is drawn by CSS
class BleedFigure extends HTMLElement {
  async connectedCallback() {
    const p = await previewFonts(this.dataset.base!);
    if (p) this.append(bleedSlider(p, 'bf-bled'));
  }
}

customElements.define('bleed-figure', BleedFigure);
