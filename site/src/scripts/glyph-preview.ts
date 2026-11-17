import { bleedSlider, draws, previewFonts, type Face } from './preview-font';

// draws each glyph built from the measuring stick in the site's own font, at the scale of the scanned tiles
class GlyphPreview extends HTMLElement {
  async connectedCallback() {
    const p = await previewFonts(this.dataset.base!);
    if (!p) return;
    const view = this.dataset.view!;
    let shown = 0;
    document.querySelectorAll<HTMLLIElement>('.glyphmap li.built[data-char]').forEach((li) => {
      const face = li.dataset.face as Face, ch = li.dataset.char!;
      const family = p.family[face];
      const slot = li.querySelector('.slot');
      if (!family || !slot || !draws(p, face, ch)) return;
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('viewBox', view);
      svg.setAttribute('class', 'preview-glyph');
      svg.setAttribute('aria-hidden', 'true');
      const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      t.setAttribute('text-anchor', 'middle');
      t.setAttribute('font-family', `"${family}"`);
      t.setAttribute('font-size', '1000');
      t.textContent = ch;
      svg.append(t);
      slot.replaceWith(svg);
      shown++;
    });
    const note = document.createElement('div');
    note.className = 'preview-note';
    note.innerHTML = `<p>The ${shown} dimmed glyphs are drawn from Nomicon Rules, the font this site is set in. Each is the measuring stick's outline with the print look applied, and Bleed sets how strong that look is.</p>`;
    note.append(bleedSlider(p, 'gp-bled'));
    this.append(note);
  }
}

customElements.define('glyph-preview', GlyphPreview);
