// The site's own Bleed axis fonts, which glyphstack.fontbuild --site writes into public/fonts, every caller keeping its normal behaviour if they fail to load

export type Face = 'regular' | 'italic';

export interface Preview {
  family: Partial<Record<Face, string>>;
  chars: Partial<Record<Face, Set<number>>>;
  urls: Partial<Record<Face, string>>;
  bleed: [number, number, number];
  weight?: [number, number, number];
}

const FAMILY: Record<Face, string> = { regular: 'Nomicon Rules Regular', italic: 'Nomicon Rules Italic' };
let pending: Promise<Preview | null> | null = null;

/** Loads the fonts listed in nomicon-rules.json under base, the site's /fonts/ path, null when they are missing */
export function previewFonts(base: string) {
  pending ??= (async () => {
    try {
      const r = await fetch(`${base}nomicon-rules.json`);
      if (!r.ok || !(r.headers.get('content-type') ?? '').includes('json')) return null;
      const info = await r.json();
      const out: Preview = { family: {}, chars: {}, urls: {}, bleed: info.axes?.BLED ?? [75, 75, 75], weight: info.axes?.wght };
      for (const [face, file] of Object.entries(info.faces) as [Face, string][]) {
        const url = `${base}${file}`;
        const f = new FontFace(FAMILY[face], `url("${url}")`);
        await f.load();
        document.fonts.add(f);
        out.family[face] = FAMILY[face];
        out.chars[face] = new Set(info.chars[face]);
        out.urls[face] = url;
      }
      document.documentElement.style.setProperty('--bled', String(out.bleed[1]));
      return out;
    } catch (err) {
      console.warn('preview fonts', err);
      return null;
    }
  })();
  return pending;
}

/** Whether a face of the preview fonts draws a character */
export const draws = (p: Preview, face: Face, ch: string) => !!p.chars[face]?.has(ch.codePointAt(0)!);

/** A Bleed slider that sets --bled on the page, which every preview glyph reads through font-variation-settings */
export function bleedSlider(p: Preview, id: string, onInput?: (v: number) => void) {
  const [lo, def, hi] = p.bleed;
  const label = document.createElement('label');
  label.className = 'nrow bleed-row';
  label.htmlFor = id;
  label.innerHTML = `Bleed <input id="${id}" type="range" min="${lo}" max="${hi}" step="1" value="${def}"> <output class="mono">${def}</output>`;
  const input = label.querySelector('input')!, out = label.querySelector('output')!;
  input.addEventListener('input', () => {
    out.textContent = input.value;
    document.documentElement.style.setProperty('--bled', input.value);
    onInput?.(+input.value);
  });
  return label;
}

/** A Weight slider that sets --wght on the page, for fonts built with a Weight axis */
export function weightSlider(range: [number, number, number], id: string, onInput?: (v: number) => void) {
  const [lo, def, hi] = range;
  const label = document.createElement('label');
  label.className = 'nrow weight-row';
  label.htmlFor = id;
  label.innerHTML = `Weight <input id="${id}" type="range" min="${lo}" max="${hi}" step="1" value="${def}"> <output class="mono">${def}</output>`;
  const input = label.querySelector('input')!, out = label.querySelector('output')!;
  input.addEventListener('input', () => {
    out.textContent = input.value;
    document.documentElement.style.setProperty('--wght', input.value);
    onInput?.(+input.value);
  });
  return label;
}

/** The current Weight value */
export const weight = () => +(getComputedStyle(document.documentElement).getPropertyValue('--wght') || 400);

/** The current Bleed value */
export const bleed = () => +(getComputedStyle(document.documentElement).getPropertyValue('--bled') || 75);

const fontData = new Map<string, Promise<string>>();

/** A face's font as a data URL, for drawing set text into an image */
export function fontDataUrl(url: string) {
  if (!fontData.has(url)) {
    fontData.set(url, fetch(url).then((r) => r.blob()).then((b) => new Promise<string>((ok) => {
      const fr = new FileReader();
      fr.onload = () => ok(String(fr.result));
      fr.readAsDataURL(b);
    })));
  }
  return fontData.get(url)!;
}

export const familyOf = (face: Face) => FAMILY[face];
