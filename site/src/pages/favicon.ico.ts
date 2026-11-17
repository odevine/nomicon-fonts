import sharp from 'sharp';
import { iconSvg } from '../lib/favicon';

// an ICO holding 32 and 16 px PNGs of the tiled n, for browsers that skip the SVG icon and ask for /favicon.ico
export async function GET() {
  const sizes = [32, 16];
  const pngs = await Promise.all(sizes.map((s) => sharp(Buffer.from(iconSvg({ tile: true })), { density: 300 }).resize(s, s).png().toBuffer()));
  const head = Buffer.alloc(6 + 16 * sizes.length);
  head.writeUInt16LE(1, 2);
  head.writeUInt16LE(sizes.length, 4);
  let at = head.length;
  sizes.forEach((s, i) => {
    const e = 6 + 16 * i;
    head.writeUInt8(s, e);
    head.writeUInt8(s, e + 1);
    head.writeUInt16LE(1, e + 4);
    head.writeUInt16LE(32, e + 6);
    head.writeUInt32LE(pngs[i].length, e + 8);
    head.writeUInt32LE(at, e + 12);
    at += pngs[i].length;
  });
  return new Response(new Uint8Array(Buffer.concat([head, ...pngs])), { headers: { 'Content-Type': 'image/x-icon' } });
}
