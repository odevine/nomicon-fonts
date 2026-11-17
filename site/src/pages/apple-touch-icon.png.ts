import sharp from 'sharp';
import { iconSvg } from '../lib/favicon';

// iOS asks for this name at the site root, so the pages don't link it, which keeps tab strips that pick the largest icon on the SVG
export async function GET() {
  const png = await sharp(Buffer.from(iconSvg({ tile: true })), { density: 300 }).resize(180, 180).png().toBuffer();
  return new Response(new Uint8Array(png), { headers: { 'Content-Type': 'image/png' } });
}
