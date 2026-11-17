import { iconSvg } from '../lib/favicon';

export const GET = () => new Response(iconSvg(), { headers: { 'Content-Type': 'image/svg+xml' } });
