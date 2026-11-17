import { defineConfig } from 'astro/config';
import servedData from './integrations/served-data.mjs';

// SITE_URL and SITE_BASE are set by the Pages workflow, local builds serve from the root
const base = process.env.SITE_BASE || '/';

export default defineConfig({
  site: process.env.SITE_URL || 'http://localhost:4321',
  base,
  // old addresses of renamed pages
  redirects: { '/printing-leaves-a-mark/': `${base.replace(/\/?$/, '/')}bleed/` },
  integrations: [servedData()],
  trailingSlash: 'always',
  build: { format: 'directory' },
});
