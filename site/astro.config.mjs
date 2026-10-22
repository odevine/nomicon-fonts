import { defineConfig } from 'astro/config';
import servedData from './integrations/served-data.mjs';

// SITE_URL and SITE_BASE are set by the Pages workflow, local builds serve from the root
export default defineConfig({
  site: process.env.SITE_URL || 'http://localhost:4321',
  base: process.env.SITE_BASE || '/',
  integrations: [servedData()],
  trailingSlash: 'always',
  build: { format: 'directory' },
});
