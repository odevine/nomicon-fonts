import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: 'tests',
  webServer: { command: 'npm run build && npm run preview', url: 'http://localhost:4321/', reuseExistingServer: false, timeout: 120_000 },
  use: { baseURL: 'http://localhost:4321' },
  projects: [{ name: 'desktop', use: { viewport: { width: 1280, height: 900 } } }, { name: 'phone', use: { viewport: { width: 390, height: 800 } } }],
});
