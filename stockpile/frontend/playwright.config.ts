import { defineConfig, devices } from "@playwright/test";

const trainerCommand = process.platform === "win32"
  ? ".venv\\Scripts\\python.exe -m stockpile play"
  // Replace Playwright's POSIX shell so SIGTERM reaches the supervisor itself.
  : "exec .venv/bin/python -m stockpile play";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  timeout: 120_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
  ],
  webServer: {
    command: trainerCommand,
    cwd: "..",
    url: "http://127.0.0.1:5173/api/v2/setup",
    reuseExistingServer: true,
    timeout: 40_000,
    gracefulShutdown: { signal: "SIGTERM", timeout: 5_000 },
  },
});
