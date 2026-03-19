playwright-cli setup (with CI)

Goal: Set up Playwright to test the app at:
https://erickwendel.github.io/vanilla-js-web-app-example/

What to include
- Install only `@playwright/cli`
- Create a `tests/` directory with a first CLI-driven test runner
- CI: GitHub Actions workflow that installs and runs only Chromium

Local setup
1) Install dev dependency
	- npm i -D @playwright/cli

2) Install browser
	- npx playwright-cli install-browser --browser=chrome

GitHub Actions
- Create .github/workflows/playwright.yml with a job that:
  - Checks out the repo
  - Sets up Node.js
  - Runs npm ci
  - Runs npm run install:browser
  - Runs npm test
  - Uploads `.playwright-cli/` as an artifact on failure
