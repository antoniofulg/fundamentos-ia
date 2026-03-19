const { spawnSync } = require("node:child_process");

const baseUrl = "https://erickwendel.github.io/vanilla-js-web-app-example/";

function runCli(args, { ignoreError = false } = {}) {
  const result = spawnSync("npx", ["playwright-cli", ...args], {
    cwd: __dirname + "/..",
    encoding: "utf8"
  });

  const stdout = result.stdout || "";

  if (stdout) process.stdout.write(stdout);
  if (result.stderr) process.stderr.write(result.stderr);

  if ((result.status !== 0 || stdout.includes("### Error")) && !ignoreError) {
    throw new Error(`playwright-cli failed: npx playwright-cli ${args.join(" ")}`);
  }

  return stdout;
}

function sessionArg(name) {
  return `-s=${name}`;
}

function runScenario(name, code) {
  const sessionName = `${name}-${Date.now()}`;

  runCli([sessionArg(sessionName), "open", baseUrl]);

  try {
    runCli([sessionArg(sessionName), "run-code", code]);
    console.log(`PASS ${name}`);
  } finally {
    runCli([sessionArg(sessionName), "close"], { ignoreError: true });
  }
}

function main() {
  runCli(["close-all"], { ignoreError: true });

  const generatedTitle = `Generated Card ${Date.now()}`;
  const submitScenario = `async page => {
    await page.waitForLoadState('domcontentloaded');
    const cards = page.locator('#card-list article');
    const initialCount = await cards.count();

    await page.getByRole('textbox', { name: 'Image Title' }).fill(${JSON.stringify(generatedTitle)});
    await page.getByRole('textbox', { name: 'Image URL' }).fill('https://picsum.photos/300/300');
    await page.getByRole('button', { name: 'Submit Form' }).click();
    await page.waitForFunction(
      expectedCount => document.querySelectorAll('#card-list article').length === expectedCount,
      initialCount + 1
    );

    const finalCount = await cards.count();
    const lastTitle = (await page.locator('#card-list .card-title').last().textContent())?.trim();

    if (finalCount !== initialCount + 1) {
      throw new Error(\`Expected card count to increase from \${initialCount} to \${initialCount + 1}, got \${finalCount}\`);
    }

    if (lastTitle !== ${JSON.stringify(generatedTitle)}) {
      throw new Error(\`Expected last title to be ${generatedTitle}, got \${lastTitle}\`);
    }

    return { initialCount, finalCount, lastTitle };
  }`;

  const validationScenario = `async page => {
    await page.waitForLoadState('domcontentloaded');
    const cards = page.locator('#card-list article');
    const initialCount = await cards.count();

    await page.getByRole('textbox', { name: 'Image Title' }).fill('Invalid URL example');
    await page.getByRole('textbox', { name: 'Image URL' }).fill('not-a-url');
    await page.getByRole('button', { name: 'Submit Form' }).click();

    const finalCount = await cards.count();
    const validationMessage = await page.getByRole('textbox', { name: 'Image URL' }).evaluate(el => el.validationMessage);

    if (validationMessage !== 'Please enter a URL.') {
      throw new Error(\`Expected validation message "Please enter a URL.", got "\${validationMessage}"\`);
    }

    if (finalCount !== initialCount) {
      throw new Error(\`Expected card count to stay at \${initialCount}, got \${finalCount}\`);
    }

    return { initialCount, finalCount, validationMessage };
  }`;

  runScenario("submit-flow", submitScenario);
  runScenario("invalid-url-validation", validationScenario);
}

main();
