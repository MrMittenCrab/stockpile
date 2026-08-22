import { expect, test } from "@playwright/test";
import type { APIRequestContext, Page, Request, Response } from "@playwright/test";
import type {
  CreateGameResponse,
  GameView,
  LegalAction,
  LiteOptions,
  SupplyBatch,
} from "../src/types";

const defaultOptions: LiteOptions = {
  market_impact: false,
  trading_fees: false,
  dividends: false,
  sell_order: false,
};

const allOptions: LiteOptions = {
  market_impact: true,
  trading_fees: true,
  dividends: true,
  sell_order: true,
};

const patterns = [
  "matrix",
  "ledger",
  "molecular",
  "chevron",
  "crosshatch",
  "wave",
] as const;

const regions = [
  "Status",
  "Market",
  "Research",
  "Stockpiles",
  "Portfolio",
  "Players",
  "Action dock",
] as const;

type SupplyPlan = SupplyBatch["plans"][number];

type Coverage = {
  views: GameView[];
  phases: Set<string>;
  decisions: Set<string>;
  causes: Set<string>;
  eventTypes: Set<string>;
  checkpoints: Array<{ kind: "demand_result" | "round_result"; round: number }>;
  demandPurchasers: Map<number, number[]>;
  maxPrice: number;
  renderedPriceAboveTen: boolean;
  backChecked: boolean;
  sawDemandWhiteOut: boolean;
  sawResearchWhiteOut: boolean;
  sawDeduplicatedSales: boolean;
  sawLocalBidStep: boolean;
  sawLocalImpactTarget: boolean;
  privacyFailures: string[];
  pendingAudits: Promise<void>[];
};

type GameHarness = {
  page: Page;
  gameId: string;
  token: string;
  view: GameView;
  coverage: Coverage;
  responseListener: (response: Response) => void;
};

function newCoverage(): Coverage {
  return {
    views: [],
    phases: new Set(),
    decisions: new Set(),
    causes: new Set(),
    eventTypes: new Set(),
    checkpoints: [],
    demandPurchasers: new Map(),
    maxPrice: Number.NEGATIVE_INFINITY,
    renderedPriceAboveTen: false,
    backChecked: false,
    sawDemandWhiteOut: false,
    sawResearchWhiteOut: false,
    sawDeduplicatedSales: false,
    sawLocalBidStep: false,
    sawLocalImpactTarget: false,
    privacyFailures: [],
    pendingAudits: [],
  };
}

function fail(coverage: Coverage, message: string) {
  if (!coverage.privacyFailures.includes(message)) coverage.privacyFailures.push(message);
}

function recordView(view: GameView, coverage: Coverage, token: string) {
  coverage.views.push(view);
  coverage.phases.add(view.phase.toLowerCase());
  coverage.decisions.add(view.pending_decision.kind);
  for (const event of view.recent_events) {
    coverage.eventTypes.add(event.event_type);
    if (event.cause) coverage.causes.add(event.cause);
    if (event.resulting_price_dollars_per_share != null) {
      coverage.maxPrice = Math.max(coverage.maxPrice, event.resulting_price_dollars_per_share);
    }
  }
  for (const company of view.companies) {
    coverage.maxPrice = Math.max(coverage.maxPrice, company.price_dollars_per_share);
  }

  if (view.viewer.player_id !== 0 || view.viewer.name !== "YOU") {
    fail(coverage, `Viewer changed to ${JSON.stringify(view.viewer)}`);
  }
  if (view.configuration.player_count !== 2 || view.configuration.round_count !== 6) {
    fail(coverage, "Browser configuration was not fixed to two players and six rounds");
  }
  if (view.players.length !== 2 || view.players[0]?.name !== "YOU" || view.players[1]?.name !== "COMPUTER") {
    fail(coverage, "Public player list was not exactly YOU and COMPUTER");
  }

  const privateKeys = Object.keys(view.private).sort();
  if (JSON.stringify(privateKeys) !== JSON.stringify([
    "available_action_cards",
    "holdings",
    "market_information",
  ])) {
    fail(coverage, `Private allowlist changed: ${privateKeys.join(",")}`);
  }

  const computer = view.players.find((player) => player.role === "computer") as Record<string, unknown> | undefined;
  if (!computer) {
    fail(coverage, "COMPUTER public record is absent");
  } else {
    for (const forbidden of ["holdings", "position_value_thousands", "position_delta_thousands"]) {
      if (forbidden in computer) fail(coverage, `COMPUTER exposed ${forbidden}`);
    }
  }

  if (view.stockpiles.length !== 4) fail(coverage, `Received ${view.stockpiles.length} Stockpiles instead of four`);
  for (const pile of view.stockpiles) {
    for (const card of pile.cards_bottom_to_top) {
      if (card.visibility === "hidden" && (Object.keys(card).length !== 1 || Object.keys(card)[0] !== "visibility")) {
        fail(coverage, "A hidden Stockpile card contained identity metadata");
      }
      if (card.visibility === "remembered" && card.face_down !== true) {
        fail(coverage, "A remembered card lost its physical face-down state");
      }
    }
  }
  for (const slot of view.private.market_information) {
    if (slot.card.visibility === "hidden" && (Object.keys(slot.card).length !== 1 || Object.keys(slot.card)[0] !== "visibility")) {
      fail(coverage, "A hidden information card contained identity metadata");
    }
  }

  for (const player of view.players) {
    if (player.bid_markers.length !== 2) {
      fail(coverage, `${player.name} exposed ${player.bid_markers.length} bid markers`);
    }
    const markerIndices = [...new Set(player.bid_markers.map((marker) => marker.marker_index))].sort();
    if (JSON.stringify(markerIndices) !== JSON.stringify([0, 1])) {
      fail(coverage, `${player.name} marker identities were ${markerIndices.join(",")}`);
    }
  }

  if (view.checkpoint) {
    const current = { kind: view.checkpoint.kind, round: view.checkpoint.round };
    const previous = coverage.checkpoints.at(-1);
    if (!previous || previous.kind !== current.kind || previous.round !== current.round) {
      coverage.checkpoints.push(current);
    }
    if (view.legal_actions.length || view.supply_batch !== null || view.decision_batch !== null || view.pending_decision.kind !== "acknowledge") {
      fail(coverage, `${view.checkpoint.kind} exposed a next-phase decision before acknowledgement`);
    }
    if (view.checkpoint.kind === "demand_result") {
      const purchasers = view.stockpiles.map((pile) => pile.purchaser_id);
      if (purchasers.some((playerId) => playerId === null)) {
        fail(coverage, `Round ${view.checkpoint.round} Demand did not settle all four piles`);
      } else {
        coverage.demandPurchasers.set(view.checkpoint.round, purchasers as number[]);
      }
    }
  }

  const wire = JSON.stringify(view).toLowerCase();
  if (wire.includes(token.toLowerCase())) fail(coverage, "A view echoed its bearer token");
  for (const forbidden of [
    "card_id",
    "information_state_id",
    "information_state_tensor",
    "history_records",
    "raw_history",
    "known_pile_cards",
    "computer_holdings",
  ]) {
    if (wire.includes(forbidden)) fail(coverage, `Response exposed ${forbidden}`);
  }
}

function isGameViewResponse(response: Response, gameId: string) {
  if (!response.ok()) return false;
  const url = new URL(response.url());
  if (!url.pathname.startsWith(`/api/v2/games/${gameId}/`)) return false;
  return ["view", "actions", "supply", "decisions", "acknowledgements"].includes(url.pathname.split("/").at(-1) ?? "");
}

async function awaitAudits(harness: GameHarness) {
  harness.page.off("response", harness.responseListener);
  for (;;) {
    const current = harness.coverage.pendingAudits.splice(0);
    if (!current.length) break;
    await Promise.all(current);
  }
  expect(harness.coverage.privacyFailures).toEqual([]);
}

async function openGame(
  page: Page,
  request: APIRequestContext,
  options: LiteOptions,
  seed: number,
): Promise<GameHarness> {
  const create = await request.post("/api/v2/games", { data: { options, seed } });
  expect(create.status()).toBe(201);
  expect(create.headers()["cache-control"]).toBe("no-store");
  const created = await create.json() as CreateGameResponse;
  expect(Object.keys(created).sort()).toEqual(["game_id", "game_url", "schema_version"]);
  expect(created.schema_version).toBe("2.0");
  expect(created).not.toHaveProperty("seed");
  expect(created).not.toHaveProperty("seats");
  expect(created).not.toHaveProperty("computer_token");

  const gameUrl = new URL(created.game_url, "http://127.0.0.1:5173");
  const token = new URLSearchParams(gameUrl.hash.slice(1)).get("seat");
  expect(token).toBeTruthy();

  const coverage = newCoverage();
  const responseListener = (response: Response) => {
    if (!isGameViewResponse(response, created.game_id)) return;
    const audit = response.json()
      .then((body: GameView) => recordView(body, coverage, token!))
      .catch((cause: unknown) => fail(coverage, `Could not inspect a V2 response: ${String(cause)}`));
    coverage.pendingAudits.push(audit);
  };
  page.on("response", responseListener);

  const initialResponse = page.waitForResponse((response) => (
    response.request().method() === "GET"
    && new URL(response.url()).pathname === `/api/v2/games/${created.game_id}/view`
    && response.ok()
  ));
  await page.goto(gameUrl.href);
  const initial = await initialResponse;
  const view = await initial.json() as GameView;
  recordView(view, coverage, token!);

  await expect(page.getByTestId("workstation")).toBeVisible();
  expect(new URL(page.url()).hash).toBe("");
  expect(await page.evaluate(
    (gameId) => sessionStorage.getItem(`stockpile.seatToken:${gameId}`),
    created.game_id,
  )).toBe(token);
  expect(await page.evaluate(() => Object.keys(sessionStorage))).toEqual([
    `stockpile.seatToken:${created.game_id}`,
  ]);

  return {
    page,
    gameId: created.game_id,
    token: token!,
    view,
    coverage,
    responseListener,
  };
}

async function acceptResponse(harness: GameHarness, response: Response) {
  const responseText = await response.text();
  expect(response.status(), responseText).toBe(200);
  expect(response.headers()["cache-control"]).toBe("no-store");
  const view = JSON.parse(responseText) as GameView;
  harness.view = view;
  recordView(view, harness.coverage, harness.token);
  await expect(harness.page.getByTestId("workstation")).toHaveAttribute(
    "data-decision-kind",
    view.pending_decision.kind,
  );
  if (view.checkpoint) {
    await expect(harness.page.getByTestId("workstation")).toHaveAttribute(
      "data-checkpoint-kind",
      view.checkpoint.kind,
    );
  }
  const aboveTen = [...view.companies]
    .sort((left, right) => right.price_dollars_per_share - left.price_dollars_per_share)
    .find((company) => company.price_dollars_per_share > 10);
  if (aboveTen) {
    await expect(harness.page.getByLabel("Market").getByText(
      `$${aboveTen.price_dollars_per_share} / SHARE`,
      { exact: true },
    ).first()).toBeVisible();
    harness.coverage.renderedPriceAboveTen = true;
  }
  return view;
}

function postResponse(page: Page, gameId: string, endpoint: "actions" | "supply" | "decisions" | "acknowledgements" | "resignations") {
  return page.waitForResponse((response) => (
    response.request().method() === "POST"
    && new URL(response.url()).pathname === `/api/v2/games/${gameId}/${endpoint}`
  ));
}

async function submitAction(harness: GameHarness, action: LegalAction) {
  const target = action.control === "stockpile"
    ? harness.page.locator(`[data-stockpile-id="${action.target_id?.split(":").at(-1)}"] [data-stockpile-target]`)
    : harness.page.locator(`[data-action-id="${action.action_id}"]`).first();
  await expect(target).toBeVisible();
  await expect(target).toBeEnabled();
  await target.click({ position: { x: 5, y: 5 } });
  const confirm = harness.page.locator('[data-context-action="confirm"]');
  await expect(confirm).toBeVisible();
  await expect(confirm).toHaveAttribute("data-action-id", String(action.action_id));
  const responsePromise = postResponse(harness.page, harness.gameId, "actions");
  await confirm.click();
  const response = await responsePromise;
  expect(response.request().postDataJSON()).toEqual({
    action_id: action.action_id,
    expected_revision: harness.view.revision,
  });
  return acceptResponse(harness, response);
}

async function submitSupply(harness: GameHarness, plan: SupplyPlan) {
  const batch = harness.view.supply_batch;
  expect(batch).not.toBeNull();
  for (const { card_ref } of batch!.cards) {
    const placement = plan.placements.find((candidate) => candidate.card_ref === card_ref);
    expect(placement, `Plan ${plan.plan_id} omitted ${card_ref}`).toBeTruthy();
    const card = harness.page.locator(`[data-supply-card-ref="${card_ref}"]`);
    await expect(card).toBeVisible();
    await card.click({ position: { x: 5, y: 5 } });

    const pile = harness.page.locator(`[data-stockpile-id="${placement!.stockpile_id}"] [data-stockpile-target]`);
    await expect(pile).toBeVisible();
    await pile.click({ position: { x: 5, y: 5 } });

    const visibility = harness.page.locator(`[data-supply-visibility="${placement!.visibility}"]`);
    await expect(visibility).toBeVisible();
    await visibility.click();
    await expect(card).toHaveAttribute("data-assigned-pile", String(placement!.stockpile_id));
    await expect(card).toHaveAttribute("data-assigned-visibility", placement!.visibility);
    await expect(card).toHaveAttribute("data-white-out", "true");
    await expect(harness.page.locator(`[data-tentative-card-ref="${card_ref}"]`)).toHaveAttribute("data-white-out", "true");
  }

  const confirm = harness.page.locator('[data-context-action="confirm"]');
  await expect(confirm).toBeEnabled();
  await expect(confirm).toHaveAttribute("data-plan-id", plan.plan_id);
  const responsePromise = postResponse(harness.page, harness.gameId, "supply");
  await confirm.click();
  const response = await responsePromise;
  expect(response.request().postDataJSON()).toEqual({
    plan_id: plan.plan_id,
    expected_revision: harness.view.revision,
  });
  return acceptResponse(harness, response);
}

async function submitDecision(harness: GameHarness) {
  const batch = harness.view.decision_batch;
  expect(batch).not.toBeNull();
  let planId: string;
  if (batch!.kind === "demand") {
    const plan = batch.plans[0];
    planId = plan.plan_id;
    const pile = harness.page.locator(`[data-stockpile-id="${plan.stockpile_id}"] [data-stockpile-target]`);
    await expect(pile).toBeVisible();
    await pile.click({ position: { x: 5, y: 5 } });
    const bid = harness.page.locator(`[data-decision-plan-id="${plan.plan_id}"]`);
    await expect(bid).toBeVisible();
    harness.coverage.sawLocalBidStep = true;
    await bid.click();
  } else {
    const plan = batch!.plans[0];
    planId = plan.plan_id;
    await harness.page.locator(`[data-impact-direction="${plan.direction}"]`).click();
    const company = harness.page.locator(`[data-decision-plan-id="${plan.plan_id}"]`);
    await expect(company).toBeVisible();
    harness.coverage.sawLocalImpactTarget = true;
    await company.click();
  }
  const confirm = harness.page.locator('[data-context-action="confirm"]');
  await expect(confirm).toHaveAttribute("data-plan-id", planId);
  const responsePromise = postResponse(harness.page, harness.gameId, "decisions");
  await confirm.click();
  const response = await responsePromise;
  expect(response.request().postDataJSON()).toEqual({
    plan_id: planId,
    expected_revision: harness.view.revision,
  });
  return acceptResponse(harness, response);
}

async function acknowledge(harness: GameHarness) {
  const checkpoint = harness.view.checkpoint;
  expect(checkpoint).not.toBeNull();
  if (checkpoint!.kind === "demand_result") {
    await expect(harness.page.locator('article[data-white-out="true"]')).toHaveCount(4);
    harness.coverage.sawDemandWhiteOut = true;
  } else {
    await expect(harness.page.getByLabel("Research")).toHaveAttribute("data-white-out", "true");
    harness.coverage.sawResearchWhiteOut = true;
  }
  const button = harness.page.locator('[data-context-action="continue"]');
  await expect(button).toBeVisible();
  await expect(button).toHaveAttribute("data-checkpoint-kind", checkpoint!.kind);
  const responsePromise = postResponse(harness.page, harness.gameId, "acknowledgements");
  await button.click();
  const response = await responsePromise;
  expect(response.request().postDataJSON()).toEqual({
    checkpoint_id: checkpoint!.checkpoint_id,
    expected_revision: harness.view.revision,
  });
  const next = await acceptResponse(harness, response);
  const back = harness.page.locator('[data-context-action="back"]');
  if (!harness.coverage.backChecked && await back.count()) {
    await back.click();
    await expect(harness.page.getByTestId("workstation")).toHaveAttribute("data-checkpoint-kind", checkpoint!.kind);
    const returnToDecision = harness.page.locator('[data-context-action="continue"]');
    await expect(returnToDecision).toBeVisible();
    await returnToDecision.click();
    await expect(harness.page.getByTestId("workstation")).toHaveAttribute("data-decision-kind", next.pending_decision.kind);
    harness.coverage.backChecked = true;
  }
  return next;
}

async function driveGame(harness: GameHarness, maximumSteps = 2_000) {
  for (let step = 0; step < maximumSteps; step += 1) {
    if (harness.view.terminal_results) return harness.view;
    if (harness.view.checkpoint) {
      await acknowledge(harness);
      continue;
    }
    if (harness.view.supply_batch) {
      await submitSupply(harness, harness.view.supply_batch.plans[0]);
      continue;
    }
    if (harness.view.decision_batch) {
      await submitDecision(harness);
      continue;
    }
    const sales = harness.view.legal_actions.filter((action) => action.control === "sell" && action.sale_preview);
    if (sales.length) {
      const identities = sales.map((action) => {
        const sale = action.sale_preview!;
        return `${sale.shares_thousands}:${sale.gross_value_thousands}:${sale.resulting_shares_thousands}`;
      });
      expect(new Set(identities).size).toBe(identities.length);
      harness.coverage.sawDeduplicatedSales = true;
    }
    const action = harness.view.legal_actions[0];
    expect(
      action,
      `No browser action at ${harness.view.phase}/${harness.view.phase_step}/${harness.view.pending_decision.kind}`,
    ).toBeTruthy();
    await submitAction(harness, action);
  }
  throw new Error(`Game did not reach Game End within ${maximumSteps} human decisions`);
}

async function expectNoHorizontalOverflow(page: Page) {
  const dimensions = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }));
  expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth + 1);
}

async function expectDisciplinedVisualLanguage(page: Page) {
  const audit = await page.evaluate(() => {
    const rootStyle = getComputedStyle(document.documentElement);
    const visible = [...document.querySelectorAll<HTMLElement>("#root *")]
      .filter((element) => element.getClientRects().length > 0);
    const unique = (values: string[]) => [...new Set(values)].sort();
    const visualViolations = visible.flatMap((element) => {
      const style = getComputedStyle(element);
      const problems: string[] = [];
      if (style.backgroundImage !== "none") problems.push("background-image");
      if (style.boxShadow !== "none") problems.push("box-shadow");
      if ([style.borderTopLeftRadius, style.borderTopRightRadius, style.borderBottomLeftRadius, style.borderBottomRightRadius].some((value) => value !== "0px")) problems.push("border-radius");
      if (style.fontWeight !== "400") problems.push(`font-weight:${style.fontWeight}`);
      if (style.fontStyle !== "normal") problems.push(`font-style:${style.fontStyle}`);
      return problems.map((problem) => `${element.tagName.toLowerCase()}:${problem}`);
    });
    const sectionNames = new Set(["MARKET", "RESEARCH", "PORTFOLIO", "PLAYERS", "ACTION"]);
    const sectionStyles = visible
      .filter((element) => element.children.length === 0 && sectionNames.has(element.textContent?.trim() ?? ""))
      .map((element) => {
        const style = getComputedStyle(element);
        return [style.color, style.fontFamily, style.fontSize, style.fontWeight, style.letterSpacing].join("|");
      });
    return {
      tokens: {
        blue: rootStyle.getPropertyValue("--blue").trim().toLowerCase(),
        black: rootStyle.getPropertyValue("--black").trim().toLowerCase(),
        grey: rootStyle.getPropertyValue("--grey").trim().toLowerCase(),
        green: rootStyle.getPropertyValue("--green").trim().toLowerCase(),
        red: rootStyle.getPropertyValue("--red").trim().toLowerCase(),
        white: rootStyle.getPropertyValue("--white").trim().toLowerCase(),
        ochre: rootStyle.getPropertyValue("--ochre").trim(),
      },
      bodyBackground: getComputedStyle(document.body).backgroundColor,
      fontFamilies: unique(visible.map((element) => getComputedStyle(element).fontFamily)),
      fontSizes: unique(visible.map((element) => getComputedStyle(element).fontSize)),
      visualViolations,
      sectionStyleCount: new Set(sectionStyles).size,
      sectionLabelCount: sectionStyles.length,
    };
  });
  expect(audit.tokens).toEqual({
    blue: "#002fa7",
    black: "#111111",
    grey: "#70747a",
    green: "#14733d",
    red: "#b21f2d",
    white: "#ffffff",
    ochre: "",
  });
  expect(audit.bodyBackground).toBe("rgb(255, 255, 255)");
  expect(audit.fontFamilies).toHaveLength(1);
  expect(audit.fontSizes.length).toBeLessThanOrEqual(2);
  expect(audit.visualViolations).toEqual([]);
  expect(audit.sectionLabelCount).toBeGreaterThanOrEqual(5);
  expect(audit.sectionStyleCount).toBe(1);
}

function expectedCheckpointSequence() {
  return Array.from({ length: 6 }, (_, index) => index + 1).flatMap((round) => [
    { kind: "demand_result" as const, round },
    { kind: "round_result" as const, round },
  ]);
}

test("Home exposes only Trainer LITE and LITE+ through one button language", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByLabel("Stockpile Trainer")).toHaveText("STOCKPILE TRAINER");
  await expect(page.getByRole("button", { name: "LITE", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "LITE+", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "PLAY", exact: true })).toHaveCount(0);
  await expect(page.locator("input, select, textarea")).toHaveCount(0);

  const homeText = (await page.locator("body").innerText()).toUpperCase();
  for (const forbidden of [
    "PLAYER COUNT",
    "PLAYER NAME",
    "OPEN SEAT",
    "CHAT",
    "LOBBY",
    "IMPACT",
    "HAND",
    "SPLIT",
    "MAJORITY",
    "INVESTOR",
    "STOCK TRACKS",
    "ANALYSIS",
    "DEEP CFR",
  ]) {
    expect(homeText).not.toContain(forbidden);
  }

  await page.getByRole("button", { name: "LITE+", exact: true }).click();
  const featureNames = ["DIVIDEND", "FEES", "SELL ORDER"];
  for (const feature of featureNames) {
    const button = page.getByRole("button", { name: feature, exact: true });
    await expect(button).toBeVisible();
    await expect(button).toHaveAttribute("aria-pressed", "false");
  }
  await expect(page.getByRole("button", { name: "PLAY", exact: true })).toHaveCount(0);

  await page.getByRole("button", { name: "DIVIDEND", exact: true }).click();
  const controls = page.getByRole("button");
  const buttonGeometry = await controls.evaluateAll((buttons) => buttons.map((button) => {
    const box = button.getBoundingClientRect();
    const style = getComputedStyle(button);
    return {
      width: box.width,
      height: box.height,
      background: style.backgroundColor,
      border: style.border,
      radius: style.borderRadius,
      weight: style.fontWeight,
      shadow: style.boxShadow,
    };
  }));
  expect(new Set(buttonGeometry.map((item) => item.width)).size).toBe(1);
  expect(new Set(buttonGeometry.map((item) => item.height)).size).toBe(1);
  expect(buttonGeometry.every((item) => item.width === 144 && item.height === 36)).toBe(true);
  expect(buttonGeometry.every((item) => item.border.includes("rgb(17, 17, 17)") && item.radius === "0px" && item.weight === "400" && item.shadow === "none")).toBe(true);

  await expect(page.getByRole("button", { name: "DIVIDEND", exact: true })).toHaveCSS("background-color", "rgb(17, 17, 17)");
  await expect(page.getByRole("button", { name: "PLAY", exact: true })).toHaveCSS("background-color", "rgb(255, 255, 255)");

  const createRequest = page.waitForRequest((request) => request.method() === "POST" && new URL(request.url()).pathname === "/api/v2/games");
  await page.getByRole("button", { name: "PLAY", exact: true }).click();
  expect((await createRequest).postDataJSON()).toEqual({ options: {
    market_impact: false,
    trading_fees: false,
    dividends: true,
    sell_order: false,
  } });
});

test("one token opens fixed Research, exact cards, and independently inspectable stacks", async ({ page, request }) => {
  await page.setViewportSize({ width: 1_280, height: 900 });
  const harness = await openGame(page, request, defaultOptions, 101);
  try {
    for (const region of regions) await expect(page.getByLabel(region)).toBeVisible();
    await expect(page.getByText("CHAT", { exact: true })).toHaveCount(0);
    await expect(page.getByText("SEAT", { exact: true })).toHaveCount(0);
    await expect(page.getByText("COMPUTER", { exact: true })).toBeVisible();
    await expect(page.getByText("YOU", { exact: true })).toBeVisible();
    await expect(page.getByText("RESEARCH", { exact: true })).toBeVisible();
    await expect(page.getByText("PRIVATE", { exact: true })).toHaveCount(0);
    await expect(page.getByText("PUBLIC", { exact: true })).toHaveCount(0);

    const piles = page.locator("article[data-stockpile-id]");
    await expect(piles).toHaveCount(4);
    const positions = await piles.evaluateAll((elements) => elements.map((element) => {
      const box = element.getBoundingClientRect();
      return { x: Math.round(box.x), y: Math.round(box.y) };
    }));
    expect(new Set(positions.map((position) => position.x)).size).toBe(2);
    expect(new Set(positions.map((position) => position.y)).size).toBe(2);
    await expectNoHorizontalOverflow(page);

    for (const company of harness.view.companies) {
      await expect(page.getByLabel("Market").getByText(`$${company.price_dollars_per_share} / SHARE`, { exact: true }).first()).toBeVisible();
    }
    for (const pattern of patterns) {
      const motif = page.getByLabel("Market").locator(`[data-stock-pattern="${pattern}"]`);
      await expect(motif).toHaveCount(1);
      await expect(motif).toHaveCSS("color", "rgb(0, 47, 167)");
    }
    await expect(page.locator('[data-player-role="human"] [data-player-metric="cash"]')).toContainText(/^CASH\$\d+K/);
    await expect(page.locator('[data-player-role="human"] [data-player-metric="position"]')).toContainText(/^POSITION\$\d+K/);
    await expect(page.locator('[data-player-role="computer"] [data-player-metric="cash"]')).toContainText(/^CASH\$\d+K/);
    await expect(page.locator('[data-player-role="computer"] [data-player-metric="position"]')).toHaveCount(0);

    await expectDisciplinedVisualLanguage(page);

    const batch = harness.view.supply_batch!;
    const samePilePlan = batch.plans.find((plan) => new Set(plan.placements.map((placement) => placement.stockpile_id)).size === 1) ?? batch.plans[0];
    const staged = samePilePlan.placements[0];
    const stagedSource = page.locator(`[data-supply-card-ref="${staged.card_ref}"]`);
    await stagedSource.click();
    await page.locator(`[data-stockpile-id="${staged.stockpile_id}"] [data-stockpile-target]`).click({ position: { x: 5, y: 5 } });
    const stagedVisibility = page.locator(`[data-supply-visibility="${staged.visibility}"]`);
    await expect(stagedVisibility).toBeVisible();
    await stagedVisibility.click();
    const tentative = page.locator(`[data-tentative-card-ref="${staged.card_ref}"]`);
    await expect(tentative).toHaveAttribute("data-white-out", "true");
    await expect(stagedSource).toHaveAttribute("data-white-out", "true");
    await tentative.dblclick();
    await expect(tentative).toHaveCount(0);
    await expect(stagedSource).not.toHaveAttribute("data-assigned-pile");

    await submitSupply(harness, samePilePlan);

    const rememberedPile = harness.view.stockpiles.find((pile) => (
      pile.cards_bottom_to_top.length >= 2
      && pile.cards_bottom_to_top.some((card) => card.visibility === "remembered")
    ));
    expect(rememberedPile, "Supply did not preserve the human's known face-down card in its actual pile").toBeTruthy();
    const pile = page.locator(`article[data-stockpile-id="${rememberedPile!.stockpile_id}"]`);
    const inspector = pile.locator("[data-stack-inspect]");
    await expect(inspector).toHaveAttribute("aria-expanded", "false");
    await expect(pile.getByText("FACE DOWN", { exact: true })).toHaveCount(0);

    const selectedSupplyCard = page.locator("[data-supply-card-ref]").first();
    if (await selectedSupplyCard.count()) {
      await selectedSupplyCard.click({ position: { x: 5, y: 5 } });
      await expect(selectedSupplyCard).not.toHaveAttribute("data-assigned-pile", /.+/);
    }
    let mutationRequests = 0;
    const countMutation = (sent: Request) => {
      if (sent.method() === "POST" && new URL(sent.url()).pathname.startsWith(`/api/v2/games/${harness.gameId}/`)) mutationRequests += 1;
    };
    page.on("request", countMutation);
    const revisionBeforeInspection = harness.view.revision;
    await inspector.dblclick();
    await expect(inspector).toHaveAttribute("aria-expanded", "true");
    await expect(pile.getByText("FACE DOWN", { exact: true })).toBeVisible();
    if (await selectedSupplyCard.count()) await expect(selectedSupplyCard).not.toHaveAttribute("data-assigned-pile", /.+/);
    await page.waitForTimeout(100);
    page.off("request", countMutation);
    expect(mutationRequests).toBe(0);
    expect(harness.view.revision).toBe(revisionBeforeInspection);

    const renderedOrder = await pile.locator("[data-stack-card]").evaluateAll((cards) => cards.map((card) => {
      const element = card as HTMLElement;
      const box = element.getBoundingClientRect();
      return {
        order: Number(element.dataset.stackOrder),
        z: Number(getComputedStyle(element).zIndex),
        x: Math.round(box.x),
        y: Math.round(box.y),
        right: Math.round(box.right),
        bottom: Math.round(box.bottom),
      };
    }));
    expect(renderedOrder.map((entry) => entry.order)).toEqual(
      rememberedPile!.cards_bottom_to_top.map((_card, index) => index),
    );
    expect(renderedOrder.map((entry) => entry.z)).toEqual(
      rememberedPile!.cards_bottom_to_top.map((_card, index) => index + 1),
    );
    expect(new Set(renderedOrder.map((entry) => `${entry.x}:${entry.y}`)).size).toBe(renderedOrder.length);
    for (let left = 0; left < renderedOrder.length; left += 1) {
      for (let right = left + 1; right < renderedOrder.length; right += 1) {
        const one = renderedOrder[left];
        const two = renderedOrder[right];
        const overlaps = one.x < two.right && one.right > two.x && one.y < two.bottom && one.bottom > two.y;
        expect(overlaps).toBe(false);
      }
    }

    const expectedCardCount = harness.view.stockpiles.reduce((count, candidate) => count + candidate.cards_bottom_to_top.length, 0);
    await expect(page.locator('[data-card-scale="stockpile"]')).toHaveCount(expectedCardCount);
    const largeCardSizes = await page.locator('[data-card-scale="stockpile"]').evaluateAll((cards) => cards.map((card) => {
      const box = card.getBoundingClientRect();
      return [Math.round(box.width), Math.round(box.height)];
    }));
    expect(new Set(largeCardSizes.map((size) => size.join("x")))).toEqual(new Set(["104x139"]));
    const smallCards = page.locator('[data-card-scale="active"], [data-card-scale="portfolio"], [data-card-scale="information"]');
    expect(await smallCards.count()).toBeGreaterThan(0);
    const smallCardSizes = await smallCards.evaluateAll((cards) => cards.map((card) => {
      const box = card.getBoundingClientRect();
      return [Math.round(box.width), Math.round(box.height)];
    }));
    expect(new Set(smallCardSizes.map((size) => size.join("x")))).toEqual(new Set(["54x72"]));
    await expect(page.getByLabel("Hidden card").first()).toHaveCSS("background-color", "rgb(0, 47, 167)");
    await expect(page.getByText("1K", { exact: true }).first()).toBeVisible();
  } finally {
    await awaitAudits(harness);
  }
});

test("RESIGN is persistent, cancellable, and returns home only after confirmation", async ({ page, request }) => {
  const harness = await openGame(page, request, defaultOptions, 211);
  try {
    let resignationRequests = 0;
    page.on("request", (sent) => {
      if (new URL(sent.url()).pathname.endsWith("/resignations")) resignationRequests += 1;
    });
    const resign = page.locator("[data-resign]");
    await expect(resign).toBeVisible();
    await resign.click();
    const confirm = page.locator('[data-context-action="confirm"][data-resign-confirm="true"]');
    await expect(confirm).toBeVisible();
    expect(resignationRequests).toBe(0);
    await resign.click();
    await expect(confirm).toHaveCount(0);
    expect(resignationRequests).toBe(0);

    await resign.click();
    const responsePromise = postResponse(page, harness.gameId, "resignations");
    await page.locator('[data-context-action="confirm"][data-resign-confirm="true"]').click();
    const response = await responsePromise;
    expect(response.status()).toBe(204);
    expect(response.request().postDataJSON()).toEqual({ expected_revision: harness.view.revision });
    await page.waitForURL("http://127.0.0.1:5173/");
    expect(await page.evaluate(
      (gameId) => sessionStorage.getItem(`stockpile.seatToken:${gameId}`),
      harness.gameId,
    )).toBeNull();
    expect(resignationRequests).toBe(1);
  } finally {
    await awaitAudits(harness);
  }
});

test("default seed 101 completes six rounds through Demand and Round acknowledgements", async ({ page, request }) => {
  test.setTimeout(180_000);
  const harness = await openGame(page, request, defaultOptions, 101);
  try {
    const terminal = await driveGame(harness);
    await expect(page.getByLabel("Game end")).toBeVisible();
    await expect(page.getByText("GAME END", { exact: true }).first()).toBeVisible();
    expect(terminal.terminal_results?.players).toHaveLength(2);
    expect(terminal.terminal_results?.winner_ids.length).toBeGreaterThan(0);
    expect(harness.coverage.checkpoints).toEqual(expectedCheckpointSequence());
    expect([...harness.coverage.phases]).toEqual(expect.arrayContaining(["supply", "demand", "selling", "terminal"]));
    expect([...harness.coverage.decisions]).toEqual(expect.arrayContaining(["supply", "bid_pile", "sell", "acknowledge", "terminal"]));
    expect(harness.coverage.sawLocalBidStep).toBe(true);
    expect([...harness.coverage.eventTypes].some((event) => event.includes("market"))).toBe(true);

    expect(harness.coverage.demandPurchasers.size).toBe(6);
    for (const purchasers of harness.coverage.demandPurchasers.values()) {
      expect(purchasers).toHaveLength(4);
      expect(purchasers.filter((playerId) => playerId === 0)).toHaveLength(2);
      expect(purchasers.filter((playerId) => playerId === 1)).toHaveLength(2);
    }
    expect(harness.coverage.views.every((view) => view.viewer.player_id === 0 && view.viewer.name === "YOU")).toBe(true);
    expect(harness.coverage.views.every((view) => view.stockpiles.length === 4)).toBe(true);
    expect(harness.coverage.views.every((view) => view.players.every((player) => player.bid_markers.length === 2))).toBe(true);
    expect(harness.coverage.backChecked).toBe(true);
    expect(harness.coverage.sawDemandWhiteOut).toBe(true);
    expect(harness.coverage.sawResearchWhiteOut).toBe(true);
    expect(harness.coverage.sawDeduplicatedSales).toBe(true);

    await expect(page.getByText("WINNER", { exact: true })).toHaveCount(0);
    const chart = page.getByTestId("terminal-chart");
    await expect(chart).toBeVisible();
    await expect(chart.locator('[data-chart-segment="position"]')).toHaveCount(2);
    await expect(chart.locator('[data-chart-segment="cash"]')).toHaveCount(2);
    await expect(chart.locator('[data-chart-segment="position"]').first()).toHaveCSS("background-color", "rgb(0, 47, 167)");
    await expect(chart.locator('[data-chart-segment="cash"]').first()).toHaveCSS("background-color", "rgb(255, 255, 255)");
    const expectedMinimum = Math.min(0, ...terminal.terminal_results!.players.map((player) => player.cash_before_liquidation_thousands));
    const expectedMaximum = Math.max(0, ...terminal.terminal_results!.players.flatMap((player) => [
      player.liquidation_value_thousands,
      player.final_cash_thousands,
      player.liquidation_value_thousands + Math.max(0, player.cash_before_liquidation_thousands),
    ]));
    await expect(chart).toHaveAttribute("data-chart-min", String(expectedMinimum));
    await expect(chart).toHaveAttribute("data-chart-max", String(expectedMaximum));
    expect(await page.getByLabel("Portfolio").locator('[data-white-out="true"]').count()).toBeGreaterThan(0);

    for (const result of terminal.terminal_results!.players) {
      expect(result.final_cash_thousands).toBe(
        result.cash_before_liquidation_thousands + result.liquidation_value_thousands,
      );
      expect(result.liquidation_value_thousands).toBe(
        result.liquidation.reduce((total, line) => total + line.value_thousands, 0),
      );
    }
  } finally {
    await awaitAudits(harness);
  }
});

test("all-options seed 2 renders Impact, ordinary prices above ten, and reaches Game End", async ({ page, request }) => {
  test.setTimeout(180_000);
  const harness = await openGame(page, request, allOptions, 2);
  try {
    const terminal = await driveGame(harness);
    await expect(page.getByLabel("Game end")).toBeVisible();
    expect(terminal.configuration.options).toEqual(allOptions);
    expect(terminal.terminal_results?.players).toHaveLength(2);
    expect(harness.coverage.checkpoints).toEqual(expectedCheckpointSequence());
    expect(harness.coverage.phases).toContain("action");
    expect(harness.coverage.sawLocalImpactTarget).toBe(true);
    expect(harness.coverage.causes).toContain("market_impact");
    expect(harness.coverage.maxPrice).toBeGreaterThan(10);
    expect(harness.coverage.renderedPriceAboveTen).toBe(true);
    expect(harness.coverage.demandPurchasers.size).toBe(6);
  } finally {
    await awaitAudits(harness);
  }
});
