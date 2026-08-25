import { expect, test } from "@playwright/test";
import type { APIRequestContext, Page, Request, Response } from "@playwright/test";
import type {
  CreateGameResponse,
  GameView,
  LegalAction,
  LiteOptions,
  StockCard,
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
  demandMetricBaselines: Map<number, {
    humanCash: number;
    humanPosition: number;
    computerCash: number;
  }>;
  maxPrice: number;
  renderedPriceAboveTen: boolean;
  backChecked: boolean;
  sawDemandWhiteOut: boolean;
  sawResearchWhiteOut: boolean;
  sawDeduplicatedSales: boolean;
  sawLocalBidStep: boolean;
  sawLocalImpactTarget: boolean;
  sawDemandMetricSettlement: boolean;
  sawDemandMetricPersistence: boolean;
  sawSaleMetricSettlement: boolean;
  sawHoldMetricClear: boolean;
  sawIndependentMetricPreservation: boolean;
  sawNewRoundMetricClear: boolean;
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

type FixtureHarness = {
  gameId: string;
  token: string;
  view: GameView;
  update: (next: GameView) => Promise<void>;
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
    demandMetricBaselines: new Map(),
    maxPrice: Number.NEGATIVE_INFINITY,
    renderedPriceAboveTen: false,
    backChecked: false,
    sawDemandWhiteOut: false,
    sawResearchWhiteOut: false,
    sawDeduplicatedSales: false,
    sawLocalBidStep: false,
    sawLocalImpactTarget: false,
    sawDemandMetricSettlement: false,
    sawDemandMetricPersistence: false,
    sawSaleMetricSettlement: false,
    sawHoldMetricClear: false,
    sawIndependentMetricPreservation: false,
    sawNewRoundMetricClear: false,
    privacyFailures: [],
    pendingAudits: [],
  };
}

function fail(coverage: Coverage, message: string) {
  if (!coverage.privacyFailures.includes(message)) coverage.privacyFailures.push(message);
}

function recordView(view: GameView, coverage: Coverage, token: string) {
  // POST responses are inspected both by the direct driver and the privacy
  // listener. Keep one chronological copy per authoritative revision.
  if (coverage.views.at(-1)?.revision !== view.revision) coverage.views.push(view);
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
  if (view.phase.toLowerCase() === "demand" && view.checkpoint === null) {
    const human = humanPlayer(view);
    const computer = computerPlayer(view);
    if (!coverage.demandMetricBaselines.has(view.round)) {
      coverage.demandMetricBaselines.set(view.round, {
        humanCash: human.cash_thousands,
        humanPosition: human.position_value_thousands,
        computerCash: computer.cash_thousands,
      });
    }
  }

  if (view.viewer.name !== "YOU" || (view.viewer.player_id !== 0 && view.viewer.player_id !== 1)) {
    fail(coverage, `Viewer changed to ${JSON.stringify(view.viewer)}`);
  }
  const human = humanPlayer(view);
  if (human.player_id !== view.viewer.player_id) {
    fail(coverage, "Human public seat does not match viewer");
  }
  if (view.configuration.player_count !== 2 || view.configuration.round_count !== 1) {
    fail(coverage, "Browser configuration was not fixed to two players and one round");
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

function humanPlayer(view: GameView) {
  const player = view.players.find((candidate) => candidate.role === "human");
  if (!player || player.role !== "human") throw new Error("Human player is absent");
  return player;
}

function computerPlayer(view: GameView) {
  const player = view.players.find((candidate) => candidate.role === "computer");
  if (!player || player.role !== "computer") throw new Error("Computer player is absent");
  return player;
}

function nullableDelta(value: number) {
  return value === 0 ? null : value;
}

async function expectPlayerMetric(
  page: Page,
  role: "human" | "computer",
  metric: "cash" | "position",
  current: number,
  delta: number | null,
) {
  const row = page.locator(`[data-player-role="${role}"] [data-player-metric="${metric}"]`);
  const values = row.locator(":scope > span");
  await expect(row).toBeVisible();
  await expect(values.nth(1)).toHaveText(`$${current}K`);
  await expect(values).toHaveCount(3);
  expect(await values.nth(1).getAttribute("data-player-value-slot")).not.toBeNull();
  expect(await values.nth(2).getAttribute("data-player-delta-slot")).not.toBeNull();
  if (delta !== null) {
    await expect(values.nth(2)).toHaveText(`${delta > 0 ? "+" : "−"}$${Math.abs(delta)}K`);
  } else {
    await expect(values.nth(2)).toBeEmpty();
  }
}

async function expectNoMetricDeltas(page: Page, view: GameView) {
  const human = humanPlayer(view);
  const computer = computerPlayer(view);
  expect(human.cash_delta_thousands).toBeNull();
  expect(human.position_delta_thousands).toBeNull();
  expect(computer.cash_delta_thousands).toBeNull();
  expect(view.companies.every((company) => company.price_delta_dollars_per_share === null)).toBe(true);
  await expectPlayerMetric(page, "human", "cash", human.cash_thousands, null);
  await expectPlayerMetric(page, "human", "position", human.position_value_thousands, null);
  await expectPlayerMetric(page, "computer", "cash", computer.cash_thousands, null);
  await expect(page.locator("[data-market-price-delta]")).toHaveCount(0);
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

async function openFixtureGame(
  page: Page,
  request: APIRequestContext,
  transform: (base: GameView) => GameView,
): Promise<FixtureHarness> {
  const create = await request.post("/api/v2/games", {
    data: { options: defaultOptions, seed: 313 },
  });
  expect(create.status()).toBe(201);
  const created = await create.json() as CreateGameResponse;
  const gameUrl = new URL(created.game_url, "http://127.0.0.1:5173");
  const token = new URLSearchParams(gameUrl.hash.slice(1)).get("seat");
  expect(token).toBeTruthy();
  const viewPath = `/api/v2/games/${created.game_id}/view`;
  const baseResponse = await request.get(viewPath, {
    headers: { Authorization: `Bearer ${token}` },
  });
  expect(baseResponse.status()).toBe(200);
  let current = transform(await baseResponse.json() as GameView);

  await page.route(`**${viewPath}`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      headers: { "Cache-Control": "no-store" },
      body: JSON.stringify(current),
    });
  });
  await page.goto(gameUrl.href);
  await expect(page.getByTestId("workstation")).toBeVisible();

  return {
    gameId: created.game_id,
    token: token!,
    view: current,
    update: async (next: GameView) => {
      current = next;
      const response = page.waitForResponse((candidate) => (
        candidate.request().method() === "GET"
        && new URL(candidate.url()).pathname === viewPath
      ));
      await page.evaluate(() => window.dispatchEvent(new Event("focus")));
      await response;
    },
  };
}

function fixtureStock(view: GameView, companyId: number, sharesThousands = 1): StockCard {
  const company = view.companies.find((candidate) => candidate.company_id === companyId)!;
  return {
    visibility: "visible",
    kind: "stock",
    company_id: companyId,
    company: company.name,
    shares_thousands: sharesThousands,
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
  const before = harness.view;
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
  const next = await acceptResponse(harness, response);
  const sale = action.sale_preview;
  if (before.configuration.options.sell_order && sale) {
    const beforeHuman = humanPlayer(before);
    const afterHuman = humanPlayer(next);
    const beforeComputer = computerPlayer(before);
    const afterComputer = computerPlayer(next);
    if (
      sale.shares_thousands > 0
      && next.phase === "selling"
      && next.active_player_id === next.viewer.player_id
      && next.pending_decision.kind === "sell"
    ) {
      const cashDelta = afterHuman.cash_thousands - beforeHuman.cash_thousands;
      const positionDelta = afterHuman.position_value_thousands - beforeHuman.position_value_thousands;
      expect(sale.gross_value_thousands).toBe(
        sale.shares_thousands * sale.price_dollars_per_share,
      );
      expect(positionDelta).toBeLessThan(0);
      expect(afterHuman.cash_delta_thousands).toBe(nullableDelta(cashDelta));
      expect(afterHuman.position_delta_thousands).toBe(nullableDelta(positionDelta));
      // A COMPUTER transition cannot overwrite either human metric.
      expect(afterComputer.cash_delta_thousands).toBe(beforeComputer.cash_delta_thousands);
      await expectPlayerMetric(
        harness.page,
        "human",
        "cash",
        afterHuman.cash_thousands,
        nullableDelta(cashDelta),
      );
      await expectPlayerMetric(
        harness.page,
        "human",
        "position",
        afterHuman.position_value_thousands,
        nullableDelta(positionDelta),
      );
      harness.coverage.sawSaleMetricSettlement = true;
      harness.coverage.sawIndependentMetricPreservation = true;
    } else if (
      sale.shares_thousands === 0
      && harness.coverage.sawSaleMetricSettlement
      && next.phase === "selling"
      && next.active_player_id === next.viewer.player_id
      && next.pending_decision.kind === "sell"
    ) {
      expect(afterHuman.cash_thousands).toBe(beforeHuman.cash_thousands);
      expect(afterHuman.position_value_thousands).toBe(beforeHuman.position_value_thousands);
      expect(afterHuman.cash_delta_thousands).toBeNull();
      expect(afterHuman.position_delta_thousands).toBeNull();
      expect(afterComputer.cash_delta_thousands).toBe(beforeComputer.cash_delta_thousands);
      await expectPlayerMetric(harness.page, "human", "cash", afterHuman.cash_thousands, null);
      await expectPlayerMetric(
        harness.page,
        "human",
        "position",
        afterHuman.position_value_thousands,
        null,
      );
      harness.coverage.sawHoldMetricClear = true;
      harness.coverage.sawIndependentMetricPreservation = true;
    }
  }
  return next;
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
  const before = harness.view;
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
  const next = await acceptResponse(harness, response);
  if (batch!.kind === "demand" && next.checkpoint?.kind === "demand_result") {
    const beforeHuman = humanPlayer(before);
    const afterHuman = humanPlayer(next);
    const afterComputer = computerPlayer(next);
    const baseline = harness.coverage.demandMetricBaselines.get(next.checkpoint.round) ?? {
      humanCash: beforeHuman.cash_thousands,
      humanPosition: beforeHuman.position_value_thousands,
      computerCash: computerPlayer(before).cash_thousands,
    };
    const humanCashDelta = nullableDelta(afterHuman.cash_thousands - baseline.humanCash);
    const humanPositionDelta = nullableDelta(
      afterHuman.position_value_thousands - baseline.humanPosition,
    );
    const computerCashDelta = nullableDelta(afterComputer.cash_thousands - baseline.computerCash);
    expect(afterHuman.cash_delta_thousands).toBe(humanCashDelta);
    expect(afterHuman.position_delta_thousands).toBe(humanPositionDelta);
    expect(afterComputer.cash_delta_thousands).toBe(computerCashDelta);
    await expectPlayerMetric(harness.page, "human", "cash", afterHuman.cash_thousands, humanCashDelta);
    await expectPlayerMetric(
      harness.page,
      "human",
      "position",
      afterHuman.position_value_thousands,
      humanPositionDelta,
    );
    await expectPlayerMetric(
      harness.page,
      "computer",
      "cash",
      afterComputer.cash_thousands,
      computerCashDelta,
    );
    harness.coverage.sawDemandMetricSettlement = true;
  }
  return next;
}

async function acknowledge(harness: GameHarness) {
  const checkpoint = harness.view.checkpoint;
  expect(checkpoint).not.toBeNull();
  const checkpointView = harness.view;
  if (checkpoint!.kind === "demand_result") {
    const resolved = harness.page.locator('article[data-stockpile-resolved="true"]');
    await expect(resolved).toHaveCount(4);
    const isolation = await resolved.evaluateAll((piles) => piles.map((pile) => {
      const stack = pile.querySelector<HTMLElement>("[data-stockpile-stack]")!;
      const bid = pile.querySelector<HTMLElement>("[data-stockpile-bid]")!;
      const bidStyle = getComputedStyle(bid);
      return {
        articleWhiteOut: pile.hasAttribute("data-white-out"),
        stackWhiteOut: stack.getAttribute("data-white-out"),
        stackOverlay: getComputedStyle(stack, "::after").backgroundColor,
        bidWhiteOut: bid.hasAttribute("data-white-out"),
        bidColor: bidStyle.color,
        bidOpacity: bidStyle.opacity,
      };
    }));
    expect(isolation).toHaveLength(4);
    expect(isolation.every((item) => (
      !item.articleWhiteOut
      && item.stackWhiteOut === "true"
      && item.stackOverlay === "rgba(255, 255, 255, 0.68)"
      && !item.bidWhiteOut
      && item.bidColor === "rgb(17, 17, 17)"
      && item.bidOpacity === "1"
    ))).toBe(true);
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
  if (
    checkpoint!.kind === "demand_result"
    && Object.values(checkpointView.configuration.options).every((enabled) => !enabled)
  ) {
    const beforeHuman = humanPlayer(checkpointView);
    const afterHuman = humanPlayer(next);
    const beforeComputer = computerPlayer(checkpointView);
    const afterComputer = computerPlayer(next);
    expect(afterHuman.cash_delta_thousands).toBe(beforeHuman.cash_delta_thousands);
    expect(afterHuman.position_delta_thousands).toBe(beforeHuman.position_delta_thousands);
    expect(afterComputer.cash_delta_thousands).toBe(beforeComputer.cash_delta_thousands);
    await expectPlayerMetric(
      harness.page,
      "human",
      "cash",
      afterHuman.cash_thousands,
      afterHuman.cash_delta_thousands,
    );
    await expectPlayerMetric(
      harness.page,
      "human",
      "position",
      afterHuman.position_value_thousands,
      afterHuman.position_delta_thousands,
    );
    await expectPlayerMetric(
      harness.page,
      "computer",
      "cash",
      afterComputer.cash_thousands,
      afterComputer.cash_delta_thousands,
    );
    harness.coverage.sawDemandMetricPersistence = true;
    harness.coverage.sawIndependentMetricPreservation = true;
  }
  if (checkpoint!.kind === "round_result" && checkpoint!.round < checkpointView.total_rounds) {
    expect(next.round).toBe(checkpoint!.round + 1);
    await expectNoMetricDeltas(harness.page, next);
    harness.coverage.sawNewRoundMetricClear = true;
  }
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
    let action = harness.view.legal_actions[0];
    if (harness.view.configuration.options.sell_order && sales.length) {
      const currentCompany = harness.view.pending_decision.company_id;
      const hasLaterHumanHolding = currentCompany !== null && harness.view.private.holdings.some(
        (holding) => holding.shares_thousands > 0 && holding.company_id > currentCompany,
      );
      if (!harness.coverage.sawSaleMetricSettlement && hasLaterHumanHolding) {
        action = sales.find((candidate) => (
          candidate.sale_preview!.shares_thousands > 0
          && candidate.sale_preview!.resulting_shares_thousands > 0
        )) ?? sales.find((candidate) => candidate.sale_preview!.shares_thousands > 0) ?? action;
      } else if (
        harness.coverage.sawSaleMetricSettlement
        && !harness.coverage.sawHoldMetricClear
        && hasLaterHumanHolding
      ) {
        action = sales.find((candidate) => candidate.sale_preview!.shares_thousands === 0) ?? action;
      }
    }
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
  return [
    { kind: "demand_result" as const, round: 1 },
    { kind: "round_result" as const, round: 1 },
  ];
}

test("Home exposes only Trainer LITE and LITE+ through one button language", async ({ page }) => {
  await page.setViewportSize({ width: 1_280, height: 900 });
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

  await page.getByRole("button", { name: "LITE", exact: true }).click();
  const litePlayBox = await page.getByRole("button", { name: "PLAY", exact: true }).boundingBox();
  const liteColumnBox = await page.getByRole("button", { name: "LITE", exact: true }).boundingBox();
  expect(litePlayBox).not.toBeNull();
  expect(liteColumnBox).not.toBeNull();
  expect(litePlayBox!.x).toBe(liteColumnBox!.x);
  expect(litePlayBox!.width).toBe(liteColumnBox!.width);
  expect(litePlayBox!.y - (liteColumnBox!.y + liteColumnBox!.height)).toBeCloseTo(56, 2);

  await page.getByRole("button", { name: "LITE+", exact: true }).click();
  const featureNames = ["DIVIDEND", "FEES", "SELL ORDER"];
  for (const feature of featureNames) {
    const button = page.getByRole("button", { name: feature, exact: true });
    await expect(button).toBeVisible();
    await expect(button).toHaveAttribute("aria-pressed", "false");
  }
  await expect(page.getByRole("button", { name: "PLAY", exact: true })).toHaveCount(0);

  await page.getByRole("button", { name: "DIVIDEND", exact: true }).click();
  const litePlusPlayBox = await page.getByRole("button", { name: "PLAY", exact: true }).boundingBox();
  const litePlusModeBox = await page.getByRole("button", { name: "LITE+", exact: true }).boundingBox();
  const feesBox = await page.getByRole("button", { name: "FEES", exact: true }).boundingBox();
  const litePlusColumnBox = await page.getByRole("button", { name: "DIVIDEND", exact: true }).boundingBox();
  const sellOrderBox = await page.getByRole("button", { name: "SELL ORDER", exact: true }).boundingBox();
  expect(litePlusPlayBox).not.toBeNull();
  expect(litePlusModeBox).not.toBeNull();
  expect(feesBox).not.toBeNull();
  expect(litePlusColumnBox).not.toBeNull();
  expect(sellOrderBox).not.toBeNull();
  expect(litePlusPlayBox!.x).toBe(litePlusColumnBox!.x);
  expect(litePlusPlayBox!.width).toBe(litePlusColumnBox!.width);
  expect(feesBox!.x).toBeLessThan(litePlusColumnBox!.x);
  expect(litePlusColumnBox!.x).toBeLessThan(sellOrderBox!.x);
  expect(litePlusColumnBox!.y - (litePlusModeBox!.y + litePlusModeBox!.height)).toBeCloseTo(56, 2);
  expect(litePlusPlayBox!.y - (litePlusColumnBox!.y + litePlusColumnBox!.height)).toBeCloseTo(56, 2);
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

test("collapsed stacks keep their bottom anchor and small-card share typography is invariant", async ({ page, request }) => {
  await page.setViewportSize({ width: 1_280, height: 900 });
  const fixture = await openFixtureGame(page, request, (base) => {
    const stock = fixtureStock(base, 0);
    return {
      ...base,
      revision: 901,
      phase: "supply",
      phase_step: "supply",
      active_player_id: 0,
      checkpoint: null,
      decision_batch: null,
      legal_actions: [],
      pending_decision: {
        kind: "supply",
        prompt: "SUPPLY",
        selected_stockpile_id: null,
        selected_action_effect: null,
        company_id: null,
      },
      stockpiles: Array.from({ length: 4 }, (_, pileId) => ({
        ...base.stockpiles[pileId],
        stockpile_id: pileId,
        cards_bottom_to_top: Array.from({ length: pileId + 1 }, () => ({ ...stock })),
        bid: null,
        locked: false,
        purchaser_id: null,
        resolved: false,
      })),
      private: {
        ...base.private,
        holdings: [{
          company_id: 0,
          company: stock.company,
          shares_thousands: 3,
          price_dollars_per_share: base.companies[0].price_dollars_per_share,
          market_value_thousands: 3 * base.companies[0].price_dollars_per_share,
        }],
      },
      supply_batch: {
        cards: [
          { card_ref: "active-stock", card: stock },
          { card_ref: "active-stock-two", card: { ...stock } },
        ],
        plans: [],
      },
      terminal_results: null,
    };
  });
  expect(fixture.view.stockpiles.map((pile) => pile.cards_bottom_to_top.length)).toEqual([1, 2, 3, 4]);

  const geometry = await page.locator("article[data-stockpile-id]").evaluateAll((piles) => piles.map((pile) => {
    const pileBox = pile.getBoundingClientRect();
    const bottom = pile.querySelector<HTMLElement>('[data-stack-bottom="true"]')!;
    const top = pile.querySelector<HTMLElement>('[data-stack-top="true"]')!;
    const bottomBox = bottom.getBoundingClientRect();
    const topBox = top.getBoundingClientRect();
    return {
      depth: pile.querySelectorAll("[data-stack-card]").length,
      bottomX: Math.round((bottomBox.x - pileBox.x) * 100) / 100,
      bottomY: Math.round((bottomBox.y - pileBox.y) * 100) / 100,
      topDx: Math.round((topBox.x - bottomBox.x) * 100) / 100,
      topDy: Math.round((topBox.y - bottomBox.y) * 100) / 100,
    };
  }));
  expect(geometry.map((item) => item.depth)).toEqual([1, 2, 3, 4]);
  for (const item of geometry) {
    // CSS grid tracks can land on adjacent browser subpixels. The physical
    // anchor is invariant when both axes stay within one twentieth of a px.
    expect(Math.abs(item.bottomX - geometry[0].bottomX), JSON.stringify(geometry)).toBeLessThanOrEqual(0.05);
    expect(Math.abs(item.bottomY - geometry[0].bottomY), JSON.stringify(geometry)).toBeLessThanOrEqual(0.05);
  }
  const stepX = geometry[1].topDx;
  const stepY = geometry[1].topDy;
  expect(stepX).toBeGreaterThan(0);
  expect(stepY).toBeGreaterThan(0);
  for (const item of geometry) {
    expect(item.topDx).toBeCloseTo(stepX * (item.depth - 1), 2);
    expect(item.topDy).toBeCloseTo(stepY * (item.depth - 1), 2);
  }

  const activeValue = page.locator('[data-card-scale="active"] [data-card-value]').first();
  const portfolioValue = page.locator('[data-card-scale="portfolio"] [data-card-value]').first();
  await expect(activeValue).toHaveText("1K");
  await expect(portfolioValue).toHaveText("3K");
  const typography = await Promise.all([activeValue, portfolioValue].map((locator) => locator.evaluate((element) => {
    const style = getComputedStyle(element);
    return {
      family: style.fontFamily,
      size: style.fontSize,
      weight: style.fontWeight,
      lineHeight: style.lineHeight,
      letterSpacing: style.letterSpacing,
    };
  })));
  expect(typography[0]).toEqual(typography[1]);
});

test("Selling shows the current holding card and only its HOLD and SELL choices", async ({ page, request }) => {
  const fixture = await openFixtureGame(page, request, (base) => {
    const company = base.companies[5];
    return {
      ...base,
      revision: 902,
      phase: "selling",
      phase_step: "selling",
      active_player_id: 0,
      checkpoint: null,
      supply_batch: null,
      decision_batch: null,
      pending_decision: {
        kind: "sell",
        prompt: "SELL",
        selected_stockpile_id: null,
        selected_action_effect: null,
        company_id: company.company_id,
      },
      private: {
        ...base.private,
        holdings: [{
          company_id: company.company_id,
          company: company.name,
          shares_thousands: 3,
          price_dollars_per_share: company.price_dollars_per_share,
          market_value_thousands: 3 * company.price_dollars_per_share,
        }],
      },
      legal_actions: [
        {
          action_id: 700,
          control: "sell",
          label: `Hold ${company.name}`,
          target_id: `company:${company.company_id}`,
          amount_thousands: 0,
          direction: null,
          sale_preview: {
            company_id: company.company_id,
            company: company.name,
            shares_thousands: 0,
            price_dollars_per_share: company.price_dollars_per_share,
            gross_value_thousands: 0,
            resulting_shares_thousands: 3,
          },
        },
        {
          action_id: 701,
          control: "sell",
          label: `Sell 1K ${company.name}`,
          target_id: `company:${company.company_id}`,
          amount_thousands: company.price_dollars_per_share,
          direction: null,
          sale_preview: {
            company_id: company.company_id,
            company: company.name,
            shares_thousands: 1,
            price_dollars_per_share: company.price_dollars_per_share,
            gross_value_thousands: company.price_dollars_per_share,
            resulting_shares_thousands: 2,
          },
        },
      ],
      terminal_results: null,
    };
  });
  const companyId = fixture.view.pending_decision.company_id!;
  const company = fixture.view.companies.find((candidate) => candidate.company_id === companyId)!;
  const dock = page.getByLabel("Action dock");
  const sellingCard = dock.locator(`[data-selling-company-id="${companyId}"]`);
  await expect(sellingCard).toHaveCount(1);
  await expect(sellingCard.locator('[data-card-scale="active"]')).toHaveAttribute(
    "aria-label",
    `${company.display_name} holding 3K shares`,
  );
  await expect(sellingCard.locator("[data-card-value]")).toHaveText("3K");
  await expect(dock.getByText("HOLD", { exact: true })).toHaveCount(1);
  await expect(dock.getByText("SELL 1K", { exact: true })).toHaveCount(1);
  await expect(dock.locator('[data-selling-company-id]:not([data-selling-company-id="5"])')).toHaveCount(0);
});

test("Market keeps the latest delta per company until a real new round", async ({ page, request }) => {
  const fixture = await openFixtureGame(page, request, (base) => ({
    ...base,
    revision: 903,
    companies: base.companies.map((company) => ({
      ...company,
      price_delta_dollars_per_share: company.company_id === 0 ? 2 : company.company_id === 1 ? -1 : null,
    })),
  }));
  const delta = (companyId: number) => page.locator(`[data-market-price-delta="${companyId}"]`);
  await expect(delta(0)).toHaveText("↑2");
  await expect(delta(1)).toHaveText("↓1");

  await page.waitForTimeout(2_600);
  await expect(delta(0)).toHaveText("↑2");
  await expect(delta(1)).toHaveText("↓1");

  const latestA: GameView = {
    ...fixture.view,
    revision: 904,
    phase: "selling",
    phase_step: "selling",
    companies: fixture.view.companies.map((company) => ({
      ...company,
      price_delta_dollars_per_share: company.company_id === 0 ? -3 : company.company_id === 1 ? -1 : null,
    })),
  };
  await fixture.update(latestA);
  await expect(delta(0)).toHaveText("↓3");
  await expect(delta(1)).toHaveText("↓1");

  const zeroAAndNewC: GameView = {
    ...latestA,
    revision: 905,
    phase: "movement",
    phase_step: "movement",
    companies: latestA.companies.map((company) => ({
      ...company,
      price_delta_dollars_per_share: company.company_id === 0 ? null : company.company_id === 1 ? -1 : company.company_id === 2 ? 4 : null,
    })),
  };
  await fixture.update(zeroAAndNewC);
  await expect(delta(0)).toHaveCount(0);
  await expect(delta(1)).toHaveText("↓1");
  await expect(delta(2)).toHaveText("↑4");

  const nextRound: GameView = {
    ...zeroAAndNewC,
    revision: 906,
    round: zeroAAndNewC.round + 1,
    phase: "supply",
    phase_step: "supply",
    companies: zeroAAndNewC.companies.map((company) => ({
      ...company,
      price_delta_dollars_per_share: null,
    })),
  };
  await fixture.update(nextRound);
  await expect(page.locator("[data-market-price-delta]")).toHaveCount(0);
});

test("Market and Players reserve fixed delta columns whether values are empty or populated", async ({ page, request }) => {
  await page.setViewportSize({ width: 1_280, height: 900 });
  const fixture = await openFixtureGame(page, request, (base) => ({
    ...base,
    revision: 907,
    companies: base.companies.map((company) => ({
      ...company,
      price_dollars_per_share: company.company_id === 0 ? 3 : company.company_id === 1 ? 47 : company.company_id === 2 ? 109 : company.price_dollars_per_share,
      price_delta_dollars_per_share: company.company_id === 0 ? null : company.company_id === 1 ? -12 : company.company_id === 2 ? 4 : null,
    })),
    players: base.players.map((player) => player.role === "human" ? {
      ...player,
      cash_thousands: 3,
      cash_delta_thousands: null,
      position_value_thousands: 141,
      position_delta_thousands: 6,
    } : {
      ...player,
      cash_thousands: 99,
      cash_delta_thousands: -4,
    }),
  }));

  type AnchorSnapshot = {
    market: Array<{ key: string; priceRight: number; deltaX: number; deltaWidth: number }>;
    players: Array<{ key: string; valueRight: number; deltaX: number; deltaWidth: number }>;
  };
  const anchors = () => page.evaluate<AnchorSnapshot>(() => ({
    market: Array.from(document.querySelectorAll<HTMLElement>("[data-company-id]")).map((row) => {
      const priceSlot = row.querySelector<HTMLElement>("[data-market-price-value]")!;
      const deltaSlot = row.querySelector<HTMLElement>("[data-market-delta-slot]")!;
      const priceBox = priceSlot.getBoundingClientRect();
      const deltaBox = deltaSlot.getBoundingClientRect();
      return {
        key: row.dataset.companyId!,
        priceRight: priceBox.right,
        deltaX: deltaBox.x,
        deltaWidth: deltaBox.width,
      };
    }),
    players: Array.from(document.querySelectorAll<HTMLElement>("[data-player-metric]")).map((row) => {
      const player = row.closest<HTMLElement>("[data-player-role]")!;
      const valueSlot = row.querySelector<HTMLElement>("[data-player-value-slot]")!;
      const deltaSlot = row.querySelector<HTMLElement>("[data-player-delta-slot]")!;
      const valueBox = valueSlot.getBoundingClientRect();
      const deltaBox = deltaSlot.getBoundingClientRect();
      return {
        key: `${player.dataset.playerRole}:${row.dataset.playerMetric}`,
        valueRight: valueBox.right,
        deltaX: deltaBox.x,
        deltaWidth: deltaBox.width,
      };
    }),
  }));
  const assertAligned = (snapshot: AnchorSnapshot) => {
    expect(new Set(snapshot.market.map((item) => item.priceRight)).size).toBe(1);
    expect(new Set(snapshot.market.map((item) => item.deltaX)).size).toBe(1);
    expect(new Set(snapshot.market.map((item) => item.deltaWidth)).size).toBe(1);
    expect(new Set(snapshot.players.map((item) => item.valueRight)).size).toBe(1);
    expect(new Set(snapshot.players.map((item) => item.deltaX)).size).toBe(1);
    expect(new Set(snapshot.players.map((item) => item.deltaWidth)).size).toBe(1);
  };

  await expect(page.locator('[data-market-delta-slot="0"]')).toBeEmpty();
  await expect(page.locator('[data-market-delta-slot="1"]')).toHaveText("↓12");
  await expect(page.locator('[data-player-role="human"] [data-player-metric="cash"] [data-player-delta-slot]')).toBeEmpty();
  const before = await anchors();
  assertAligned(before);

  const next: GameView = {
    ...fixture.view,
    revision: 908,
    companies: fixture.view.companies.map((company) => ({
      ...company,
      price_dollars_per_share: company.company_id === 0 ? 123 : company.company_id === 1 ? 4 : company.price_dollars_per_share,
      price_delta_dollars_per_share: company.company_id === 0 ? 8 : company.company_id === 1 ? null : company.company_id === 2 ? -2 : null,
    })),
    players: fixture.view.players.map((player) => player.role === "human" ? {
      ...player,
      cash_thousands: 123,
      cash_delta_thousands: 17,
      position_value_thousands: 4,
      position_delta_thousands: null,
    } : {
      ...player,
      cash_thousands: 7,
      cash_delta_thousands: null,
    }),
  };
  await fixture.update(next);
  await expect(page.locator('[data-market-delta-slot="0"]')).toHaveText("↑8");
  await expect(page.locator('[data-market-delta-slot="1"]')).toBeEmpty();
  await expect(page.locator('[data-player-role="human"] [data-player-metric="cash"] [data-player-delta-slot]')).toHaveText("+$17K");
  await expect(page.locator('[data-player-role="human"] [data-player-metric="position"] [data-player-delta-slot]')).toBeEmpty();
  const after = await anchors();
  assertAligned(after);

  for (const prior of before.market) {
    const current = after.market.find((item) => item.key === prior.key)!;
    expect(Math.abs(current.priceRight - prior.priceRight), `${prior.key}:priceRight`).toBeLessThanOrEqual(0.05);
    expect(Math.abs(current.deltaX - prior.deltaX), `${prior.key}:deltaX`).toBeLessThanOrEqual(0.05);
    expect(Math.abs(current.deltaWidth - prior.deltaWidth), `${prior.key}:deltaWidth`).toBeLessThanOrEqual(0.05);
  }
  for (const prior of before.players) {
    const current = after.players.find((item) => item.key === prior.key)!;
    expect(Math.abs(current.valueRight - prior.valueRight), `${prior.key}:valueRight`).toBeLessThanOrEqual(0.05);
    expect(Math.abs(current.deltaX - prior.deltaX), `${prior.key}:deltaX`).toBeLessThanOrEqual(0.05);
    expect(Math.abs(current.deltaWidth - prior.deltaWidth), `${prior.key}:deltaWidth`).toBeLessThanOrEqual(0.05);
  }
});

test("Bankruptcy stays on Round Result until CONTINUE, then reveals the reset market", async ({ page, request }) => {
  await page.setViewportSize({ width: 1_280, height: 900 });
  let afterAcknowledgement!: GameView;
  const fixture = await openFixtureGame(page, request, (base) => {
    const company = base.companies[0];
    afterAcknowledgement = {
      ...base,
      revision: 910,
      round: base.round + 1,
      phase: "supply",
      phase_step: "supply",
      checkpoint: null,
      companies: base.companies.map((candidate) => candidate.company_id === company.company_id ? {
        ...candidate,
        price_dollars_per_share: 5,
        price_delta_dollars_per_share: null,
      } : { ...candidate, price_delta_dollars_per_share: null }),
      private: {
        ...base.private,
        holdings: base.private.holdings.map((holding) => holding.company_id === company.company_id ? {
          ...holding,
          shares_thousands: 0,
          price_dollars_per_share: 5,
          market_value_thousands: 0,
        } : holding),
      },
      recent_events: [],
    };
    return {
      ...base,
      revision: 909,
      phase: "ROUND_RESULT",
      phase_step: "acknowledge",
      active_player_id: null,
      companies: base.companies.map((candidate) => candidate.company_id === company.company_id ? {
        ...candidate,
        price_dollars_per_share: 0,
        price_delta_dollars_per_share: -1,
      } : candidate),
      private: {
        ...base.private,
        holdings: base.private.holdings.map((holding) => holding.company_id === company.company_id ? {
          ...holding,
          shares_thousands: 1,
          price_dollars_per_share: 0,
          market_value_thousands: 0,
        } : holding),
      },
      pending_decision: {
        kind: "acknowledge",
        prompt: "CONTINUE",
        selected_stockpile_id: null,
        selected_action_effect: null,
        company_id: null,
      },
      legal_actions: [],
      supply_batch: null,
      decision_batch: null,
      checkpoint: {
        checkpoint_id: "bankruptcy-round-result",
        kind: "round_result",
        round: base.round,
      },
      recent_events: [
        ...base.recent_events,
        {
          event_id: 909,
          event_type: "bankruptcy",
          cause: "market_forecast",
          round: base.round,
          company_id: company.company_id,
          prior_price_dollars_per_share: 1,
          price_delta: -1,
          resulting_price_dollars_per_share: 0,
          forecast: -1,
          cash_effect_thousands: null,
          direction: "down",
        },
      ],
      terminal_results: null,
    };
  });

  const companyId = 0;
  const bankruptRow = page.locator(`[data-bankrupt-company="${companyId}"]`);
  await expect(bankruptRow).toHaveCount(1);
  await expect(bankruptRow.locator(`[data-market-company-name="${companyId}"]`)).toHaveAttribute("data-white-out", "true");
  await expect(bankruptRow.locator(`[data-market-price-value="${companyId}"]`)).toHaveAttribute("data-white-out", "true");
  await expect(bankruptRow.locator(`[data-market-price-value="${companyId}"]`)).toHaveText("$0 / SHARE");
  await expect(bankruptRow.locator(`[data-market-delta-slot="${companyId}"]`)).toHaveText("↓1");
  await expect(bankruptRow.locator(`[data-market-delta-slot="${companyId}"]`)).not.toHaveAttribute("data-white-out", "true");
  await expect(page.locator(`[data-portfolio-company-id="${companyId}"]`)).toHaveAttribute("data-white-out", "true");

  let acknowledgementBody: unknown;
  await page.route(`**/api/v2/games/${fixture.gameId}/acknowledgements`, async (route) => {
    acknowledgementBody = route.request().postDataJSON();
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      headers: { "Cache-Control": "no-store" },
      body: JSON.stringify(afterAcknowledgement),
    });
  });
  await page.locator('[data-context-action="continue"]').click();
  await expect(page.locator(`[data-bankrupt-company="${companyId}"]`)).toHaveCount(0);
  await expect(page.locator(`[data-market-price-value="${companyId}"]`)).toHaveText("$5 / SHARE");
  await expect(page.locator(`[data-market-delta-slot="${companyId}"]`)).toBeEmpty();
  await expect(page.locator(`[data-portfolio-company-id="${companyId}"]`)).toHaveCount(0);
  expect(acknowledgementBody).toEqual({
    checkpoint_id: "bankruptcy-round-result",
    expected_revision: 909,
  });
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

test("default seed 101 completes one round through Demand and Round acknowledgements", async ({ page, request }) => {
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

    expect(harness.coverage.demandPurchasers.size).toBe(2);
    for (const purchasers of harness.coverage.demandPurchasers.values()) {
      expect(purchasers).toHaveLength(4);
      expect(purchasers.filter((playerId) => playerId === 0)).toHaveLength(2);
      expect(purchasers.filter((playerId) => playerId === 1)).toHaveLength(2);
    }
    expect(harness.coverage.views.every((view) => view.viewer.name === "YOU")).toBe(true);
    expect(harness.coverage.views.every((view) => view.viewer.player_id === harness.view.viewer.player_id)).toBe(true);
    expect(harness.coverage.views.every((view) => view.stockpiles.length === 4)).toBe(true);
    expect(harness.coverage.views.every((view) => view.players.every((player) => player.bid_markers.length === 2))).toBe(true);
    expect(harness.coverage.backChecked).toBe(true);
    expect(harness.coverage.sawDemandWhiteOut).toBe(true);
    expect(harness.coverage.sawResearchWhiteOut).toBe(true);
    expect(harness.coverage.sawDeduplicatedSales).toBe(true);
    expect(harness.coverage.sawDemandMetricSettlement).toBe(true);
    expect(harness.coverage.sawDemandMetricPersistence).toBe(true);
    expect(harness.coverage.sawIndependentMetricPreservation).toBe(true);

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
      const row = chart.locator(`[data-chart-player="${result.player_id === terminal.viewer.player_id ? "human" : "computer"}"]`);
      await expect(row.locator("[data-chart-position]")).toHaveAttribute(
        "data-chart-position",
        String(result.liquidation_value_thousands),
      );
      await expect(row.locator("[data-chart-cash]")).toHaveAttribute(
        "data-chart-cash",
        String(result.cash_before_liquidation_thousands),
      );
    }
    const computerResult = terminal.terminal_results!.players.find(
      (player) => player.player_id !== terminal.viewer.player_id,
    )!;
    if (computerResult.liquidation_value_thousands > 0) {
      await expect(
        page.getByLabel("Players").locator('[data-player-role="computer"] [data-player-metric="position"]'),
      ).toContainText(`$${computerResult.liquidation_value_thousands}K`);
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
    expect(harness.coverage.demandPurchasers.size).toBe(2);
    expect(harness.coverage.sawSaleMetricSettlement).toBe(true);
    expect(harness.coverage.sawHoldMetricClear).toBe(true);
    expect(harness.coverage.sawIndependentMetricPreservation).toBe(true);
  } finally {
    await awaitAudits(harness);
  }
});
