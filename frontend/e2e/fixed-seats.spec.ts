import { expect, test } from "@playwright/test";
import type { APIRequestContext, Browser, BrowserContext, Page } from "@playwright/test";

const defaultOptions = {
  market_impact: false,
  starting_share: false,
  trading_fees: false,
  dividends: false,
  sell_order: false,
};

const visibleFeatureOptions = {
  market_impact: true,
  starting_share: false,
  trading_fees: true,
  dividends: true,
  sell_order: true,
};

const companyPresentation = [
  ["COSMIC", "matrix"],
  ["BOTTOMLINE", "ledger"],
  ["LEADING", "molecular"],
  ["AMERICAN", "chevron"],
  ["STANFORD", "crosshatch"],
  ["EPIC", "wave"],
] as const;

const workstationRegions = [
  "Status",
  "Market",
  "Private information",
  "Public information",
  "Stockpiles",
  "Portfolio",
  "Players",
  "Action dock",
] as const;

type WireAction = {
  action_id: number;
  control: string;
  target_id: string | null;
  amount: number | null;
  sale_preview: { quantity: number } | null;
};

type WireMarker = {
  player_id: number;
  marker_index: number;
  status: string;
  stockpile_id: number | null;
  bid: number | null;
};

type WireView = {
  revision: number;
  round: number;
  total_rounds: number;
  phase: string;
  viewer: { player_id: number; name: string };
  active_player_id: number | null;
  companies: Array<{
    company_id: number;
    display_name: string;
    pattern: string;
    price: number;
    color: string;
  }>;
  stockpiles: Array<{
    stockpile_id: number;
    purchaser_id: number | null;
    visible_cards: Array<Record<string, unknown>>;
    hidden_cards: Array<Record<string, unknown>>;
    marker: WireMarker | null;
  }>;
  players: Array<{
    player_id: number;
    name: string;
    cash: number;
    active: boolean;
    status: string;
    bid_markers: WireMarker[];
  }>;
  private: {
    hand: Array<{ kind: string; effect?: string }>;
    market_information: Array<{
      visibility: "private" | "public" | "hidden";
      card: { visibility: "hidden" | "visible"; company_id?: number; forecast?: number | string };
    }>;
    holdings: unknown[];
    known_pile_cards: unknown[];
    available_action_cards: unknown[];
  };
  pending_decision: { kind: string; selected_card_index: number | null };
  legal_actions: WireAction[];
  recent_events: Array<{
    event_type: string;
    company_id: number | null;
    actual_delta: number | null;
    resulting_price: number | null;
  }>;
  terminal_results: null | {
    players: Array<{ player_id: number; final_cash: number; winner: boolean }>;
  };
};

function privateForecastPairs(view: WireView) {
  return new Set(
    view.private.market_information
      .filter((slot) => slot.visibility === "private" && slot.card.visibility === "visible")
      .map((slot) => `${slot.card.company_id}:${slot.card.forecast}`),
  );
}

type Coverage = {
  phases: Set<string>;
  decisions: Set<string>;
  activePlayersBySeat: Array<Set<number>>;
  markerIndicesByPlayer: Map<number, Set<number>>;
  outbidMarkers: Set<string>;
  sawOutbid: boolean;
  sawRebid: boolean;
  sawPriceAboveTen: boolean;
  sawRenderedPriceAboveTen: boolean;
  settledRounds: Map<number, number[]>;
  privacyFailures: string[];
  capturedViews: WireView[][];
  viewsByRevision: Map<number, Array<WireView | undefined>>;
  sawOffsetStack: boolean;
  sawStockpileDominance: boolean;
  sawPortfolioScale: boolean;
};

type SeatTable = {
  context: BrowserContext;
  pages: Page[];
  views: Array<WireView | null>;
  coverage: Coverage;
};

function markerKey(marker: WireMarker) {
  return `${marker.player_id}:${marker.marker_index}`;
}

function recordView(view: WireView, coverage: Coverage, expectedSeat: number) {
  coverage.capturedViews[expectedSeat].push(view);
  if (view.viewer.player_id !== expectedSeat) {
    coverage.privacyFailures.push(`Seat ${expectedSeat} received viewer ${view.viewer.player_id}`);
  }

  coverage.phases.add(view.phase.toLowerCase());
  if (view.recent_events.some((event) => event.event_type === "market_movement" || event.event_type === "market_reveal")) {
    coverage.phases.add("movement");
  }
  coverage.decisions.add(view.pending_decision.kind);
  if (view.active_player_id !== null) coverage.activePlayersBySeat[expectedSeat].add(view.active_player_id);

  const expectedPrivateKeys = [
    "available_action_cards",
    "hand",
    "holdings",
    "known_pile_cards",
    "market_information",
  ];
  if (JSON.stringify(Object.keys(view.private).sort()) !== JSON.stringify(expectedPrivateKeys)) {
    coverage.privacyFailures.push(`Seat ${expectedSeat} private payload changed its allowlist`);
  }

  for (const player of view.players) {
    if (view.players.length === 2 && player.bid_markers.length !== 2) {
      coverage.privacyFailures.push(`Player ${player.player_id} exposed ${player.bid_markers.length} bid positions`);
    }
    const indices = coverage.markerIndicesByPlayer.get(player.player_id) ?? new Set<number>();
    for (const marker of player.bid_markers) {
      indices.add(marker.marker_index);
      const key = markerKey(marker);
      if (marker.status === "outbid") {
        coverage.sawOutbid = true;
        coverage.outbidMarkers.add(key);
      }
      if (marker.status === "rebidding" || (coverage.outbidMarkers.has(key) && ["placed", "locked"].includes(marker.status))) {
        coverage.sawRebid = true;
      }
    }
    coverage.markerIndicesByPlayer.set(player.player_id, indices);
  }

  coverage.sawPriceAboveTen ||=
    view.companies.some((company) => company.price > 10)
    || view.recent_events.some((event) => (event.resulting_price ?? 0) > 10);

  for (const pile of view.stockpiles) {
    for (const hidden of pile.hidden_cards) {
      if (Object.keys(hidden).length !== 1 || hidden.visibility !== "hidden") {
        coverage.privacyFailures.push("Hidden Stockpile card contained extra fields");
      }
    }
  }
  for (const slot of view.private.market_information) {
    if (slot.card.visibility === "hidden" && Object.keys(slot.card).length !== 1) {
      coverage.privacyFailures.push("Hidden information card contained extra fields");
    }
  }

  if (view.stockpiles.length > 0 && view.stockpiles.every((pile) => pile.purchaser_id !== null)) {
    coverage.settledRounds.set(view.round, view.stockpiles.map((pile) => pile.purchaser_id!));
  }

  const revisionViews = coverage.viewsByRevision.get(view.revision) ?? [];
  revisionViews[expectedSeat] = view;
  coverage.viewsByRevision.set(view.revision, revisionViews);
  if (revisionViews[0] && revisionViews[1]) {
    const left = privateForecastPairs(revisionViews[0]);
    const right = privateForecastPairs(revisionViews[1]);
    const overlap = [...left].filter((pair) => right.has(pair));
    if (overlap.length > 0) {
      coverage.privacyFailures.push(`Revision ${view.revision} transmitted cross-seat private pairs: ${overlap.join(", ")}`);
    }
  }

  const wire = JSON.stringify(view).toLowerCase();
  for (const forbidden of [
    "information_state_id",
    "information_state_tensor",
    "raw_history",
    "correlation",
    "card_id",
  ]) {
    if (wire.includes(forbidden)) coverage.privacyFailures.push(`Response exposed ${forbidden}`);
  }
  if (view.phase.toLowerCase() === "selling" && view.active_player_id === null && view.players.some((player) => player.active)) {
    coverage.privacyFailures.push("Sealed selling exposed an active player");
  }
}

function newCoverage(): Coverage {
  return {
    phases: new Set(),
    decisions: new Set(),
    activePlayersBySeat: [new Set(), new Set()],
    markerIndicesByPlayer: new Map(),
    outbidMarkers: new Set(),
    sawOutbid: false,
    sawRebid: false,
    sawPriceAboveTen: false,
    sawRenderedPriceAboveTen: false,
    settledRounds: new Map(),
    privacyFailures: [],
    capturedViews: [[], []],
    viewsByRevision: new Map(),
    sawOffsetStack: false,
    sawStockpileDominance: false,
    sawPortfolioScale: false,
  };
}

async function createSeatTable(
  browser: Browser,
  request: APIRequestContext,
  configuration: { round_count: number; seed: number; options: typeof defaultOptions },
): Promise<SeatTable> {
  const created = await request.post("/api/v1/games", {
    data: {
      player_count: 2,
      player_names: ["Ada", "Lin"],
      ...configuration,
    },
  });
  expect(created.status()).toBe(201);
  const game = await created.json() as { game_id: string; seats: Array<{ url: string }> };

  const context = await browser.newContext({ viewport: { width: 1_280, height: 900 } });
  const pages = await Promise.all([context.newPage(), context.newPage()]);
  const views: Array<WireView | null> = [null, null];
  const coverage = newCoverage();

  for (const [seat, page] of pages.entries()) {
    page.on("response", (response) => {
      if (
        !response.url().includes("/api/v1/games/")
        || (!response.url().endsWith("/view") && !response.url().endsWith("/actions"))
        || !response.ok()
      ) return;
      void response.json()
        .then((body: WireView) => {
          views[seat] = body;
          recordView(body, coverage, seat);
        })
        .catch(() => {
          coverage.privacyFailures.push(`Seat ${seat} response could not be inspected`);
        });
    });
  }

  await Promise.all(pages.map(async (page, seat) => {
    const initial = page.waitForResponse((response) => response.url().endsWith("/view") && response.ok());
    await page.goto(game.seats[seat].url);
    views[seat] = await (await initial).json() as WireView;
    recordView(views[seat]!, coverage, seat);

    const seatUrl = new URL(game.seats[seat].url, "http://127.0.0.1:5173");
    const gameId = seatUrl.pathname.split("/").at(-1)!;
    const expectedToken = seatUrl.hash.replace(/^#seat=/, "");
    const storedToken = await page.evaluate(
      (key) => sessionStorage.getItem(key),
      `stockpile.seatToken:${gameId}`,
    );
    expect(storedToken).toBe(expectedToken);
    expect(new URL(page.url()).hash).toBe("");
  }));

  const tokens = await Promise.all(pages.map((page) => page.evaluate(
    (gameId) => sessionStorage.getItem(`stockpile.seatToken:${gameId}`),
    game.game_id,
  )));
  expect(tokens[0]).toBeTruthy();
  expect(tokens[0]).not.toBe(tokens[1]);
  return { context, pages, views, coverage };
}

async function refreshSeat(table: SeatTable, seat: number) {
  const response = table.pages[seat].waitForResponse(
    (candidate) => candidate.url().endsWith("/view") && candidate.ok(),
    { timeout: 4_000 },
  );
  await table.pages[seat].evaluate(() => window.dispatchEvent(new Event("focus")));
  table.views[seat] = await (await response).json() as WireView;
  recordView(table.views[seat]!, table.coverage, seat);
}

async function recordVisualCoverage(table: SeatTable) {
  if (
    table.coverage.sawOffsetStack
    && table.coverage.sawStockpileDominance
    && table.coverage.sawPortfolioScale
    && table.coverage.sawRenderedPriceAboveTen
  ) return;

  for (const [seat, page] of table.pages.entries()) {
    const observation = await page.evaluate(() => {
      const width = (selector: string) => document.querySelector<HTMLElement>(selector)?.getBoundingClientRect().width ?? null;
      let exposedStack = false;
      for (const pile of document.querySelectorAll<HTMLElement>("article[data-stockpile-id]")) {
        const cards = [...pile.querySelectorAll<HTMLElement>("[data-stack-card]")];
        if (cards.length < 2) continue;
        const positions = cards.map((card) => {
          const box = card.getBoundingClientRect();
          return { x: box.x, y: box.y };
        });
        exposedStack = positions.slice(1).every((position, index) => {
          const previous = positions[index];
          return Math.abs(position.x - previous.x) + Math.abs(position.y - previous.y) >= 4;
        });
        if (exposedStack) break;
      }
      const stockpile = width('[data-card-scale="stockpile"]');
      const active = width('[data-card-scale="active"]');
      const information = width('[data-card-scale="information"]');
      const portfolio = width('[data-card-scale="portfolio"]');
      const secondary = active ?? information;
      return {
        exposedStack,
        stockpileDominates: stockpile !== null && secondary !== null && stockpile > secondary * 1.2,
        portfolioIsSmaller: stockpile !== null && portfolio !== null && stockpile > portfolio * 1.2,
      };
    });

    table.coverage.sawOffsetStack ||= observation.exposedStack;
    table.coverage.sawStockpileDominance ||= observation.stockpileDominates;
    table.coverage.sawPortfolioScale ||= observation.portfolioIsSmaller;

    const pricedCompany = table.views[seat]?.companies.find((company) => company.price > 10);
    if (pricedCompany) {
      const market = page.getByLabel("Market");
      try {
        await expect(market.getByText(pricedCompany.display_name, { exact: true })).toBeVisible({ timeout: 1_000 });
        await expect(market.getByText(String(pricedCompany.price), { exact: true }).first()).toBeVisible({ timeout: 1_000 });
        table.coverage.sawRenderedPriceAboveTen = true;
      } catch {
        // A newer response may already have replaced the transient price; later observations retry.
      }
    }
  }
}

async function clickAction(table: SeatTable, seat: number, action: WireAction) {
  const button = table.pages[seat].locator(`[data-action-id="${action.action_id}"]`);
  await expect(button).toBeVisible();
  await expect(button).toBeEnabled();
  const responsePromise = table.pages[seat].waitForResponse(
    (candidate) => candidate.url().endsWith("/actions") && candidate.request().method() === "POST",
  );
  await button.click();
  const response = await responsePromise;
  expect(response.ok()).toBe(true);
  expect(Object.keys(response.request().postDataJSON() as Record<string, unknown>).sort()).toEqual([
    "action_id",
    "expected_revision",
  ]);
  table.views[seat] = await response.json() as WireView;
  recordView(table.views[seat]!, table.coverage, seat);

  for (let other = 0; other < table.pages.length; other += 1) {
    if (other !== seat) await refreshSeat(table, other);
  }
  await recordVisualCoverage(table);
}

function firstAction(view: WireView): WireAction {
  return view.legal_actions[0];
}

function allOptionsAction(view: WireView): WireAction {
  if (view.pending_decision.kind === "supply_card") {
    const boom = view.legal_actions.find((action) => {
      const index = Number(action.target_id?.split(":")[1]);
      const card = view.private.hand[index];
      return card?.kind === "action" && card.effect?.toLowerCase() === "boom";
    });
    if (boom) return boom;
  }
  if (view.legal_actions[0]?.control === "stockpile") {
    const preferred = view.viewer.player_id === 0 ? 1 : 0;
    return view.legal_actions.find((action) => action.target_id === `stockpile:${preferred}`) ?? firstAction(view);
  }
  if (view.pending_decision.kind === "action_card") {
    return view.legal_actions.find((action) => action.target_id === "action:boom") ?? firstAction(view);
  }
  if (view.pending_decision.kind === "action_company") {
    const descending = [...view.companies].sort((left, right) => right.price - left.price);
    for (const company of descending) {
      const action = view.legal_actions.find((candidate) => candidate.target_id === `company:${company.company_id}`);
      if (action) return action;
    }
  }
  if (view.pending_decision.kind === "sell") {
    return [...view.legal_actions].sort(
      (left, right) => (right.sale_preview?.quantity ?? 0) - (left.sale_preview?.quantity ?? 0),
    )[0];
  }
  return firstAction(view);
}

async function driveThroughGame(
  table: SeatTable,
  choose: (view: WireView) => WireAction = firstAction,
  maximumActions = 1_500,
) {
  await recordVisualCoverage(table);
  for (let step = 0; step < maximumActions; step += 1) {
    const terminalSeat = table.views.findIndex((view) => view?.terminal_results !== null);
    if (terminalSeat >= 0) return table.views[terminalSeat]!;

    let actingSeat = table.views.findIndex((view) => (view?.legal_actions.length ?? 0) > 0);
    if (actingSeat < 0) {
      await Promise.all(table.pages.map((_page, seat) => refreshSeat(table, seat)));
      actingSeat = table.views.findIndex((view) => (view?.legal_actions.length ?? 0) > 0);
    }
    expect(actingSeat, `No actionable fixed seat at step ${step}`).toBeGreaterThanOrEqual(0);
    await clickAction(table, actingSeat, choose(table.views[actingSeat]!));
  }
  throw new Error(`Game did not reach Game End within ${maximumActions} UI actions`);
}

function pilePositions(page: Page) {
  return page.getByLabel("Stockpiles").locator("article[data-stockpile-id]").evaluateAll((elements) => elements.map((element) => {
    const box = element.getBoundingClientRect();
    return { x: Math.round(box.x), y: Math.round(box.y) };
  }));
}

async function expectNoHorizontalOverflow(page: Page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth)).toBe(false);
}

async function expectDesktopWorkstationContained(page: Page) {
  const measurements = await page.evaluate((labels) => {
    const regions = labels.map((label) => {
      const element = document.querySelector<HTMLElement>(`[aria-label="${label}"]`);
      if (!element) return { label, missing: true, top: 0, bottom: 0 };
      const bounds = element.getBoundingClientRect();
      return { label, missing: false, top: bounds.top, bottom: bounds.bottom };
    });
    const piles = [...document.querySelectorAll<HTMLElement>("article[data-stockpile-id]")].map((element) => {
      const bounds = element.getBoundingClientRect();
      return { top: bounds.top, bottom: bounds.bottom };
    });
    return {
      viewportHeight: window.innerHeight,
      documentHeight: document.documentElement.scrollHeight,
      regions,
      piles,
    };
  }, [...workstationRegions]);

  expect(measurements.documentHeight).toBeLessThanOrEqual(measurements.viewportHeight + 1);
  for (const region of measurements.regions) {
    expect(region.missing, `${region.label} is missing`).toBe(false);
    expect(region.top, `${region.label} begins above the viewport`).toBeGreaterThanOrEqual(-1);
    expect(region.bottom, `${region.label} ends below the viewport`).toBeLessThanOrEqual(measurements.viewportHeight + 1);
  }
  for (const [index, pile] of measurements.piles.entries()) {
    expect(pile.top, `Stockpile ${index + 1} begins above the viewport`).toBeGreaterThanOrEqual(-1);
    expect(pile.bottom, `Stockpile ${index + 1} ends below the viewport`).toBeLessThanOrEqual(measurements.viewportHeight + 1);
  }
}

test("Home exposes only the four supported features and creates opaque seat links", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("STOCKPILE", { exact: true })).toBeVisible();
  await expect(page.getByText("LITE", { exact: true })).toBeVisible();
  for (const count of [2, 3, 4, 5]) await expect(page.getByRole("button", { name: `${count} players` })).toBeVisible();
  for (const feature of ["DIVIDEND", "FEES", "IMPACT", "SELL ORDER"]) {
    await expect(page.getByRole("button", { name: feature, exact: true })).toBeVisible();
  }
  for (const unsupported of ["HAND", "SPLIT", "MAJORITY", "INVESTOR", "STOCK TRACKS"]) {
    await expect(page.getByText(unsupported, { exact: true })).toHaveCount(0);
  }

  await page.getByRole("button", { name: "3 players" }).click();
  for (const feature of ["DIVIDEND", "FEES", "IMPACT", "SELL ORDER"]) {
    await page.getByRole("button", { name: feature, exact: true }).click();
  }
  const createRequest = page.waitForRequest((request) => request.url().endsWith("/api/v1/games") && request.method() === "POST");
  await page.getByRole("button", { name: "START", exact: true }).click();
  const payload = (await createRequest).postDataJSON() as {
    player_count: number;
    player_names: string[];
    round_count: number;
    options: typeof defaultOptions;
  };
  expect(payload).toEqual({
    player_count: 3,
    player_names: ["Player 1", "Player 2", "Player 3"],
    round_count: 6,
    options: visibleFeatureOptions,
  });
  await expect(page.getByRole("link", { name: "Open Seat" })).toHaveCount(3);
  const seatUrls = await page.getByRole("link", { name: "Open Seat" }).evaluateAll((links) => links.map((link) => (link as HTMLAnchorElement).href));
  expect(seatUrls.every((url) => new URL(url).hash.startsWith("#seat="))).toBe(true);
  expect(new Set(seatUrls.map((url) => new URL(url).hash)).size).toBe(3);
});

test("separate pages in one browser context retain fixed seats and the workstation contract", async ({ browser, request }) => {
  const table = await createSeatTable(browser, request, { round_count: 1, seed: 13, options: defaultOptions });
  try {
    const [left, right] = table.pages;
    for (const region of workstationRegions) await expect(left.getByLabel(region)).toBeVisible();
    await expect(left.getByLabel("Chat")).toHaveCount(0);
    await expect(left.getByTestId("workstation")).toBeVisible();
    await expect(left.getByTestId("stockpile-field")).toBeVisible();

    const piles = left.getByLabel("Stockpiles").locator("article[data-stockpile-id]");
    await expect(piles).toHaveCount(4);
    const expectedCards = table.views[0]!.stockpiles.reduce(
      (total, pile) => total + pile.visible_cards.length + pile.hidden_cards.length,
      0,
    );
    await expect(left.locator("[data-stack-card]")).toHaveCount(expectedCards);

    expect(table.views[0]!.companies.map((company) => [company.display_name, company.pattern])).toEqual(companyPresentation);
    expect(new Set(table.views[0]!.companies.map((company) => company.color.toUpperCase()))).toEqual(new Set(["#002FA7"]));
    for (const [, pattern] of companyPresentation) {
      const sample = left.getByLabel("Market").locator(`[data-stock-pattern="${pattern}"]`);
      await expect(sample).toHaveCount(1);
      expect(await sample.evaluate((element) => getComputedStyle(element).color)).toBe("rgb(0, 47, 167)");
    }

    const sizing = await left.evaluate(() => {
      const workstation = document.querySelector<HTMLElement>('[data-testid="workstation"]')!.getBoundingClientRect();
      const field = document.querySelector<HTMLElement>('[data-testid="stockpile-field"]')!.getBoundingClientRect();
      return { workstationWidth: workstation.width, fieldWidth: field.width };
    });
    expect(sizing.fieldWidth / sizing.workstationWidth).toBeGreaterThanOrEqual(0.5);
    const positions = await pilePositions(left);
    expect(new Set(positions.map((position) => position.x)).size).toBe(2);
    expect(new Set(positions.map((position) => position.y)).size).toBe(2);
    await expectNoHorizontalOverflow(left);
    await expectDesktopWorkstationContained(left);

    await left.setViewportSize({ width: 1_920, height: 1_080 });
    const widePositions = await pilePositions(left);
    expect(new Set(widePositions.map((position) => position.x)).size).toBe(2);
    expect(new Set(widePositions.map((position) => position.y)).size).toBe(2);
    await expectNoHorizontalOverflow(left);
    await expectDesktopWorkstationContained(left);

    await left.setViewportSize({ width: 1_024, height: 768 });
    await expect(left.getByLabel("Portfolio")).toBeVisible();
    await expect(left.getByLabel("Players")).toBeVisible();
    await expectNoHorizontalOverflow(left);
    await expectDesktopWorkstationContained(left);

    expect(table.views[0]!.viewer.player_id).toBe(0);
    expect(table.views[1]!.viewer.player_id).toBe(1);
    expect(table.views[0]!.private).not.toEqual(table.views[1]!.private);
    const privatePairs = table.views.map((view) => privateForecastPairs(view!));
    expect(privatePairs[0].size).toBeGreaterThan(0);
    expect(privatePairs[1].size).toBeGreaterThan(0);
    expect([...privatePairs[0]].filter((pair) => privatePairs[1].has(pair))).toEqual([]);
    expect(table.coverage.privacyFailures).toEqual([]);

    for (const page of [left, right]) {
      const body = (await page.locator("body").innerText()).toLowerCase();
      for (const forbidden of [
        "chat",
        "activity log",
        "history",
        "deep cfr",
        "recommendation",
        "expected value",
        "exploitability",
        "advantage",
        "analysis",
      ]) expect(body).not.toContain(forbidden);
    }
  } finally {
    await table.context.close();
  }
});

for (const playerCount of [3, 4, 5]) {
  test(`${playerCount}-player workstation reflows without horizontal overflow`, async ({ page }) => {
    await page.setViewportSize({ width: 1_280, height: 900 });
    await page.goto("/");
    await page.getByRole("button", { name: `${playerCount} players` }).click();
    await page.getByRole("button", { name: "START", exact: true }).click();
    const seat = page.getByRole("link", { name: "Open Seat" }).first();
    const url = await seat.getAttribute("href");
    expect(url).toBeTruthy();
    await page.goto(url!);

    for (const region of workstationRegions) await expect(page.getByLabel(region)).toBeVisible();
    const piles = page.getByLabel("Stockpiles").locator("article[data-stockpile-id]");
    await expect(piles).toHaveCount(playerCount);
    await expectNoHorizontalOverflow(page);
    await expectDesktopWorkstationContained(page);
    if (playerCount === 4) {
      const positions = await pilePositions(page);
      expect(new Set(positions.map((position) => position.x)).size).toBe(2);
      expect(new Set(positions.map((position) => position.y)).size).toBe(2);
    }
    if (playerCount === 5) {
      const positions = await pilePositions(page);
      expect(new Set(positions.map((position) => position.y)).size).toBeGreaterThan(1);
    }
  });
}

test("default six-round game completes through fixed-seat pages with two persistent bid positions", async ({ browser, request }) => {
  test.setTimeout(180_000);
  const table = await createSeatTable(browser, request, { round_count: 6, seed: 101, options: defaultOptions });
  try {
    const terminal = await driveThroughGame(table);
    const terminalPage = table.pages[terminal.viewer.player_id];
    await expect(terminalPage.getByLabel("Game end")).toBeVisible();
    await expect(terminalPage.getByText("GAME END", { exact: true }).first()).toBeVisible();
    expect(terminal.terminal_results?.players).toHaveLength(2);
    expect([...table.coverage.phases]).toEqual(expect.arrayContaining(["supply", "demand", "selling", "movement"]));
    expect([...table.coverage.decisions]).toEqual(expect.arrayContaining([
      "supply_card",
      "bid_pile",
      "bid_amount",
      "sell",
      "terminal",
    ]));
    expect(table.coverage.sawOutbid).toBe(true);
    expect(table.coverage.sawRebid).toBe(true);
    expect(table.coverage.privacyFailures).toEqual([]);

    for (const playerId of [0, 1]) {
      expect([...(table.coverage.markerIndicesByPlayer.get(playerId) ?? [])].sort()).toEqual([0, 1]);
    }
    expect(table.coverage.settledRounds.size).toBe(6);
    for (const purchasers of table.coverage.settledRounds.values()) {
      expect(purchasers).toHaveLength(4);
      expect(purchasers.filter((player) => player === 0)).toHaveLength(2);
      expect(purchasers.filter((player) => player === 1)).toHaveLength(2);
    }
    for (const [seat, activePlayers] of table.coverage.activePlayersBySeat.entries()) {
      expect([...activePlayers].sort(), `Seat ${seat} did not observe both changing turns`).toEqual([0, 1]);
      expect(table.coverage.capturedViews[seat].every((view) => view.viewer.player_id === seat)).toBe(true);
    }

    expect(table.coverage.sawOffsetStack).toBe(true);
    expect(table.coverage.sawStockpileDominance).toBe(true);
    expect(table.coverage.sawPortfolioScale).toBe(true);
  } finally {
    await table.context.close();
  }
});

test("all visible Lite features complete with Impact targeting and a rendered price above ten", async ({ browser, request }) => {
  test.setTimeout(120_000);
  const table = await createSeatTable(browser, request, { round_count: 1, seed: 5, options: visibleFeatureOptions });
  try {
    const terminal = await driveThroughGame(table, allOptionsAction, 500);
    const terminalPage = table.pages[terminal.viewer.player_id];
    await expect(terminalPage.getByLabel("Game end")).toBeVisible();
    expect([...table.coverage.phases]).toContain("action");
    expect([...table.coverage.decisions]).toContain("action_company");
    expect(table.coverage.sawPriceAboveTen).toBe(true);
    expect(table.coverage.sawRenderedPriceAboveTen).toBe(true);
    expect(table.coverage.privacyFailures).toEqual([]);
    expect(terminal.terminal_results?.players).toHaveLength(2);
  } finally {
    await table.context.close();
  }
});
