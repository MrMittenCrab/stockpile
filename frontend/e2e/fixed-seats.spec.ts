import { expect, test } from "@playwright/test";
import type { APIRequestContext, Browser, BrowserContext, Page } from "@playwright/test";

const options = {
  market_impact: false,
  starting_share: false,
  trading_fees: false,
  dividends: false,
  sell_order: false,
};

type WireAction = {
  action_id: number;
  control: string;
  target_id: string | null;
  amount: number | null;
  sale_preview: { quantity: number } | null;
};

type WireView = {
  revision: number;
  round: number;
  phase: string;
  viewer: { player_id: number; name: string };
  active_player_id: number | null;
  companies: Array<{ company_id: number; price: number }>;
  stockpiles: Array<{ purchaser_id: number | null; hidden_cards: Array<Record<string, unknown>> }>;
  players: Array<{ player_id: number; active: boolean; status: string; bid_markers: Array<{ status: string }> }>;
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
  recent_events: Array<{ event_type: string; resulting_price: number | null }>;
  terminal_results: null | { players: Array<{ player_id: number; final_cash: number; winner: boolean }> };
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
  sawOutbid: boolean;
  sawPriceAboveTen: boolean;
  purchases: Map<number, Set<number>>;
  settledRounds: Map<number, number[]>;
  privacyFailures: string[];
};

type SeatTable = {
  contexts: BrowserContext[];
  pages: Page[];
  views: Array<WireView | null>;
  coverage: Coverage;
};

function recordView(view: WireView, coverage: Coverage) {
  coverage.phases.add(view.phase.toLowerCase());
  if (view.recent_events.some((event) => event.event_type === "market_movement" || event.event_type === "market_reveal")) coverage.phases.add("movement");
  coverage.decisions.add(view.pending_decision.kind);
  coverage.sawOutbid ||= view.players.some((player) => player.bid_markers.some((marker) => marker.status === "outbid" || marker.status === "rebidding"));
  coverage.sawPriceAboveTen ||= view.companies.some((company) => company.price > 10) || view.recent_events.some((event) => (event.resulting_price ?? 0) > 10);
  for (const pile of view.stockpiles) {
    if (pile.purchaser_id !== null) {
      const purchases = coverage.purchases.get(pile.purchaser_id) ?? new Set<number>();
      purchases.add(pile.purchaser_id * 100 + pile.purchaser_id + view.round * 10_000 + view.stockpiles.indexOf(pile));
      coverage.purchases.set(pile.purchaser_id, purchases);
    }
    for (const hidden of pile.hidden_cards) {
      if (Object.keys(hidden).length !== 1 || hidden.visibility !== "hidden") coverage.privacyFailures.push("Hidden card contained extra fields");
    }
  }
  if (view.stockpiles.length > 0 && view.stockpiles.every((pile) => pile.purchaser_id !== null)) {
    coverage.settledRounds.set(view.round, view.stockpiles.map((pile) => pile.purchaser_id!));
  }
  const wire = JSON.stringify(view).toLowerCase();
  for (const forbidden of ["information_state_id", "information_state_tensor", "raw_history", "card_id"]) {
    if (wire.includes(forbidden)) coverage.privacyFailures.push(`Response exposed ${forbidden}`);
  }
  if (view.phase.toLowerCase() === "selling" && view.active_player_id === null) {
    if (view.players.some((player) => player.active)) coverage.privacyFailures.push("Sealed selling exposed an active player");
  }
}

async function createSeatTable(
  browser: Browser,
  request: APIRequestContext,
  configuration: { round_count: number; seed: number; options: typeof options },
): Promise<SeatTable> {
  const created = await request.post("/api/v1/games", {
    data: {
      player_count: 2,
      player_names: ["Ada", "Lin"],
      ...configuration,
    },
  });
  expect(created.status()).toBe(201);
  const game = await created.json() as { seats: Array<{ url: string }> };
  const contexts = await Promise.all([browser.newContext(), browser.newContext()]);
  const pages = await Promise.all(contexts.map((context) => context.newPage()));
  const views: Array<WireView | null> = [null, null];
  const coverage: Coverage = { phases: new Set(), decisions: new Set(), sawOutbid: false, sawPriceAboveTen: false, purchases: new Map(), settledRounds: new Map(), privacyFailures: [] };
  for (const [seat, page] of pages.entries()) {
    page.on("response", async (response) => {
      if (!response.url().includes("/api/v1/games/") || (!response.url().endsWith("/view") && !response.url().endsWith("/actions")) || !response.ok()) return;
      const view = await response.json() as WireView;
      views[seat] = view;
      recordView(view, coverage);
    });
  }
  await Promise.all(pages.map(async (page, seat) => {
    const initial = page.waitForResponse((response) => response.url().endsWith("/view") && response.ok());
    await page.goto(game.seats[seat].url);
    views[seat] = await (await initial).json() as WireView;
    recordView(views[seat]!, coverage);
  }));
  return { contexts, pages, views, coverage };
}

async function refreshSeat(table: SeatTable, seat: number) {
  const response = table.pages[seat].waitForResponse((candidate) => candidate.url().endsWith("/view") && candidate.ok(), { timeout: 3_000 });
  await table.pages[seat].evaluate(() => window.dispatchEvent(new Event("focus")));
  table.views[seat] = await (await response).json() as WireView;
  recordView(table.views[seat]!, table.coverage);
}

async function clickAction(table: SeatTable, seat: number, action: WireAction) {
  const button = table.pages[seat].locator(`[data-action-id="${action.action_id}"]`);
  await expect(button).toBeVisible();
  await expect(button).toBeEnabled();
  const response = table.pages[seat].waitForResponse((candidate) => candidate.url().endsWith("/actions") && candidate.request().method() === "POST");
  await button.click();
  const result = await response;
  expect(result.ok()).toBe(true);
  table.views[seat] = await result.json() as WireView;
  recordView(table.views[seat]!, table.coverage);
  for (let other = 0; other < table.pages.length; other += 1) {
    if (other !== seat) await refreshSeat(table, other);
  }
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
    return [...view.legal_actions].sort((left, right) => (right.sale_preview?.quantity ?? 0) - (left.sale_preview?.quantity ?? 0))[0];
  }
  return firstAction(view);
}

async function driveThroughGame(
  table: SeatTable,
  choose: (view: WireView) => WireAction = firstAction,
  maximumActions = 1_500,
) {
  for (let step = 0; step < maximumActions; step += 1) {
    const terminalSeat = table.views.findIndex((view) => view?.terminal_results !== null);
    if (terminalSeat >= 0) return table.views[terminalSeat]!;
    let actingSeat = table.views.findIndex((view) => (view?.legal_actions.length ?? 0) > 0);
    if (actingSeat < 0) {
      await Promise.all(table.pages.map((_page, seat) => refreshSeat(table, seat)));
      actingSeat = table.views.findIndex((view) => (view?.legal_actions.length ?? 0) > 0);
    }
    expect(actingSeat, `No actionable fixed seat at step ${step}`).toBeGreaterThanOrEqual(0);
    const view = table.views[actingSeat]!;
    await clickAction(table, actingSeat, choose(view));
  }
  throw new Error(`Game did not reach Game End within ${maximumActions} UI actions`);
}

test("separate contexts open fixed seats and preserve desktop hierarchy", async ({ browser, request }) => {
  const created = await request.post("/api/v1/games", {
    data: {
      player_count: 2,
      player_names: ["Ada", "Lin"],
      round_count: 6,
      options,
      seed: 13,
    },
  });
  expect(created.status()).toBe(201);
  const game = await created.json() as { game_id: string; seats: Array<{ url: string }> };
  const contexts = await Promise.all([browser.newContext(), browser.newContext()]);
  try {
    const pages = await Promise.all(contexts.map((context) => context.newPage()));
    const intercepted = pages.map((page) => page.waitForResponse((response) => response.url().endsWith("/view") && response.ok()));
    await Promise.all(pages.map((page, index) => page.goto(game.seats[index].url)));
    const viewBodies = await Promise.all(intercepted.map(async (response) => await (await response).json() as WireView));
    await expect(pages[0].getByText("Ada", { exact: true }).first()).toBeVisible();
    await expect(pages[1].getByText("Lin", { exact: true }).first()).toBeVisible();
    const piles = pages[0].getByLabel("Stockpiles").locator("article");
    await expect(piles).toHaveCount(4);
    await expect(pages[0].getByRole("heading", { name: "CHAT" })).toBeVisible();
    const sizing = await pages[0].evaluate(() => {
      const layout = document.querySelector('[class*="layout"]')!;
      const board = document.querySelector('main[class*="board"]')!;
      return { layout: layout.getBoundingClientRect().width, board: board.getBoundingClientRect().width, overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth };
    });
    expect(sizing.board / sizing.layout).toBeGreaterThan(0.67);
    expect(sizing.board / sizing.layout).toBeLessThan(0.77);
    expect(sizing.overflow).toBe(false);
    const pilePositions = await piles.evaluateAll((elements) => elements.map((element) => {
      const box = element.getBoundingClientRect();
      return { x: Math.round(box.x), y: Math.round(box.y) };
    }));
    expect(new Set(pilePositions.map((position) => position.x)).size).toBe(2);
    expect(new Set(pilePositions.map((position) => position.y)).size).toBe(2);
    await pages[0].setViewportSize({ width: 1_920, height: 1_080 });
    const widePilePositions = await piles.evaluateAll((elements) => elements.map((element) => {
      const box = element.getBoundingClientRect();
      return { x: Math.round(box.x), y: Math.round(box.y) };
    }));
    expect(new Set(widePilePositions.map((position) => position.x)).size).toBe(2);
    expect(new Set(widePilePositions.map((position) => position.y)).size).toBe(2);
    expect(await pages[0].evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth)).toBe(false);
    await pages[0].setViewportSize({ width: 1_024, height: 768 });
    expect(await pages[0].evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth)).toBe(false);
    expect(viewBodies[0].private).not.toEqual(viewBodies[1].private);
    const privatePairs = viewBodies.map(privateForecastPairs);
    expect(privatePairs[0].size).toBeGreaterThan(0);
    expect(privatePairs[1].size).toBeGreaterThan(0);
    expect([...privatePairs[0]].filter((pair) => privatePairs[1].has(pair))).toEqual([]);
    for (const [seat, payload] of viewBodies.entries()) {
      expect(payload.viewer.player_id).toBe(seat);
      expect(Object.keys(payload.private).sort()).toEqual([
        "available_action_cards",
        "hand",
        "holdings",
        "known_pile_cards",
        "market_information",
      ]);
      for (const otherSeatPair of privatePairs[1 - seat]) {
        expect(privateForecastPairs(payload).has(otherSeatPair)).toBe(false);
      }
      const wire = JSON.stringify(payload).toLowerCase();
      expect(wire).not.toContain("information_state_id");
      expect(wire).not.toContain("information_state_tensor");
      expect(wire).not.toContain("raw_history");
    }
    const body = (await pages[0].locator("body").innerText()).toLowerCase();
    for (const forbidden of ["deep cfr", "recommendation", "expected value", "exploitability", "advantage"]) expect(body).not.toContain(forbidden);
  } finally {
    await Promise.all(contexts.map((context) => context.close()));
  }
});

test("five-player setup reflows five-plus piles without horizontal overflow", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Player count").selectOption("5");
  await expect(page.getByLabel("Player 5 name")).toBeVisible();
  await page.getByRole("button", { name: "Create game" }).click();
  const seat = page.getByRole("link", { name: /Open Player 1/ });
  const url = await seat.getAttribute("href");
  expect(url).toBeTruthy();
  await page.goto(url!);
  await expect(page.getByLabel("Stockpiles").locator("article")).toHaveCount(5);
  expect(await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth)).toBe(false);
});

test("default six-round game completes entirely through fixed-seat UI controls", async ({ browser, request }) => {
  test.setTimeout(180_000);
  const table = await createSeatTable(browser, request, { round_count: 6, seed: 101, options });
  try {
    const terminal = await driveThroughGame(table);
    await expect(table.pages[terminal.viewer.player_id].getByText("GAME END")).toBeVisible();
    await expect(table.pages[terminal.viewer.player_id].getByRole("heading", { name: "FINAL MARKET MOVEMENT" })).toBeVisible();
    expect(terminal.terminal_results?.players).toHaveLength(2);
    expect([...table.coverage.phases]).toEqual(expect.arrayContaining(["supply", "demand", "selling", "movement"]));
    expect([...table.coverage.decisions]).toEqual(expect.arrayContaining(["supply_card", "bid_pile", "bid_amount", "sell", "terminal"]));
    expect(table.coverage.privacyFailures).toEqual([]);
    expect(table.coverage.sawOutbid).toBe(true);
    expect(table.coverage.purchases.get(0)?.size).toBeGreaterThanOrEqual(2);
    expect(table.coverage.purchases.get(1)?.size).toBeGreaterThanOrEqual(2);
    expect(table.coverage.settledRounds.size).toBe(6);
    for (const purchasers of table.coverage.settledRounds.values()) {
      expect(purchasers).toHaveLength(4);
      expect(purchasers.filter((player) => player === 0)).toHaveLength(2);
      expect(purchasers.filter((player) => player === 1)).toHaveLength(2);
    }
  } finally {
    await Promise.all(table.contexts.map((context) => context.close()));
  }
});

test("all Lite options complete through UI with Action targeting and an uncapped price", async ({ browser, request }) => {
  test.setTimeout(120_000);
  const allOptions = { market_impact: true, starting_share: true, trading_fees: true, dividends: true, sell_order: true };
  const table = await createSeatTable(browser, request, { round_count: 1, seed: 2, options: allOptions });
  try {
    const terminal = await driveThroughGame(table, allOptionsAction, 500);
    await expect(table.pages[terminal.viewer.player_id].getByText("GAME END")).toBeVisible();
    await expect(table.pages[terminal.viewer.player_id].getByRole("heading", { name: "FINAL MARKET MOVEMENT" })).toBeVisible();
    expect([...table.coverage.phases]).toContain("action");
    expect([...table.coverage.decisions]).toContain("action_company");
    expect(table.coverage.sawPriceAboveTen).toBe(true);
    expect(table.coverage.privacyFailures).toEqual([]);
  } finally {
    await Promise.all(table.contexts.map((context) => context.close()));
  }
});
