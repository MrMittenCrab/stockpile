import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, expect, it, vi } from "vitest";
import { SetupPage } from "../SetupPage";
import setupCss from "../SetupPage.module.css?raw";
import { server } from "./server";

afterEach(cleanup);

it("creates one fixed YOU versus COMPUTER game and navigates directly", async () => {
  let request: unknown;
  const navigate = vi.fn();
  server.use(http.post("/api/v2/games", async ({ request: incoming }) => {
    request = await incoming.json();
    return HttpResponse.json({ schema_version: "2.0", game_id: "g1", game_url: "/game/g1#seat=secret" }, { status: 201 });
  }));

  render(<SetupPage navigate={navigate} />);

  for (const name of ["DIVIDEND", "FEES", "IMPACT", "SELL ORDER"]) {
    const control = await screen.findByRole("button", { name });
    expect(control).toHaveAttribute("aria-pressed", "false");
    await userEvent.click(control);
    expect(control).toHaveAttribute("aria-pressed", "true");
  }
  expect(screen.queryByText("PLAYERS")).not.toBeInTheDocument();
  expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  expect(screen.queryByText(/seat/i)).not.toBeInTheDocument();

  await userEvent.click(screen.getByRole("button", { name: "START" }));
  await waitFor(() => expect(request).toEqual({ options: {
    market_impact: true,
    trading_fees: true,
    dividends: true,
    sell_order: true,
  } }));
  expect(navigate).toHaveBeenCalledWith("/game/g1#seat=secret");
});

it("uses no special blue START surface or decorative setup copy", async () => {
  render(<SetupPage navigate={() => undefined} />);
  await screen.findByRole("button", { name: "START" });
  expect(screen.getByText("STOCKPILE")).toBeInTheDocument();
  expect(screen.getByText("LITE")).toBeInTheDocument();
  expect(setupCss).not.toMatch(/gradient|shadow|border-radius|font-style\s*:\s*italic/i);
  expect(setupCss).not.toMatch(/\.start[^}]*background\s*:\s*var\(--blue\)/s);
  for (const name of ["HAND", "SPLIT", "MAJORITY", "INVESTOR", "STOCK TRACKS"]) {
    expect(screen.queryByText(name)).not.toBeInTheDocument();
  }
});
