import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SetupPage } from "../SetupPage";
import setupCss from "../SetupPage.module.css?raw";
import primitiveCss from "../components/Primitives.module.css?raw";
import { server } from "./server";

afterEach(cleanup);

describe("Stockpile Trainer home", () => {
  it("starts without a selected mode and creates plain LITE with every option off", async () => {
    let request: unknown;
    const navigate = vi.fn();
    server.use(http.post("/api/v2/games", async ({ request: incoming }) => {
      request = await incoming.json();
      return HttpResponse.json({ schema_version: "2.0", game_id: "g1", game_url: "/game/g1#seat=secret" }, { status: 201 });
    }));

    render(<SetupPage navigate={navigate} />);
    expect(await screen.findByLabelText("Stockpile Trainer")).toHaveTextContent("STOCKPILE TRAINER");
    expect(screen.queryByRole("button", { name: "PLAY" })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "LITE" }));
    await userEvent.click(screen.getByRole("button", { name: "PLAY" }));
    await waitFor(() => expect(request).toEqual({ options: {
      market_impact: false,
      trading_fees: false,
      dividends: false,
      sell_order: false,
    } }));
    expect(navigate).toHaveBeenCalledWith("/game/g1#seat=secret");
  });

  it("requires a retained LITE+ feature selection and exposes exactly three options", async () => {
    let request: unknown;
    server.use(http.post("/api/v2/games", async ({ request: incoming }) => {
      request = await incoming.json();
      return HttpResponse.json({ schema_version: "2.0", game_id: "g2", game_url: "/game/g2#seat=secret" }, { status: 201 });
    }));
    render(<SetupPage navigate={() => undefined} />);
    await screen.findByRole("button", { name: "LITE+" });
    await userEvent.click(screen.getByRole("button", { name: "LITE+" }));

    for (const name of ["DIVIDEND", "FEES", "SELL ORDER"]) {
      expect(screen.getByRole("button", { name })).toHaveAttribute("aria-pressed", "false");
    }
    expect(screen.queryByText("MARKET ACTIONS")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "PLAY" })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "DIVIDEND" }));
    expect(screen.getByRole("button", { name: "PLAY" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "LITE" }));
    await userEvent.click(screen.getByRole("button", { name: "LITE+" }));
    expect(screen.getByRole("button", { name: "DIVIDEND" })).toHaveAttribute("aria-pressed", "true");

    await userEvent.click(screen.getByRole("button", { name: "PLAY" }));
    await waitFor(() => expect(request).toEqual({ options: {
      market_impact: false,
      trading_fees: false,
      dividends: true,
      sell_order: false,
    } }));
  });

  it("uses one exact universal control geometry and no setup decoration", async () => {
    render(<SetupPage navigate={() => undefined} />);
    await screen.findByRole("button", { name: "LITE" });
    expect(primitiveCss).toMatch(/\.button\s*\{[^}]*width:\s*144px;[^}]*height:\s*36px;/s);
    expect(setupCss).toContain("grid-template-columns: repeat(auto-fit, 144px)");
    expect(`${setupCss}\n${primitiveCss}`).not.toMatch(/gradient|shadow|border-radius|font-style\s*:\s*italic|ochre/i);
    for (const forbidden of ["FEATURES", "PLAYERS", "HAND", "SPLIT", "MAJORITY", "INVESTOR", "STOCK TRACKS", "START"]) {
      expect(screen.queryByText(forbidden)).not.toBeInTheDocument();
    }
  });
});
