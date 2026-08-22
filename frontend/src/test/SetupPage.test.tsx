import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, expect, it } from "vitest";
import { SetupPage } from "../SetupPage";
import { server, setupResponse } from "./server";

afterEach(cleanup);

it("creates a game from the minimal authoritative setup", async () => {
  let request: Record<string, unknown> | undefined;

  server.use(
    http.get("/api/v1/setup", () => HttpResponse.json({
      ...setupResponse,
      options: setupResponse.options.map((option) => (
        option.key === "starting_share" ? { ...option, default: true } : option
      )),
    })),
    http.post("/api/v1/games", async ({ request: incoming }) => {
      request = await incoming.json() as Record<string, unknown>;
      const body = request as { player_count: number; player_names: string[] };
      return HttpResponse.json({
        schema_version: "1.0",
        game_id: "g1",
        seats: Array.from({ length: body.player_count }, (_, playerId) => ({
          player_id: playerId,
          player_name: body.player_names[playerId],
          url: `/game/g1#seat=${playerId}`,
        })),
      }, { status: 201 });
    }),
  );

  render(<SetupPage />);

  const twoPlayers = await screen.findByRole("button", { name: "2 players" });
  expect(twoPlayers).toHaveAttribute("aria-pressed", "true");
  expect(screen.getAllByRole("button", { name: /players$/ })).toHaveLength(4);

  const featureNames = ["DIVIDEND", "FEES", "IMPACT", "SELL ORDER"];
  for (const name of featureNames) {
    const control = screen.getByRole("button", { name });
    expect(control).toHaveAttribute("aria-pressed", "false");
    await userEvent.click(control);
    expect(control).toHaveAttribute("aria-pressed", "true");
  }

  expect(screen.queryByRole("button", { name: /starting share/i })).not.toBeInTheDocument();
  expect(screen.queryByRole("spinbutton")).not.toBeInTheDocument();
  expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
  expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();

  await userEvent.click(screen.getByRole("button", { name: "5 players" }));
  await userEvent.click(screen.getByRole("button", { name: "START" }));

  await waitFor(() => expect(request).toEqual({
    player_count: 5,
    player_names: ["Player 1", "Player 2", "Player 3", "Player 4", "Player 5"],
    round_count: 6,
    options: {
      market_impact: true,
      starting_share: false,
      trading_fees: true,
      dividends: true,
      sell_order: true,
    },
  }));

  expect(await screen.findAllByRole("link", { name: "Open Seat" })).toHaveLength(5);
  expect(screen.getByText("P1")).toBeInTheDocument();
  expect(screen.getByText("P5")).toBeInTheDocument();
  expect(screen.getAllByRole("link", { name: "Open Seat" })[0]).toHaveAttribute(
    "href",
    "/game/g1#seat=0",
  );
});

it("shows no decorative setup or unsupported feature copy", async () => {
  render(<SetupPage />);
  await screen.findByRole("button", { name: "2 players" });

  expect(screen.getByText("STOCKPILE")).toBeInTheDocument();
  expect(screen.getByText("LITE")).toBeInTheDocument();
  expect(screen.queryByText(/Take your seat/i)).not.toBeInTheDocument();
  expect(screen.queryByText(/local table/i)).not.toBeInTheDocument();

  for (const name of ["HAND", "SPLIT", "MAJORITY", "INVESTOR", "STOCK TRACKS"]) {
    expect(screen.queryByText(name)).not.toBeInTheDocument();
  }
});
