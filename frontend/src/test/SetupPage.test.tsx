import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { expect, it } from "vitest";
import { SetupPage } from "../SetupPage";
import { server } from "./server";

it("uses authoritative setup defaults and option descriptors", async () => {
  let request: Record<string, unknown> | undefined;
  server.use(http.post("/api/v1/games", async ({ request: incoming }) => {
    request = await incoming.json() as Record<string, unknown>;
    return HttpResponse.json({ schema_version: "1.0", game_id: "g1", seats: [{ player_id: 0, player_name: "Player 1", url: "/game/g1#seat=a" }, { player_id: 1, player_name: "Player 2", url: "/game/g1#seat=b" }] }, { status: 201 });
  }));
  render(<SetupPage />);
  expect(await screen.findByRole("spinbutton", { name: "Round count" })).toHaveValue(6);
  expect(screen.getByRole("combobox", { name: "Player count" })).toHaveValue("2");
  expect(screen.getByRole("checkbox", { name: /Market Impact/ })).not.toBeChecked();
  await userEvent.click(screen.getByRole("checkbox", { name: /Market Impact/ }));
  await userEvent.click(screen.getByRole("button", { name: "Create game" }));
  await waitFor(() => expect(request).toMatchObject({ player_count: 2, round_count: 6, options: { market_impact: true } }));
  expect(await screen.findByRole("link", { name: /Open Player 1/ })).toHaveAttribute("href", "/game/g1#seat=a");
});
