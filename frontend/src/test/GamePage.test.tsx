import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { GamePage } from "../components/GamePage";
import { gameView, server } from "./server";

describe("fixed-seat game surface", () => {
  it("renders unusual server prices, pile counts, and bids without deriving choices", async () => {
    let submitted: unknown;
    server.use(http.post("/api/v1/games/unusual/actions", async ({ request }) => {
      submitted = await request.json();
      return HttpResponse.json({ ...gameView, revision: 8, legal_actions: [] });
    }));
    render(<GamePage gameId="unusual" token="seat-secret" />);
    expect(await screen.findByText("47")).toBeInTheDocument();
    expect(screen.getByText("STOCKPILE 7")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Bid 37K" })).toBeInTheDocument();
    expect(screen.queryByText("$5K")).not.toBeInTheDocument();
    expect(screen.getByText("Fees due: $4K · $7K")).toBeInTheDocument();
    expect(screen.getByLabelText("Private pile knowledge")).toHaveTextContent("Pile 2");
    expect(screen.getByLabelText("Latest market movement")).toHaveTextContent("Arc");
    expect(screen.getByLabelText("Latest market movement")).toHaveTextContent("Bolt");
    expect(screen.getByText("STOCKPILE 7").closest("article")?.className).toContain("selectedPile");
    await userEvent.click(screen.getByRole("button", { name: "Bid 37K" }));
    await waitFor(() => expect(submitted).toEqual({ action_id: 9123, expected_revision: 7 }));
  });

  it("keeps a hidden card opaque and contains no engine output", async () => {
    render(<GamePage gameId="unusual" token="seat-secret" />);
    expect((await screen.findAllByLabelText("Hidden card")).length).toBeGreaterThan(0);
    const text = document.body.textContent?.toLowerCase() ?? "";
    for (const forbidden of ["deep cfr", "expected value", "exploitability", "policy percentage", "advantage value"]) expect(text).not.toContain(forbidden);
  });
});
