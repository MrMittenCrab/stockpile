import { expect, it, vi } from "vitest";
import { claimFragmentToken, seatStorageKey } from "../api";

it("moves a fragment seat secret into session storage and removes it from the URL", () => {
  window.history.replaceState(null, "", "/game/g1#seat=very-secret");
  const replace = vi.spyOn(window.history, "replaceState");
  expect(claimFragmentToken("g1")).toBe("very-secret");
  expect(sessionStorage.getItem(seatStorageKey("g1"))).toBe("very-secret");
  expect(replace).toHaveBeenCalledWith(null, "", "/game/g1");
});
