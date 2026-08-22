"""Replaceable, player-scoped policies for the browser's computer seat."""

from __future__ import annotations

import random
from typing import Protocol, Sequence

from .. import stockpile_platform as platform


class ComputerPolicy(Protocol):
    """Choose from actions visible in one player's information state only."""

    def choose_action(
        self,
        information: platform.InformationState,
        legal_actions: Sequence[platform.LegalAction],
        rng: random.Random,
    ) -> int:
        """Return one action ID from ``legal_actions``."""


class RandomComputerPolicy:
    """Deliberately trivial uniform policy used until an engine policy is supplied."""

    def choose_action(
        self,
        information: platform.InformationState,
        legal_actions: Sequence[platform.LegalAction],
        rng: random.Random,
    ) -> int:
        del information
        if not legal_actions:
            raise ValueError("computer policy requires at least one legal action")
        return int(rng.choice(tuple(legal_actions)).action_id)


RandomPolicy = RandomComputerPolicy


__all__ = ["ComputerPolicy", "RandomComputerPolicy", "RandomPolicy"]
