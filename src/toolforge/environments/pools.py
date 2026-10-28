"""Shared deterministic content pools for fixture generation.

Keeping names/emails consistent across environments makes multi-family tasks
read naturally ("email alice about the meeting she's in") and guarantees that
``default_state(seed)`` is identical across code paths that use the same seed.
"""

from __future__ import annotations

import random

PEOPLE = [
    ("Alice Nguyen", "alice@example.com"),
    ("Bob Martinez", "bob@example.com"),
    ("Carol Wu", "carol@example.com"),
    ("Dave Okafor", "dave@example.com"),
    ("Erin Kowalski", "erin@example.com"),
    ("Frank Zhou", "frank@example.com"),
]

USER_EMAIL = "user@example.com"
USER_NAME = "user"

# Short display names used when tasks refer to people informally.
FIRST_NAMES = [name.split()[0] for name, _ in PEOPLE]
EMAILS = [email for _, email in PEOPLE]


def pick(rng: random.Random, items: list):
    return items[rng.randrange(len(items))]


def sample(rng: random.Random, items: list, k: int) -> list:
    """Deterministic sample preserving the original order of ``items``."""
    chosen = rng.sample(range(len(items)), min(k, len(items)))
    return [items[i] for i in sorted(chosen)]
