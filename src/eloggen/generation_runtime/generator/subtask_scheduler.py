"""Deterministic schedulers for validated partial-order subtask graphs."""

from __future__ import annotations

import random

from eloggen.generation_runtime.config.subtask_graph import SubtaskGraph


class SubtaskScheduler:
    def __init__(self, graph: SubtaskGraph, seed: int = 0):
        self.graph = graph
        self.seed = int(seed)
        self.orders = graph.legal_orders()
        if not self.orders:
            raise ValueError("subtask_graph has no legal execution order")
        self.random = random.Random(seed)

    def select(
        self,
        strategy: str,
        fixed_order: list[str] | tuple[str, ...] | None = None,
        quota_targets: dict[str, int] | None = None,
        quota_counts: dict[str, int] | None = None,
    ) -> tuple[str, ...]:
        if strategy == "fixed":
            order = tuple(fixed_order or self.orders[0])
            if not self.graph.is_valid_order(order):
                raise ValueError(f"invalid fixed subtask order: {order}")
            return order
        if strategy == "random_topological":
            return self.random.choice(self.orders)
        if strategy == "ordered_quota":
            targets = quota_targets or {}
            counts = quota_counts or {}
            remaining = []
            for order in self.orders:
                name = self.order_name(order)
                deficit = int(targets.get(name, 0)) - int(counts.get(name, 0))
                if deficit > 0:
                    remaining.extend([order] * deficit)
            if not remaining:
                raise StopIteration("all subtask order quotas are complete")
            return self.random.choice(remaining)
        raise ValueError(f"unknown subtask scheduling strategy: {strategy}")

    def select_for_attempt(
        self,
        strategy: str,
        attempt_index: int,
        fixed_order: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        """Select without depending on choices consumed by earlier attempts."""
        if strategy == "fixed":
            return self.select(strategy, fixed_order=fixed_order)
        if strategy != "random_topological":
            raise ValueError(f"attempt-indexed selection does not support {strategy!r}")
        randomizer = random.Random(f"{self.seed}:{int(attempt_index)}:drawer-subtask-order")
        return randomizer.choice(self.orders)

    def apply_resource_events(self, state: dict[str, str], node: str) -> dict[str, str]:
        updated = dict(state)
        for resource, operation, token in self.graph.resource_events.get(node, ()):
            if operation == "acquire":
                if resource in updated:
                    raise ValueError(f"resource {resource!r} already holds {updated[resource]!r}")
                updated[resource] = token
            elif updated.get(resource) != token:
                raise ValueError(
                    f"resource {resource!r} cannot release {token!r}; current={updated.get(resource)!r}"
                )
            else:
                del updated[resource]
        return updated

    def resource_states_before(self, order: tuple[str, ...]) -> list[dict[str, str]]:
        if not self.graph.is_valid_order(order):
            raise ValueError(f"invalid subtask order: {order}")
        states = []
        current = {}
        for node in order:
            states.append(dict(current))
            current = self.apply_resource_events(current, node)
        return states

    def order_name(self, order: tuple[str, ...]) -> str:
        first_acquire = next(
            (node, token)
            for node in order
            for _, operation, token in self.graph.resource_events.get(node, ())
            if operation == "acquire"
        )
        first_pick_node, token = first_acquire
        safe_token = token.replace(" ", "_")
        if order.index(first_pick_node) > 0:
            return f"open_first_{safe_token}"
        return f"{safe_token}_pick_first"
