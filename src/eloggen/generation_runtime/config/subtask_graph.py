"""Validated partial-order configuration for long-horizon subtasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Iterable


@dataclass(frozen=True)
class SubtaskGraph:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    atomic_chains: tuple[tuple[str, ...], ...] = ()
    resource_constraints: dict[str, tuple[str, ...]] = field(default_factory=dict)
    resource_events: dict[str, tuple[tuple[str, str, str], ...]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict) -> "SubtaskGraph":
        graph = cls(
            nodes=tuple(value.get("nodes", ())),
            edges=tuple(tuple(edge) for edge in value.get("edges", ())),
            atomic_chains=tuple(tuple(chain) for chain in value.get("atomic_chains", ())),
            resource_constraints={
                str(resource): tuple(nodes)
                for resource, nodes in value.get("resource_constraints", {}).items()
            },
            resource_events={
                str(node): tuple(
                    (str(event["resource"]), str(event["operation"]), str(event["token"]))
                    for event in events
                )
                for node, events in value.get("resource_events", {}).items()
            },
        )
        graph.validate()
        return graph

    def validate(self) -> None:
        node_set = set(self.nodes)
        if not self.nodes or len(node_set) != len(self.nodes):
            raise ValueError("subtask_graph nodes must be non-empty and unique")
        for source, target in self.edges:
            if source not in node_set or target not in node_set:
                raise ValueError(f"unknown edge node: {(source, target)}")
            if source == target:
                raise ValueError(f"self cycle: {source}")
        chain_nodes = [node for chain in self.atomic_chains for node in chain]
        if len(chain_nodes) != len(set(chain_nodes)) or not set(chain_nodes).issubset(node_set):
            raise ValueError("atomic_chains must contain known, non-overlapping nodes")
        for resource, nodes in self.resource_constraints.items():
            unknown = set(nodes) - node_set
            if unknown:
                raise ValueError(f"resource {resource!r} references unknown nodes: {sorted(unknown)}")
        for node, events in self.resource_events.items():
            if node not in node_set:
                raise ValueError(f"resource event references unknown node: {node!r}")
            for resource, operation, token in events:
                if not resource or not token or operation not in {"acquire", "release"}:
                    raise ValueError(f"invalid resource event for {node!r}: {(resource, operation, token)}")
        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        incoming = {node: 0 for node in self.nodes}
        outgoing = {node: [] for node in self.nodes}
        for source, target in self.edges:
            incoming[target] += 1
            outgoing[source].append(target)
        ready = [node for node in self.nodes if incoming[node] == 0]
        visited = 0
        while ready:
            node = ready.pop()
            visited += 1
            for target in outgoing[node]:
                incoming[target] -= 1
                if incoming[target] == 0:
                    ready.append(target)
        if visited != len(self.nodes):
            raise ValueError("subtask_graph contains a cycle")

    def is_valid_order(self, order: Iterable[str]) -> bool:
        order = tuple(order)
        if len(order) != len(self.nodes) or set(order) != set(self.nodes):
            return False
        positions = {node: index for index, node in enumerate(order)}
        if any(positions[source] >= positions[target] for source, target in self.edges):
            return False
        for chain in self.atomic_chains:
            indices = [positions[node] for node in chain]
            if indices != list(range(indices[0], indices[0] + len(chain))):
                return False
        if not self._resource_lifecycle_is_valid(order):
            return False
        return True

    def _resource_lifecycle_is_valid(self, order: tuple[str, ...]) -> bool:
        """Validate exclusive gripper occupancy for pick/place task nodes."""
        held_tokens = {}
        for node in order:
            for resource, operation, token in self.resource_events.get(node, ()):
                if operation == "acquire":
                    if resource in held_tokens:
                        return False
                    held_tokens[resource] = token
                elif held_tokens.get(resource) != token:
                    return False
                else:
                    del held_tokens[resource]
        return not held_tokens

    def legal_orders(self) -> list[tuple[str, ...]]:
        return [order for order in permutations(self.nodes) if self.is_valid_order(order)]
