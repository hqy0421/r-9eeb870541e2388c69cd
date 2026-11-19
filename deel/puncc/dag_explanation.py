# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements explanations of conformal classification prediction
sets with concepts of a multi-parent DAG (directed acyclic graph) of labels.

Fine-grained prediction sets produced by conformal classifiers (such as
:meth:`deel.puncc.classification.LAC` or :meth:`deel.puncc.classification.RAPS`)
can be hard to read when many classes are involved. When labels are organized
in a DAG of coarser concepts, each prediction set can be re-expressed as a
minimally costly combination of such concepts: each node of the DAG stands for
the whole set of its descendant leaves, and a prediction set is explained by a
collection of nodes whose descendant leaves union **exactly** to the original
set (no missing class, no extra class, overlaps allowed).

For each prediction set, the explanation is computed as the solution of a
minimum-cost exact cover problem, solved to global optimality by an exact
branch-and-bound solver (no greedy tree heuristic). When several optimal
explanations exist, one is returned deterministically.
"""

import logging
from numbers import Integral
from typing import Any
from typing import Dict
from typing import FrozenSet
from typing import Hashable
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np

from deel.puncc import metrics

logger = logging.getLogger(__name__)


class DAGExplanationError(Exception):
    """Base class for the errors raised by DAG label explanation."""


class DAGCycleError(DAGExplanationError):
    """Raised when the provided label graph contains a cycle."""


class InvalidMappingError(DAGExplanationError):
    """Raised when the class column to leaf label mapping is inconsistent
    with the DAG or with the classes of the predictor."""


class InvalidCostsError(DAGExplanationError):
    """Raised when node costs are missing, unknown or not positive
    integers."""


class ExplanationSolverError(DAGExplanationError):
    """Raised when the exact cover solver fails to produce an optimal
    explanation."""


def _check_hashable(label: Any, kind: str = "node") -> None:
    """Raise a DAGExplanationError if `label` is not hashable."""
    if not isinstance(label, Hashable):
        raise DAGExplanationError(
            f"The {kind} label {label!r} is not hashable. "
            "DAG labels must be hashable objects."
        )


def _solve_min_cost_cover(
    keys: Tuple[Hashable, ...],
    covers: Tuple[FrozenSet[Hashable], ...],
    costs: Tuple[int, ...],
    target: FrozenSet[Hashable],
    max_expand: int,
) -> Tuple[Tuple[Hashable, ...], int]:
    """Exact minimum-cost cover of `target` by candidate leaf sets.

    Solves, to global optimality, the following set-cover problem: select a
    collection of candidates whose leaf sets union exactly to `target` (each
    candidate leaf set is already a subset of `target`) while minimizing the
    sum of their costs. Overlaps between selected candidates are allowed.

    The solver is an exact branch-and-bound procedure (no greedy heuristic):
    dominated candidates are first pruned, a greedy solution provides an
    initial incumbent, and the search branches on the least covered element
    with a valid lower bound on the additional cost.

    :param tuple keys: candidate node labels.
    :param tuple covers: `covers[i]` is the (nonempty) leaf set of `keys[i]`,
        included in `target`.
    :param tuple costs: `costs[i]` is the positive integer cost of `keys[i]`.
    :param frozenset target: leaf set to cover exactly.
    :param int max_expand: maximum number of search nodes expanded before
        raising an :class:`ExplanationSolverError`.

    :returns: tuple of the selected node labels (sorted deterministically)
        and the total cost.
    :rtype: Tuple[Tuple, int]

    :raises ExplanationSolverError: no exact cover exists or the search
        budget `max_expand` is exceeded.
    """
    if not target:
        return (), 0

    n_cand = len(keys)

    # ~~~~ Domination pruning: a candidate u is useless if another candidate
    # v covers a superset of its leaves for a cost no larger (ties are broken
    # deterministically on the repr of the labels).
    order = sorted(range(n_cand), key=lambda i: (costs[i], repr(keys[i])))
    keep = []
    for i in order:
        set_i, cost_i = covers[i], costs[i]
        dominated = False
        for j in order:
            if i == j or costs[j] > cost_i:
                continue
            if set_i <= covers[j] and (
                costs[j] < cost_i or set_i < covers[j] or repr(keys[j]) < repr(keys[i])
            ):
                dominated = True
                break
        if not dominated:
            keep.append(i)

    keys = tuple(keys[i] for i in keep)
    covers = tuple(covers[i] for i in keep)
    costs = tuple(costs[i] for i in keep)
    n_cand = len(keys)

    # ~~~~ Feasibility check: every leaf of the target must be coverable.
    for leaf in target:
        if not any(leaf in covers[i] for i in range(n_cand)):
            raise ExplanationSolverError(
                f"No candidate node covers leaf {leaf!r}: the prediction "
                "set cannot be explained exactly."
            )

    # ~~~~ Greedy incumbent (cost per newly covered leaf) to initialize the
    # upper bound of the branch-and-bound.
    uncovered = set(target)
    greedy_selection: List[int] = []
    greedy_cost = 0
    while uncovered:
        best_i, best_key = -1, None
        for i in range(n_cand):
            gain = len(covers[i] & uncovered)
            if gain == 0:
                continue
            crit = (costs[i] / gain, costs[i], repr(keys[i]))
            if best_key is None or crit < best_key:
                best_key, best_i = crit, i
        if best_i < 0:  # pragma: no cover - prevented by feasibility check
            raise ExplanationSolverError(
                "Greedy covering failed: the prediction set cannot be "
                "explained exactly."
            )
        greedy_selection.append(best_i)
        greedy_cost += costs[best_i]
        uncovered -= covers[best_i]

    best_cost = greedy_cost
    best_selection: Tuple[int, ...] = tuple(greedy_selection)
    expansions = 0

    def search(uncovered, avail, chosen, cost):
        """Branch-and-bound recursion.

        :param frozenset uncovered: leaves of the target not covered yet.
        :param tuple avail: indices of candidates covering at least one
            uncovered leaf (candidates covering only covered leaves can
            never belong to a minimum-cost solution since costs are
            positive).
        :param tuple chosen: indices of the candidates selected so far.
        :param int cost: cost of the candidates selected so far.
        """
        nonlocal best_cost, best_selection, expansions

        if cost >= best_cost:
            return
        if not uncovered:
            best_cost = cost
            best_selection = chosen
            return

        expansions += 1
        if expansions > max_expand:
            raise ExplanationSolverError(
                f"The exact cover solver exceeded its search budget "
                f"({max_expand} expansions) while explaining a set of "
                f"{len(target)} leaves. Increase `max_expand` or simplify "
                "the DAG."
            )

        # Lower bound on the additional cost: each further candidate covers
        # at most `max_cover` uncovered leaves, hence at least `need` more
        # candidates, costing at least the `need` cheapest available costs.
        max_cover = max(len(covers[i] & uncovered) for i in avail)
        need = -(-len(uncovered) // max_cover)  # ceil division
        cheapest = sorted(costs[i] for i in avail)[:need]
        if cost + sum(cheapest) >= best_cost:
            return

        # Branch on the uncovered leaf coverable by the fewest candidates.
        def n_covering(leaf):
            return sum(1 for i in avail if leaf in covers[i])

        leaf = min(uncovered, key=lambda x: (n_covering(x), repr(x)))
        branching = sorted(
            (i for i in avail if leaf in covers[i]),
            key=lambda i: (costs[i], repr(keys[i])),
        )
        for i in branching:
            new_uncovered = uncovered - covers[i]
            new_avail = tuple(j for j in avail if j != i and covers[j] & new_uncovered)
            search(new_uncovered, new_avail, chosen + (i,), cost + costs[i])

    search(
        frozenset(target),
        tuple(range(n_cand)),
        (),
        0,
    )

    selected = tuple(sorted((keys[i] for i in best_selection), key=repr))
    return selected, best_cost


class DAGLabelExplainer:
    """Explains conformal classification prediction sets with the concepts
    of a multi-parent DAG of labels.

    The DAG relates fine-grained classes (its leaves) to coarser concepts
    (its internal nodes). A node may have **several parents**. Each node
    stands for the whole set of its descendant leaves (itself if it is a
    leaf) and is associated with a positive integer cost.

    Given a fitted conformal classifier (e.g.
    :class:`deel.puncc.classification.LAC` or
    :class:`deel.puncc.classification.RAPS`), :meth:`explain` first lets the
    predictor produce its prediction sets on new data (the predicted leaf
    class sets are frozen at this point), then re-expresses each set as a
    minimum-total-cost collection of nodes whose descendant leaves union
    exactly to the predicted set. The optimization is exact
    (branch-and-bound): it is **not** a greedy tree heuristic, overlapping
    nodes are allowed and, when several optimal explanations exist, one is
    returned deterministically. Empty prediction sets are supported (empty
    explanation, zero cost).

    :param dict children: adjacency of the DAG, mapping each node label to
        the iterable of its children node labels. Nodes that never appear as
        keys (or map to an empty iterable) are the leaves. A node may appear
        in the child lists of several parents (multi-parent DAG).
    :param dict costs: mapping of every node label to its cost. Costs must
        be positive integers.
    :param dict class_to_leaf: mapping of each class column of the predictor
        (integer, as produced in the prediction sets) to the corresponding
        leaf label of the DAG. The mapping must be a bijection between the
        class columns and the leaves of the DAG.
    :param int max_expand: maximum number of search nodes expanded by the
        exact solver for a single prediction set before raising an
        :class:`ExplanationSolverError`. Defaults to 100000.

    :raises DAGCycleError: the graph defined by `children` contains a cycle.
    :raises InvalidCostsError: a cost is missing, is defined for an unknown
        node, or is not a positive integer.
    :raises InvalidMappingError: `class_to_leaf` is not a bijection between
        integer class columns and the leaves of the DAG.

    .. _example dag explainer:

    Example::

        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        from deel.puncc.api.prediction import BasePredictor
        from deel.puncc.classification import LAC
        from deel.puncc.dag_explanation import DAGLabelExplainer

        # Generate a random 4-class classification problem
        X, y = make_classification(n_samples=500, n_features=6, n_informative=4,
                                   n_classes=4, random_state=0, shuffle=False)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2, random_state=0)
        X_fit, X_calib, y_fit, y_calib = train_test_split(
            X_train, y_train, test_size=.25, random_state=0)

        # Wrap a random forest so that predict returns class probabilities
        class RFPredictor(BasePredictor):
            def predict(self, X, **kwargs):
                return self.model.predict_proba(X, **kwargs)

        rf_predictor = RFPredictor(RandomForestClassifier(random_state=0))

        # Fit LAC conformal classifier
        lac_cp = LAC(rf_predictor)
        lac_cp.fit(X_fit=X_fit, y_fit=y_fit, X_calib=X_calib, y_calib=y_calib)

        # DAG of labels: leaves are the classifier classes
        children = {
            "animal": ["mammal", "bird", "fish"],
            "mammal": ["cat", "dog"],
        }
        costs = {"animal": 5, "mammal": 3,
                 "cat": 1, "dog": 1, "bird": 1, "fish": 1}
        class_to_leaf = {0: "cat", 1: "dog", 2: "bird", 3: "fish"}

        explainer = DAGLabelExplainer(children, costs, class_to_leaf)

        # Predict and explain the prediction sets
        result = explainer.explain(lac_cp, X_test, alpha=.1)

        # Actual predictions, explanations and costs
        print(result.y_pred)
        print(result.explanations)
        print(result.costs)

        # Inverse resolution: explanations back to leaf class sets, ready
        # for the native coverage and set size metrics
        leaf_sets = result.resolve()
        print(result.mean_coverage(y_test), result.mean_size())

        # Membership query: is class 0 (resp. concept "mammal") in row 0 ?
        print(result.is_member(0, 0), result.is_member(0, "mammal"))
    """

    def __init__(
        self,
        children: Mapping[Hashable, Iterable[Hashable]],
        costs: Mapping[Hashable, int],
        class_to_leaf: Mapping[int, Hashable],
        max_expand: int = 100000,
    ):
        self._set_structure(children)
        self._set_costs(costs)
        self._set_mapping(class_to_leaf)

        # ~~~~~~~~~~~~~~~~ Solver budget ~~~~~~~~~~~~~~~~
        if (
            isinstance(max_expand, bool)
            or not isinstance(max_expand, Integral)
            or int(max_expand) < 0
        ):
            raise DAGExplanationError(
                "Argument `max_expand` must be a non-negative integer, got "
                f"{max_expand!r}."
            )
        self._max_expand = int(max_expand)

    # ~~~~~~~~~~~~~~~~ Initialization helpers ~~~~~~~~~~~~~~~~

    def _set_structure(self, children: Mapping[Hashable, Iterable[Hashable]]) -> None:
        """Normalize and validate the DAG adjacency, then precompute the
        parent relations and the descendant leaf sets."""
        if not isinstance(children, Mapping):
            raise DAGExplanationError(
                "Argument `children` must be a mapping of node labels to "
                f"iterables of node labels, got {type(children)}."
            )

        norm_children: Dict[Hashable, FrozenSet[Hashable]] = {}
        for node, childs in children.items():
            _check_hashable(node)
            try:
                childs = frozenset(childs)
            except TypeError as exc:
                raise DAGExplanationError(
                    f"Children of node {node!r} must be an iterable of "
                    f"node labels, got {childs!r}."
                ) from exc
            for child in childs:
                _check_hashable(child, "child")
            norm_children[node] = childs

        nodes = set(norm_children)
        for childs in norm_children.values():
            nodes |= childs
        self._nodes = frozenset(nodes)
        # Nodes with no children entry are leaves (empty children set).
        self._children = {n: norm_children.get(n, frozenset()) for n in self._nodes}
        self._leaves = frozenset(n for n in self._nodes if not self._children[n])

        if not self._nodes:
            raise DAGExplanationError("The provided label DAG is empty.")

        self._check_acyclic()

        # Parents (multi-parent DAG)
        self._parents = {n: set() for n in self._nodes}
        for parent, childs in self._children.items():
            for child in childs:
                self._parents[child].add(parent)
        self._parents = {n: frozenset(p) for n, p in self._parents.items()}

        # Descendant leaf sets
        self._leaf_sets = self._compute_leaf_sets()
        # Deterministic ordering of the (node, leaf set) pairs.
        self._node_leaf_items = tuple(
            sorted(self._leaf_sets.items(), key=lambda kv: repr(kv[0]))
        )

    def _set_costs(self, costs: Mapping[Hashable, int]) -> None:
        """Validate the node costs (positive integers, one per node)."""
        if not isinstance(costs, Mapping):
            raise InvalidCostsError(
                "Argument `costs` must be a mapping of node labels to "
                f"positive integer costs, got {type(costs)}."
            )
        unknown_cost_nodes = set(costs) - self._nodes
        if unknown_cost_nodes:
            raise InvalidCostsError(
                "Costs are provided for unknown node(s) "
                f"{sorted(map(repr, unknown_cost_nodes))}."
            )
        self._costs: Dict[Hashable, int] = {}
        for node in self._nodes:
            if node not in costs:
                raise InvalidCostsError(
                    f"No cost provided for node {node!r}. Every node of "
                    "the DAG must have a positive integer cost."
                )
            cost = costs[node]
            if isinstance(cost, bool) or not isinstance(cost, Integral):
                raise InvalidCostsError(
                    f"Cost of node {node!r} must be a positive integer, "
                    f"got {cost!r}."
                )
            cost = int(cost)
            if cost < 1:
                raise InvalidCostsError(
                    f"Cost of node {node!r} must be a positive integer, " f"got {cost}."
                )
            self._costs[node] = cost

    def _set_mapping(self, class_to_leaf: Mapping[int, Hashable]) -> None:
        """Validate the class column to leaf label mapping (bijection with
        the DAG leaves)."""
        if not isinstance(class_to_leaf, Mapping):
            raise InvalidMappingError(
                "Argument `class_to_leaf` must be a mapping of integer "
                f"class columns to leaf labels, got {type(class_to_leaf)}."
            )
        self._class_to_leaf: Dict[int, Hashable] = {}
        leaf_to_class: Dict[Hashable, int] = {}
        for cls, leaf in class_to_leaf.items():
            if isinstance(cls, bool) or not isinstance(cls, Integral):
                raise InvalidMappingError(
                    "Class columns must be non-negative integers, got key " f"{cls!r}."
                )
            cls = int(cls)
            if cls < 0:
                raise InvalidMappingError(
                    "Class columns must be non-negative integers, got key " f"{cls}."
                )
            _check_hashable(leaf, "leaf")
            if leaf not in self._nodes:
                raise InvalidMappingError(
                    f"Class column {cls} is mapped to {leaf!r} which is "
                    "not a node of the DAG."
                )
            if leaf not in self._leaves:
                raise InvalidMappingError(
                    f"Class column {cls} is mapped to {leaf!r} which is "
                    "not a leaf of the DAG."
                )
            if leaf in leaf_to_class:
                raise InvalidMappingError(
                    f"Leaf {leaf!r} is mapped to both class columns "
                    f"{leaf_to_class[leaf]} and {cls}. The mapping must be "
                    "a bijection between class columns and DAG leaves."
                )
            self._class_to_leaf[cls] = leaf
            leaf_to_class[leaf] = cls

        if not self._class_to_leaf:
            raise InvalidMappingError(
                "The class column to leaf label mapping is empty."
            )

        unmapped_leaves = self._leaves - set(leaf_to_class)
        if unmapped_leaves:
            raise InvalidMappingError(
                "DAG leave(s) "
                f"{sorted(map(repr, unmapped_leaves))} are not mapped to "
                "any class column. The mapping must be a bijection between "
                "class columns and DAG leaves."
            )
        self._leaf_to_class = leaf_to_class

    def _check_acyclic(self) -> None:
        """Depth-first cycle detection.

        :raises DAGCycleError: the graph contains a cycle (the error message
            reports one such cycle).
        """
        white, gray, black = 0, 1, 2
        color = {n: white for n in self._nodes}
        exhausted = object()  # sentinel (None may be a legit node label)
        for root in sorted(self._nodes, key=repr):
            if color[root] != white:
                continue
            color[root] = gray
            path = [root]
            stack = [(root, iter(sorted(self._children[root], key=repr)))]
            while stack:
                node, it = stack[-1]
                child = next(it, exhausted)
                if child is exhausted:
                    color[node] = black
                    stack.pop()
                    path.pop()
                    continue
                if color[child] == white:
                    color[child] = gray
                    path.append(child)
                    stack.append((child, iter(sorted(self._children[child], key=repr))))
                elif color[child] == gray:
                    idx = path.index(child)
                    cycle = path[idx:] + [child]
                    raise DAGCycleError(
                        "The provided label graph contains a cycle: "
                        + " -> ".join(repr(n) for n in cycle)
                        + ". The label graph must be a DAG."
                    )
                # black children: already fully explored, skip.

    def _compute_leaf_sets(self) -> Dict[Hashable, FrozenSet[Hashable]]:
        """Compute the set of descendant leaves of every node (itself for
        a leaf) with an iterative post-order traversal."""
        leaf_sets: Dict[Hashable, FrozenSet[Hashable]] = {}
        for start in self._nodes:
            if start in leaf_sets:
                continue
            stack = [start]
            while stack:
                node = stack[-1]
                if node in leaf_sets:
                    stack.pop()
                    continue
                pending = [c for c in self._children[node] if c not in leaf_sets]
                if pending:
                    stack.extend(pending)
                    continue
                stack.pop()
                if not self._children[node]:
                    leaf_sets[node] = frozenset({node})
                else:
                    leaves: set = set()
                    for child in self._children[node]:
                        leaves |= leaf_sets[child]
                    leaf_sets[node] = frozenset(leaves)
        return leaf_sets

    # ~~~~~~~~~~~~~~~~ Public structure inspection ~~~~~~~~~~~~~~~~

    @property
    def nodes(self) -> FrozenSet[Hashable]:
        """All node labels of the DAG."""
        return self._nodes

    @property
    def leaves(self) -> FrozenSet[Hashable]:
        """Leaf labels of the DAG (the classifier classes)."""
        return self._leaves

    @property
    def n_classes(self) -> int:
        """Number of mapped class columns."""
        return len(self._class_to_leaf)

    @property
    def costs(self) -> Dict[Hashable, int]:
        """Mapping of node labels to their (positive integer) cost."""
        return dict(self._costs)

    @property
    def class_to_leaf(self) -> Dict[int, Hashable]:
        """Mapping of class columns to leaf labels."""
        return dict(self._class_to_leaf)

    def children_of(self, node: Hashable) -> FrozenSet[Hashable]:
        """Children of `node` in the DAG.

        :param node: a node label.

        :returns: children of `node`.
        :rtype: frozenset

        :raises DAGExplanationError: unknown node.
        """
        if node not in self._nodes:
            raise DAGExplanationError(f"Unknown node {node!r}.")
        return self._children[node]

    def parents_of(self, node: Hashable) -> FrozenSet[Hashable]:
        """Parents of `node` in the DAG (possibly several, possibly none).

        :param node: a node label.

        :returns: parents of `node`.
        :rtype: frozenset

        :raises DAGExplanationError: unknown node.
        """
        if node not in self._nodes:
            raise DAGExplanationError(f"Unknown node {node!r}.")
        return self._parents[node]

    def leaf_set(self, node: Hashable) -> FrozenSet[Hashable]:
        """All descendant leaves of `node` (itself if it is a leaf).

        :param node: a node label.

        :returns: descendant leaves of `node`.
        :rtype: frozenset

        :raises DAGExplanationError: unknown node.
        """
        if node not in self._nodes:
            raise DAGExplanationError(f"Unknown node {node!r}.")
        return self._leaf_sets[node]

    def invert(self, nodes: Iterable[Hashable]) -> List[int]:
        """Inverse resolution of an explanation: compute the leaf class
        columns covered by a collection of nodes.

        :param Iterable nodes: node labels.

        :returns: sorted list of the covered class columns.
        :rtype: List[int]

        :raises DAGExplanationError: unknown node.
        """
        leaves: set = set()
        for node in nodes:
            if node not in self._nodes:
                raise DAGExplanationError(f"Unknown node {node!r}.")
            leaves |= self._leaf_sets[node]
        return sorted(self._leaf_to_class[leaf] for leaf in leaves)

    # ~~~~~~~~~~~~~~~~ Explanation ~~~~~~~~~~~~~~~~

    def _check_mapping_consistency(self, n_classes: int) -> None:
        """Check that the class columns of the mapping are exactly
        {0, ..., n_classes - 1}.

        :raises InvalidMappingError: the mapping does not match the classes
            of the predictor.
        """
        expected = set(range(n_classes))
        provided = set(self._class_to_leaf)
        if provided != expected:
            missing = sorted(expected - provided)
            extra = sorted(provided - expected)
            raise InvalidMappingError(
                f"The predictor outputs {n_classes} classes (columns 0 to "
                f"{n_classes - 1}) but the class to leaf mapping does not "
                f"match them: missing class column(s) {missing}, unexpected "
                f"class column(s) {extra}."
            )

    def _normalize_sets(
        self, set_pred: Sequence[Iterable], n_rows: int, n_classes: int
    ) -> List[FrozenSet[int]]:
        """Validate and freeze the prediction sets as frozensets of integer
        class columns.

        :raises DAGExplanationError: malformed prediction sets.
        """
        if isinstance(set_pred, tuple):
            set_pred = list(set_pred)
        if not isinstance(set_pred, list) or len(set_pred) != n_rows:
            raise DAGExplanationError(
                "The prediction sets must be provided as a list of "
                f"{n_rows} collections of class columns, got "
                f"{type(set_pred)} of length "
                f"{len(set_pred) if hasattr(set_pred, '__len__') else '?'}."
            )
        frozen_sets = []
        for i, row in enumerate(set_pred):
            try:
                iterator = iter(row)
            except TypeError as exc:
                raise DAGExplanationError(
                    f"Prediction set of row {i} is not an iterable of "
                    f"class columns: {row!r}."
                ) from exc
            row_set = set()
            for cls in iterator:
                if isinstance(cls, bool) or not isinstance(cls, Integral):
                    raise DAGExplanationError(
                        f"Prediction set of row {i} contains {cls!r} which "
                        "is not an integer class column."
                    )
                cls = int(cls)
                if not 0 <= cls < n_classes:
                    raise DAGExplanationError(
                        f"Prediction set of row {i} contains class column "
                        f"{cls} but the predictor outputs {n_classes} "
                        "classes."
                    )
                row_set.add(cls)
            frozen_sets.append(frozenset(row_set))
        return frozen_sets

    def _solve(self, target: FrozenSet[Hashable]) -> Tuple[Tuple, int]:
        """Minimum-cost exact cover of a target leaf set."""
        if not target:
            return (), 0
        keys, covers = [], []
        for node, leaf_set in self._node_leaf_items:
            if leaf_set <= target:
                keys.append(node)
                covers.append(leaf_set)
        costs = tuple(self._costs[node] for node in keys)
        return _solve_min_cost_cover(
            tuple(keys),
            tuple(covers),
            costs,
            target,
            self._max_expand,
        )

    def _build_result(
        self,
        frozen_sets: List[FrozenSet[int]],
        y_pred: Optional[Iterable],
        alpha: Optional[float],
    ) -> "DAGExplanationResult":
        """Explain each frozen prediction set and build the result."""
        explanations, explanation_costs = [], []
        for row_set in frozen_sets:
            target = frozenset(self._class_to_leaf[cls] for cls in row_set)
            selected, cost = self._solve(target)
            explanations.append(selected)
            explanation_costs.append(cost)
        return DAGExplanationResult(
            explainer=self,
            prediction_sets=frozen_sets,
            explanations=explanations,
            costs=explanation_costs,
            y_pred=y_pred,
            alpha=alpha,
        )

    def explain(
        self, cp_predictor: Any, X: Iterable, alpha: float
    ) -> "DAGExplanationResult":
        """Predict with a fitted conformal classifier and explain the
        resulting prediction sets.

        The conformal predictor (e.g. a fitted
        :class:`deel.puncc.classification.LAC` or
        :class:`deel.puncc.classification.RAPS`) is called **once** on `X`;
        the produced leaf class sets are frozen and each of them is
        explained by a minimum-total-cost collection of DAG nodes.

        :param cp_predictor: fitted conformal classifier whose `predict`
            method returns a `(y_pred, prediction_sets)` tuple, where
            `y_pred` has shape `(n_samples, n_classes)` and
            `prediction_sets` is a list of collections of class columns.
        :param Iterable X: features of the new samples.
        :param float alpha: target maximum miscoverage.

        :returns: actual predictions, explanations and costs, with inverse
            resolution and membership queries.
        :rtype: DAGExplanationResult

        :raises DAGExplanationError: the predictor output is not a
            classification `(y_pred, prediction_sets)` tuple.
        :raises InvalidMappingError: the class to leaf mapping does not
            match the classes of the predictor.
        :raises ExplanationSolverError: the exact solver failed.
        """
        output = cp_predictor.predict(X, alpha=alpha)
        if not isinstance(output, tuple) or len(output) != 2:
            raise DAGExplanationError(
                "DAG label explanation expects a classification conformal "
                "predictor whose `predict` method returns a "
                "`(y_pred, prediction_sets)` tuple, got "
                f"{type(output)} of length "
                f"{len(output) if hasattr(output, '__len__') else '?'}."
            )
        y_pred, set_pred = output

        try:
            y_pred_arr = np.asarray(y_pred)
        except Exception as exc:
            raise DAGExplanationError(
                "Could not convert the predictor output `y_pred` to an array."
            ) from exc
        if y_pred_arr.ndim != 2:
            raise DAGExplanationError(
                "The predictor output `y_pred` must have shape "
                f"(n_samples, n_classes), got shape {y_pred_arr.shape}."
            )
        n_rows, n_classes = y_pred_arr.shape

        self._check_mapping_consistency(n_classes)
        frozen_sets = self._normalize_sets(set_pred, n_rows, n_classes)

        logger.info(
            "Explaining %d prediction sets over %d classes.",
            n_rows,
            n_classes,
        )
        return self._build_result(frozen_sets, y_pred=y_pred, alpha=alpha)

    def explain_sets(
        self,
        prediction_sets: Sequence[Iterable],
        n_classes: Optional[int] = None,
    ) -> "DAGExplanationResult":
        """Explain already computed (frozen) prediction sets.

        :param prediction_sets: list of collections of class columns.
        :param int n_classes: number of classes of the predictor. Defaults
            to the largest mapped class column plus one.

        :returns: explanations and costs, with inverse resolution and
            membership queries.
        :rtype: DAGExplanationResult

        :raises DAGExplanationError: malformed prediction sets.
        :raises InvalidMappingError: the class to leaf mapping does not
            match `n_classes`.
        :raises ExplanationSolverError: the exact solver failed.
        """
        if n_classes is None:
            n_classes = max(self._class_to_leaf) + 1
        if (
            isinstance(n_classes, bool)
            or not isinstance(n_classes, Integral)
            or int(n_classes) < 1
        ):
            raise DAGExplanationError(
                "Argument `n_classes` must be a positive integer, got "
                f"{n_classes!r}."
            )
        n_classes = int(n_classes)
        self._check_mapping_consistency(n_classes)
        prediction_sets = list(prediction_sets)
        frozen_sets = self._normalize_sets(
            prediction_sets, len(prediction_sets), n_classes
        )
        return self._build_result(frozen_sets, y_pred=None, alpha=None)


class DAGExplanationResult:
    """Container of the DAG label explanations of conformal prediction sets.

    Stores the actual predictions of the conformal classifier, the frozen
    leaf class prediction sets, their explanations (minimum-cost collections
    of DAG nodes) and the explanation costs. Provides inverse resolution of
    the explanations back to leaf class sets (in the native prediction set
    format, ready for :mod:`deel.puncc.metrics`) and membership queries.

    Instances of this class are built by
    :meth:`DAGLabelExplainer.explain` and
    :meth:`DAGLabelExplainer.explain_sets`.
    """

    def __init__(
        self,
        explainer: DAGLabelExplainer,
        prediction_sets: List[FrozenSet[int]],
        explanations: List[Tuple[Hashable, ...]],
        costs: List[int],
        y_pred: Optional[Iterable],
        alpha: Optional[float],
    ):
        self._explainer = explainer
        self._prediction_sets = tuple(prediction_sets)
        self._explanations = tuple(explanations)
        self._costs = tuple(costs)
        self.y_pred = y_pred
        self.alpha = alpha

        # Inverse resolution of the explanations: leaf labels and class
        # columns covered by the selected nodes of each row.
        self._resolved_leaves: List[FrozenSet[Hashable]] = []
        self._resolved_classes: List[FrozenSet[int]] = []
        for selected in self._explanations:
            leaves: set = set()
            for node in selected:
                leaves |= explainer.leaf_set(node)
            self._resolved_leaves.append(frozenset(leaves))
            self._resolved_classes.append(
                frozenset(explainer._leaf_to_class[leaf] for leaf in leaves)
            )

    def __len__(self) -> int:
        """Number of explained prediction sets."""
        return len(self._explanations)

    def __repr__(self) -> str:
        return (
            f"DAGExplanationResult(n_rows={len(self)}, "
            f"alpha={self.alpha}, total_cost={self.total_cost()})"
        )

    def _check_row(self, i: int) -> int:
        """Validate a row index."""
        if (
            isinstance(i, bool)
            or not isinstance(i, Integral)
            or not 0 <= int(i) < len(self)
        ):
            raise IndexError(
                f"Row index {i!r} out of range for {len(self)} explained "
                "prediction sets."
            )
        return int(i)

    @property
    def prediction_sets(self) -> List[List[int]]:
        """Frozen leaf class prediction sets (sorted class columns), as
        produced by the conformal predictor."""
        return [sorted(s) for s in self._prediction_sets]

    @property
    def explanations(self) -> List[Tuple[Hashable, ...]]:
        """Explanation of each prediction set: tuple of the selected DAG
        node labels per row."""
        return list(self._explanations)

    @property
    def costs(self) -> List[int]:
        """Total cost of the explanation of each prediction set."""
        return list(self._costs)

    def total_cost(self) -> int:
        """Sum of the explanation costs over all the prediction sets."""
        return sum(self._costs)

    def selected_nodes(self, i: int) -> Tuple[Hashable, ...]:
        """Nodes selected to explain the prediction set of row `i`.

        :param int i: row index.

        :returns: tuple of the selected node labels.
        :rtype: tuple
        """
        return self._explanations[self._check_row(i)]

    def covered_leaves(self, i: int) -> FrozenSet[Hashable]:
        """Leaf labels covered by the explanation of row `i` (inverse
        resolution at the DAG label level).

        :param int i: row index.

        :returns: covered leaf labels.
        :rtype: frozenset
        """
        return self._resolved_leaves[self._check_row(i)]

    def covered_classes(self, i: int) -> List[int]:
        """Class columns covered by the explanation of row `i` (inverse
        resolution at the class column level).

        :param int i: row index.

        :returns: sorted covered class columns.
        :rtype: List[int]
        """
        return sorted(self._resolved_classes[self._check_row(i)])

    def resolve(self, i: Optional[int] = None):
        """Inverse resolution of the explanations: leaf class sets covered
        by the selected nodes, in the native prediction set format.

        The output can be passed directly to
        :func:`deel.puncc.metrics.classification_mean_coverage` and
        :func:`deel.puncc.metrics.classification_mean_size`: the metrics are
        evaluated on the resolved leaf classes, not on the numbers of
        selected nodes.

        :param int i: optional row index. If provided, only the resolved
            class set of row `i` is returned.

        :returns: list of sorted class column lists (one per row), or the
            sorted class column list of row `i`.
        :rtype: List[List[int]] or List[int]
        """
        if i is not None:
            return self.covered_classes(i)
        return [sorted(s) for s in self._resolved_classes]

    def is_member(self, i: int, label: Any) -> bool:
        """Membership query for the explained prediction set of row `i`.

        If `label` is a node of the DAG, the query checks whether the
        concept is fully covered by the explanation of row `i`, i.e. whether
        all its descendant leaves belong to the explained set (this holds
        whether the node itself or other nodes were selected to cover them).
        Otherwise, if `label` is an integer class column of the predictor,
        the query checks its membership to the explained leaf class set.

        :param int i: row index.
        :param label: DAG node label or integer class column.

        :returns: True if `label` belongs to the explained set of row `i`.
        :rtype: bool

        :raises DAGExplanationError: `label` is neither a DAG node nor a
            class column of the predictor.
        """
        i = self._check_row(i)
        if isinstance(label, Hashable) and label in self._explainer.nodes:
            return self._explainer.leaf_set(label) <= self._resolved_leaves[i]
        if isinstance(label, Integral) and not isinstance(label, bool):
            label = int(label)
            if label not in self._explainer._class_to_leaf:
                raise DAGExplanationError(
                    f"Class column {label} is not a class of the predictor."
                )
            return label in self._resolved_classes[i]
        raise DAGExplanationError(
            f"Label {label!r} is neither a node of the DAG nor a class "
            "column of the predictor."
        )

    def mean_coverage(self, y_true: Iterable) -> float:
        """Empirical coverage of the explained prediction sets, computed
        with the native
        :func:`deel.puncc.metrics.classification_mean_coverage` metric on
        the inverse-resolved leaf class sets.

        :param Iterable y_true: observed labels.

        :returns: average coverage.
        :rtype: float
        """
        return metrics.classification_mean_coverage(y_true, self.resolve())

    def mean_size(self) -> float:
        """Average size of the explained prediction sets, computed with the
        native :func:`deel.puncc.metrics.classification_mean_size` metric on
        the inverse-resolved leaf class sets (leaf classes are counted, not
        the selected upper-level nodes).

        :returns: average prediction set size.
        :rtype: float
        """
        return metrics.classification_mean_size(self.resolve())
