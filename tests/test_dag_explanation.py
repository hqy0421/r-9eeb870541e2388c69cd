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
import itertools
import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from deel.puncc import metrics
from deel.puncc.api.prediction import BasePredictor
from deel.puncc.classification import LAC
from deel.puncc.classification import RAPS
from deel.puncc.dag_explanation import DAGCycleError
from deel.puncc.dag_explanation import DAGExplanationError
from deel.puncc.dag_explanation import DAGLabelExplainer
from deel.puncc.dag_explanation import ExplanationSolverError
from deel.puncc.dag_explanation import InvalidCostsError
from deel.puncc.dag_explanation import InvalidMappingError

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Fixtures and helpers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def make_animal_dag():
    """Multi-parent DAG: 'eagle' has two parents ('bird' and 'flyer')."""
    children = {
        "animal": ["bird", "mammal", "fish"],
        "bird": ["eagle", "penguin"],
        "flyer": ["eagle", "bee"],
        "mammal": ["cat", "dog"],
    }
    costs = {
        "animal": 8,
        "bird": 3,
        "flyer": 3,
        "mammal": 3,
        "eagle": 1,
        "penguin": 1,
        "bee": 1,
        "cat": 1,
        "dog": 1,
        "fish": 1,
    }
    class_to_leaf = {
        0: "cat",
        1: "dog",
        2: "eagle",
        3: "penguin",
        4: "bee",
        5: "fish",
    }
    return children, costs, class_to_leaf


def make_explainer(**kwargs):
    children, costs, class_to_leaf = make_animal_dag()
    return DAGLabelExplainer(children, costs, class_to_leaf, **kwargs)


def brute_force_min_cost(explainer, target_classes):
    """Reference minimum-cost exact cover by exhaustive enumeration."""
    target = frozenset(explainer.class_to_leaf[c] for c in target_classes)
    if not target:
        return 0
    candidates = [
        (node, explainer.leaf_set(node))
        for node in explainer.nodes
        if explainer.leaf_set(node) <= target
    ]
    best = None
    for r in range(1, len(candidates) + 1):
        for combo in itertools.combinations(candidates, r):
            union = frozenset().union(*(c[1] for c in combo))
            if union == target:
                cost = sum(explainer.costs[c[0]] for c in combo)
                if best is None or cost < best:
                    best = cost
    return best


def fit_lac(train=True, **lac_kwargs):
    """Fit a LAC conformal classifier on synthetic 6-class data."""
    X, y = make_classification(
        n_samples=400,
        n_features=8,
        n_informative=6,
        n_classes=6,
        n_clusters_per_class=1,
        random_state=0,
        shuffle=False,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, test_size=0.34, random_state=0
    )

    class RFPredictor(BasePredictor):
        def predict(self, X, **kwargs):
            return self.model.predict_proba(X, **kwargs)

    rf_predictor = RFPredictor(RandomForestClassifier(n_estimators=30, random_state=0))
    lac_cp = LAC(rf_predictor, **lac_kwargs)
    lac_cp.fit(X_fit=X_fit, y_fit=y_fit, X_calib=X_calib, y_calib=y_calib)
    return lac_cp, X_test, y_test


def fit_raps(**raps_kwargs):
    """Fit a RAPS conformal classifier on synthetic 6-class data."""
    X, y = make_classification(
        n_samples=400,
        n_features=8,
        n_informative=6,
        n_classes=6,
        n_clusters_per_class=1,
        random_state=0,
        shuffle=False,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, test_size=0.34, random_state=0
    )

    class RFPredictor(BasePredictor):
        def predict(self, X, **kwargs):
            return self.model.predict_proba(X, **kwargs)

    rf_predictor = RFPredictor(RandomForestClassifier(n_estimators=30, random_state=0))
    raps_cp = RAPS(rf_predictor, **raps_kwargs)
    raps_cp.fit(X_fit=X_fit, y_fit=y_fit, X_calib=X_calib, y_calib=y_calib)
    return raps_cp, X_test, y_test


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Structure validation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class TestDAGStructure(unittest.TestCase):
    def test_multi_parent_structure(self):
        explainer = make_explainer()
        # eagle has two parents
        self.assertEqual(explainer.parents_of("eagle"), frozenset({"bird", "flyer"}))
        self.assertEqual(explainer.parents_of("animal"), frozenset())
        self.assertEqual(explainer.children_of("bird"), frozenset({"eagle", "penguin"}))
        # leaf sets
        self.assertEqual(
            explainer.leaf_set("animal"),
            frozenset({"cat", "dog", "eagle", "penguin", "fish"}),
        )
        self.assertEqual(explainer.leaf_set("flyer"), frozenset({"eagle", "bee"}))
        self.assertEqual(explainer.leaf_set("cat"), frozenset({"cat"}))
        # leaves / nodes
        self.assertEqual(
            explainer.leaves,
            frozenset({"cat", "dog", "eagle", "penguin", "bee", "fish"}),
        )
        self.assertEqual(explainer.n_classes, 6)

    def test_leaf_only_dag(self):
        # DAG reduced to its leaves (flat structure)
        explainer = DAGLabelExplainer(
            children={"a": [], "b": []},
            costs={"a": 1, "b": 2},
            class_to_leaf={0: "a", 1: "b"},
        )
        res = explainer.explain_sets([[0, 1], [1], []])
        self.assertEqual(res.explanations, [("a", "b"), ("b",), ()])
        self.assertEqual(res.costs, [3, 2, 0])

    def test_node_defined_only_as_child(self):
        # leaves appear only as children values, never as keys
        explainer = DAGLabelExplainer(
            children={"root": ["x", "y"]},
            costs={"root": 1, "x": 1, "y": 1},
            class_to_leaf={0: "x", 1: "y"},
        )
        self.assertEqual(explainer.leaves, frozenset({"x", "y"}))
        res = explainer.explain_sets([[0, 1]])
        self.assertEqual(res.explanations, [("root",)])
        self.assertEqual(res.costs, [1])

    def test_cycle_raises(self):
        with self.assertRaises(DAGCycleError) as ctx:
            DAGLabelExplainer(
                children={"a": ["b"], "b": ["c"], "c": ["a"]},
                costs={"a": 1, "b": 1, "c": 1},
                class_to_leaf={0: "a"},
            )
        self.assertIn("cycle", str(ctx.exception).lower())

    def test_self_loop_raises(self):
        with self.assertRaises(DAGCycleError):
            DAGLabelExplainer(
                children={"a": ["a"]},
                costs={"a": 1},
                class_to_leaf={0: "a"},
            )

    def test_empty_dag_raises(self):
        with self.assertRaises(DAGExplanationError):
            DAGLabelExplainer(children={}, costs={}, class_to_leaf={})

    def test_non_mapping_children_raises(self):
        with self.assertRaises(DAGExplanationError):
            DAGLabelExplainer(
                children=[("a", ["b"])],
                costs={"a": 1, "b": 1},
                class_to_leaf={0: "b"},
            )

    def test_unhashable_labels_raise(self):
        with self.assertRaises(DAGExplanationError):
            DAGLabelExplainer(
                children={"a": [["b"]]},
                costs={"a": 1},
                class_to_leaf={0: "a"},
            )


class TestCostsValidation(unittest.TestCase):
    def setUp(self):
        self.children, _, self.class_to_leaf = make_animal_dag()
        self.valid_costs = make_animal_dag()[1]

    def test_missing_cost_raises(self):
        costs = dict(self.valid_costs)
        del costs["animal"]
        with self.assertRaises(InvalidCostsError) as ctx:
            DAGLabelExplainer(self.children, costs, self.class_to_leaf)
        self.assertIn("animal", str(ctx.exception))

    def test_unknown_node_cost_raises(self):
        costs = dict(self.valid_costs)
        costs["ghost"] = 1
        with self.assertRaises(InvalidCostsError):
            DAGLabelExplainer(self.children, costs, self.class_to_leaf)

    def test_non_positive_cost_raises(self):
        for bad in (0, -3):
            costs = dict(self.valid_costs)
            costs["mammal"] = bad
            with self.assertRaises(InvalidCostsError):
                DAGLabelExplainer(self.children, costs, self.class_to_leaf)

    def test_non_integer_cost_raises(self):
        for bad in (1.5, "2", True, None):
            costs = dict(self.valid_costs)
            costs["mammal"] = bad
            with self.assertRaises(InvalidCostsError):
                DAGLabelExplainer(self.children, costs, self.class_to_leaf)

    def test_numpy_integer_cost_accepted(self):
        costs = dict(self.valid_costs)
        costs["mammal"] = np.int64(4)
        explainer = DAGLabelExplainer(self.children, costs, self.class_to_leaf)
        self.assertEqual(explainer.costs["mammal"], 4)


class TestMappingValidation(unittest.TestCase):
    def setUp(self):
        self.children, self.costs, self.class_to_leaf = make_animal_dag()

    def test_unknown_leaf_raises(self):
        mapping = dict(self.class_to_leaf)
        mapping[0] = "ghost"
        with self.assertRaises(InvalidMappingError):
            DAGLabelExplainer(self.children, self.costs, mapping)

    def test_non_leaf_target_raises(self):
        mapping = dict(self.class_to_leaf)
        mapping[0] = "mammal"
        with self.assertRaises(InvalidMappingError) as ctx:
            DAGLabelExplainer(self.children, self.costs, mapping)
        self.assertIn("not a leaf", str(ctx.exception))

    def test_duplicate_leaf_raises(self):
        mapping = dict(self.class_to_leaf)
        mapping[1] = "cat"
        with self.assertRaises(InvalidMappingError) as ctx:
            DAGLabelExplainer(self.children, self.costs, mapping)
        self.assertIn("bijection", str(ctx.exception))

    def test_unmapped_dag_leaf_raises(self):
        mapping = dict(self.class_to_leaf)
        del mapping[5]  # 'fish' no longer mapped
        with self.assertRaises(InvalidMappingError) as ctx:
            DAGLabelExplainer(self.children, self.costs, mapping)
        self.assertIn("fish", str(ctx.exception))

    def test_non_integer_class_column_raises(self):
        for bad in ("0", 1.5, True, -2):
            mapping = dict(self.class_to_leaf)
            mapping[bad] = mapping.pop(0)
            with self.assertRaises(InvalidMappingError):
                DAGLabelExplainer(self.children, self.costs, mapping)

    def test_empty_mapping_raises(self):
        with self.assertRaises(InvalidMappingError):
            DAGLabelExplainer(self.children, self.costs, {})

    def test_mapping_mismatch_at_explain_time(self):
        # Mapping covers 6 classes but predictor outputs 3
        explainer = make_explainer()
        y_pred = np.full((2, 3), 1 / 3)

        class StubCP:
            def predict(self, X, alpha):
                return y_pred, [[0, 1], [2]]

        with self.assertRaises(InvalidMappingError):
            explainer.explain(StubCP(), X=np.zeros((2, 1)), alpha=0.1)

    def test_explain_sets_n_classes_mismatch(self):
        explainer = make_explainer()
        with self.assertRaises(InvalidMappingError):
            explainer.explain_sets([[0, 1]], n_classes=3)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Exact solver ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class TestExactSolver(unittest.TestCase):
    def test_overlap_required_optimum(self):
        # Optimal explanation of {x, y, z} is the overlapping pair (P, Q):
        # P = {x, y}, Q = {y, z}, cost 2 + 2 = 4. Any non-overlapping cover
        # costs at least 2 + 4 = 6 (P + z) or 4 + 4 = 8 (x + y + z is 12).
        children = {"P": ["x", "y"], "Q": ["y", "z"]}
        costs = {"P": 2, "Q": 2, "x": 4, "y": 4, "z": 4}
        class_to_leaf = {0: "x", 1: "y", 2: "z"}
        explainer = DAGLabelExplainer(children, costs, class_to_leaf)
        res = explainer.explain_sets([[0, 1, 2]])
        self.assertEqual(res.costs, [4])
        self.assertEqual(set(res.explanations[0]), {"P", "Q"})
        self.assertEqual(res.resolve(), [[0, 1, 2]])

    def test_not_tree_greedy(self):
        # A tree-style disjoint cover cannot use both A and B (they overlap
        # at y): its best is A + z or x + B with cost 7, or C with cost 8.
        # The global optimum is the overlapping pair (A, B) with cost 6.
        children = {"C": ["A", "B"], "A": ["x", "y"], "B": ["y", "z"]}
        costs = {"C": 8, "A": 3, "B": 3, "x": 4, "y": 4, "z": 4}
        class_to_leaf = {0: "x", 1: "y", 2: "z"}
        explainer = DAGLabelExplainer(children, costs, class_to_leaf)
        res = explainer.explain_sets([[0, 1, 2]])
        self.assertEqual(res.costs, [6])
        self.assertEqual(set(res.explanations[0]), {"A", "B"})

    def test_multiple_optima_deterministic(self):
        # Three optimal explanations with cost 4: (P, Q), (P, R) and (Q, R)
        children = {"P": ["x", "y"], "Q": ["y", "z"], "R": ["x", "z"]}
        costs = {"P": 2, "Q": 2, "R": 2, "x": 5, "y": 5, "z": 5}
        class_to_leaf = {0: "x", 1: "y", 2: "z"}
        explainer = DAGLabelExplainer(children, costs, class_to_leaf)
        res1 = explainer.explain_sets([[0, 1, 2]])
        res2 = explainer.explain_sets([[0, 1, 2]])
        self.assertEqual(res1.costs, [4])
        self.assertEqual(len(res1.explanations[0]), 2)
        # Deterministic choice among the multiple optima
        self.assertEqual(res1.explanations, res2.explanations)
        self.assertEqual(res1.resolve(), [[0, 1, 2]])

    def test_empty_prediction_set(self):
        explainer = make_explainer()
        res = explainer.explain_sets([[], [0], []])
        self.assertEqual(res.explanations, [(), ("cat",), ()])
        self.assertEqual(res.costs, [0, 1, 0])
        self.assertEqual(res.resolve(), [[], [0], []])
        self.assertEqual(res.prediction_sets, [[], [0], []])

    def test_optimality_against_brute_force(self):
        # Random multi-parent DAGs: the solver cost must match the
        # exhaustive minimum and the union must be exact.
        rng = np.random.default_rng(1234)
        for trial in range(8):
            n_leaves = 5
            n_internal = 4
            leaves = [f"l{i}" for i in range(n_leaves)]
            internal = [f"n{i}" for i in range(n_internal)]
            # Random DAG: internal nodes form layers, edges go from higher
            # to lower layers (acyclic by construction), multi-parent allowed
            children = {leaf: [] for leaf in leaves}
            layers = [leaves] + [[internal[0]], internal[1:]]
            for layer_idx in range(1, len(layers)):
                for node in layers[layer_idx]:
                    pool = [x for lower in layers[:layer_idx] for x in lower]
                    n_child = rng.integers(1, 4)
                    chosen = rng.choice(
                        pool, size=min(n_child, len(pool)), replace=False
                    )
                    children[node] = [str(c) for c in chosen]
            nodes = leaves + internal
            costs = {n: int(rng.integers(1, 8)) for n in nodes}
            class_to_leaf = {i: leaf for i, leaf in enumerate(leaves)}
            explainer = DAGLabelExplainer(children, costs, class_to_leaf)

            # Random prediction sets (including the empty set)
            targets = [set() for _ in range(2)]
            for _ in range(12):
                size = rng.integers(1, n_leaves + 1)
                targets.append(set(rng.choice(n_leaves, size=size, replace=False)))
            sets = [sorted(t) for t in targets]
            res = explainer.explain_sets(sets)
            for row, target in enumerate(targets):
                expected_cost = brute_force_min_cost(explainer, target)
                self.assertEqual(
                    res.costs[row],
                    expected_cost,
                    msg=f"trial {trial}, target {sorted(target)}",
                )
                # Exactness: union of selected nodes == target leaf set
                self.assertEqual(res.resolve(row), sorted(target))

    def test_solver_budget_exceeded_raises(self):
        explainer = make_explainer(max_expand=0)
        with self.assertRaises(ExplanationSolverError):
            explainer.explain_sets([[0, 1, 2]])
        # Empty sets are explained without any search
        res = explainer.explain_sets([[]])
        self.assertEqual(res.explanations, [()])

    def test_invalid_max_expand_raises(self):
        children, costs, class_to_leaf = make_animal_dag()
        for bad in (-1, 2.5, True):
            with self.assertRaises(DAGExplanationError):
                DAGLabelExplainer(children, costs, class_to_leaf, max_expand=bad)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Explanation result ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class TestExplanationResult(unittest.TestCase):
    def setUp(self):
        self.explainer = make_explainer()
        self.res = self.explainer.explain_sets(
            [[0, 1], [2, 3, 4], [], [0, 1, 2, 3, 4, 5]]
        )

    def test_exactness_of_explanations(self):
        # Union of the descendant leaves of the selected nodes is exactly
        # the predicted set: no missing class, no extra class.
        for i in range(len(self.res)):
            predicted = set(self.res.prediction_sets[i])
            explained = set(self.res.resolve(i))
            self.assertEqual(predicted, explained)
            union = set()
            for node in self.res.selected_nodes(i):
                union |= self.explainer.leaf_set(node)
            self.assertEqual(
                union,
                {self.explainer.class_to_leaf[c] for c in predicted},
            )

    def test_inverse_resolution(self):
        res = self.res
        self.assertEqual(res.resolve(0), [0, 1])
        self.assertEqual(res.resolve(1), [2, 3, 4])
        self.assertEqual(res.resolve(2), [])
        self.assertEqual(res.resolve(), [[0, 1], [2, 3, 4], [], [0, 1, 2, 3, 4, 5]])
        self.assertEqual(res.covered_leaves(0), frozenset({"cat", "dog"}))
        self.assertEqual(res.covered_classes(1), [2, 3, 4])
        # Explainer-level inverse resolution
        self.assertEqual(self.explainer.invert(["mammal"]), [0, 1])
        self.assertEqual(self.explainer.invert([]), [])
        with self.assertRaises(DAGExplanationError):
            self.explainer.invert(["ghost"])

    def test_membership_queries(self):
        res = self.res
        # Row 0 is {cat, dog} explained by ('cat', 'dog') (cost 2 < mammal 3)
        self.assertTrue(res.is_member(0, 0))  # class column
        self.assertTrue(res.is_member(0, "cat"))  # leaf label
        self.assertFalse(res.is_member(0, 2))
        # Concept membership: all descendant leaves covered
        self.assertTrue(res.is_member(0, "mammal"))
        self.assertFalse(res.is_member(0, "animal"))
        self.assertFalse(res.is_member(0, "bird"))
        # Row 1 is {eagle, penguin, bee}: 'flyer' is fully covered even
        # though the optimal explanation may not select the node itself
        self.assertTrue(res.is_member(1, "flyer"))
        self.assertTrue(res.is_member(1, "bird"))
        self.assertFalse(res.is_member(1, "animal"))
        # Empty row: nothing is a member
        self.assertFalse(res.is_member(2, 0))
        self.assertFalse(res.is_member(2, "mammal"))
        # Unknown labels
        with self.assertRaises(DAGExplanationError):
            res.is_member(0, "ghost")
        with self.assertRaises(DAGExplanationError):
            res.is_member(0, 42)
        with self.assertRaises(DAGExplanationError):
            res.is_member(0, 1.5)
        # Row index validation
        with self.assertRaises(IndexError):
            res.is_member(10, 0)

    def test_native_metrics_on_resolved_sets(self):
        res = self.res
        y_true = np.array([0, 2, 1, 5])
        # The resolved leaf class sets feed the native metrics
        expected_cov = metrics.classification_mean_coverage(y_true, res.resolve())
        expected_size = metrics.classification_mean_size(res.resolve())
        self.assertEqual(res.mean_coverage(y_true), expected_cov)
        self.assertEqual(res.mean_size(), expected_size)
        # Coverage/size on resolved sets equal those on the predicted sets
        self.assertEqual(
            res.mean_coverage(y_true),
            metrics.classification_mean_coverage(y_true, res.prediction_sets),
        )
        self.assertEqual(
            res.mean_size(),
            metrics.classification_mean_size(res.prediction_sets),
        )

    def test_size_counts_leaves_not_nodes(self):
        # A single upper-level node explaining several leaves: the set size
        # must count leaf classes, not selected nodes.
        children = {"pair": ["a", "b"]}
        costs = {"pair": 1, "a": 2, "b": 2}
        class_to_leaf = {0: "a", 1: "b"}
        explainer = DAGLabelExplainer(children, costs, class_to_leaf)
        res = explainer.explain_sets([[0, 1]])
        self.assertEqual(res.explanations, [("pair",)])  # 1 node
        self.assertEqual(res.costs, [1])
        self.assertEqual(len(res.explanations[0]), 1)
        # Native mean size on the resolved sets: 2 leaf classes
        self.assertEqual(res.mean_size(), 2.0)
        self.assertEqual(
            res.mean_size(),
            metrics.classification_mean_size(res.resolve()),
        )

    def test_result_container_api(self):
        res = self.res
        self.assertEqual(len(res), 4)
        self.assertEqual(res.total_cost(), sum(res.costs))
        self.assertIsNone(res.y_pred)
        self.assertIsNone(res.alpha)
        self.assertIn("DAGExplanationResult", repr(res))
        # prediction_sets returns copies
        sets = res.prediction_sets
        sets[0].append(999)
        self.assertEqual(res.prediction_sets[0], [0, 1])
        with self.assertRaises(IndexError):
            res.selected_nodes(4)
        with self.assertRaises(IndexError):
            res.covered_classes(-1)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Integration with LAC / RAPS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class TestConformalIntegration(unittest.TestCase):
    def assert_explanations_valid(self, explainer, result, set_pred, y_test):
        n = len(result)
        # Actual predictions are returned
        self.assertIsNotNone(result.y_pred)
        # The leaf class sets are frozen copies of the predicted sets
        self.assertEqual(
            [set(s) for s in result.prediction_sets],
            [set(map(int, s)) for s in set_pred],
        )
        # One explanation and one cost per row
        self.assertEqual(len(result.explanations), n)
        self.assertEqual(len(result.costs), n)
        for i in range(n):
            # Exactness: no missing class, no extra class
            self.assertEqual(set(result.resolve(i)), set(map(int, set_pred[i])))
            # Cost consistency
            self.assertEqual(
                result.costs[i],
                sum(explainer.costs[node] for node in result.explanations[i]),
            )
        # Inverse resolution feeds the native metrics: coverage and size on
        # the resolved sets match those on the original prediction sets
        self.assertAlmostEqual(
            result.mean_coverage(y_test),
            metrics.classification_mean_coverage(y_test, set_pred),
        )
        self.assertAlmostEqual(
            result.mean_size(), metrics.classification_mean_size(set_pred)
        )

    def test_explain_lac(self):
        lac_cp, X_test, y_test = fit_lac()
        explainer = make_explainer()
        result = explainer.explain(lac_cp, X_test, alpha=0.1)
        _, set_pred = lac_cp.predict(X_test, alpha=0.1)
        self.assert_explanations_valid(explainer, result, set_pred, y_test)
        self.assertEqual(result.alpha, 0.1)

    def test_explain_raps(self):
        raps_cp, X_test, y_test = fit_raps(k_reg=2, lambd=0.5, rand=False)
        explainer = make_explainer()
        result = explainer.explain(raps_cp, X_test, alpha=0.1)
        _, set_pred = raps_cp.predict(X_test, alpha=0.1)
        self.assert_explanations_valid(explainer, result, set_pred, y_test)

    def test_explain_raps_randomized(self):
        np.random.seed(0)
        raps_cp, X_test, y_test = fit_raps(k_reg=2, lambd=0.5, rand=True)
        explainer = make_explainer()
        result = explainer.explain(raps_cp, X_test, alpha=0.1)
        # Invariants hold whatever the randomized sets are
        for i in range(len(result)):
            self.assertEqual(set(result.resolve(i)), set(result.prediction_sets[i]))
        self.assertTrue(0 <= result.mean_coverage(y_test) <= 1)

    def test_predict_called_once_and_sets_frozen(self):
        calls = []

        class CountingCP:
            def __init__(self):
                self.sets = [[0, 1], [2], []]

            def predict(self, X, alpha):
                calls.append(1)
                y_pred = np.full((3, 6), 1 / 6)
                return y_pred, self.sets

        cp = CountingCP()
        explainer = make_explainer()
        result = explainer.explain(cp, X=np.zeros((3, 2)), alpha=0.1)
        self.assertEqual(len(calls), 1)
        # Mutating the predictor output after the call does not affect the
        # frozen result
        cp.sets[0].append(5)
        self.assertEqual(result.prediction_sets[0], [0, 1])
        self.assertEqual(result.resolve(0), [0, 1])

    def test_unfitted_predictor_raises(self):
        class RFPredictor(BasePredictor):
            def predict(self, X, **kwargs):
                return self.model.predict_proba(X, **kwargs)

        rf_predictor = RFPredictor(RandomForestClassifier())
        lac_cp = LAC(rf_predictor)
        explainer = make_explainer()
        with self.assertRaises(RuntimeError):
            explainer.explain(lac_cp, np.zeros((2, 4)), alpha=0.1)

    def test_non_classification_output_raises(self):
        class RegressionCP:
            def predict(self, X, alpha):
                return None, np.zeros(2), np.ones(2)  # 3-tuple

        explainer = make_explainer()
        with self.assertRaises(DAGExplanationError):
            explainer.explain(RegressionCP(), np.zeros((2, 2)), alpha=0.1)

    def test_malformed_prediction_sets_raise(self):
        explainer = make_explainer()

        class BadSetsCP:
            def predict(self, X, alpha):
                y_pred = np.full((2, 6), 1 / 6)
                return y_pred, [[0, 1.5], [2]]  # non integer class

        with self.assertRaises(DAGExplanationError):
            explainer.explain(BadSetsCP(), np.zeros((2, 2)), alpha=0.1)

        class OutOfRangeCP:
            def predict(self, X, alpha):
                y_pred = np.full((2, 6), 1 / 6)
                return y_pred, [[0, 6], [2]]  # class column out of range

        with self.assertRaises(DAGExplanationError):
            explainer.explain(OutOfRangeCP(), np.zeros((2, 2)), alpha=0.1)

        class BadYPredCP:
            def predict(self, X, alpha):
                return np.zeros(3), [[0], [1], [2]]  # 1-D y_pred

        with self.assertRaises(DAGExplanationError):
            explainer.explain(BadYPredCP(), np.zeros((3, 2)), alpha=0.1)

    def test_numpy_int_classes_in_sets(self):
        # Prediction sets with numpy integer classes (as produced by RAPS)
        explainer = make_explainer()
        res = explainer.explain_sets([[np.int64(0), np.int32(1)], [np.int16(2)]])
        self.assertEqual(res.resolve(), [[0, 1], [2]])


if __name__ == "__main__":
    unittest.main()
