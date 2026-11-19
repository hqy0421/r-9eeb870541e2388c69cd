.. _dag_explanation:

🏷️ DAG label explanation
=========================

Conformal classification prediction sets may contain many fine-grained
classes. When the classes are organized in a multi-parent DAG (directed
acyclic graph) of coarser concepts, each prediction set can be explained by a
minimum-cost collection of concepts: every node stands for the whole set of
its descendant leaves, and the selected nodes union exactly to the original
prediction set.

Given a fitted conformal classifier (such as
:class:`deel.puncc.classification.LAC` or
:class:`deel.puncc.classification.RAPS`), the explainer first lets the
predictor produce its prediction sets on new data, then re-expresses each set
as a minimum-total-cost collection of nodes. The optimization is exact
(branch-and-bound, no greedy tree heuristic), overlapping nodes are allowed
and empty prediction sets are supported. The inverse resolution of the
explanations back to leaf class sets feeds the native
:doc:`coverage and set size metrics <metrics>`.

.. autoclass:: deel.puncc.dag_explanation.DAGLabelExplainer

.. autoclass:: deel.puncc.dag_explanation.DAGExplanationResult
