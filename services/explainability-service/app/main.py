"""
Explainability Service

Generates human-readable explanations for scoring decisions using
a dual-mode approach: Gemini LLM narration with a rule-based fallback
that guarantees sub-50ms response when the LLM is unavailable.

Explanation sources:
  - TreeSHAP feature attribution from LightGBM and Autoencoder
  - PGExplainer edge importance from the TGN graph model
  - Graph evidence: ring membership, hub role, and SCC component
"""
# TODO: Implement LLM narration (Gemini 1.5 Flash, 6-second timeout)
# TODO: Implement rule-based fallback template engine
# TODO: Implement PGExplainer for TGN graph-level edge importance
