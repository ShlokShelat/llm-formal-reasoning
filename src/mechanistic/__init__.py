"""
mechanistic
===========
Mechanistic analysis pipeline — Section 4.2 and Appendix G.11.

Modules
-------
build_adfa_variants.py
    Generates four contrast variant types (V1-V4) from Tier 4 positives:
      V1: same DFA, symbols renamed (alphabet swap)
      V2: different regex, isomorphic DFA (structural sibling)
      V3: complex-looking regex, trivial 2-state DFA (complexity decoy)
      V4: same example with step-level correctness annotations (VGNS)

compute_ace_scores.py
    Activation Difference Scoring (ADS) across all 28 transformer layers.
    Samples 2,000 neurons per layer (56,000 neuron-layer pairs total).
    Identifies good (facilitating) and bad (inhibiting) neurons.
    Key finding: Layer 27 hosts 22 of the top-25 ADS neurons (execution peak).

extract_steering_vectors.py
    Extracts layer-wise steering vectors via three methods:
      ActAdd       : mean difference vector (fixed global shift)
      Probe dir    : logistic regression coefficient at Layer 10
      SADI         : sparse masking — top-10% dimensions by absolute value
    Key finding: Layer 10 encodes 87.1% of outcome information before
    the model completes half its computation.

run_steering_eval.py
    Evaluates all steering conditions on 285 Tier 4 mechanistic examples:
      Neuron-level : good/bad neuron amplification and suppression
      Representation: ActAdd, probe direction, SADI adaptive steering
      VGNS         : Verification-Guided Neuron Steering (4-round iterative)
    Key finding: VGNS recovers +2.4 pp — 35 examples remain unsolvable.
"""
