"""
build_adfa_variants.py  --  Phase 1, Step 1B
=============================================
PURPOSE:
  For each positive (correct) Tier 4 example, generate 4 variant types.
  These variants are used by filter_adfa_neurons.py to check whether
  a neuron fires for the RIGHT reason (true construction) vs surface
  reasons (format, alphabet, task difficulty).

READS:
  data/adfa/contrastive_positives.jsonl

PRODUCES:
  data/adfa/variants_v1_alphabet_swap.jsonl
  data/adfa/variants_v2_structural_sibling.jsonl
  data/adfa/variants_v3_complexity_decoy.jsonl
  data/adfa/variants_v4_step_specific.jsonl

THE 4 VARIANT TYPES:

  V1 -- Alphabet Swap:
    Same regex structure, different alphabet symbols.
    (a|b)*abb over {a,b}  ->  (0|1)*011 over {0,1}
    A TRUE construction neuron fires on BOTH the original and V1.

  V2 -- Structural Sibling:
    A different Tier 4 regex with same complexity level.
    A TRUE construction neuron fires on BOTH the original and V2.

  V3 -- Complexity Decoy:
    A simple regex that LOOKS complex but has an easy DFA (2-3 states).
    A TRUE construction neuron should NOT fire here.

  V4 -- Step-Specific:
    The SAME example as the original, annotated with step-level
    correctness labels for use in VGNS (Phase 5).

ADFA FILTER RULE (enforced later in filter_adfa_neurons.py):
  A neuron i is a TRUE DFA-construction neuron if:
    ACE_good on V1 > threshold
    ACE_good on V2 > threshold
    ACE_good on V3 < threshold
  All three conditions must hold simultaneously.

USAGE:
  python build_adfa_variants.py \
      --positives_file data/adfa/contrastive_positives.jsonl \
      --output_dir     data/adfa
"""

import os
import re
import json
import copy
import random
import argparse
import logging
from typing import Optional, List

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)
random.seed(42)


# ===========================
#  VARIANT TYPE 1 -- ALPHABET SWAP
# ===========================

ALPHABET_SWAP_MAP = {
    frozenset(["a", "b"]):       ["0", "1"],
    frozenset(["0", "1"]):       ["a", "b"],
    frozenset(["a", "b", "c"]): ["x", "y", "z"],
    frozenset(["x", "y", "z"]): ["a", "b", "c"],
    frozenset(["0", "1", "2"]): ["a", "b", "c"],
}


def swap_symbols_in_string(text: str, old_syms: list, new_syms: list) -> str:
    result = text
    for i, sym in enumerate(old_syms):
        result = result.replace(sym, f"__SYM{i}__")
    for i, sym in enumerate(new_syms):
        result = result.replace(f"__SYM{i}__", sym)
    return result


def build_variant1_alphabet_swap(ex: dict) -> Optional[dict]:
    alphabet = ex["metadata"].get("alphabet", ["a", "b"])
    new_alphabet = ALPHABET_SWAP_MAP.get(frozenset(alphabet))
    if new_alphabet is None:
        return None

    old_sorted = sorted(alphabet)
    new_sorted  = sorted(new_alphabet)

    variant = copy.deepcopy(ex)

    user_content = variant["messages"][1]["content"]
    user_content = swap_symbols_in_string(user_content, old_sorted, new_sorted)
    variant["messages"][1]["content"] = user_content

    variant["metadata"] = copy.deepcopy(ex["metadata"])
    variant["metadata"]["alphabet"]          = new_sorted
    variant["metadata"]["original_alphabet"] = old_sorted

    if variant.get("gold_dfa"):
        old_dfa = variant["gold_dfa"]
        new_trans = {}
        for state, trans in old_dfa["transitions"].items():
            new_row = {}
            for sym, target in trans.items():
                new_sym = (new_sorted[old_sorted.index(sym)]
                           if sym in old_sorted else sym)
                new_row[new_sym] = target
            new_trans[state] = new_row
        variant["gold_dfa"]["transitions"] = new_trans
        variant["gold_dfa"]["alphabet"]    = new_sorted

    variant["variant_type"] = "v1_alphabet_swap"
    variant["source_regex"] = ex["metadata"].get("regex", "")
    variant["old_alphabet"] = old_sorted
    variant["new_alphabet"] = new_sorted
    variant["source_id"]    = ex["metadata"].get("id", "")

    return variant


# ===========================
#  VARIANT TYPE 2 -- STRUCTURAL SIBLING
# ===========================

def build_variant2_structural_sibling(ex: dict,
                                      all_positives: List[dict],
                                      ex_index: int = -1) -> Optional[dict]:
    alphabet     = ex["metadata"].get("alphabet", ["a", "b"])
    source_regex = ex["metadata"].get("regex", "")

    def is_different(p, p_idx):
        if p_idx == ex_index:
            return False
        if source_regex and p["metadata"].get("regex", "") == source_regex:
            return False
        return True

    gold_dfa    = ex.get("gold_dfa") or {}
    state_count = len(gold_dfa.get("states", []))

    candidates = [
        (p, i) for i, p in enumerate(all_positives)
        if is_different(p, i)
        and set(p["metadata"].get("alphabet", [])) == set(alphabet)
        and state_count > 0
        and abs(len((p.get("gold_dfa") or {}).get(
            "states", [])) - state_count) <= 2
    ]

    if not candidates:
        candidates = [
            (p, i) for i, p in enumerate(all_positives)
            if is_different(p, i)
            and set(p["metadata"].get("alphabet", [])) == set(alphabet)
        ]

    if not candidates:
        candidates = [
            (p, i) for i, p in enumerate(all_positives)
            if is_different(p, i)
        ]

    if not candidates:
        return None

    sibling, _ = random.choice(candidates)

    variant = copy.deepcopy(sibling)
    variant["variant_type"]  = "v2_structural_sibling"
    variant["source_id"]     = ex["metadata"].get("id", "")
    variant["source_regex"]  = ex["metadata"].get("regex", "")
    variant["sibling_id"]    = sibling["metadata"].get("id", "")
    variant["sibling_regex"] = sibling["metadata"].get("regex", "")

    return variant


# ===========================
#  VARIANT TYPE 3 -- COMPLEXITY DECOY
# ===========================

TIER2_REGEXES = {
    frozenset(["a", "b"]): [
        ("a*b",    "recognises strings of zero or more a's followed by one b"),
        ("ab*",    "recognises one a followed by zero or more b's"),
        ("a+b+",   "recognises one or more a's followed by one or more b's"),
        ("(a|b)*", "recognises any string over {a,b}"),
        ("a*b*",   "recognises zero or more a's followed by zero or more b's"),
        ("(ab)+",  "recognises one or more occurrences of ab"),
    ],
    frozenset(["0", "1"]): [
        ("0*1",    "recognises zero or more 0s followed by one 1"),
        ("01*",    "recognises one 0 followed by zero or more 1s"),
        ("(0|1)*", "recognises any binary string"),
        ("0+1+",   "recognises one or more 0s followed by one or more 1s"),
    ],
    frozenset(["a", "b", "c"]): [
        ("a*b*c*",   "recognises zero or more a's, b's, then c's"),
        ("(a|b|c)*", "recognises any string over {a,b,c}"),
        ("abc+",     "recognises ab followed by one or more c's"),
    ],
}


def build_variant3_complexity_decoy(ex: dict) -> Optional[dict]:
    alphabet = ex["metadata"].get("alphabet", ["a", "b"])
    decoys   = TIER2_REGEXES.get(frozenset(alphabet))
    if not decoys:
        return None

    decoy_regex, decoy_description = random.choice(decoys)

    orig_regex   = ex["metadata"].get("regex", "")
    user_content = ex["messages"][1]["content"]

    if orig_regex and orig_regex in user_content:
        new_user_content = user_content.replace(orig_regex, decoy_regex, 1)
    else:
        new_user_content = re.sub(
            r'`[^`]+`',
            f'`{decoy_regex}`',
            user_content,
            count=1
        )

    variant = copy.deepcopy(ex)
    variant["messages"][1]["content"] = new_user_content
    variant["metadata"] = copy.deepcopy(ex["metadata"])
    variant["metadata"]["regex"]          = decoy_regex
    variant["metadata"]["tier"]           = 2
    variant["metadata"]["tier_original"]  = ex["metadata"].get("tier", 4)

    variant["gold_dfa"]      = None
    variant["model_output"]  = None
    variant["variant_type"]  = "v3_complexity_decoy"
    variant["source_id"]     = ex["metadata"].get("id", "")
    variant["source_regex"]  = orig_regex
    variant["decoy_regex"]   = decoy_regex

    return variant


# ===========================
#  VARIANT TYPE 4 -- STEP-SPECIFIC
# ===========================

COT_STEP_PATTERNS = {
    "step1_language":  r"## Step 1",
    "step2_nfa_start": r"## Step 2",
    "step3_nfa_table": r"## Step 3",
    "step4_power_set": r"## Step 4",
    "step5_subset":    r"## Step 5",
    "step6_hopcroft":  r"## Step 6",
    "step7_final_dfa": r"## Step 7",
    "step8_verify":    r"## Step 8",
}


def check_step_presence(model_output: str) -> dict:
    return {
        step_name: bool(re.search(pattern, model_output, re.IGNORECASE))
        for step_name, pattern in COT_STEP_PATTERNS.items()
    }


def check_step3_nfa(model_output: str, alphabet: list) -> dict:
    step3_match = re.search(
        r'## Step 3.*?\n(.*?)(?=## Step 4|\Z)',
        model_output, re.DOTALL | re.IGNORECASE
    )
    if not step3_match:
        return {"present": False, "has_table": False,
                "col_count_correct": False}

    step3_text = step3_match.group(1)
    has_table  = bool(re.search(r'\|.*\|.*\|', step3_text))
    col_correct = False
    if has_table:
        header_match = re.search(r'\|([^\n]+)\|', step3_text)
        if header_match:
            cols = [c.strip() for c in header_match.group(1).split('|')
                    if c.strip()]
            col_correct = len(cols) >= len(alphabet) + 1

    return {
        "present":           True,
        "has_table":         has_table,
        "col_count_correct": col_correct,
    }


def check_step5_subset(model_output: str) -> dict:
    step5_match = re.search(
        r'## Step 5.*?\n(.*?)(?=## Step 6|\Z)',
        model_output, re.DOTALL | re.IGNORECASE
    )
    if not step5_match:
        return {"present": False, "has_table": False}
    step5_text = step5_match.group(1)
    has_table  = bool(re.search(r'\|.*D\d+.*\|', step5_text))
    return {"present": True, "has_table": has_table}


def check_step6_hopcroft(model_output: str) -> dict:
    step6_match = re.search(
        r'## Step 6.*?\n(.*?)(?=## Step 7|\Z)',
        model_output, re.DOTALL | re.IGNORECASE
    )
    if not step6_match:
        return {"present": False}
    step6_text = step6_match.group(1).lower()
    has_partition = any(kw in step6_text for kw in
                        ["partition", "equivalen", "minimis",
                         "minimiz", "class"])
    return {"present": True, "has_partition_language": has_partition}


def build_variant4_step_specific(ex: dict) -> dict:
    model_output = ex.get("model_output", "")
    alphabet     = ex["metadata"].get("alphabet", ["a", "b"])

    step_annotations = {
        "steps_present":   check_step_presence(model_output),
        "step3_nfa":       check_step3_nfa(model_output, alphabet),
        "step5_subset":    check_step5_subset(model_output),
        "step6_hopcroft":  check_step6_hopcroft(model_output),
        "overall_correct": ex.get("correct", False),
    }

    variant = copy.deepcopy(ex)
    variant["variant_type"]     = "v4_step_specific"
    variant["source_id"]        = ex["metadata"].get("id", "")
    variant["step_annotations"] = step_annotations

    return variant


# ===========================
#  MAIN
# ===========================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--positives_file", required=True)
    p.add_argument("--output_dir",     default="data/adfa")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    positives = []
    with open(args.positives_file) as f:
        for line in f:
            line = line.strip()
            if line:
                positives.append(json.loads(line))
    logger.info(f"Loaded {len(positives)} positive examples")

    # Variant 1: Alphabet Swap
    v1_list = []
    for ex in positives:
        v = build_variant1_alphabet_swap(ex)
        if v:
            v1_list.append(v)

    v1_path = os.path.join(args.output_dir, "variants_v1_alphabet_swap.jsonl")
    with open(v1_path, "w") as f:
        for v in v1_list:
            f.write(json.dumps(v) + "\n")
    logger.info(f"V1 (alphabet swap):       {len(v1_list):>3} variants -> {v1_path}")

    # Variant 2: Structural Sibling
    v2_list = []
    for ex_index, ex in enumerate(positives):
        v = build_variant2_structural_sibling(ex, positives, ex_index)
        if v:
            v2_list.append(v)

    v2_path = os.path.join(
        args.output_dir, "variants_v2_structural_sibling.jsonl")
    with open(v2_path, "w") as f:
        for v in v2_list:
            f.write(json.dumps(v) + "\n")
    logger.info(
        f"V2 (structural sibling):  {len(v2_list):>3} variants -> {v2_path}")

    # Variant 3: Complexity Decoy
    v3_list = []
    for ex in positives:
        v = build_variant3_complexity_decoy(ex)
        if v:
            v3_list.append(v)

    v3_path = os.path.join(
        args.output_dir, "variants_v3_complexity_decoy.jsonl")
    with open(v3_path, "w") as f:
        for v in v3_list:
            f.write(json.dumps(v) + "\n")
    logger.info(
        f"V3 (complexity decoy):    {len(v3_list):>3} variants -> {v3_path}")

    # Variant 4: Step-Specific
    v4_list = []
    for ex in positives:
        v = build_variant4_step_specific(ex)
        v4_list.append(v)

    v4_path = os.path.join(
        args.output_dir, "variants_v4_step_specific.jsonl")
    with open(v4_path, "w") as f:
        for v in v4_list:
            f.write(json.dumps(v) + "\n")
    logger.info(
        f"V4 (step-specific):       {len(v4_list):>3} variants -> {v4_path}")

    # Step annotation summary
    step3_present = sum(
        1 for v in v4_list
        if v["step_annotations"]["step3_nfa"]["present"])
    step5_present = sum(
        1 for v in v4_list
        if v["step_annotations"]["step5_subset"]["present"])
    step6_present = sum(
        1 for v in v4_list
        if v["step_annotations"]["step6_hopcroft"]["present"])
    all_steps = sum(
        1 for v in v4_list
        if all(v["step_annotations"]["steps_present"].values()))

    logger.info(f"\nStep presence in correct Tier 4 model outputs:")
    logger.info(f"  Step 3 (NFA table):      {step3_present}/{len(v4_list)}")
    logger.info(f"  Step 5 (subset constr.): {step5_present}/{len(v4_list)}")
    logger.info(f"  Step 6 (Hopcroft):       {step6_present}/{len(v4_list)}")
    logger.info(f"  All 8 steps present:     {all_steps}/{len(v4_list)}")

    summary = {
        "n_positives":   len(positives),
        "v1_count":      len(v1_list),
        "v2_count":      len(v2_list),
        "v3_count":      len(v3_list),
        "v4_count":      len(v4_list),
        "v1_coverage":   f"{len(v1_list)/len(positives)*100:.0f}%",
        "v2_coverage":   f"{len(v2_list)/len(positives)*100:.0f}%",
        "v3_coverage":   f"{len(v3_list)/len(positives)*100:.0f}%",
        "step3_present": step3_present,
        "step5_present": step5_present,
        "step6_present": step6_present,
    }
    summary_path = os.path.join(args.output_dir, "variants_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 55)
    logger.info("STEP 1B COMPLETE")
    logger.info(f"  V1 coverage: {summary['v1_coverage']} of positives")
    logger.info(f"  V2 coverage: {summary['v2_coverage']} of positives")
    logger.info(f"  V3 coverage: {summary['v3_coverage']} of positives")
    logger.info(f"  V4 coverage: 100% (same example + annotations)")
    logger.info("=" * 55)
    logger.info("Next step: run compute_ace_scores.py (Phase 2)")


if __name__ == "__main__":
    main()
