"""
run_steering_eval.py  --  Phase 3/4/5: Steering Evaluation
=========================
Evaluates ALL steering methods on the full Tier 4 test set:
  - Functional antagonism (good/bad neurons)
  - ActAdd (mean difference vector)
  - SADI adaptive (sparse masking)
  - Cross-model transfer (No-CoT direction)
  - VGNS (verification-guided adaptive loop)

Reads pre-computed:
  - good_neurons.json / bad_neurons.json  (from compute_ace_scores.py)
  - steering_vectors.json                 (from extract_steering_vectors.py)
  - cross_model_vectors.json              (from extract_steering_vectors.py)

Outputs:
  steering_results.json   -- accuracy table for all conditions
  vgns_results.json       -- per-example VGNS round breakdown

Usage:
    python run_steering_eval.py \
        --base_model      Qwen/Qwen2.5-7B-Instruct \
        --adapter_dir     checkpoints/cot_7b \
        --test_file       data/regex_dfa_dataset_test.jsonl \
        --ace_dir         results/ace_cot_7b \
        --steering_dir    results/steering_cot_7b \
        --output_dir      results/steering_eval_7b \
        --conditions      all
"""

import os
import re
import json
import logging
import argparse
from typing import Optional, List, Dict, Tuple
from itertools import product

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ===========================
#  DFA PARSER + VERIFIER
# ===========================

def parse_dfa(text: str, alphabet: list) -> Optional[dict]:
    match = re.search(
        r'## Step 7.*?Transition function.*?\n\|.*?\n\|[-| ]+\n'
        r'((?:\|.*?\n)+)',
        text, re.DOTALL
    )
    if not match:
        match = re.search(
            r'\| State \|.*?Accept.*?\n\|[-| ]+\n((?:\|.*?\n)+)',
            text, re.DOTALL
        )
    if not match:
        return None

    table_text = match.group(1)
    rows = [r.strip() for r in table_text.strip().split('\n')
            if r.strip().startswith('|')]
    transitions  = {}
    accept_states = set()
    states        = []
    sa            = sorted(alphabet)

    for row in rows:
        cells = [c.strip() for c in row.split('|') if c.strip()]
        if len(cells) < len(sa) + 2:
            continue
        m = re.match(r'D(\d+)', cells[0])
        if not m:
            continue
        sid = int(m.group(1))
        states.append(sid)
        # Accept marker: Y or checkmark
        if cells[-1].strip() in ('Y', 'y'):
            accept_states.add(sid)
        transitions[sid] = {}
        for j, sym in enumerate(sa):
            tm = re.match(r'D(\d+)', cells[j + 1])
            if tm:
                transitions[sid][sym] = int(tm.group(1))

    if not states:
        return None
    return {
        "states":      sorted(set(states)),
        "alphabet":    sa,
        "start":       0,
        "accept":      sorted(accept_states),
        "transitions": transitions,
    }


def dfa_accepts(dfa: dict, s: str) -> bool:
    state = dfa["start"]
    for ch in s:
        if ch not in dfa["transitions"].get(state, {}):
            return False
        state = dfa["transitions"][state][ch]
    return state in dfa["accept"]


def is_exact(pred: dict, gold: dict, alphabet: list,
             max_len: int = 6) -> bool:
    for length in range(max_len + 1):
        for chars in product(alphabet, repeat=length):
            s = "".join(chars)
            if dfa_accepts(pred, s) != dfa_accepts(gold, s):
                return False
    return True


def verify(pred_text: str, gold_text: str, alphabet: list) -> bool:
    pred_dfa = parse_dfa(pred_text, alphabet)
    gold_dfa = parse_dfa(gold_text, alphabet)
    if pred_dfa is None or gold_dfa is None:
        return False
    return is_exact(pred_dfa, gold_dfa, alphabet)


# ===========================
#  STEERING HOOKS
# ===========================

def make_antagonism_hooks(good_neurons: List[dict],
                          bad_neurons: List[dict],
                          alpha_good: float = 0.5,
                          alpha_bad: float = 0.5):
    good_by_layer = {}
    for n in good_neurons:
        good_by_layer.setdefault(n["layer"], []).append(n["neuron"])

    bad_by_layer = {}
    for n in bad_neurons:
        bad_by_layer.setdefault(n["layer"], []).append(n["neuron"])

    all_layers = set(
        list(good_by_layer.keys()) + list(bad_by_layer.keys()))
    hooks = {}

    for layer_idx in all_layers:
        good_idx = good_by_layer.get(layer_idx, [])
        bad_idx  = bad_by_layer.get(layer_idx, [])

        def make_hook(g_idx, b_idx, a_good, a_bad):
            def hook(module, input, output):
                out = (output[0] if isinstance(output, tuple)
                       else output)
                out = out.clone()
                if g_idx:
                    out[:, :, g_idx] *= (1.0 + a_good)
                if b_idx:
                    out[:, :, b_idx] *= (1.0 - a_bad)
                if isinstance(output, tuple):
                    return (out,) + output[1:]
                return out
            hook._hook_type = 'neuron'
            return hook

        hooks[layer_idx] = make_hook(
            good_idx, bad_idx, alpha_good, alpha_bad)

    return hooks


def make_static_vector_hook(vector: np.ndarray,
                            layer_idx: int,
                            alpha: float = 1.0):
    """ActAdd / mean diff: add fixed vector to hidden state."""
    v_tensor = torch.tensor(vector, dtype=torch.float32)

    def hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        out = out.clone()
        v   = v_tensor.to(out.device).to(out.dtype)
        out += alpha * v.unsqueeze(0).unsqueeze(0)
        if isinstance(output, tuple):
            return (out,) + output[1:]
        return out
    return hook


def make_sadi_hook(vector: np.ndarray, mask: np.ndarray,
                   layer_idx: int, delta: float = 2.0):
    """
    SADI adaptive hook: scale intervention by input's own activation.
    h_new[:, :, mask] += delta * sign(q[:, mask]) * v_sadi[mask]
    """
    v_tensor    = torch.tensor(vector, dtype=torch.float32)
    mask_tensor = torch.tensor(mask,   dtype=torch.bool)

    def hook(module, input, output):
        out  = output[0] if isinstance(output, tuple) else output
        out  = out.clone()
        v    = v_tensor.to(out.device).to(out.dtype)
        mask = mask_tensor.to(out.device)
        q    = out[:, -1, :]
        intervention = delta * q[:, mask].sign() * v[mask]
        out[:, :, mask] += intervention.unsqueeze(1)
        if isinstance(output, tuple):
            return (out,) + output[1:]
        return out
    return hook


# ===========================
#  GENERATION WITH HOOKS
# ===========================

def generate_with_hooks(model, tokenizer, messages,
                        hooks: Dict[int, callable],
                        max_new_tokens: int = 3500) -> str:
    prompt = tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True
    )
    inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
    handles = []

    for layer_idx, hook_fn in hooks.items():
        hook_type = getattr(hook_fn, '_hook_type', 'representation')
        if hook_type == 'neuron':
            layer = model.model.model.layers[layer_idx].mlp.act_fn
        else:
            layer = model.model.model.layers[layer_idx]
        handle = layer.register_forward_hook(hook_fn)
        handles.append(handle)

    try:
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        for h in handles:
            h.remove()

    return tokenizer.decode(
        out_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


# ===========================
#  EVALUATION CONDITIONS
# ===========================

def evaluate_condition(model, tokenizer, tier4_data,
                       hooks, condition_name):
    results = []
    for ex in tqdm(tier4_data, desc=condition_name):
        messages  = ex["messages"]
        alphabet  = ex.get("metadata", {}).get("alphabet", ["a", "b"])
        gold_text = messages[2]["content"]
        pred_text = generate_with_hooks(model, tokenizer, messages, hooks)
        correct   = verify(pred_text, gold_text, alphabet)
        results.append({
            "correct": correct,
            "tier":    ex.get("metadata", {}).get("tier", 4),
        })

    accuracy = (sum(r["correct"] for r in results) / len(results)
                if results else 0)
    return accuracy, results


# ===========================
#  VGNS LOOP
# ===========================

def run_vgns(model, tokenizer, tier4_data,
             good_neurons, bad_neurons,
             v_sadi, mask_sadi, best_layer,
             alpha_neuron=0.5, delta_sadi=2.0):
    """
    Verification-Guided Neuron Steering.
    For each example: try escalating steering until correct or exhausted.
    """
    vgns_results = []

    for ex in tqdm(tier4_data, desc="VGNS"):
        messages  = ex["messages"]
        alphabet  = ex.get("metadata", {}).get("alphabet", ["a", "b"])
        gold_text = messages[2]["content"]

        result = {
            "metadata":    ex.get("metadata", {}),
            "rounds":      [],
            "solved":      False,
            "solved_round": None,
        }

        # Round 1: No steering
        pred    = generate_with_hooks(model, tokenizer, messages, {})
        correct = verify(pred, gold_text, alphabet)
        result["rounds"].append(
            {"round": 1, "method": "none", "correct": correct})
        if correct:
            result["solved"]       = True
            result["solved_round"] = 1
            vgns_results.append(result)
            continue

        # Round 2: Neuron antagonism
        antag_hooks = make_antagonism_hooks(
            good_neurons, bad_neurons, alpha_neuron, alpha_neuron)
        pred    = generate_with_hooks(
            model, tokenizer, messages, antag_hooks)
        correct = verify(pred, gold_text, alphabet)
        result["rounds"].append(
            {"round": 2, "method": "antagonism", "correct": correct})
        if correct:
            result["solved"]       = True
            result["solved_round"] = 2
            vgns_results.append(result)
            continue

        # Round 3: SADI representation steering
        sadi_hooks = {
            best_layer: make_sadi_hook(
                v_sadi, mask_sadi, best_layer, delta_sadi)
        }
        pred    = generate_with_hooks(
            model, tokenizer, messages, sadi_hooks)
        correct = verify(pred, gold_text, alphabet)
        result["rounds"].append(
            {"round": 3, "method": "sadi", "correct": correct})
        if correct:
            result["solved"]       = True
            result["solved_round"] = 3
            vgns_results.append(result)
            continue

        # Round 4: Combined (stronger alpha)
        combined_hooks = make_antagonism_hooks(
            good_neurons, bad_neurons,
            alpha_neuron * 2.0, alpha_neuron * 2.0
        )
        combined_hooks[best_layer] = make_sadi_hook(
            v_sadi, mask_sadi, best_layer, delta_sadi * 2.0)
        pred    = generate_with_hooks(
            model, tokenizer, messages, combined_hooks)
        correct = verify(pred, gold_text, alphabet)
        result["rounds"].append(
            {"round": 4, "method": "combined", "correct": correct})
        result["solved"]       = correct
        result["solved_round"] = 4 if correct else None

        vgns_results.append(result)

    solved_counts = {1: 0, 2: 0, 3: 0, 4: 0, None: 0}
    for r in vgns_results:
        solved_counts[r["solved_round"]] = (
            solved_counts.get(r["solved_round"], 0) + 1)

    n       = len(vgns_results)
    summary = {
        "total":         n,
        "solved_total":  sum(r["solved"] for r in vgns_results),
        "final_accuracy": (sum(r["solved"] for r in vgns_results) / n
                           if n > 0 else 0),
        "solved_round1": solved_counts.get(1,    0),
        "solved_round2": solved_counts.get(2,    0),
        "solved_round3": solved_counts.get(3,    0),
        "solved_round4": solved_counts.get(4,    0),
        "unsolvable":    solved_counts.get(None, 0),
    }
    return vgns_results, summary


# ===========================
#  MAIN
# ===========================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",   required=True)
    p.add_argument("--adapter_dir",  required=True)
    p.add_argument("--test_file",    required=True)
    p.add_argument("--ace_dir",      required=True)
    p.add_argument("--steering_dir", required=True)
    p.add_argument("--output_dir",   default="results/steering_eval")
    p.add_argument("--conditions",   default="all")
    p.add_argument("--alpha_neuron", type=float, default=0.5)
    p.add_argument("--delta_sadi",   type=float, default=2.0)
    p.add_argument("--alpha_static", type=float, default=1.0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    # Load Tier 4 test data
    logger.info("Loading Tier 4 test data...")
    tier4_data = []
    with open(args.test_file) as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                if ex.get("metadata", {}).get("tier") == 4:
                    tier4_data.append(ex)
    logger.info(f"Found {len(tier4_data)} Tier 4 examples")

    # Load pre-computed neurons
    with open(os.path.join(args.ace_dir, "good_neurons.json")) as f:
        good_neurons = json.load(f)
    with open(os.path.join(args.ace_dir, "bad_neurons.json")) as f:
        bad_neurons = json.load(f)
    logger.info(
        f"Loaded {len(good_neurons)} good, {len(bad_neurons)} bad neurons")

    # Load steering vectors
    with open(
            os.path.join(args.steering_dir, "steering_vectors.json")
    ) as f:
        sv_data    = json.load(f)
    best_layer = sv_data["meta"]["best_layer"]
    logger.info(
        f"Best layer: {best_layer} "
        f"(probe acc={sv_data['meta']['best_layer_probe_acc']:.3f})")

    v_mean  = np.array(sv_data["vectors_mean"][str(best_layer)])
    v_probe = np.array(sv_data["vectors_probe"][str(best_layer)])
    v_sadi  = np.array(sv_data["vectors_sadi"][str(best_layer)])
    mask    = np.array(
        sv_data["masks_sadi"][str(best_layer)], dtype=bool)

    # Load transfer vector if available
    transfer_path = os.path.join(
        args.steering_dir, "cross_model_vectors.json")
    v_transfer          = None
    best_transfer_layer = best_layer
    if os.path.exists(transfer_path):
        with open(transfer_path) as f:
            transfer_data = json.load(f)
        best_transfer_layer = transfer_data["meta"]["best_transfer_layer"]
        v_transfer = np.array(
            transfer_data["transfer_vectors"][str(best_transfer_layer)])
        logger.info(
            f"Loaded cross-model transfer vector "
            f"(best layer={best_transfer_layer})")

    all_conditions = args.conditions == "all"
    results_table  = {}

    # Condition 1: Baseline
    if all_conditions or "baseline" in args.conditions:
        logger.info("--- Condition: Baseline (no steering) ---")
        acc, _ = evaluate_condition(
            model, tokenizer, tier4_data, {}, "Baseline")
        results_table["baseline"] = acc
        logger.info(f"Baseline Tier 4: {acc:.3f}")

    # Conditions 2-5: Neuron-level
    if all_conditions or "antagonism" in args.conditions:
        logger.info("--- Condition: Good neurons only ---")
        hooks = make_antagonism_hooks(
            good_neurons, [], args.alpha_neuron, 0)
        acc, _ = evaluate_condition(
            model, tokenizer, tier4_data, hooks, "Good only")
        results_table["good_only"] = acc
        logger.info(f"Good only Tier 4: {acc:.3f}")

        logger.info("--- Condition: Bad neurons only ---")
        hooks = make_antagonism_hooks(
            [], bad_neurons, 0, args.alpha_neuron)
        acc, _ = evaluate_condition(
            model, tokenizer, tier4_data, hooks, "Bad only")
        results_table["bad_only"] = acc
        logger.info(f"Bad only Tier 4: {acc:.3f}")

        logger.info("--- Condition: Functional antagonism ---")
        hooks = make_antagonism_hooks(
            good_neurons, bad_neurons,
            args.alpha_neuron, args.alpha_neuron)
        acc, _ = evaluate_condition(
            model, tokenizer, tier4_data, hooks, "Antagonism")
        results_table["antagonism"] = acc
        logger.info(f"Antagonism Tier 4: {acc:.3f}")

        logger.info("--- Condition: Random neurons (control) ---")
        n_layers         = model.config.num_hidden_layers
        intermediate_dim = model.config.intermediate_size
        rng = np.random.default_rng(42)
        random_good = [
            {"layer":  int(rng.integers(n_layers)),
             "neuron": int(rng.integers(intermediate_dim))}
            for _ in range(len(good_neurons))
        ]
        random_bad = [
            {"layer":  int(rng.integers(n_layers)),
             "neuron": int(rng.integers(intermediate_dim))}
            for _ in range(len(bad_neurons))
        ]
        hooks = make_antagonism_hooks(
            random_good, random_bad,
            args.alpha_neuron, args.alpha_neuron)
        acc, _ = evaluate_condition(
            model, tokenizer, tier4_data, hooks, "Random control")
        results_table["random_control"] = acc
        logger.info(f"Random control Tier 4: {acc:.3f}")

    # Conditions 6-8: Representation-level
    if all_conditions or "sadi" in args.conditions:
        logger.info("--- Condition: ActAdd (mean vector) ---")
        hooks = {best_layer: make_static_vector_hook(
            v_mean, best_layer, args.alpha_static)}
        acc, _ = evaluate_condition(
            model, tokenizer, tier4_data, hooks, "ActAdd")
        results_table["actadd_mean"] = acc
        logger.info(f"ActAdd mean Tier 4: {acc:.3f}")

        logger.info("--- Condition: Probe direction (static) ---")
        hooks = {best_layer: make_static_vector_hook(
            v_probe, best_layer, args.alpha_static)}
        acc, _ = evaluate_condition(
            model, tokenizer, tier4_data, hooks, "Probe dir")
        results_table["probe_direction"] = acc
        logger.info(f"Probe direction Tier 4: {acc:.3f}")

        logger.info("--- Condition: SADI adaptive ---")
        hooks = {best_layer: make_sadi_hook(
            v_sadi, mask, best_layer, args.delta_sadi)}
        acc, _ = evaluate_condition(
            model, tokenizer, tier4_data, hooks, "SADI")
        results_table["sadi_adaptive"] = acc
        logger.info(f"SADI adaptive Tier 4: {acc:.3f}")

    # Condition 9: Cross-model transfer
    if v_transfer is not None and (
            all_conditions or "transfer" in args.conditions):
        logger.info("--- Condition: Cross-model transfer ---")
        for alpha in [0.5, 1.0, 2.0, 5.0]:
            hooks = {best_transfer_layer: make_static_vector_hook(
                v_transfer, best_transfer_layer, alpha)}
            acc, _ = evaluate_condition(
                model, tokenizer, tier4_data, hooks,
                f"Transfer alpha={alpha}")
            results_table[f"transfer_alpha_{alpha}"] = acc
            logger.info(f"Transfer alpha={alpha} Tier 4: {acc:.3f}")

    # Condition 10: VGNS
    if all_conditions or "vgns" in args.conditions:
        logger.info(
            "--- Condition: VGNS (Verification-Guided Neuron Steering) ---")
        vgns_results, vgns_summary = run_vgns(
            model, tokenizer, tier4_data,
            good_neurons, bad_neurons,
            v_sadi, mask, best_layer,
            args.alpha_neuron, args.delta_sadi,
        )
        results_table["vgns"] = vgns_summary["final_accuracy"]
        logger.info(f"VGNS Tier 4: {vgns_summary['final_accuracy']:.3f}")
        logger.info(
            f"  Solved round 1: {vgns_summary['solved_round1']}")
        logger.info(
            f"  Solved round 2: {vgns_summary['solved_round2']}")
        logger.info(
            f"  Solved round 3: {vgns_summary['solved_round3']}")
        logger.info(
            f"  Solved round 4: {vgns_summary['solved_round4']}")
        logger.info(
            f"  Unsolvable:     {vgns_summary['unsolvable']}")

        vgns_path = os.path.join(args.output_dir, "vgns_results.json")
        with open(vgns_path, "w") as f:
            json.dump({
                "summary":     vgns_summary,
                "per_example": vgns_results,
            }, f, indent=2)
        logger.info(f"Saved VGNS results -> {vgns_path}")

    # Save results table
    results_path = os.path.join(args.output_dir, "steering_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing = json.load(f)
        existing.update(results_table)
        results_table = existing
    with open(results_path, "w") as f:
        json.dump(results_table, f, indent=2)
    logger.info(f"\nSaved results table -> {results_path}")

    # Print summary
    logger.info("\n" + "=" * 55)
    logger.info("STEERING RESULTS -- TIER 4 ACCURACY")
    logger.info("=" * 55)
    for condition, acc in sorted(
            results_table.items(), key=lambda x: -x[1]):
        logger.info(
            f"  {condition:<35} {acc:.3f}  ({acc*100:.1f}%)")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
