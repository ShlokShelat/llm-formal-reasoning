"""
compute_ace_scores.py  --  Phase 2: Activation Difference Scoring (ADS)
===========================
Replaces CE-based ACE scoring with activation difference scoring,
which is standard in mechanistic interpretability literature
(cf. Meng et al. 2022 "Locating and Editing Factual Associations in GPT").

METHOD:
  For every MLP neuron i in every layer l, compute:

  ADS_good(l, i) = mean_activation(i | correct examples)
                 - mean_activation(i | incorrect examples)
  -> Positive = neuron fires MORE on correct DFA construction
  -> High ADS_good = facilitating neuron (good neuron)

  ADS_bad(l, i)  = mean_activation(i | incorrect examples)
                 - mean_activation(i | correct examples)
  -> Positive = neuron fires MORE on incorrect outputs
  -> High ADS_bad = inhibiting neuron (bad neuron)

  Activations are collected at the act_fn output (SiLU output) of
  Qwen2MLP: gate_proj -> SiLU -> * up_proj -> down_proj
  Shape: (batch, seq_len, intermediate_dim=18944)
  We take the mean over all completion token positions.

OUTPUT FILES:
  ace_scores.json          -- full ADS scores for all neurons
  good_neurons.json        -- Top-K facilitating neurons
  bad_neurons.json         -- Top-K inhibiting neurons
  mixed_neurons.json       -- Top-K neurons high on both
  ace_layer_summary.json   -- per-layer aggregated stats

USAGE:
    python compute_ace_scores.py \
        --base_model   Qwen/Qwen2.5-7B-Instruct \
        --adapter_dir  checkpoints/qwen7b_lora \
        --adfa_file    data/adfa/contrastive_combined.jsonl \
        --output_dir   results/ace_cot_7b \
        --top_k        100 \
        --neurons_per_layer_sample 200
"""

import os
import json
import logging
import argparse
from collections import defaultdict

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
#  ACTIVATION COLLECTION
# ===========================

def collect_mlp_activations(model, tokenizer, examples,
                             layer_indices, device):
    """
    Collect mean MLP intermediate activations (SiLU output) for a list
    of examples. Returns dict: layer_idx -> np.array
    (n_examples, intermediate_dim).

    Activations are averaged over completion token positions only,
    giving one vector per example per layer.
    """
    activations = {l: [] for l in layer_indices}
    handles = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            activations[layer_idx].append(output.detach().cpu())
        return hook

    for layer_idx in layer_indices:
        handle = (
            model.model.model.layers[layer_idx]
            .mlp.act_fn.register_forward_hook(make_hook(layer_idx))
        )
        handles.append(handle)

    per_example_acts = {l: [] for l in layer_indices}

    try:
        for ex in tqdm(examples, desc="Collecting activations", leave=False):
            prompt = tokenizer.apply_chat_template(
                ex["messages"][:2],
                tokenize=False,
                add_generation_prompt=True
            )
            gold      = ex["messages"][2]["content"]
            full_text = prompt + gold

            inputs = tokenizer(full_text, return_tensors="pt").to(device)
            prompt_len = tokenizer(
                prompt, return_tensors="pt")["input_ids"].shape[1]

            for l in layer_indices:
                activations[l] = []

            with torch.no_grad():
                model(**inputs)

            for l in layer_indices:
                if activations[l]:
                    act = activations[l][0]
                    completion_act = act[0, prompt_len:, :]
                    if completion_act.shape[0] > 0:
                        mean_act = completion_act.mean(dim=0)
                    else:
                        mean_act = act[0].mean(dim=0)
                    per_example_acts[l].append(
                        mean_act.float().numpy())
    finally:
        for h in handles:
            h.remove()

    result = {}
    for l in layer_indices:
        if per_example_acts[l]:
            result[l] = np.stack(per_example_acts[l], axis=0)
        else:
            result[l] = np.zeros((0, 1))
    return result


# ===========================
#  GET MODEL STRUCTURE INFO
# ===========================

def get_model_info(model) -> dict:
    try:
        cfg = model.config
        n_layers         = cfg.num_hidden_layers
        hidden_dim       = cfg.hidden_size
        intermediate_dim = getattr(cfg, "intermediate_size", hidden_dim * 4)
        return {
            "n_layers":         n_layers,
            "hidden_dim":       hidden_dim,
            "intermediate_dim": intermediate_dim,
        }
    except Exception as e:
        logger.warning(f"Could not read model config: {e}")
        return {
            "n_layers": 28, "hidden_dim": 3584, "intermediate_dim": 18944}


# ===========================
#  MAIN SCORING LOOP
# ===========================

def compute_ace_scores(args):
    # Load model
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True)

    logger.info("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    logger.info(f"Loading adapter from {args.adapter_dir}...")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    info             = get_model_info(model)
    n_layers         = info["n_layers"]
    intermediate_dim = info["intermediate_dim"]
    device           = next(model.parameters()).device
    logger.info(
        f"Model: {n_layers} layers, "
        f"intermediate_dim={intermediate_dim}, device={device}")

    # Load ADFA contrastive data
    logger.info(f"Loading ADFA data from {args.adfa_file}...")
    positives, negatives = [], []
    with open(args.adfa_file) as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                if ex["label"] == 1:
                    positives.append(ex)
                else:
                    negatives.append(ex)
    logger.info(
        f"Loaded {len(positives)} positives, {len(negatives)} negatives")

    if args.max_examples:
        positives = positives[:args.max_examples]
        negatives = negatives[:args.max_examples]
        logger.info(
            f"Capped to {len(positives)} positives, "
            f"{len(negatives)} negatives")

    if len(negatives) == 0:
        logger.error(
            "No negatives found. "
            "Use contrastive_combined.jsonl (not just positives).")
        raise ValueError("No negatives in ADFA file.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Decide which layers to score
    if n_layers <= 32:
        layers_to_score = list(range(n_layers))
    else:
        layers_to_score = list(range(0, n_layers, 2))
    logger.info(f"Scoring {len(layers_to_score)} layers")

    # Collect activations
    logger.info("Collecting activations for CORRECT examples...")
    pos_acts = collect_mlp_activations(
        model, tokenizer, positives, layers_to_score, device)

    logger.info("Collecting activations for INCORRECT examples...")
    neg_acts = collect_mlp_activations(
        model, tokenizer, negatives, layers_to_score, device)

    # Compute ADS scores
    rng      = np.random.default_rng(42)
    n_sample = args.neurons_per_layer_sample

    ace_good = defaultdict(dict)
    ace_bad  = defaultdict(dict)

    logger.info(
        f"Computing ADS scores: "
        f"{len(layers_to_score)} layers x {n_sample} neurons sampled")

    for layer_idx in tqdm(layers_to_score, desc="ADS scoring"):
        pos = pos_acts.get(layer_idx)
        neg = neg_acts.get(layer_idx)

        if (pos is None or neg is None
                or pos.shape[0] == 0 or neg.shape[0] == 0):
            logger.warning(
                f"Layer {layer_idx}: missing activations, skipping")
            continue

        mean_pos = pos.mean(axis=0)
        mean_neg = neg.mean(axis=0)
        diff     = mean_pos - mean_neg

        neuron_sample = rng.choice(
            intermediate_dim,
            size=min(n_sample, intermediate_dim),
            replace=False
        ).tolist()

        for neuron_idx in neuron_sample:
            d = float(diff[neuron_idx])
            ace_good[layer_idx][neuron_idx] = max(d,  0.0)
            ace_bad[layer_idx][neuron_idx]  = max(-d, 0.0)

    # Save full scores
    scores_path = os.path.join(args.output_dir, "ace_scores.json")
    scores_data = {
        "ace_good": {
            str(l): {str(n): v for n, v in neurons.items()}
            for l, neurons in ace_good.items()},
        "ace_bad": {
            str(l): {str(n): v for n, v in neurons.items()}
            for l, neurons in ace_bad.items()},
        "meta": {
            "method":           "activation_difference_scoring_ADS",
            "n_layers":         n_layers,
            "intermediate_dim": intermediate_dim,
            "n_positives":      len(positives),
            "n_negatives":      len(negatives),
            "layers_scored":    layers_to_score,
            "neurons_per_layer": n_sample,
        }
    }
    with open(scores_path, "w") as f:
        json.dump(scores_data, f)
    logger.info(f"Saved ADS scores -> {scores_path}")

    # Identify top-K neurons
    logger.info(f"Identifying top-{args.top_k} good and bad neurons...")

    all_neurons = []
    for layer_idx, neurons in ace_good.items():
        for neuron_idx, good_score in neurons.items():
            bad_score = ace_bad.get(layer_idx, {}).get(neuron_idx, 0.0)
            all_neurons.append({
                "layer":    int(layer_idx),
                "neuron":   int(neuron_idx),
                "ace_good": good_score,
                "ace_bad":  bad_score,
                "net_good": good_score - bad_score,
                "net_bad":  bad_score - good_score,
            })

    good_neurons  = sorted(
        all_neurons, key=lambda x: x["net_good"], reverse=True)[:args.top_k]
    bad_neurons   = sorted(
        all_neurons, key=lambda x: x["net_bad"],  reverse=True)[:args.top_k]
    mixed_neurons = sorted(
        all_neurons,
        key=lambda x: x["ace_good"] + x["ace_bad"],
        reverse=True)[:args.top_k // 2]

    with open(os.path.join(args.output_dir, "good_neurons.json"), "w") as f:
        json.dump(good_neurons, f, indent=2)
    with open(os.path.join(args.output_dir, "bad_neurons.json"), "w") as f:
        json.dump(bad_neurons, f, indent=2)
    with open(os.path.join(args.output_dir, "mixed_neurons.json"), "w") as f:
        json.dump(mixed_neurons, f, indent=2)

    logger.info(
        f"Top good neuron: layer={good_neurons[0]['layer']}, "
        f"neuron={good_neurons[0]['neuron']}, "
        f"ads_good={good_neurons[0]['ace_good']:.4f}")
    logger.info(
        f"Top bad  neuron: layer={bad_neurons[0]['layer']}, "
        f"neuron={bad_neurons[0]['neuron']}, "
        f"ads_bad={bad_neurons[0]['ace_bad']:.4f}")

    # Per-layer summary
    layer_summary = []
    for layer_idx in layers_to_score:
        neurons = ace_good.get(layer_idx, {})
        if not neurons:
            continue
        good_vals = list(neurons.values())
        bad_vals  = [
            ace_bad.get(layer_idx, {}).get(n, 0) for n in neurons]
        layer_summary.append({
            "layer":          layer_idx,
            "mean_ace_good":  float(np.mean(good_vals)),
            "max_ace_good":   float(np.max(good_vals)),
            "mean_ace_bad":   float(np.mean(bad_vals)),
            "max_ace_bad":    float(np.max(bad_vals)),
            "n_good_neurons": sum(1 for v in good_vals if v > 0.01),
            "n_bad_neurons":  sum(1 for v in bad_vals  if v > 0.01),
        })

    with open(
            os.path.join(args.output_dir, "ace_layer_summary.json"), "w"
    ) as f:
        json.dump(layer_summary, f, indent=2)
    logger.info("ADS scoring complete.")

    # Sanity check
    top_good = good_neurons[0]['ace_good'] if good_neurons else 0
    top_bad  = bad_neurons[0]['ace_bad']   if bad_neurons  else 0
    logger.info(
        f"Sanity: top ADS_good={top_good:.4f}, top ADS_bad={top_bad:.4f}")
    if top_good < 0.001 and top_bad < 0.001:
        logger.warning(
            "All ADS scores near zero -- check that positives/negatives "
            "differ in model behavior.")
    else:
        logger.info(
            "Scores look healthy. Proceed with filter_adfa_neurons.py.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",   required=True)
    p.add_argument("--adapter_dir",  required=True)
    p.add_argument("--adfa_file",    required=True)
    p.add_argument("--output_dir",   default="results/ace_cot_7b")
    p.add_argument("--top_k",        type=int, default=100)
    p.add_argument("--neurons_per_layer_sample", type=int, default=200)
    p.add_argument("--max_examples", type=int, default=None)
    args = p.parse_args()
    compute_ace_scores(args)


if __name__ == "__main__":
    main()
