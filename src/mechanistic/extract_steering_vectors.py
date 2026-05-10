"""
extract_steering_vectors.py  --  Phase 3 + 4: Representation-Level Steering
=========================
Extracts layer-wise steering vectors from the ADFA contrastive dataset.
Implements: Mean difference (ActAdd), Linear probe (CAA),
SADI sparse masking. Also extracts the cross-model transfer vector
(No-CoT direction -> CoT model).

Outputs:
  steering_vectors.json        -- all vectors for all layers, all methods
  probe_accuracy.json          -- linear probe accuracy by layer
  cross_model_vectors.json     -- No-CoT advantage vector (if both given)

Usage (single model):
    python extract_steering_vectors.py \
        --base_model  Qwen/Qwen2.5-7B-Instruct \
        --adapter_dir checkpoints/cot_7b \
        --adfa_file   data/adfa/adfa_contrastive.jsonl \
        --output_dir  results/steering_cot_7b

Usage (cross-model transfer):
    python extract_steering_vectors.py \
        --base_model      Qwen/Qwen2.5-7B-Instruct \
        --adapter_dir     checkpoints/cot_7b \
        --nocot_adapter   checkpoints/no_cot_7b \
        --adfa_file       data/adfa/adfa_contrastive.jsonl \
        --output_dir      results/steering_crossmodel_7b \
        --cross_model
"""

import os
import json
import logging
import argparse
from typing import Dict, List, Optional

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ===========================
#  HIDDEN STATE EXTRACTOR
# ===========================

class HiddenStateCollector:
    """Registers hooks on all transformer layers to collect hidden states."""

    def __init__(self, model, layers: List[int]):
        self.model     = model
        self.layers    = layers
        self.collected = {}
        self.handles   = []

    def __enter__(self):
        self.collected = {}
        for layer_idx in self.layers:
            def make_hook(idx):
                def hook(module, input, output):
                    out = (output[0] if isinstance(output, tuple)
                           else output)
                    self.collected[idx] = (
                        out[0, -1, :].detach().cpu().float())
                return hook
            layer  = self.model.model.model.layers[layer_idx]
            handle = layer.register_forward_hook(make_hook(layer_idx))
            self.handles.append(handle)
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles = []


def collect_hidden_states(model, tokenizer, examples: List[dict],
                          layers: List[int]) -> Dict[int, np.ndarray]:
    """
    Run inference on all examples and collect hidden states at each layer.
    Returns: {layer_idx: np.array of shape (n_examples, hidden_dim)}
    """
    all_states = {l: [] for l in layers}

    for ex in tqdm(examples, desc="Collecting hidden states"):
        messages = ex["messages"]
        prompt   = tokenizer.apply_chat_template(
            messages[:2], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with HiddenStateCollector(model, layers) as collector:
            with torch.no_grad():
                model(**inputs)
            for layer_idx in layers:
                if layer_idx in collector.collected:
                    all_states[layer_idx].append(
                        collector.collected[layer_idx].numpy()
                    )

    return {
        l: np.stack(states)
        for l, states in all_states.items() if len(states) > 0
    }


# ===========================
#  STEERING VECTOR EXTRACTION METHODS
# ===========================

def extract_mean_diff(h_pos: np.ndarray,
                      h_neg: np.ndarray) -> np.ndarray:
    """ActAdd / mean difference vector."""
    return h_pos.mean(axis=0) - h_neg.mean(axis=0)


def extract_probe_direction(h_pos: np.ndarray, h_neg: np.ndarray):
    """Linear probe -- logistic regression coefficient vector."""
    X = np.vstack([h_pos, h_neg])
    y = np.array([1] * len(h_pos) + [0] * len(h_neg))

    clf_cv = LogisticRegression(max_iter=2000, C=1.0)
    if len(X) >= 10:
        scores      = cross_val_score(
            clf_cv, X, y, cv=min(5, len(y) // 2))
        cv_accuracy = float(scores.mean())
    else:
        cv_accuracy = 0.0

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, y)
    train_accuracy = float(clf.score(X, y))

    return clf.coef_[0], cv_accuracy, train_accuracy


def extract_sadi_vector(h_pos: np.ndarray, h_neg: np.ndarray,
                        top_percentile: float = 90.0):
    """
    SADI binary masking: sparse vector keeping only top-p% dimensions.
    Returns: (sparse_vector, binary_mask)
    """
    mean_diff  = h_pos.mean(axis=0) - h_neg.mean(axis=0)
    abs_diff   = np.abs(mean_diff)
    threshold  = np.percentile(abs_diff, top_percentile)
    mask       = (abs_diff >= threshold).astype(float)
    sparse_vec = mean_diff * mask
    return sparse_vec, mask


# ===========================
#  MAIN EXTRACTION
# ===========================

def load_model(base_model_path: str, adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def extract_all_vectors(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load ADFA data
    logger.info("Loading ADFA data...")
    positives, negatives = [], []

    def _load_jsonl(path, default_label=None):
        items = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                if default_label is not None:
                    ex["label"] = default_label
                items.append(ex)
        return items

    if args.adfa_pos and args.adfa_neg:
        positives = _load_jsonl(args.adfa_pos, default_label=1)
        negatives = _load_jsonl(args.adfa_neg, default_label=0)
    else:
        for ex in _load_jsonl(args.adfa_file):
            if ex.get("label", 1) == 1:
                positives.append(ex)
            else:
                negatives.append(ex)
    logger.info(
        f"Loaded {len(positives)} positives, {len(negatives)} negatives")

    # Load CoT model
    logger.info("Loading CoT model...")
    cot_model, tokenizer = load_model(args.base_model, args.adapter_dir)

    n_layers   = cot_model.config.num_hidden_layers
    hidden_dim = cot_model.config.hidden_size
    layers     = list(range(n_layers))
    logger.info(f"Model: {n_layers} layers, hidden_dim={hidden_dim}")

    # Collect hidden states (CoT model)
    logger.info("Collecting hidden states from CoT model (positives)...")
    h_pos = collect_hidden_states(cot_model, tokenizer, positives, layers)

    logger.info("Collecting hidden states from CoT model (negatives)...")
    h_neg = collect_hidden_states(cot_model, tokenizer, negatives, layers)

    # Extract steering vectors at each layer
    logger.info("Extracting steering vectors...")
    vectors_mean    = {}
    vectors_probe   = {}
    vectors_sadi    = {}
    masks_sadi      = {}
    probe_acc_cv    = {}
    probe_acc_train = {}

    for layer_idx in tqdm(layers, desc="Layers"):
        if layer_idx not in h_pos or layer_idx not in h_neg:
            continue
        hp = h_pos[layer_idx]
        hn = h_neg[layer_idx]

        v_mean = extract_mean_diff(hp, hn)
        vectors_mean[layer_idx] = v_mean

        v_probe, cv_acc, train_acc = extract_probe_direction(hp, hn)
        vectors_probe[layer_idx]   = v_probe
        probe_acc_cv[layer_idx]    = cv_acc
        probe_acc_train[layer_idx] = train_acc

        v_sadi, mask = extract_sadi_vector(hp, hn, top_percentile=90.0)
        vectors_sadi[layer_idx] = v_sadi
        masks_sadi[layer_idx]   = mask

    # Find best layer
    best_layer = max(probe_acc_cv, key=probe_acc_cv.get)
    logger.info(
        f"Best layer (highest probe CV acc): {best_layer} "
        f"(acc={probe_acc_cv[best_layer]:.3f})")

    # Save steering vectors
    sv_data = {
        "meta": {
            "n_layers":                n_layers,
            "hidden_dim":              hidden_dim,
            "n_positives":             len(positives),
            "n_negatives":             len(negatives),
            "best_layer":              best_layer,
            "best_layer_probe_acc":    probe_acc_cv[best_layer],
        },
        "vectors_mean":    {l: v.tolist() for l, v in vectors_mean.items()},
        "vectors_probe":   {l: v.tolist() for l, v in vectors_probe.items()},
        "vectors_sadi":    {l: v.tolist() for l, v in vectors_sadi.items()},
        "masks_sadi":      {l: v.tolist() for l, v in masks_sadi.items()},
        "probe_acc_cv":    {l: float(v) for l, v in probe_acc_cv.items()},
        "probe_acc_train": {
            l: float(v) for l, v in probe_acc_train.items()},
    }
    sv_path = os.path.join(args.output_dir, "steering_vectors.json")
    with open(sv_path, "w") as f:
        json.dump(sv_data, f)
    logger.info(f"Saved steering vectors -> {sv_path}")

    # Save probe accuracy summary
    probe_summary = [
        {"layer":          l,
         "cv_accuracy":    probe_acc_cv[l],
         "train_accuracy": probe_acc_train.get(l, 0.0)}
        for l in sorted(probe_acc_cv.keys())
    ]
    probe_path = os.path.join(args.output_dir, "probe_accuracy.json")
    with open(probe_path, "w") as f:
        json.dump(probe_summary, f, indent=2)
    logger.info(f"Saved probe accuracy by layer -> {probe_path}")

    # Cross-model transfer (optional)
    if args.cross_model and args.nocot_adapter:
        logger.info("=" * 60)
        logger.info("CROSS-MODEL TRANSFER: Extracting No-CoT advantage vector")
        logger.info("=" * 60)

        del cot_model
        torch.cuda.empty_cache()

        logger.info("Loading No-CoT model...")
        nocot_model, _ = load_model(args.base_model, args.nocot_adapter)

        logger.info(
            "Collecting hidden states from No-CoT model (positives)...")
        h_nocot_pos = collect_hidden_states(
            nocot_model, tokenizer, positives, layers)

        transfer_vectors = {}
        transfer_norms   = {}
        for layer_idx in layers:
            if layer_idx not in h_nocot_pos or layer_idx not in h_pos:
                continue
            h_nc   = h_nocot_pos[layer_idx]
            h_c    = h_pos[layer_idx]
            min_n  = min(len(h_nc), len(h_c))
            v_tf   = h_nc[:min_n].mean(axis=0) - h_c[:min_n].mean(axis=0)
            transfer_vectors[layer_idx] = v_tf
            transfer_norms[layer_idx]   = float(np.linalg.norm(v_tf))

        best_transfer_layer = max(transfer_norms, key=transfer_norms.get)
        logger.info(
            f"Best transfer layer: {best_transfer_layer} "
            f"(norm={transfer_norms[best_transfer_layer]:.4f})")

        transfer_data = {
            "meta": {
                "n_layers":             n_layers,
                "hidden_dim":           hidden_dim,
                "best_transfer_layer":  best_transfer_layer,
            },
            "transfer_vectors": {
                l: v.tolist() for l, v in transfer_vectors.items()},
            "transfer_norms": {
                l: float(n) for l, n in transfer_norms.items()},
        }
        transfer_path = os.path.join(
            args.output_dir, "cross_model_vectors.json")
        with open(transfer_path, "w") as f:
            json.dump(transfer_data, f)
        logger.info(
            f"Saved cross-model transfer vectors -> {transfer_path}")

    logger.info("Steering vector extraction complete.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",    required=True)
    p.add_argument("--adapter_dir",   required=True,
                   help="CoT model adapter path")
    p.add_argument("--nocot_adapter", default=None,
                   help="No-CoT adapter path (for cross-model transfer)")
    p.add_argument("--adfa_pos",      default=None)
    p.add_argument("--adfa_neg",      default=None)
    p.add_argument("--adfa_file",     default=None,
                   help="Single JSONL with label field")
    p.add_argument("--output_dir",    default="results/steering_cot_7b")
    p.add_argument("--cross_model",   action="store_true")
    args = p.parse_args()

    if not args.adfa_file and not (args.adfa_pos and args.adfa_neg):
        p.error("Provide --adfa_pos + --adfa_neg  OR  --adfa_file")
    if args.cross_model and not args.nocot_adapter:
        p.error("--cross_model requires --nocot_adapter")

    extract_all_vectors(args)


if __name__ == "__main__":
    main()
