"""
run_patching_multi.py  --  MULTI-LAYER prompt-only activation patching (Tier 4)
===============================================================================
Extends run_patching_fast.py: instead of patching ONE layer at a time, this
patches a SET of layers simultaneously in a single forward pass, each layer
with its own mean-correct vector. This tests whether the failure is recoverable
by correcting a *combination* of layers, even when no single layer recovers it.

WHY
  Single-layer patching (run_patching_fast.py) recovered 0/82 at every layer.
  That rules out single-layer localization but NOT a distributed cause. This
  script patches multiple layers together, so we can test:
    * ALL layers at once   -> is the failure recoverable by representation
                              injection AT ALL? (the decisive upper bound)
    * cumulative 0..L       -> how much of the early stack must be corrected?
    * windows [a..b]        -> which region holds it?

HOW YOU CHOOSE THE SETS  (--layer_sets, a ';'-separated list of ranges)
  --layer_sets "0-27"                       -> one set: all layers at once
  --layer_sets "0-1;0-2;0-3;...;0-27"       -> cumulative from start
  --layer_sets "0-4;5-9;10-14;15-19;20-27"  -> sliding windows
  Ranges are inclusive. A single layer is just "10".

DESIGN (unchanged from the validated single-layer version, extended to sets)
  * Prompt-only: patch the residual-stream hidden state on the PROMPT forward
    pass, then let generation run untouched.
  * At each patched layer L in the set, overwrite the prompt positions
    (attention_mask==1) with mean_correct[L]. Left-pad positions untouched.
  * Batched (left padding), attn_implementation='sdpa', bf16.
  * Each layer in the set fires its own hook exactly once, on prefill.

CORRECTNESS
  Validated by test_patching_multi_equiv.sh: multi-layer batched results must
  match an unbatched reference on the same examples. Do not run the full sweep
  until that passes.

USAGE
  python run_patching_multi.py \
      --base_model  <BASE_MODEL_DIR> \
      --adapter_dir <ADAPTER_DIR> \
      --test_file   <TEST_FILE> \
      --output_dir  <OUTPUT_DIR> \
      --layer_sets  "0-27" \
      --batch_size  24 --max_new_tokens 3500
"""

import os
import re
import json
import logging
import argparse
from itertools import product
from typing import Optional, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =====================================================================
#  DFA PARSER + VERIFIER  (identical fixed parser)
# =====================================================================

def parse_dfa(text: str, alphabet: list) -> Optional[dict]:
    sa = sorted(alphabet)
    m7 = re.search(r'##\s*Step\s*7\b.*?(?=##\s*Step\s*8\b|\Z)', text, re.DOTALL)
    section = m7.group(0) if m7 else text
    tbl = re.search(
        r'\|\s*State\s*\|.*?Accept.*?\n\|[-\s|]+\n((?:\|.*\n?)+)',
        section, re.DOTALL)
    if not tbl:
        return None
    rows = [r for r in tbl.group(1).split('\n') if r.strip().startswith('|')]
    transitions, accept_states, states = {}, set(), []
    n_expected = len(sa) + 2
    for row in rows:
        parts = row.split('|')
        if parts and parts[0].strip() == '':
            parts = parts[1:]
        if parts and parts[-1].strip() == '':
            parts = parts[:-1]
        cells = [c.strip() for c in parts]
        if len(cells) < n_expected:
            continue
        m = re.match(r'D(\d+)', cells[0])
        if not m:
            continue
        sid = int(m.group(1))
        states.append(sid)
        accept_cell = cells[n_expected - 1]
        if accept_cell in ('Y', 'y', '\u2713') or accept_cell.startswith('\u2713'):
            accept_states.add(sid)
        transitions[sid] = {}
        for j, sym in enumerate(sa):
            tm = re.match(r'D(\d+)', cells[j + 1])
            if tm:
                transitions[sid][sym] = int(tm.group(1))
    if not states:
        return None
    return {"states": sorted(set(states)), "alphabet": sa, "start": 0,
            "accept": sorted(accept_states), "transitions": transitions}


def dfa_accepts(dfa: dict, s: str) -> bool:
    state = dfa["start"]
    for ch in s:
        if ch not in dfa["transitions"].get(state, {}):
            return False
        state = dfa["transitions"][state][ch]
    return state in dfa["accept"]


def is_exact(pred: dict, gold: dict, alphabet: list, max_len: int = 6) -> bool:
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


# =====================================================================
#  MODEL HELPERS
# =====================================================================

def get_layers(model):
    for c in (lambda m: m.base_model.model.model.layers,
              lambda m: m.model.model.layers,
              lambda m: m.model.layers):
        try:
            layers = c(model)
            if layers is not None and len(layers) > 0:
                return layers
        except AttributeError:
            continue
    raise AttributeError("Could not locate decoder layers.")


def build_prompt(tokenizer, messages) -> str:
    return tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True)


def gen_kwargs(tokenizer, max_new_tokens):
    return dict(max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def parse_range(tok: str, n_layers: int) -> List[int]:
    tok = tok.strip()
    if "-" in tok:
        a, b = tok.split("-")
        a, b = int(a), int(b)
        rng = list(range(a, b + 1))
    else:
        rng = [int(tok)]
    for l in rng:
        if not (0 <= l < n_layers):
            raise ValueError(f"layer {l} out of range [0,{n_layers-1}]")
    return rng


def parse_layer_sets(arg: str, n_layers: int) -> List[List[int]]:
    """'0-27' -> [[0..27]]  ;  '0-1;0-2' -> [[0,1],[0,1,2]]"""
    sets = []
    for part in arg.split(";"):
        part = part.strip()
        if not part:
            continue
        sets.append(sorted(set(parse_range(part, n_layers))))
    return sets


def set_label(layer_set: List[int]) -> str:
    if len(layer_set) == 1:
        return f"L{layer_set[0]}"
    return f"L{layer_set[0]}-{layer_set[-1]}" if layer_set == list(
        range(layer_set[0], layer_set[-1] + 1)) else "L" + "_".join(map(str, layer_set))


# =====================================================================
#  BATCHED BASELINE
# =====================================================================

def generate_batch(model, tokenizer, batch_messages, max_new_tokens):
    import torch
    prompts = [build_prompt(tokenizer, m) for m in batch_messages]
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    in_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs(tokenizer, max_new_tokens))
    return [tokenizer.decode(g, skip_special_tokens=True) for g in out[:, in_len:]]


# =====================================================================
#  MEAN PROMPT HIDDEN STATES for a set of layers (masked mean, batched)
# =====================================================================

def collect_prompt_hidden_states(model, tokenizer, examples, layer_indices, batch_size):
    import torch
    layers = get_layers(model)
    captured = {l: None for l in layer_indices}
    handles = []

    def make_hook(li):
        def hook(module, inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[li] = hs.detach()
        return hook
    for l in layer_indices:
        handles.append(layers[l].register_forward_hook(make_hook(l)))

    per_example = {l: [] for l in layer_indices}
    try:
        for batch in tqdm(list(chunks(examples, batch_size)),
                          desc="Prompt hidden states", leave=False):
            prompts = [build_prompt(tokenizer, ex["messages"]) for ex in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            mask = enc["attention_mask"]
            for l in layer_indices:
                captured[l] = None
            with torch.no_grad():
                model(**enc)
            for l in layer_indices:
                hs = captured[l].float()
                m = mask.unsqueeze(-1).float()
                mean = ((hs * m).sum(1) / m.sum(1).clamp(min=1.0)).cpu().numpy()
                for b in range(mean.shape[0]):
                    per_example[l].append(mean[b])
    finally:
        for h in handles:
            h.remove()
    return {l: np.stack(v, axis=0) for l, v in per_example.items()}


# =====================================================================
#  MULTI-LAYER PROMPT PATCH
#  One hook per layer in the set; each fires once on prefill and overwrites the
#  real (mask==1) prompt positions with that layer's source vector.
# =====================================================================

class OneLayerPatch:
    def __init__(self, source_vec, attn_mask):
        import torch
        self.v = torch.tensor(np.asarray(source_vec), dtype=torch.float32)
        self.mask = attn_mask
        self.fired = False

    def __call__(self, module, inp, output):
        import torch
        hs = output[0] if isinstance(output, tuple) else output
        if self.fired or hs.shape[1] != self.mask.shape[1]:
            return None
        self.fired = True
        hs = hs.clone()
        vv = self.v.to(hs.device).to(hs.dtype)
        m = self.mask.to(hs.device).unsqueeze(-1).bool()
        hs = torch.where(m, vv.view(1, 1, -1).expand_as(hs), hs)
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs


def generate_batch_multi_patch(model, tokenizer, batch_messages, layer_set,
                               source_by_layer, max_new_tokens):
    """Patch every layer in layer_set simultaneously (each with its own vector)."""
    import torch
    layers = get_layers(model)
    prompts = [build_prompt(tokenizer, m) for m in batch_messages]
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    in_len = enc["input_ids"].shape[1]
    hooks, handles = [], []
    for l in layer_set:
        hk = OneLayerPatch(source_by_layer[l], enc["attention_mask"])
        hooks.append(hk)
        handles.append(layers[l].register_forward_hook(hk))
    try:
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs(tokenizer, max_new_tokens))
    finally:
        for h in handles:
            h.remove()
    for l, hk in zip(layer_set, hooks):
        if not hk.fired:
            raise RuntimeError(f"patch hook at layer {l} never fired")
    return [tokenizer.decode(g, skip_special_tokens=True) for g in out[:, in_len:]]


# =====================================================================
#  MAIN
# =====================================================================

def load_model(base_model, adapter_dir):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_dir", required=True)
    p.add_argument("--test_file", required=True)
    p.add_argument("--output_dir", default="results_multi")
    p.add_argument("--layer_sets", required=True,
                   help="';'-separated inclusive ranges, e.g. '0-27' or '0-1;0-2;0-3'")
    p.add_argument("--max_new_tokens", type=int, default=3500)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--subset_size", type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    import torch
    assert torch.cuda.is_available(), "CUDA not available -- refusing CPU run"

    model, tokenizer = load_model(args.base_model, args.adapter_dir)
    n_layers = model.config.num_hidden_layers
    layer_sets = parse_layer_sets(args.layer_sets, n_layers)
    all_layers_needed = sorted(set(l for s in layer_sets for l in s))
    logger.info(f"Model: {n_layers} layers | {len(layer_sets)} layer-set(s) | "
                f"batch {args.batch_size} | dev {next(model.parameters()).device}")
    logger.info(f"Layer sets: {[set_label(s) for s in layer_sets]}")

    # ---- load Tier 4 ----
    tier4 = []
    with open(args.test_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if ex.get("metadata", {}).get("tier") == 4:
                tier4.append(ex)
    if args.subset_size:
        tier4 = tier4[:args.subset_size]
    logger.info(f"Tier 4 examples: {len(tier4)}")

    # ---- STEP 1: baseline ----
    logger.info("STEP 1: batched baseline...")
    correct_ex, incorrect_ex, split_records = [], [], []
    for batch in tqdm(list(chunks(tier4, args.batch_size)), desc="Baseline"):
        preds = generate_batch(model, tokenizer, [e["messages"] for e in batch],
                               args.max_new_tokens)
        for ex, pred in zip(batch, preds):
            alphabet = ex.get("metadata", {}).get("alphabet", ["a", "b"])
            gold = ex["messages"][2]["content"]
            ok = verify(pred, gold, alphabet)
            (correct_ex if ok else incorrect_ex).append(ex)
            split_records.append({"hash": ex.get("metadata", {}).get("hash"),
                                  "correct": bool(ok)})
    n_c, n_x = len(correct_ex), len(incorrect_ex)
    logger.info(f"Baseline: {n_c} correct, {n_x} incorrect (acc={n_c/len(tier4):.3f})")
    with open(os.path.join(args.output_dir, "split.json"), "w") as f:
        json.dump({"n_total": len(tier4), "n_correct": n_c, "n_incorrect": n_x,
                   "records": split_records}, f, indent=2)
    if n_x == 0 or n_c == 0:
        logger.warning("need both correct and incorrect. exiting."); return

    # ---- STEP 2: mean-correct hidden states for every layer any set needs ----
    logger.info("STEP 2: mean-correct hidden states...")
    hs_correct = collect_prompt_hidden_states(model, tokenizer, correct_ex,
                                              all_layers_needed, args.batch_size)
    mean_correct = {l: hs_correct[l].mean(axis=0) for l in all_layers_needed}

    # ---- STEP 3: patch each layer-set (all its layers at once) ----
    logger.info("STEP 3: multi-layer patching...")
    results = {}
    per_example = []
    for layer_set in layer_sets:
        label = set_label(layer_set)
        rec = 0
        for batch in tqdm(list(chunks(incorrect_ex, args.batch_size)),
                          desc=label, leave=False):
            preds = generate_batch_multi_patch(
                model, tokenizer, [e["messages"] for e in batch],
                layer_set, mean_correct, args.max_new_tokens)
            for ex, pred in zip(batch, preds):
                alphabet = ex.get("metadata", {}).get("alphabet", ["a", "b"])
                gold = ex["messages"][2]["content"]
                ok = verify(pred, gold, alphabet)
                rec += int(ok)
                per_example.append({"set": label,
                                    "hash": ex.get("metadata", {}).get("hash"),
                                    "recovered": bool(ok)})
        results[label] = rec
        logger.info(f"Set {label:>10}: recovered {rec}/{n_x} "
                    f"({100.0*rec/n_x:.1f}%)")

    with open(os.path.join(args.output_dir, "multi_results.json"), "w") as f:
        json.dump({"n_incorrect": n_x, "n_correct": n_c,
                   "sets": [set_label(s) for s in layer_sets],
                   "recovered_by_set": results}, f, indent=2)
    with open(os.path.join(args.output_dir, "multi_per_example.json"), "w") as f:
        json.dump(per_example, f, indent=2)

    best = max(results, key=lambda k: results[k])
    logger.info("=" * 60)
    logger.info(f"BEST set: {best} recovered {results[best]}/{n_x}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
