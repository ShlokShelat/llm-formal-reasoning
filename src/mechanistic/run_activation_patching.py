"""
run_patching_fast.py  --  BATCHED prompt-only activation patching (Tier 4)
==========================================================================
Same experiment as run_patching_eval.py, but BATCHED for H100 throughput.

WHAT'S DIFFERENT (and why it's still correct)
  * Baseline and patched generation run in batches (--batch_size), not one
    example at a time. This is the ~10x speedup (batch-1 used ~17% of GPU mem).
  * Decoder-only batched generation requires LEFT padding, so all prompts end
    aligned. We set tokenizer.padding_side='left'.
  * The prompt-patch hook must overwrite ONLY the real prompt positions, never
    the left-pad tokens. We pass the attention_mask into the hook and patch only
    where mask==1. (Patching pad positions would be both wrong and pointless.)
  * The hook still fires ONCE, on the prefill forward pass (seq_len>1), and is a
    no-op on every decode step (seq_len==1), so generation is never corrupted.
  * attn_implementation='sdpa' (built into torch; flash-attn lib is broken here).

CORRECTNESS
  This file is validated by test_patching_equiv.py, which runs the SAME examples
  through both the unbatched reference (run_patching_eval.py) and this batched
  version and asserts identical correct/incorrect and recovery results. Do NOT
  run the full experiment until that test passes.

USAGE
  python run_patching_fast.py \
      --base_model  <BASE_MODEL_DIR> \
      --adapter_dir <ADAPTER_DIR> \
      --test_file   <TEST_FILE> \
      --output_dir  <OUTPUT_DIR> \
      --batch_size  16 \
      --max_new_tokens 3500
"""

import os
import re
import json
import logging
import argparse
from itertools import product
from typing import Optional, Dict, List

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =====================================================================
#  DFA PARSER + VERIFIER  (identical to the fixed run_patching_eval.py)
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
    candidates = [
        lambda m: m.base_model.model.model.layers,
        lambda m: m.model.model.layers,
        lambda m: m.model.layers,
    ]
    for c in candidates:
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


def gen_kwargs(tokenizer, max_new_tokens: int) -> dict:
    return dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# =====================================================================
#  BATCHED BASELINE GENERATION (no patching)
# =====================================================================

def generate_batch(model, tokenizer, batch_messages, max_new_tokens):
    """Left-padded batched generation. Returns list of decoded completions."""
    import torch
    prompts = [build_prompt(tokenizer, m) for m in batch_messages]
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    in_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs(tokenizer, max_new_tokens))
    # with left padding, new tokens are appended after the (padded) input block
    gen = out[:, in_len:]
    return [tokenizer.decode(g, skip_special_tokens=True) for g in gen]


# =====================================================================
#  MEAN PROMPT HIDDEN STATES (batched capture, masked mean over real positions)
# =====================================================================

def collect_prompt_hidden_states(model, tokenizer, examples, layer_indices,
                                 batch_size):
    """
    Batched forward over prompts; capture residual-stream hidden state at each
    requested layer; average over REAL prompt positions (mask==1) per example.
    Returns dict layer -> np.ndarray (n_examples, hidden_dim), preserving order.
    """
    import torch
    layers = get_layers(model)
    captured: Dict[int, object] = {l: None for l in layer_indices}
    handles = []

    def make_hook(layer_idx):
        def hook(module, inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hs.detach()
        return hook

    for l in layer_indices:
        handles.append(layers[l].register_forward_hook(make_hook(l)))

    per_example = {l: [] for l in layer_indices}
    try:
        for batch in tqdm(list(chunks(examples, batch_size)),
                          desc="Prompt hidden states", leave=False):
            prompts = [build_prompt(tokenizer, ex["messages"]) for ex in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            mask = enc["attention_mask"]  # (B, T), 1 = real token
            for l in layer_indices:
                captured[l] = None
            with torch.no_grad():
                model(**enc)
            for l in layer_indices:
                hs = captured[l].float()               # (B, T, H)
                m = mask.unsqueeze(-1).float()         # (B, T, 1)
                summed = (hs * m).sum(dim=1)           # (B, H)
                counts = m.sum(dim=1).clamp(min=1.0)   # (B, 1)
                mean = (summed / counts).cpu().numpy() # (B, H) masked mean
                for b in range(mean.shape[0]):
                    per_example[l].append(mean[b])
    finally:
        for h in handles:
            h.remove()

    return {l: np.stack(v, axis=0) for l, v in per_example.items()}


# =====================================================================
#  BATCHED PROMPT-ONLY PATCH HOOK
#  Fires once on prefill (seq_len>1). Overwrites the hidden state with the
#  source vector ONLY at real prompt positions (attention_mask==1), leaving
#  left-pad positions untouched. No-op on decode steps (seq_len==1).
# =====================================================================

class BatchedPromptPatchHook:
    def __init__(self, source_vec: np.ndarray, attn_mask):
        import torch
        self.v = torch.tensor(np.asarray(source_vec), dtype=torch.float32)
        self.mask = attn_mask  # (B, T_prompt)
        self.fired = False

    def __call__(self, module, inp, output):
        import torch
        hs = output[0] if isinstance(output, tuple) else output
        # only act on the prefill pass (the one whose seq len matches the prompt)
        if self.fired or hs.shape[1] != self.mask.shape[1]:
            return None
        self.fired = True
        hs = hs.clone()
        vv = self.v.to(hs.device).to(hs.dtype)          # (H,)
        m = self.mask.to(hs.device).unsqueeze(-1).bool()  # (B, T, 1)
        # broadcast source across positions, keep original where mask==0 (pad)
        src = vv.view(1, 1, -1).expand_as(hs)
        hs = torch.where(m, src, hs)
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs


def generate_batch_with_patch(model, tokenizer, batch_messages, layer_idx,
                              source_vec, max_new_tokens):
    import torch
    layers = get_layers(model)
    prompts = [build_prompt(tokenizer, m) for m in batch_messages]
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    in_len = enc["input_ids"].shape[1]
    hook = BatchedPromptPatchHook(source_vec, enc["attention_mask"])
    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs(tokenizer, max_new_tokens))
    finally:
        handle.remove()
    if not hook.fired:
        raise RuntimeError(f"Patch hook at layer {layer_idx} never fired "
                           f"(prompt len {enc['attention_mask'].shape[1]}).")
    gen = out[:, in_len:]
    return [tokenizer.decode(g, skip_special_tokens=True) for g in gen]


# =====================================================================
#  UTIL
# =====================================================================

def leave_one_out_mean(mat: np.ndarray, idx: int) -> np.ndarray:
    n = mat.shape[0]
    return (mat.sum(axis=0) - mat[idx]) / (n - 1)


def parse_layers(arg, n_layers):
    if not arg:
        return list(range(n_layers))
    out = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        li = int(tok)
        if not (0 <= li < n_layers):
            raise ValueError(f"layer {li} out of range [0,{n_layers-1}]")
        out.append(li)
    return sorted(set(out))


def load_model(base_model, adapter_dir):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # decoder-only batched generation needs LEFT padding
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tok


# =====================================================================
#  MAIN
# =====================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",  required=True)
    p.add_argument("--adapter_dir", required=True)
    p.add_argument("--test_file",   required=True)
    p.add_argument("--output_dir",  default="results_patch")
    p.add_argument("--max_new_tokens", type=int, default=3500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--subset_size", type=int, default=None)
    p.add_argument("--layers", type=str, default=None)
    p.add_argument("--skip_control", action="store_true")
    p.add_argument("--control_layers", type=str, default=None,
                   help="comma-separated layers to run the (slow, unbatched) "
                        "control on; default = all patched layers. Use a small "
                        "subset to save time, since control cannot be batched.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    import torch
    assert torch.cuda.is_available(), "CUDA not available -- refusing to run on CPU"

    model, tokenizer = load_model(args.base_model, args.adapter_dir)
    n_layers = model.config.num_hidden_layers
    layer_indices = parse_layers(args.layers, n_layers)
    if args.control_layers:
        control_layers = set(parse_layers(args.control_layers, n_layers))
    else:
        control_layers = set(layer_indices)
    logger.info(f"Model: {n_layers} layers | patch layers {layer_indices} "
                f"| batch_size {args.batch_size} | dev {next(model.parameters()).device}")

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

    # ---- STEP 1: batched baseline split ----
    logger.info("STEP 1: batched baseline...")
    correct_ex, incorrect_ex, split_records = [], [], []
    idx = 0
    for batch in tqdm(list(chunks(tier4, args.batch_size)), desc="Baseline"):
        preds = generate_batch(model, tokenizer, [e["messages"] for e in batch],
                               args.max_new_tokens)
        for ex, pred in zip(batch, preds):
            alphabet = ex.get("metadata", {}).get("alphabet", ["a", "b"])
            gold = ex["messages"][2]["content"]
            ok = verify(pred, gold, alphabet)
            (correct_ex if ok else incorrect_ex).append(ex)
            split_records.append({"idx": idx, "hash": ex.get("metadata", {}).get("hash"),
                                  "correct": bool(ok)})
            idx += 1
    n_c, n_x = len(correct_ex), len(incorrect_ex)
    logger.info(f"Baseline: {n_c} correct, {n_x} incorrect (acc={n_c/len(tier4):.3f})")
    with open(os.path.join(args.output_dir, "split.json"), "w") as f:
        json.dump({"n_total": len(tier4), "n_correct": n_c, "n_incorrect": n_x,
                   "records": split_records}, f, indent=2)
    if n_x == 0 or n_c == 0:
        logger.warning("Need both correct and incorrect examples. Exiting.")
        return
    run_control = (not args.skip_control) and n_x >= 2

    # ---- STEP 2: mean prompt hidden states ----
    logger.info("STEP 2: prompt hidden states (correct)...")
    hs_correct = collect_prompt_hidden_states(model, tokenizer, correct_ex,
                                              layer_indices, args.batch_size)
    mean_correct = {l: hs_correct[l].mean(axis=0) for l in layer_indices}
    hs_incorrect = None
    if run_control:
        logger.info("STEP 2b: prompt hidden states (incorrect, control)...")
        hs_incorrect = collect_prompt_hidden_states(model, tokenizer, incorrect_ex,
                                                    layer_indices, args.batch_size)

    # ---- STEP 3: batched patching per layer ----
    logger.info("STEP 3: batched prompt-only patching...")
    curve_c, curve_x, per_example = {}, {}, []
    for l in layer_indices:
        rec_c = 0
        rec_x = 0
        # correct-source: same source vector for every failing example -> batch freely
        for batch in tqdm(list(chunks(incorrect_ex, args.batch_size)),
                          desc=f"L{l} correct", leave=False):
            preds = generate_batch_with_patch(
                model, tokenizer, [e["messages"] for e in batch], l,
                mean_correct[l], args.max_new_tokens)
            for ex, pred in zip(batch, preds):
                alphabet = ex.get("metadata", {}).get("alphabet", ["a", "b"])
                gold = ex["messages"][2]["content"]
                ok = verify(pred, gold, alphabet)
                rec_c += int(ok)
                per_example.append({"layer": l, "hash": ex.get("metadata", {}).get("hash"),
                                    "recovered_correct_source": bool(ok),
                                    "recovered_incorrect_control": None})
        # control: leave-one-out source differs per example -> cannot share a batch
        # vector; run these one-at-a-time. Only for the requested control layers,
        # since this is the slow (unbatched) part.
        if run_control and l in control_layers:
            for j, ex in enumerate(tqdm(incorrect_ex, desc=f"L{l} control", leave=False)):
                src_x = leave_one_out_mean(hs_incorrect[l], j)
                pred = generate_batch_with_patch(
                    model, tokenizer, [ex["messages"]], l, src_x, args.max_new_tokens)[0]
                alphabet = ex.get("metadata", {}).get("alphabet", ["a", "b"])
                gold = ex["messages"][2]["content"]
                ok = verify(pred, gold, alphabet)
                rec_x += int(ok)
                # attach control result to the matching per_example record
                for rec in per_example:
                    if rec["layer"] == l and rec["hash"] == ex.get("metadata", {}).get("hash"):
                        rec["recovered_incorrect_control"] = bool(ok)
                        break
        curve_c[l] = rec_c
        ran_ctrl = run_control and l in control_layers
        curve_x[l] = rec_x if ran_ctrl else None
        ctrl = f"{rec_x}/{n_x}" if ran_ctrl else "n/a"
        logger.info(f"Layer {l:2d}: correct-source {rec_c}/{n_x} | control {ctrl}")

    with open(os.path.join(args.output_dir, "patch_curve.json"), "w") as f:
        json.dump({"n_incorrect": n_x, "n_correct": n_c, "layers": layer_indices,
                   "batch_size": args.batch_size,
                   "recovered_by_layer_correct_source": {str(k): v for k, v in curve_c.items()},
                   "recovered_by_layer_incorrect_control": {str(k): v for k, v in curve_x.items()}},
                  f, indent=2)
    with open(os.path.join(args.output_dir, "patch_per_example.json"), "w") as f:
        json.dump(per_example, f, indent=2)

    best = max(curve_c, key=lambda k: curve_c[k])
    logger.info("=" * 60)
    logger.info(f"BEST layer (correct-source): L{best} recovered {curve_c[best]}/{n_x}")
    if run_control:
        logger.info(f"control at that layer: {curve_x[best]}/{n_x}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
