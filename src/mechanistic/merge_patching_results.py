"""
merge_and_plot.py -- combine the two GPU runs (layers 0-13 and 14-27) into one
recovery-vs-layer curve, and print the numbers to cite in the paper.

Run on the login node (in the venv) after both jobs finish:
    python merge_and_plot.py \
        --dir_a <RESULTS_DIR_A> \
        --dir_b <RESULTS_DIR_B> \
        --out   <OUTPUT_DIR>
"""
import json, os, argparse

def load_curve(d):
    c = json.load(open(os.path.join(d, "patch_curve.json")))
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir_a", required=True)
    ap.add_argument("--dir_b", required=True)
    ap.add_argument("--out", default="patch_figure")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ca = load_curve(args.dir_a)
    cb = load_curve(args.dir_b)

    # sanity: both must agree on the failing-set size (same baseline split)
    n_x = ca["n_incorrect"]
    if ca["n_incorrect"] != cb["n_incorrect"] or ca["n_correct"] != cb["n_correct"]:
        print(f"WARNING: split differs between runs! "
              f"A: {ca['n_correct']}/{ca['n_incorrect']}  "
              f"B: {cb['n_correct']}/{cb['n_incorrect']}")
        print("  (deterministic generation should give identical splits; investigate)")

    # merge the per-layer dicts
    rec_c = {}
    rec_x = {}
    for c in (ca, cb):
        for k, v in c["recovered_by_layer_correct_source"].items():
            rec_c[int(k)] = v
        for k, v in c["recovered_by_layer_incorrect_control"].items():
            rec_x[int(k)] = v

    layers = sorted(rec_c.keys())
    n_correct = ca["n_correct"]

    # save merged json
    merged = {
        "n_incorrect": n_x, "n_correct": n_correct,
        "layers": layers,
        "recovered_by_layer_correct_source": {str(l): rec_c[l] for l in layers},
        "recovered_by_layer_incorrect_control": {str(l): rec_x.get(l) for l in layers},
    }
    with open(os.path.join(args.out, "patch_curve_merged.json"), "w") as f:
        json.dump(merged, f, indent=2)

    # ---- print the numbers for the paper ----
    print("="*64)
    print(f"Baseline split: {n_correct} correct / {n_x} incorrect")
    print(f"Failing examples patched: {n_x}")
    print("-"*64)
    print(f"{'Layer':>5} | {'correct-src recovered':>22} | {'control recovered':>18}")
    print("-"*64)
    for l in layers:
        cx = rec_x.get(l)
        cxs = f"{cx}/{n_x}" if cx is not None else "n/a"
        pct = 100.0*rec_c[l]/n_x if n_x else 0
        print(f"{l:>5} | {rec_c[l]:>3}/{n_x}  ({pct:4.1f}%)      | {cxs:>18}")
    print("-"*64)
    best = max(layers, key=lambda l: rec_c[l])
    print(f"Best layer: L{best}  recovered {rec_c[best]}/{n_x} "
          f"({100.0*rec_c[best]/n_x:.1f}%)  vs control {rec_x.get(best)}")
    # early vs late summary
    early = [l for l in layers if l <= 13]
    late  = [l for l in layers if l >= 14]
    if early and late:
        be = max(early, key=lambda l: rec_c[l])
        bl = max(late,  key=lambda l: rec_c[l])
        print(f"Best EARLY layer (<=13): L{be} -> {rec_c[be]}/{n_x}")
        print(f"Best LATE  layer (>=14): L{bl} -> {rec_c[bl]}/{n_x}")
    print("="*64)

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3.2))
        plt.plot(layers, [rec_c[l] for l in layers], marker="o",
                 label="patch mean-correct")
        if all(rec_x.get(l) is not None for l in layers):
            plt.plot(layers, [rec_x[l] for l in layers], marker="s",
                     linestyle="--", label="control (mean-incorrect)")
        plt.xlabel("Patched layer")
        plt.ylabel(f"Failing examples recovered (of {n_x})")
        plt.title("Prompt-only activation patching, Tier 4")
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(args.out, "patch_curve.png")
        plt.savefig(out_png, dpi=200)
        print(f"Saved plot -> {out_png}")
    except Exception as e:
        print("plot skipped:", e)

if __name__ == "__main__":
    main()
