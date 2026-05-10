# Tree-of-Thought (ToT) Experiments

This directory contains all scripts, prompts, and representative outputs used for the Tree-of-Thought experiments described in the paper:

**“Position: Testing the Limits of Large Language Models on Regular Languages.”**

The goal of these experiments is to evaluate whether LLMs (GPT-5.x, Grok-4, Gemini-2.5) can construct deterministic finite automata (DFAs) given a regular expression, under three different prompting paradigms:
1. **Intuitive Construction**  
2. **Derivative-Based Construction**  
3. **Cross-Consistency Evaluation**

Each method exposes different failure modes in symbolic reasoning.

---

##  Directory Structure

```
tot_experiments/
│
├── run_experiment_tot.py        # Main ToT experiment runner
│
├── prompts/                     # Prompting templates used in the paper
│   ├── prompt_intuitive.txt
│   ├── prompt_derivative.txt
│   └── prompt_cross_consistency.txt
│
└── example_outputs/             # Clean, representative outputs
    ├── intuitive/
    ├── derivative/
    └── cross_consistency/
```

---

## What These Experiments Test

### **1. Intuitive Method**
Tests whether the model can directly interpret a regex and produce a DFA without explicit symbolic steps.

### **2. Derivative-Based Method**
Evaluates the model’s ability to perform formal reasoning:
- nullability  
- derivative computation  
- canonicalization  
- DFA state expansion  

### **3. Cross-Consistency Tasks**
Checks whether the model can remain consistent across tasks:
- language description  
- membership classification  
- DFA construction  

This is where failure modes such as **Indexing Drift**, **Nullability Collapse**, and **Transition Hallucination** appear.

---

## Running Experiments

The script automatically:
- loads the dataset specified in `DATASET_PATH` inside the script  
- loads the prompts  
- calls the GPT-5.x API with retry and backoff logic  
- enforces strict JSON extraction  
- stores raw and processed outputs  

Run:

```bash
python run_experiment_tot.py
```

To ensure GPT-5.x compatibility, the script uses:

```python
max_completion_tokens
```

instead of the older:

```python
max_tokens
```

---

## Prompt Templates

### **Intuitive Prompt**
Requests an intuitive DFA construction and outputs only a transition table.

### **Derivative Prompt**
Enforces Brzozowski derivative-based DFA construction.

### **Cross-Consistency Prompt**
Requires:
1. language description  
2. membership classification  
3. formal DFA construction  

All templates are stored in `prompts/`.

---

## Example Outputs

The `example_outputs/` directory contains representative samples for each ToT method.  
These match the examples referenced in the paper.

---

## Requirements

- Python 3.10  
- Internet access  
- **API access** to GPT-5.x (or equivalent models)

No GPU is required for running these experiments.

---

## Reproducibility

To reproduce the ToT experiments:
- use the included prompts  
- run `run_experiment_tot.py`  
- inspect outputs in `example_outputs/`  

All code paths are deterministic other than model-side nondeterminism.

---

## Questions?

Open an issue in the root repository.

