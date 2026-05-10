"""
Regex -> DFA Dataset Generator for Qwen 2.5 Instruct LoRA Fine-tuning
=================================
Generates Chain-of-Thought (CoT) examples following:
  1. Thompson's NFA construction
  2. epsilon-closure computation
  3. Subset (Powerset) construction -> DFA
  4. DFA minimisation (Hopcroft's procedure)
  5. Full CoT trace formatted in Qwen ChatML template

Dataset tiers for curriculum learning:
  Tier 1 - concatenation, simple union                (~15%)
  Tier 2 - Kleene star, plus, optional                (~25%)
  Tier 3 - Combined: (a|b)*abb style                  (~35%)
  Tier 4 - Complex nested / multi-operator            (~25%)

FIXES vs v1:
  - Minimised DFA transition table was wrong (remap logic fully rewritten)
  - Accept states in minimised DFA were sometimes wrong (wrong rep state chosen)
  - Verification now cross-checks NFA == DFA == minDFA on all strings up to len 7
  - Verification examples now include meaningful accepted strings (not just len 0/1)
  - Minimisation CoT shows actual partition refinement steps (not just "apply Hopcroft")
  - "Already minimal" case is reported cleanly
  - Dead-state handling fully decoupled from live-state processing
"""

import random
import json
import hashlib
import itertools
import argparse
from collections import defaultdict, deque
from copy import deepcopy
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, total=None, **kwargs): self.n = 0; self.total = total
        def update(self, n=1):
            self.n += n
            print(f"\r  {self.n}/{self.total}", end="", flush=True)
        def __enter__(self): return self
        def __exit__(self, *a): print()


# ===================
#  1.  REGEX AST
# ===================

class RegexNode: pass

class Literal(RegexNode):
    def __init__(self, char): self.char = char
    def __repr__(self): return f"Lit({self.char!r})"

class Epsilon(RegexNode):
    def __repr__(self): return "eps"

class Concat(RegexNode):
    def __init__(self, left, right): self.left = left; self.right = right

class Union(RegexNode):
    def __init__(self, left, right): self.left = left; self.right = right

class Star(RegexNode):
    def __init__(self, child): self.child = child

class Plus(RegexNode):       # sugar: a+ = aa*
    def __init__(self, child): self.child = child

class Optional_(RegexNode):  # sugar: a? = a|eps
    def __init__(self, child): self.child = child


def desugar(node: RegexNode) -> RegexNode:
    """Rewrite Plus/Optional_ into core nodes (Concat/Union/Star)."""
    if isinstance(node, (Literal, Epsilon)):
        return node
    if isinstance(node, Concat):
        return Concat(desugar(node.left), desugar(node.right))
    if isinstance(node, Union):
        return Union(desugar(node.left), desugar(node.right))
    if isinstance(node, Star):
        return Star(desugar(node.child))
    if isinstance(node, Plus):
        d = desugar(node.child)
        return Concat(d, Star(deepcopy(d)))
    if isinstance(node, Optional_):
        return Union(desugar(node.child), Epsilon())
    raise ValueError(f"Unknown node {node}")


def regex_to_string(node: RegexNode, parent_prec: int = 0) -> str:
    """Convert AST back to a human-readable regex string."""
    if isinstance(node, Literal):   return node.char
    if isinstance(node, Epsilon):   return "eps"
    if isinstance(node, Star):
        inner = regex_to_string(node.child, 3)
        return f"({inner})*" if isinstance(node.child, (Concat, Union)) else f"{inner}*"
    if isinstance(node, Plus):
        inner = regex_to_string(node.child, 3)
        return f"({inner})+" if isinstance(node.child, (Concat, Union)) else f"{inner}+"
    if isinstance(node, Optional_):
        inner = regex_to_string(node.child, 3)
        return f"({inner})?" if isinstance(node.child, (Concat, Union)) else f"{inner}?"
    if isinstance(node, Concat):
        l = regex_to_string(node.left, 2)
        r = regex_to_string(node.right, 2)
        s = l + r
        return f"({s})" if parent_prec > 2 else s
    if isinstance(node, Union):
        l = regex_to_string(node.left, 1)
        r = regex_to_string(node.right, 1)
        s = f"{l}|{r}"
        return f"({s})" if parent_prec > 1 else s
    return "?"


# ===================
#  2.  RANDOM REGEX GENERATOR  (tiered)
# ===================

ALPHABETS = {
    "binary": ["0", "1"],
    "ab":     ["a", "b"],
    "abc":    ["a", "b", "c"],
    "abcd":   ["a", "b", "c", "d"],
    "digits": ["0", "1", "2", "3"],
}


def random_regex(tier: int, alphabet: list, rng: random.Random,
                 depth: int = 0) -> RegexNode:
    """
    Generate a random regex AST.
    Tier controls allowed operators AND how aggressively we recurse.
    Higher tiers have:
      - deeper trees (more states)
      - higher star/concat probability
      - lower leaf probability at shallow depth
    """
    max_depth = {1: 4, 2: 5, 3: 6, 4: 7}[tier]

    # Leaf probability: lower at shallow depth for higher tiers.
    # This forces the tree to grow before terminating.
    base_leaf = {1: 0.40, 2: 0.32, 3: 0.25, 4: 0.20}[tier]
    leaf_prob = min(0.92, base_leaf + 0.14 * depth)

    if depth >= max_depth or rng.random() < leaf_prob:
        return Literal(rng.choice(alphabet))

    if tier == 1:
        # Pure concatenation + union only; produces linear DFAs.
        ops, weights = ["concat", "union", "concat3"], [0.50, 0.35, 0.15]
    elif tier == 2:
        # Introduce closures; tends to produce small looping DFAs.
        ops, weights = (
            ["concat", "union", "star", "plus", "optional", "concat3"],
            [0.28, 0.20, 0.22, 0.14, 0.08, 0.08]
        )
    elif tier == 3:
        # Mix: closures + longer concatenations -> mid-size DFAs.
        ops, weights = (
            ["concat", "union", "star", "plus", "concat3", "star_union"],
            [0.28, 0.20, 0.22, 0.12, 0.10, 0.08]
        )
    else:
        # Deeply nested: closures of concatenations -> larger DFAs.
        ops, weights = (
            ["concat", "union", "star", "plus", "concat3", "star_union", "star_concat"],
            [0.22, 0.18, 0.22, 0.12, 0.10, 0.08, 0.08]
        )

    op = rng.choices(ops, weights=weights)[0]

    if op == "concat":
        return Concat(
            random_regex(tier, alphabet, rng, depth + 1),
            random_regex(tier, alphabet, rng, depth + 1))
    if op == "concat3":
        # Three-way concatenation: produces longer accepted strings -> more DFA states.
        n1 = random_regex(tier, alphabet, rng, depth + 1)
        n2 = random_regex(tier, alphabet, rng, depth + 1)
        n3 = random_regex(tier, alphabet, rng, depth + 1)
        return Concat(Concat(n1, n2), n3)
    if op == "union":
        return Union(
            random_regex(tier, alphabet, rng, depth + 1),
            random_regex(tier, alphabet, rng, depth + 1))
    if op == "star_union":
        # (X|Y)* style: very common pattern, produces interesting DFAs.
        return Star(Union(
            random_regex(tier, alphabet, rng, depth + 1),
            random_regex(tier, alphabet, rng, depth + 1)))
    if op == "star_concat":
        # (XY)* style: cyclic DFAs with 2+ states per cycle.
        return Star(Concat(
            random_regex(tier, alphabet, rng, depth + 1),
            random_regex(tier, alphabet, rng, depth + 1)))
    child = random_regex(tier, alphabet, rng, depth + 1)
    if op == "star":     return Star(child)
    if op == "plus":     return Plus(child)
    return Optional_(child)


# ===================
#  3.  THOMPSON'S NFA CONSTRUCTION
# ===================

class NFAState:
    _counter = 0

    def __init__(self):
        NFAState._counter += 1
        self.id = NFAState._counter
        self.transitions: dict = defaultdict(list)
        self.epsilon_transitions: list = []

    def __repr__(self):  return f"q{self.id}"
    def __hash__(self):  return hash(self.id)
    def __eq__(self, o): return isinstance(o, NFAState) and self.id == o.id


class NFA:
    def __init__(self, start: NFAState, accept: NFAState):
        self.start  = start
        self.accept = accept   # single accept state (Thompson invariant)

    def all_states(self):
        visited, stack = set(), [self.start]
        while stack:
            s = stack.pop()
            if s in visited: continue
            visited.add(s)
            for tgts in s.transitions.values():
                stack.extend(t for t in tgts if t not in visited)
            stack.extend(t for t in s.epsilon_transitions if t not in visited)
        return visited


def thompson(node: RegexNode) -> NFA:
    """Build NFA from desugared regex AST using Thompson's construction."""
    if isinstance(node, Literal):
        s, a = NFAState(), NFAState()
        s.transitions[node.char].append(a)
        return NFA(s, a)

    if isinstance(node, Epsilon):
        s, a = NFAState(), NFAState()
        s.epsilon_transitions.append(a)
        return NFA(s, a)

    if isinstance(node, Concat):
        n1 = thompson(node.left)
        n2 = thompson(node.right)
        n1.accept.epsilon_transitions.append(n2.start)
        return NFA(n1.start, n2.accept)

    if isinstance(node, Union):
        n1 = thompson(node.left)
        n2 = thompson(node.right)
        s, a = NFAState(), NFAState()
        s.epsilon_transitions.extend([n1.start, n2.start])
        n1.accept.epsilon_transitions.append(a)
        n2.accept.epsilon_transitions.append(a)
        return NFA(s, a)

    if isinstance(node, Star):
        n = thompson(node.child)
        s, a = NFAState(), NFAState()
        s.epsilon_transitions.extend([n.start, a])
        n.accept.epsilon_transitions.extend([n.start, a])
        return NFA(s, a)

    raise ValueError(f"thompson: unexpected node type {type(node)}")


# ===================
#  4.  EPSILON-CLOSURE + SUBSET CONSTRUCTION
# ===================

def epsilon_closure(states: frozenset) -> frozenset:
    stack, closure = list(states), set(states)
    while stack:
        s = stack.pop()
        for t in s.epsilon_transitions:
            if t not in closure:
                closure.add(t)
                stack.append(t)
    return frozenset(closure)


def nfa_move(states: frozenset, char: str) -> frozenset:
    result = set()
    for s in states:
        result.update(s.transitions.get(char, []))
    return frozenset(result)


class DFA:
    def __init__(self):
        self.states        = []
        self.state_index   = {}
        self.transitions   = {}   # (state_idx, char) -> state_idx
        self.start         = 0
        self.accept_states = set()
        self.alphabet      = []

    def accepts(self, s: str) -> bool:
        cur = self.start
        for c in s:
            if (cur, c) not in self.transitions:
                return False
            cur = self.transitions[(cur, c)]
        return cur in self.accept_states


def subset_construction(nfa: NFA, alphabet: list) -> DFA:
    dfa = DFA()
    dfa.alphabet = sorted(alphabet)

    start_closure = epsilon_closure(frozenset([nfa.start]))
    dfa.states.append(start_closure)
    dfa.state_index[start_closure] = 0
    if nfa.accept in start_closure:
        dfa.accept_states.add(0)

    queue = deque([start_closure])
    while queue:
        current = queue.popleft()
        cur_idx = dfa.state_index[current]
        for char in dfa.alphabet:
            nxt = epsilon_closure(nfa_move(current, char))
            if not nxt:
                continue
            if nxt not in dfa.state_index:
                idx = len(dfa.states)
                dfa.states.append(nxt)
                dfa.state_index[nxt] = idx
                if nfa.accept in nxt:
                    dfa.accept_states.add(idx)
                queue.append(nxt)
            dfa.transitions[(cur_idx, char)] = dfa.state_index[nxt]

    return dfa


# ===================
#  5.  DFA MINIMISATION -- Hopcroft's procedure
#      Completely rewritten; previous version had remap bugs.
# ===================

def minimise_dfa(dfa: DFA) -> tuple:
    """
    Returns (min_dfa, partition_log).
      min_dfa       : new DFA with states renumbered 0..k-1, start = 0
      partition_log : list[str] of refinement steps for CoT
    """
    n    = len(dfa.states)
    DEAD = n   # virtual trap/dead state

    # Complete transition table.
    full = {}
    for s in range(n):
        for c in dfa.alphabet:
            full[(s, c)] = dfa.transitions.get((s, c), DEAD)
    for c in dfa.alphabet:
        full[(DEAD, c)] = DEAD

    all_states = set(range(n + 1))   # 0..n-1 live, n dead

    accepting     = frozenset(dfa.accept_states)
    non_accepting = frozenset(s for s in range(n) if s not in dfa.accept_states)
    dead_group    = frozenset({DEAD})

    P = set()
    if accepting:     P.add(accepting)
    if non_accepting: P.add(non_accepting)
    P.add(dead_group)

    partition_log = []
    step_num = 0

    def pstr(partition):
        parts = []
        for g in sorted(partition, key=lambda g: min(g)):
            labels = ", ".join(
                f"D{s}" if s != DEAD else "dead" for s in sorted(g))
            parts.append("{" + labels + "}")
        return "[" + ", ".join(parts) + "]"

    partition_log.append(f"Initial partition P = {pstr(P)}")

    W = set(P)
    while W:
        A = W.pop()
        for c in dfa.alphabet:
            X = frozenset(s for s in all_states if full[(s, c)] in A)
            if not X:
                continue
            new_P = set()
            changed = False
            for Y in P:
                inter = Y & X
                diff  = Y - X
                if inter and diff:
                    new_P.add(inter)
                    new_P.add(diff)
                    changed = True
                    step_num += 1
                    partition_log.append(
                        f"  Split on '{c}' (splitter={pstr({A})}): "
                        f"{pstr({Y})} -> {pstr({inter})} and {pstr({diff})}"
                    )
                    if Y in W:
                        W.discard(Y)
                        W.add(inter)
                        W.add(diff)
                    else:
                        W.add(inter if len(inter) <= len(diff) else diff)
                else:
                    new_P.add(Y)
            if changed:
                P = new_P

    # Drop the dead group.
    live_groups = [g for g in P if g != dead_group]

    # Map every live old-state to its group index.
    state_to_gi: dict = {}
    for gi, group in enumerate(live_groups):
        for s in group:
            state_to_gi[s] = gi

    # Choose representative = lowest-numbered original state in each group.
    def rep(gi: int) -> int:
        return min(s for s in live_groups[gi] if s != DEAD)

    # BFS from start group to get a clean 0-indexed renaming.
    start_gi = state_to_gi[dfa.start]
    gi_to_new: dict = {}
    queue2 = deque([start_gi])
    visited2: set = set()
    counter = 0

    while queue2:
        gi = queue2.popleft()
        if gi in visited2: continue
        visited2.add(gi)
        gi_to_new[gi] = counter
        counter += 1
        r = rep(gi)
        for c in dfa.alphabet:
            tgt = full[(r, c)]
            if tgt != DEAD:
                tgt_gi = state_to_gi.get(tgt)
                if tgt_gi is not None and tgt_gi not in visited2:
                    queue2.append(tgt_gi)

    # Any unreachable live group (defensive).
    for gi in range(len(live_groups)):
        if gi not in gi_to_new:
            gi_to_new[gi] = counter
            counter += 1

    # Build the minimised DFA.
    min_dfa = DFA()
    min_dfa.alphabet = dfa.alphabet
    min_dfa.states   = list(range(len(live_groups)))
    min_dfa.start    = 0

    for gi, group in enumerate(live_groups):
        new_idx = gi_to_new[gi]
        r       = rep(gi)
        # Accept state: if any member was an accept state.
        if any(s in dfa.accept_states for s in group if s != DEAD):
            min_dfa.accept_states.add(new_idx)
        # Transitions.
        for c in dfa.alphabet:
            tgt = full[(r, c)]
            if tgt != DEAD:
                tgt_gi = state_to_gi.get(tgt)
                if tgt_gi is not None:
                    min_dfa.transitions[(new_idx, c)] = gi_to_new[tgt_gi]

    if step_num == 0:
        partition_log.append(
            "  No splits occurred -- the DFA is already minimal.")
    partition_log.append(
        f"Final: {len(live_groups)} equivalence classes "
        f"-> minimised DFA has {len(live_groups)} states."
    )

    return min_dfa, partition_log


# ===================
#  6.  VERIFICATION (NFA == DFA == minDFA on all strings <= len 7)
# ===================

def strings_up_to_length(alphabet: list, max_len: int):
    for length in range(0, max_len + 1):
        for combo in itertools.product(alphabet, repeat=length):
            yield "".join(combo)


def nfa_accepts_str(nfa: NFA, s: str) -> bool:
    states = epsilon_closure(frozenset([nfa.start]))
    for c in s:
        states = epsilon_closure(nfa_move(states, c))
    return nfa.accept in states


def verify_all(nfa: NFA, dfa: DFA, min_dfa: DFA, alphabet: list,
               max_len: int = 7) -> tuple:
    """Returns (ok: bool, error_msg: str)."""
    for s in strings_up_to_length(alphabet, max_len):
        a = nfa_accepts_str(nfa, s)
        b = dfa.accepts(s)
        c = min_dfa.accepts(s)
        if a != b:
            return False, f"NFA/DFA mismatch on '{s}': NFA={a}, DFA={b}"
        if a != c:
            return False, f"NFA/minDFA mismatch on '{s}': NFA={a}, minDFA={c}"
    return True, ""


# ===================
#  7.  CoT TRACE BUILDER
# ===================

def fmt_q(s: NFAState) -> str: return f"q{s.id}"
def fmt_D(i: int) -> str:      return f"D{i}"

def fmt_nfa_set(fs: frozenset) -> str:
    ids = sorted(s.id for s in fs)
    return "{" + ", ".join(f"q{i}" for i in ids) + "}"


def build_cot_trace(
    regex_str: str,
    ast_raw:   RegexNode,
    nfa:       NFA,
    dfa:       DFA,
    min_dfa:   DFA,
    partition_log: list,
    alphabet:  list,
) -> str:
    lines = []
    sa = sorted(alphabet)

    # Step 1: Structure
    lines += [
        "## Step 1: Analyse the Regular Expression\n",
        f"Given regex: **{regex_str}**\n",
        f"Alphabet S = {{{', '.join(sa)}}}\n",
        "Parse tree structure:",
        _describe_structure(ast_raw),
    ]

    # Step 2: Thompson's Rules
    lines += ["", "## Step 2: Thompson's Construction Rules Applied\n"]
    rules = _collect_thompson_rules(ast_raw)
    lines.append("We apply the following Thompson construction rules:\n")
    for r in rules:
        lines.append(f"  - {r}")

    # Step 3: NFA State Listing
    nfa_states = sorted(nfa.all_states(), key=lambda s: s.id)
    lines += [
        "",
        "## Step 3: NFA States and Transitions\n",
        f"NFA states: {{{', '.join(fmt_q(s) for s in nfa_states)}}}",
        f"Start state: {fmt_q(nfa.start)}",
        f"Accept state: {fmt_q(nfa.accept)}\n",
        "Transition function d_NFA:\n",
        "| State | Symbol | Next States |",
        "|-------|--------|-------------|",
    ]
    for s in nfa_states:
        for char, targets in sorted(s.transitions.items()):
            tstr = "{" + ", ".join(
                fmt_q(t) for t in sorted(targets, key=lambda x: x.id)) + "}"
            lines.append(f"| {fmt_q(s)} | {char} | {tstr} |")
        if s.epsilon_transitions:
            tstr = "{" + ", ".join(
                fmt_q(t) for t in sorted(
                    s.epsilon_transitions, key=lambda x: x.id)) + "}"
            lines.append(f"| {fmt_q(s)} | eps | {tstr} |")
    lines.append("")

    # Step 4: epsilon-closure of start
    start_ec = epsilon_closure(frozenset([nfa.start]))
    lines += [
        "## Step 4: Compute epsilon-closure of Start State\n",
        f"eps-closure({fmt_q(nfa.start)}) = {fmt_nfa_set(start_ec)}",
        "This becomes DFA start state D0.\n",
    ]
    _explain_eps_closure(nfa.start, lines)

    # Step 5: Subset Construction Table
    lines += [
        "",
        "## Step 5: Subset Construction (Powerset Construction)\n",
        "We expand each DFA state by computing move() then "
        "eps-closure for each symbol.\n",
    ]
    header = ("| DFA State | NFA States | "
              + " | ".join(sa) + " | Accept? |")
    sep    = ("|-----------|------------|"
              + "|".join(["--------"] * len(sa)) + "|---------|")
    lines += [header, sep]

    for i, nfa_set in enumerate(dfa.states):
        row = []
        for c in sa:
            t = dfa.transitions.get((i, c))
            row.append(fmt_D(t) if t is not None else "empty")
        acc = "Y" if i in dfa.accept_states else ""
        lines.append(
            f"| {fmt_D(i)} | {fmt_nfa_set(nfa_set)} | "
            + " | ".join(row) + f" | {acc} |")

    lines += [
        "",
        f"Total DFA states before minimisation: **{len(dfa.states)}**",
        f"Start state: {fmt_D(dfa.start)}",
        f"Accept states: {{{', '.join(fmt_D(i) for i in sorted(dfa.accept_states))}}}\n",
    ]

    # Step 5b: Detailed walkthrough
    lines.append("### Subset Construction Walkthrough\n")
    for i, nfa_set in enumerate(dfa.states[:min(8, len(dfa.states))]):
        lines.append(f"**{fmt_D(i)}** = {fmt_nfa_set(nfa_set)}:")
        for c in sa:
            moved = nfa_move(nfa_set, c)
            ec    = epsilon_closure(moved)
            t     = dfa.transitions.get((i, c))
            if ec:
                lines.append(
                    f"  - On '{c}': move = {fmt_nfa_set(moved)}, "
                    f"eps-closure = {fmt_nfa_set(ec)} -> {fmt_D(t)}"
                )
            else:
                lines.append(
                    f"  - On '{c}': move = empty "
                    f"-> dead/trap state (no transition recorded)")
        lines.append("")
    if len(dfa.states) > 8:
        lines.append(
            f"*(Remaining {len(dfa.states) - 8} states omitted for brevity.)*\n")

    # Step 6: Minimisation
    lines += [
        "## Step 6: DFA Minimisation (Hopcroft's Procedure)\n",
        "**Partition refinement trace:**\n",
    ]
    for entry in partition_log:
        lines.append(entry)
    lines.append("")

    if len(min_dfa.states) == len(dfa.states):
        lines.append(
            "The DFA is **already minimal** -- all states are distinguishable. "
            "The minimised DFA is structurally identical to the "
            "pre-minimisation DFA.\n"
        )
    else:
        lines.append(
            f"Minimisation reduced **{len(dfa.states)} -> "
            f"{len(min_dfa.states)} states** "
            f"by merging indistinguishable equivalence classes.\n"
        )

    # Step 7: Final Minimised DFA
    lines += [
        "## Step 7: Final Minimised DFA\n",
        f"States Q = {{{', '.join(fmt_D(i) for i in range(len(min_dfa.states)))}}}",
        f"Start state: {fmt_D(min_dfa.start)}",
        f"Accept states F = "
        f"{{{', '.join(fmt_D(i) for i in sorted(min_dfa.accept_states))}}}",
        f"Alphabet S = {{{', '.join(sa)}}}\n",
        "Transition function d (-- = no transition / leads to dead state):\n",
    ]
    h2 = "| State | " + " | ".join(sa) + " | Accept? |"
    s2 = "|-------|" + "|".join(["-------"] * len(sa)) + "|---------|"
    lines += [h2, s2]
    for s in range(len(min_dfa.states)):
        row = []
        for c in sa:
            t = min_dfa.transitions.get((s, c))
            row.append(fmt_D(t) if t is not None else "--")
        acc = "Y" if s in min_dfa.accept_states else ""
        lines.append(f"| {fmt_D(s)} | " + " | ".join(row) + f" | {acc} |")
    lines.append("")

    # Step 8: Verification
    lines += [
        "## Step 8: Verification\n",
        "We test the minimised DFA against representative strings "
        "(both accepted Y and rejected N):\n",
        "| String | Result | State Trace |",
        "|--------|--------|-------------|",
    ]
    for word, accepted, trace in _verification_examples(min_dfa, alphabet):
        label = "Y Accept" if accepted else "N Reject"
        disp  = f'"{word}"' if word else '"eps" (empty string)'
        lines.append(f"| {disp} | {label} | {trace} |")

    # Summary
    sample = _sample_accepted(min_dfa, alphabet)
    lines += [
        "",
        "## Summary\n",
        f"The regex `{regex_str}` over S = {{{', '.join(sa)}}} is recognised "
        f"by a minimised DFA with:",
        f"  - **{len(min_dfa.states)} states**: "
        + ", ".join(fmt_D(i) for i in range(len(min_dfa.states))),
        f"  - **Start state**: {fmt_D(min_dfa.start)}",
        f"  - **Accept state(s)**: "
        + (", ".join(fmt_D(i) for i in sorted(min_dfa.accept_states))
           or "none"),
        f"  - **{len(min_dfa.transitions)} transition(s)** total",
        f"  - Example accepted strings: {sample}",
    ]

    return "\n".join(lines)


# -- helpers ---------

def _sample_accepted(dfa: DFA, alphabet: list) -> str:
    found = []
    for length in range(0, 8):
        for combo in itertools.product(alphabet, repeat=length):
            word = "".join(combo)
            if dfa.accepts(word):
                found.append(f'"{word}"' if word else '"eps"')
        if len(found) >= 3:
            break
    return ", ".join(found) if found else "none (empty language)"


def _describe_structure(node: RegexNode, depth: int = 0) -> str:
    pad = "  " * depth
    if isinstance(node, Literal):   return f"{pad}Literal '{node.char}'"
    if isinstance(node, Epsilon):   return f"{pad}eps (epsilon)"
    if isinstance(node, Star):
        return (f"{pad}Kleene Star (*)\n"
                f"{_describe_structure(node.child, depth + 1)}")
    if isinstance(node, Plus):
        return (f"{pad}One-or-more (+)\n"
                f"{_describe_structure(node.child, depth + 1)}")
    if isinstance(node, Optional_):
        return (f"{pad}Optional (?)\n"
                f"{_describe_structure(node.child, depth + 1)}")
    if isinstance(node, Concat):
        return (f"{pad}Concatenation\n"
                f"{_describe_structure(node.left,  depth + 1)}\n"
                f"{_describe_structure(node.right, depth + 1)}")
    if isinstance(node, Union):
        return (f"{pad}Union (|)\n"
                f"{_describe_structure(node.left,  depth + 1)}\n"
                f"{_describe_structure(node.right, depth + 1)}")
    return f"{pad}?"


def _collect_thompson_rules(node: RegexNode, seen: set = None) -> list:
    if seen is None: seen = set()
    rules = []
    key = type(node).__name__
    if key not in seen:
        seen.add(key)
        if isinstance(node, Literal):
            rules.append(
                f"**Symbol rule** for '{node.char}': "
                f"two states s, a with s --'{node.char}'--> a"
            )
        if isinstance(node, Epsilon):
            rules.append(
                "**Epsilon rule**: two states s, a with s --eps--> a")
        if isinstance(node, Concat):
            rules.append(
                "**Concatenation rule**: build N(r1) and N(r2) independently, "
                "then add eps-transition from N(r1)'s accept to N(r2)'s start"
            )
        if isinstance(node, Union):
            rules.append(
                "**Union rule**: new start q0 with eps-transitions to "
                "N(r1) and N(r2) starts; both accepts eps-connect to new accept qa"
            )
        if isinstance(node, Star):
            rules.append(
                "**Kleene Star rule**: new start q0 and accept qa; "
                "q0 --eps--> N(r).start and q0 --eps--> qa (allow skip); "
                "N(r).accept --eps--> N(r).start (loop) and --eps--> qa (exit)"
            )
    children = (
        [node.left, node.right] if isinstance(node, (Concat, Union))
        else [node.child] if isinstance(node, (Star, Plus, Optional_))
        else []
    )
    for child in children:
        rules += _collect_thompson_rules(child, seen)
    return rules


def _explain_eps_closure(start: NFAState, lines: list):
    visited, queue, steps = {start}, deque([start]), []
    while queue:
        s = queue.popleft()
        for t in s.epsilon_transitions:
            if t not in visited:
                visited.add(t)
                queue.append(t)
                steps.append(f"  {fmt_q(s)} --eps--> {fmt_q(t)}")
    if steps:
        lines.append("eps-closure BFS expansion:")
        lines.extend(steps)
    else:
        lines.append(
            "  (Start state has no outgoing eps-transitions; "
            "eps-closure = {start state})")


def _verification_examples(min_dfa: DFA, alphabet: list) -> list:
    """Up to 3 accepted + 3 rejected strings, with full state traces."""
    accepted, rejected = [], []
    for length in range(0, 9):
        for combo in itertools.product(alphabet, repeat=length):
            word = "".join(combo)
            if min_dfa.accepts(word) and len(accepted) < 3:
                accepted.append(word)
            elif not min_dfa.accepts(word) and len(rejected) < 3:
                rejected.append(word)
        if len(accepted) >= 3 and len(rejected) >= 3:
            break

    results = []
    for word in accepted + rejected:
        cur, parts, dead = min_dfa.start, [fmt_D(min_dfa.start)], False
        for c in word:
            nxt = min_dfa.transitions.get((cur, c))
            if nxt is None:
                parts.append(f"--{c}-->empty")
                dead = True
                break
            parts.append(f"--{c}-->{fmt_D(nxt)}")
            cur = nxt
        ok = (not dead) and (cur in min_dfa.accept_states)
        results.append((word, ok, " ".join(parts)))
    return results


# ===================
#  8.  QWEN ChatML FORMATTER
# ===================

SYSTEM_PROMPT = (
    "You are an expert in formal language theory and automata. "
    "When given a regular expression, you convert it to a Deterministic "
    "Finite Automaton (DFA) using Thompson's construction to build an NFA, "
    "followed by the subset (powerset) construction to convert to a DFA, "
    "and finally Hopcroft's procedure to minimise the DFA. "
    "You always show your full working step by step."
)

USER_TEMPLATES = [
    "Convert the regular expression `{regex}` over the alphabet S = {{{alpha}}} "
    "to a minimised DFA. Show all steps.",
    "Given the regex `{regex}` with alphabet {{{alpha}}}, construct a minimised "
    "DFA using Thompson's construction followed by subset construction. "
    "Show your full work.",
    "Use Thompson's NFA construction and the powerset construction to build a "
    "DFA for the regex `{regex}` (S = {{{alpha}}}), then minimise it using "
    "Hopcroft's procedure.",
    "Construct a minimised DFA that accepts exactly the language described by "
    "the regular expression `{regex}` over S = {{{alpha}}}. Show the NFA, DFA, "
    "and minimised DFA.",
    (
        "For the regular expression `{regex}` over alphabet {{{alpha}}}, perform:\n"
        "1. Thompson's NFA construction\n"
        "2. epsilon-closure computation\n"
        "3. Subset construction to DFA\n"
        "4. DFA minimisation\n"
        "Show all intermediate steps."
    ),
]


def format_qwen_chatml(regex_str: str, alphabet: list, cot: str,
                       rng: random.Random) -> dict:
    alpha_str = ", ".join(sorted(alphabet))
    user_msg  = rng.choice(USER_TEMPLATES).format(
        regex=regex_str, alpha=alpha_str)
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": cot},
        ]
    }


# ===================
#  9.  GENERATION PIPELINE
# ===================

TIER_DIST = {1: 0.15, 2: 0.25, 3: 0.35, 4: 0.25}

MAX_DFA_STATES     = 24
MAX_MIN_DFA_STATES = 18

# Minimum min-DFA states by tier: enforces complexity per tier.
MIN_DFA_BY_TIER = {1: 2, 2: 2, 3: 3, 4: 4}

HANDCRAFTED = [
    # Classic textbook examples
    ("(a|b)*abb",              ["a", "b"]),
    ("(a|b)*a(a|b)",           ["a", "b"]),
    ("(0|1)*00",               ["0", "1"]),
    ("(0|1)*0(0|1)",           ["0", "1"]),
    ("(0|1)*101",              ["0", "1"]),
    ("(a|b)*ab",               ["a", "b"]),
    ("(a|b)*aabb(a|b)*",       ["a", "b"]),
    ("(a|b)*bab",              ["a", "b"]),
    ("(0|1)*010(0|1)*",        ["0", "1"]),
    ("(0|1)*11(0|1)*",         ["0", "1"]),
    # Kleene / closure patterns
    ("a*b*",                   ["a", "b"]),
    ("a*ba*",                  ["a", "b"]),
    ("(aa|b)*",                ["a", "b"]),
    ("(0|1)*(01)(0|1)*",       ["0", "1"]),
    ("(ab|ba)*",               ["a", "b"]),
    ("(ab|cd)*",               ["a", "b", "c", "d"]),
    ("(abc|bca|cab)*",         ["a", "b", "c"]),
    ("(aab|b)*",               ["a", "b"]),
    ("(a|bb)*",                ["a", "b"]),
    # Plus and optional
    ("a+b+",                   ["a", "b"]),
    ("(ab)+",                  ["a", "b"]),
    ("a*(b|c)+a*",             ["a", "b", "c"]),
    ("a?b?c?",                 ["a", "b", "c"]),
    ("(a|b)?c",                ["a", "b", "c"]),
    # Pure concatenation
    ("abbba",                  ["a", "b"]),
    ("1011",                   ["0", "1"]),
    ("abcd",                   ["a", "b", "c", "d"]),
    ("0110",                   ["0", "1"]),
    ("abab",                   ["a", "b"]),
    # Multi-symbol alphabet complexity
    ("(a|b|c)*abc",            ["a", "b", "c"]),
    ("a(b|c)*",                ["a", "b", "c"]),
    # Harder patterns
    ("(a|b)*aba(a|b)*",        ["a", "b"]),
    ("(0|1)*001(0|1)*",        ["0", "1"]),
    ("(a|b)*abba(a|b)*",       ["a", "b"]),
    ("(0|1)*0101",             ["0", "1"]),
    ("(a|b)(a|b)(a|b)",        ["a", "b"]),
    ("(0|1)*(00|11)(0|1)*",    ["0", "1"]),
    ("a(ba)*",                 ["a", "b"]),
    ("(ab)*(ba)*",             ["a", "b"]),
    # Strings of even / divisible length
    ("(aa)*",                  ["a", "b"]),
    ("(aaa)*",                 ["a", "b"]),
    ("(ab|ba|aa|bb)*",         ["a", "b"]),
    # Contains specific substring
    ("(a|b)*aa(a|b)*",         ["a", "b"]),
    ("(0|1)*00(0|1)*",         ["0", "1"]),
    ("(a|b)*bb(a|b)*",         ["a", "b"]),
]


class _Parser:
    def __init__(self, s): self.s = s; self.pos = 0

    def peek(self):
        return self.s[self.pos] if self.pos < len(self.s) else None

    def consume(self, c=None):
        ch = self.s[self.pos]
        if c and ch != c:
            raise ValueError(f"Expected {c!r} got {ch!r}")
        self.pos += 1
        return ch

    def parse(self):
        node = self._union()
        if self.pos != len(self.s):
            raise ValueError(f"Leftover at {self.pos}")
        return node

    def _union(self):
        left = self._concat()
        while self.peek() == '|':
            self.consume('|')
            left = Union(left, self._concat())
        return left

    def _concat(self):
        nodes = []
        while self.peek() not in (None, ')', '|'):
            nodes.append(self._quantified())
        if not nodes:
            raise ValueError("Empty concat")
        result = nodes[0]
        for n in nodes[1:]:
            result = Concat(result, n)
        return result

    def _quantified(self):
        base = self._atom()
        q = self.peek()
        if q == '*': self.consume(); return Star(base)
        if q == '+': self.consume(); return Plus(base)
        if q == '?': self.consume(); return Optional_(base)
        return base

    def _atom(self):
        c = self.peek()
        if c == '(':
            self.consume('(')
            node = self._union()
            self.consume(')')
            return node
        if c and c not in (')', '|', '*', '+', '?'):
            self.consume()
            return Literal(c)
        raise ValueError(f"Unexpected char {c!r}")


def parse_regex(s: str) -> RegexNode:
    return _Parser(s).parse()


def _build_entry(regex_str: str, ast_raw: RegexNode, alphabet: list,
                 rng: random.Random, tier) -> Optional[dict]:
    NFAState._counter = 0
    try:
        ast     = desugar(ast_raw)
        nfa     = thompson(ast)
        dfa     = subset_construction(nfa, alphabet)
        if len(dfa.states) > MAX_DFA_STATES:
            return None
        min_dfa, plog = minimise_dfa(dfa)
        min_states_floor = (MIN_DFA_BY_TIER.get(tier, 2)
                            if isinstance(tier, int) else 2)
        if (len(min_dfa.states) > MAX_MIN_DFA_STATES
                or len(min_dfa.states) < min_states_floor):
            return None
    except Exception:
        return None

    ok, err = verify_all(nfa, dfa, min_dfa, alphabet, max_len=7)
    if not ok:
        # Silently skip: verification failure means a bug slipped through.
        return None

    cot   = build_cot_trace(
        regex_str, ast_raw, nfa, dfa, min_dfa, plog, alphabet)
    entry = format_qwen_chatml(regex_str, alphabet, cot, rng)
    entry["metadata"] = {
        "regex":          regex_str,
        "alphabet":       sorted(alphabet),
        "tier":           tier,
        "nfa_states":     len(nfa.all_states()),
        "dfa_states":     len(dfa.states),
        "min_dfa_states": len(min_dfa.states),
        "accept_states":  sorted(min_dfa.accept_states),
        "transitions":    len(min_dfa.transitions),
        "hash":           hashlib.md5(
            (regex_str + str(sorted(alphabet))).encode()
        ).hexdigest()[:8],
    }
    return entry


def generate_handcrafted(rng: random.Random) -> list:
    entries = []
    for regex_str, alphabet in HANDCRAFTED:
        try:
            ast_raw = parse_regex(regex_str)
        except Exception as e:
            print(f"  [WARN] Parse failed '{regex_str}': {e}")
            continue
        entry = _build_entry(regex_str, ast_raw, alphabet, rng, "handcrafted")
        if entry is None:
            print(f"  [WARN] Build/verify failed for '{regex_str}'")
        else:
            entries.append(entry)
    return entries


def generate_random_entry(rng: random.Random, tier: int) -> Optional[dict]:
    alpha_name = rng.choice(list(ALPHABETS.keys()))
    alphabet   = ALPHABETS[alpha_name]
    ast_raw    = random_regex(tier, alphabet, rng)
    regex_str  = regex_to_string(ast_raw)
    if len(regex_str) < 2:
        return None
    return _build_entry(regex_str, ast_raw, alphabet, rng, tier)


# ===================
#  10.  STATISTICS + ENTRY POINT
# ===================

def print_stats(dataset: list):
    print("\n" + "=" * 62)
    print("  DATASET STATISTICS")
    print("=" * 62)
    tiers, sc = {}, []
    for ex in dataset:
        m = ex["metadata"]
        tiers[m["tier"]] = tiers.get(m["tier"], 0) + 1
        sc.append(m["min_dfa_states"])
    print(f"  Total examples   : {len(dataset):,}")
    for t in sorted(tiers.keys(), key=str):
        print(f"  Tier {str(t):<13} : {tiers[t]:,}")
    print(f"  Min DFA states   : avg={sum(sc)/len(sc):.2f}, "
          f"min={min(sc)}, max={max(sc)}")
    ac = {}
    for ex in dataset:
        k = tuple(ex["metadata"]["alphabet"])
        ac[k] = ac.get(k, 0) + 1
    print("  Alphabet dist    :")
    for k, v in sorted(ac.items(), key=lambda x: -x[1]):
        print(f"    {list(k)}: {v:,}")
    print("=" * 62 + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Generate Regex->DFA CoT dataset for Qwen LoRA fine-tuning"
    )
    ap.add_argument("--n",          type=int,   default=25000)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--out",        type=str,   default="regex_dfa_dataset.jsonl")
    ap.add_argument("--val_split",  type=float, default=0.05)
    ap.add_argument("--test_split", type=float, default=0.05)
    args = ap.parse_args()

    rng     = random.Random(args.seed)
    dataset = []
    seen    = set()

    print("Generating handcrafted examples...")
    for ex in generate_handcrafted(rng):
        h = ex["metadata"]["hash"]
        if h not in seen:
            seen.add(h)
            dataset.append(ex)
    print(f"  Added {len(dataset)} handcrafted examples.")

    print(f"Generating up to {args.n:,} random examples...")
    max_attempts = args.n * 25

    with tqdm(total=args.n) as pbar:
        pbar.update(len(dataset))
        attempts = 0
        while len(dataset) < args.n and attempts < max_attempts:
            attempts += 1
            tier  = rng.choices(
                list(TIER_DIST), weights=list(TIER_DIST.values()))[0]
            entry = generate_random_entry(rng, tier)
            if entry is None: continue
            h = entry["metadata"]["hash"]
            if h in seen: continue
            seen.add(h)
            dataset.append(entry)
            pbar.update(1)

    print(f"\nGenerated {len(dataset):,} unique examples "
          f"in {attempts:,} attempts.")
    print_stats(dataset)

    rng.shuffle(dataset)
    n       = len(dataset)
    n_val   = int(n * args.val_split)
    n_test  = int(n * args.test_split)
    n_train = n - n_val - n_test

    splits = {
        "train": dataset[:n_train],
        "val":   dataset[n_train:n_train + n_val],
        "test":  dataset[n_train + n_val:],
    }
    base = args.out.replace(".jsonl", "")
    for name, data in splits.items():
        path = f"{base}_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Wrote {len(data):,} examples -> {path}")

    with open(args.out, "w", encoding="utf-8") as f:
        for ex in dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\n  Full dataset -> {args.out}")
    print("\nDone")


if __name__ == "__main__":
    main()
