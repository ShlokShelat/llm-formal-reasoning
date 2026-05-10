#!/usr/bin/env python3

import os, json, time, re
from datetime import datetime
from typing import Optional
from openai import OpenAI

MODEL            = "gpt-5.2"
TEMPERATURE      = 0.0
MAX_TOKENS       = 4000
PROMPT_FILES     = {
    "direct":     "prompts/analysis_cot_branch_direct.txt",
    "derivative": "prompts/analysis_cot_branch_derivative.txt"
}
DATASET_PATH     = "data/analysis_new_final.json"
OUTPUT_RAW_DIR   = "outputs/model/raw_tot"
OUTPUT_TABLE_DIR = "outputs/model/tables_tot"
OUTPUT_META_DIR  = "outputs/model/meta_tot"
RETRIES          = 3
RETRY_SLEEP      = 1.5
RATE_LIMIT_SLEEP = 0.3

client = OpenAI()
for d in [OUTPUT_RAW_DIR, OUTPUT_TABLE_DIR, OUTPUT_META_DIR]:
    os.makedirs(d, exist_ok=True)

def timestamp():
    return datetime.utcnow().strftime(
    
def safe_filename(s):
    return re.sub(r"[^0-9A-Za-z._-]", "_", s)[:200]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def make_json_safe(obj):
    if obj is None or isinstance(obj, (str,int,float,bool)):
        return obj
    if isinstance(obj, list):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k,v in obj.items()}
    if hasattr(obj, "__dict__"):
        return make_json_safe(vars(obj))
    return str(obj)

def try_parse_json(text):
    if not text: return None
    cleaned = re.sub(
        r"```(?:json)?", "", text,
        flags=re.IGNORECASE).strip("` \n")
    try: return json.loads(cleaned)
    except Exception: pass
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        try: return json.loads(m.group(0))
        except Exception: pass
    return None

def completion_kwargs():
    if MODEL.startswith("gpt-5"):
        return {"max_completion_tokens": MAX_TOKENS}
    return {"max_tokens": MAX_TOKENS}

def build_messages(template, regex, alphabet):
    filled = (template
        .replace("{{REGEX}}", regex)
        .replace("{{ALPHABET}}", json.dumps(alphabet)))
    return [
        {"role": "system", "content":
            "You are an expert in formal languages. "
            "Output only the DFA JSON."},
        {"role": "user", "content": filled}
    ]

def call_model(messages):
    response = client.chat.completions.create(
        model=MODEL, messages=messages,
        temperature=TEMPERATURE, **completion_kwargs())
    text = response.choices[0].message.content or ""
    meta = make_json_safe({
        "finish_reason": response.choices[0].finish_reason,
        "usage": getattr(response, "usage", None)})
    return text, meta

def run_single_branch(entry, branch_name, template):
    regex_id = entry["id"]
    regex    = entry["regex"]
    alphabet = entry.get("alphabet", [])
    ts       = timestamp()
    messages = build_messages(template, regex, alphabet)
    raw_text, meta, parsed, attempts = "", None, None, 0

    while attempts < RETRIES:
        attempts += 1
        try:
            raw_text, meta = call_model(messages)
        except Exception as e:
            meta = {"error": str(e)}
            time.sleep(RETRY_SLEEP); continue
        parsed = try_parse_json(raw_text)
        if parsed: break
        messages.append({"role": "user",
            "content": "Output only the DFA JSON."})
        time.sleep(RETRY_SLEEP)

    base = safe_filename(
        f"{regex_id}_{branch_name}_{ts}")
    save_json(
        os.path.join(OUTPUT_RAW_DIR, base+".json"),
        make_json_safe({"id": regex_id,
            "branch": branch_name, "regex": regex,
            "alphabet": alphabet,
            "raw_output": raw_text,
            "attempts": attempts, "meta": meta,
            "model": MODEL, "timestamp": ts}))
    save_json(
        os.path.join(OUTPUT_META_DIR,
            base+"_meta.json"),
        make_json_safe(meta))
    if parsed is None:
        parsed = {"error": "GENERATION_FAILED",
            "reason": "Invalid or non-JSON output",
            "regex_id": regex_id,
            "branch": branch_name,
            "model": MODEL, "attempts": attempts,
            "meta": meta,
            "raw_output_snippet": raw_text[:1000]}
    save_json(
        os.path.join(OUTPUT_TABLE_DIR, base+".json"),
        parsed)

def main():
    dataset = load_json(DATASET_PATH)
    templates = {k: load_text(v)
        for k,v in PROMPT_FILES.items()}
    for entry in dataset:
        for branch, tmpl in templates.items():
            run_single_branch(entry, branch, tmpl)
            time.sleep(RATE_LIMIT_SLEEP)

if __name__ == "__main__":
    main()
