import os
import time
import random
import numpy as np
import torch
import csv
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from tqdm import tqdm

# ===================
# CONFIGURATION & REPRODUCIBILITY
# ===================
SEED = 42
set_seed(SEED)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"

MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"

TRAIN_FILE = "dataset_reasoning/train_ps_10.jsonl"
IID_FILE   = "dataset_reasoning/iid_ps_final_10.jsonl"
OOD_FILE   = "dataset_reasoning/ood_ps_010.jsonl"

OUTPUT_DIR = "lora_output_qwen7b_cot"

MAX_SEQ_LEN = 8192

EVAL_STEPS = 200
SAVE_STEPS = 200
LOG_STEPS  = 10

IID_EVAL_SAMPLES = 200
OOD_EVAL_SAMPLES = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print(f"Random seed: {SEED} | GPU: {torch.cuda.get_device_name(0)}")
print("=" * 80)

# ===================
# SYSTEM PROMPT
# ===================
SYSTEM_PROMPT = (
    "You are a DFA construction assistant. "
    "You must output the TRACE and FINAL DFA exactly in the same format "
    "as training data. "
    "Do not output any explanation outside the TRACE markers."
)

# ===================
# CHAT TEMPLATE FORMATTER
# ===================
def build_chat_text(tokenizer, prompt, output):
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": output},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

def format_sample(example, tokenizer):
    prompt = example["prompt"]
    output = example["output"]
    text   = build_chat_text(tokenizer, prompt, output)
    return {"text": text}

# ===================
# CHECKPOINT FINDER
# ===================
def find_last_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
    ]
    if not checkpoints:
        return None
    checkpoints = sorted(
        checkpoints, key=lambda x: int(x.split("-")[-1]))
    return os.path.join(output_dir, checkpoints[-1])

# ===================
# CALLBACKS (TQDM + CSV LOGGER)
# ===================
class MetricsLogger(TrainerCallback):
    def __init__(self, output_dir):
        self.metrics_file = os.path.join(
            output_dir, "training_metrics.csv")
        with open(self.metrics_file, "w", newline="") as f:
            csv.writer(f).writerow(
                ["step", "epoch", "loss", "eval_loss"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.metrics_file, "a", newline="") as f:
                csv.writer(f).writerow([
                    state.global_step,
                    round(state.epoch, 4) if state.epoch else "",
                    logs.get("loss", ""),
                    logs.get("eval_loss", "")
                ])

class TqdmCallback(TrainerCallback):
    def __init__(self):
        self.pbar = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.pbar = tqdm(
            total=state.max_steps, desc="Training", unit="step")

    def on_step_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.close()

# ===================
# MAIN
# ===================
def main():
    # -----------------------------
    # 1. LOAD TOKENIZER
    # -----------------------------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LEN

    # -----------------------------
    # 2. LOAD DATASETS
    # -----------------------------
    print("Loading datasets...")
    train_data = load_dataset(
        "json", data_files=TRAIN_FILE, split="train")
    iid_data   = load_dataset(
        "json", data_files=IID_FILE,   split="train")
    ood_data   = load_dataset(
        "json", data_files=OOD_FILE,   split="train")
    print(f"Train: {len(train_data):,} | "
          f"IID: {len(iid_data):,} | "
          f"OOD: {len(ood_data):,}")

    # -----------------------------
    # 3. FORMAT INTO CHAT TEMPLATE
    # -----------------------------
    print("Formatting train dataset...")
    train_data = train_data.map(
        lambda x: format_sample(x, tokenizer),
        remove_columns=train_data.column_names,
        num_proc=4
    )
    print("Formatting IID dataset...")
    iid_data = iid_data.map(
        lambda x: format_sample(x, tokenizer),
        remove_columns=iid_data.column_names,
        num_proc=4
    )
    print("Formatting OOD dataset...")
    ood_data = ood_data.map(
        lambda x: format_sample(x, tokenizer),
        remove_columns=ood_data.column_names,
        num_proc=4
    )
    train_data = train_data.shuffle(seed=SEED)

    # -----------------------------
    # 4. BUILD EVAL SET (MERGED IID + OOD)
    # -----------------------------
    if len(iid_data) > IID_EVAL_SAMPLES:
        iid_eval = iid_data.shuffle(seed=SEED).select(
            range(IID_EVAL_SAMPLES))
    else:
        iid_eval = iid_data

    if len(ood_data) > OOD_EVAL_SAMPLES:
        ood_eval = ood_data.shuffle(seed=SEED).select(
            range(OOD_EVAL_SAMPLES))
    else:
        ood_eval = ood_data

    eval_dataset = concatenate_datasets([iid_eval, ood_eval])
    print(f"Eval dataset size: {len(eval_dataset):,}")

    # -----------------------------
    # 5. LOAD MODEL
    # -----------------------------
    print("Loading full bf16 Qwen model (LoRA training)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # -----------------------------
    # 6. APPLY LoRA
    # -----------------------------
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        use_dora=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -----------------------------
    # 7. TRAINING ARGUMENTS
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=False,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,

        num_train_epochs=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,

        bf16=True,
        tf32=True,

        logging_steps=LOG_STEPS,

        eval_strategy="steps",
        eval_steps=EVAL_STEPS,

        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,

        report_to="none",
        optim="adamw_torch_fused",

        gradient_checkpointing=True,
        remove_unused_columns=True,

        dataloader_num_workers=4,
        seed=SEED,
    )

    # -----------------------------
    # 8. TRAINER
    # -----------------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        callbacks=[TqdmCallback(), MetricsLogger(OUTPUT_DIR)],
    )

    # -----------------------------
    # 9. TRAIN (WITH RESUME SUPPORT)
    # -----------------------------
    last_ckpt = find_last_checkpoint(OUTPUT_DIR)
    if last_ckpt:
        print(f"Resuming from checkpoint: {last_ckpt}")

    print("\nStarting Training...")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=last_ckpt)
    print(f"\nTraining finished in "
          f"{(time.time() - t0) / 3600:.2f} hours")

    # -----------------------------
    # 10. SAVE
    # -----------------------------
    print("Saving LoRA adapter and tokenizer...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done. Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
