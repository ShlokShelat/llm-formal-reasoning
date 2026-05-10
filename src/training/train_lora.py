"""
train_lora.py — LoRA SFT Training for Qwen2.5 Regex→DFA
=======================
Supports: Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct
Hardware: Optimised for H100 80GB (AU HPC, cn5/cn6)
"""

import os
import json
import math
import logging
import argparse
import random
from typing import List, Dict, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
    EarlyStoppingCallback,
)
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


# ══════════════════════
#  1. ARGUMENT PARSING
# ══════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="LoRA SFT — Qwen2.5 Regex→DFA")

    p.add_argument("--model_path",  required=True)
    p.add_argument("--train_file",  required=True)
    p.add_argument("--val_file",    required=True)
    p.add_argument("--output_dir",  default="./lora_output")
    p.add_argument("--replay_file", default=None)

    p.add_argument("--lora_r",           type=int,   default=32)
    p.add_argument("--lora_alpha",       type=int,   default=64)
    p.add_argument("--lora_dropout",     type=float, default=0.05)
    p.add_argument("--lora_target_modules", nargs="+",
                   default=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"])

    p.add_argument("--num_train_epochs",            type=int,   default=3)
    p.add_argument("--per_device_train_batch_size", type=int,   default=2)
    p.add_argument("--per_device_eval_batch_size",  type=int,   default=2)
    p.add_argument("--gradient_accumulation_steps", type=int,   default=8)
    p.add_argument("--learning_rate",               type=float, default=2e-4)
    p.add_argument("--weight_decay",                type=float, default=0.01)
    p.add_argument("--warmup_ratio",                type=float, default=0.05)
    p.add_argument("--lr_scheduler_type",           type=str,   default="cosine")
    p.add_argument("--max_seq_length",              type=int,   default=4096)
    p.add_argument("--seed",                        type=int,   default=42)
    p.add_argument("--replay_ratio",                type=float, default=0.10)

    p.add_argument("--bf16",    action="store_true", default=True)
    p.add_argument("--no_bf16", dest="bf16", action="store_false")
    p.add_argument("--fp16",    action="store_true", default=False)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing",
                   dest="gradient_checkpointing", action="store_false")

    p.add_argument("--save_steps",               type=int, default=200)
    p.add_argument("--eval_steps",               type=int, default=200)
    p.add_argument("--logging_steps",            type=int, default=10)
    p.add_argument("--save_total_limit",         type=int, default=3)
    p.add_argument("--load_best_model_at_end",   action="store_true", default=True)
    p.add_argument("--early_stopping_patience",  type=int, default=5)

    p.add_argument("--deepspeed",  type=str, default=None)
    p.add_argument("--report_to",  type=str, default="tensorboard",
                   choices=["tensorboard","wandb","none"])
    p.add_argument("--run_name",   type=str, default=None)

    # CHANGE 1 OF 2: accept checkpoint path from submit script
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = p.parse_args()

    if args.fp16:
        args.bf16 = False

    if args.run_name is None:
        model_shortname = os.path.basename(args.model_path.rstrip("/"))
        args.run_name = f"{model_shortname}_lora_r{args.lora_r}"

    return args


# ══════════════════════
#  2. DATA LOADING AND TOKENISATION
# ══════════════════════

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def tokenise_example(example: Dict, tokenizer, max_seq_length: int) -> Optional[Dict]:
    messages = example["messages"]

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=False)["input_ids"]
    labels = [-100] * len(full_ids)

    prefix_messages = []
    for msg in messages:
        if msg["role"] == "assistant":
            prefix_text = tokenizer.apply_chat_template(
                prefix_messages, tokenize=False, add_generation_prompt=True
            )
            prefix_ids = tokenizer(
                prefix_text, add_special_tokens=False, truncation=False
            )["input_ids"]

            end_text = tokenizer.apply_chat_template(
                prefix_messages + [msg], tokenize=False, add_generation_prompt=False
            )
            end_ids = tokenizer(
                end_text, add_special_tokens=False, truncation=False
            )["input_ids"]

            for i in range(len(prefix_ids), min(len(end_ids), len(labels))):
                labels[i] = full_ids[i]

        prefix_messages.append(msg)

    full_ids = full_ids[:max_seq_length]
    labels   = labels[:max_seq_length]

    if all(l == -100 for l in labels):
        return None

    return {
        "input_ids":      full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels":         labels,
    }


def build_dataset(data: List[Dict], tokenizer, max_seq_length: int, desc: str) -> Dataset:
    tokenised, skipped = [], 0
    for ex in data:
        try:
            tok = tokenise_example(ex, tokenizer, max_seq_length)
            if tok is None:
                skipped += 1
            else:
                tokenised.append(tok)
        except Exception as e:
            logger.warning(f"Skipping example: {e}")
            skipped += 1

    if skipped:
        logger.warning(f"{desc}: skipped {skipped} examples")
    logger.info(f"{desc}: {len(tokenised)} examples ready")
    return Dataset.from_list(tokenised)


# ══════════════════════
#  3. MODEL SETUP
# ══════════════════════

def load_model_and_tokenizer(args):
    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if args.bf16 else (
        torch.float16 if args.fp16 else torch.float32
    )
    logger.info(f"Using dtype: {torch_dtype}")

    attn_impl = "eager"
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        logger.info("Flash Attention 2 enabled")
    except ImportError:
        logger.info("Flash Attention 2 not found — using eager attention")

    logger.info(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
        attn_implementation=attn_impl,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("Gradient checkpointing enabled")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ══════════════════════
#  4. CUSTOM TRAINER
# ══════════════════════

class RegexDFATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss

        if self.state.global_step % 50 == 0 and self.state.global_step > 0:
            total_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5
            self.log({"grad_norm": total_norm})

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, *args, **kwargs):
        output = super().evaluate(*args, **kwargs)
        if "eval_loss" in output:
            try:
                output["eval_perplexity"] = math.exp(output["eval_loss"])
            except OverflowError:
                output["eval_perplexity"] = float("inf")
        return output


# ══════════════════════
#  5. MAIN
# ══════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("  Qwen2.5 LoRA SFT — Regex→DFA")
    logger.info(f"  Model:     {args.model_path}")
    logger.info(f"  Output:    {args.output_dir}")
    logger.info(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    logger.info(f"  LR={args.learning_rate}, epochs={args.num_train_epochs}")
    if args.resume_from_checkpoint:
        logger.info(f"  Resuming:  {args.resume_from_checkpoint}")
    logger.info("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args)

    train_raw = load_jsonl(args.train_file)
    val_raw   = load_jsonl(args.val_file)
    logger.info(f"Raw data — train: {len(train_raw)}, val: {len(val_raw)}")

    if args.replay_file and os.path.exists(args.replay_file):
        replay_raw = load_jsonl(args.replay_file)
        n_replay   = int(len(train_raw) * args.replay_ratio)
        rng = random.Random(args.seed)
        train_raw = train_raw + rng.choices(replay_raw, k=n_replay)
        rng.shuffle(train_raw)
        logger.info(f"Replay: added {n_replay} examples ({args.replay_ratio*100:.0f}%)")

    train_dataset = build_dataset(train_raw, tokenizer, args.max_seq_length, "Train")
    val_dataset   = build_dataset(val_raw,   tokenizer, args.max_seq_length, "Val")

    lengths = [len(ex["input_ids"]) for ex in train_dataset]
    logger.info(
        f"Seq lengths — min:{min(lengths)}, max:{max(lengths)}, "
        f"mean:{sum(lengths)/len(lengths):.0f}, "
        f"p95:{sorted(lengths)[int(0.95*len(lengths))]}"
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,

        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim="adamw_torch_fused",

        bf16=args.bf16,
        fp16=args.fp16,
        tf32=True,

        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        report_to=args.report_to if args.report_to != "none" else [],

        deepspeed=args.deepspeed,
        seed=args.seed,
        remove_unused_columns=False,
        prediction_loss_only=True,
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        ))

    trainer = RegexDFATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    n_gpus   = max(torch.cuda.device_count(), 1)
    eff_bs   = args.per_device_train_batch_size * args.gradient_accumulation_steps * n_gpus
    n_steps  = math.ceil(len(train_dataset) / eff_bs * args.num_train_epochs)
    logger.info(f"Effective batch size: {eff_bs} | Total steps: {n_steps}")

    logger.info("Starting training...")
    # CHANGE 2 OF 2: pass resume path to trainer.train()
    # When None (fresh run) this is silently ignored by HF Trainer
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logger.info("Saving adapter...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(val_dataset)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"  eval_loss:       {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    logger.info(f"  eval_perplexity: {eval_metrics.get('eval_perplexity', 'N/A'):.2f}")
    logger.info(f"  adapter saved:   {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
