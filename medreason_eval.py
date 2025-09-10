# evaluate_with_vllm.py
# -*- coding: utf-8 -*-
import os
import re
import json
import math
import time
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from datasets import load_dataset, Dataset
from tqdm import tqdm

from model.vllmModel import vllmModel


# ----------------------------
# Utilities and Parsing
# ----------------------------
LETTER_LIST_4 = ["A", "B", "C", "D"]
LETTER_LIST_5 = ["A", "B", "C", "D", "E"]
LETTER_LIST_10 = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def letters_from_options(options: Dict[str, str]) -> List[str]:
    order = {c: i for i, c in enumerate(LETTER_LIST_10)}
    seen = []
    for k in options.keys():
        kk = str(k).strip().upper()
        if kk in order and kk not in seen:
            seen.append(kk)
    return sorted(seen, key=lambda x: order[x])

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def safe_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", s)

def _compile_dynamic_patterns(letters: List[str]):
    letters_group = "(" + "|".join([re.escape(L) for L in letters]) + ")"
    num_group = "(" + "|".join(str(i) for i in range(1, len(letters) + 1)) + ")"

    patterns_letter = [
        rf"final\s*answer\s*[:：]?\s*\(?\s*{letters_group}\s*\)?\b",
        rf"answer\s*[:：]?\s*\(?\s*{letters_group}\s*\)?\b",
        rf"option\s*{letters_group}\b",
    ]
    patterns_num = [
        rf"final\s*answer\s*[:：]?\s*\(?\s*{num_group}\s*\)?\b",
        rf"answer\s*[:：]?\s*\(?\s*{num_group}\s*\)?\b",
        rf"option\s*{num_group}\b",
    ]
    line_pat_letter = rf"^\s*\(?\s*{letters_group}\s*\)?\s*\.?\s*$"
    line_pat_num = rf"^\s*\(?\s*{num_group}\s*\)?\s*\.?\s*$"

    return patterns_letter, patterns_num, line_pat_letter, line_pat_num


def parse_final_answer(text: str, letters: List[str]) -> Optional[str]:
    if not text:
        return None
    t = text.strip()
    if not letters:
        return None

    patterns_letter, patterns_num, line_pat_letter, line_pat_num = _compile_dynamic_patterns(letters)

    for pat in patterns_letter:
        m_all = list(re.finditer(pat, t, flags=re.IGNORECASE))
        if m_all:
            return m_all[-1].group(1).upper()

    for pat in patterns_num:
        m_all = list(re.finditer(pat, t, flags=re.IGNORECASE))
        if m_all:
            idx = int(m_all[-1].group(1)) - 1
            if 0 <= idx < len(letters):
                return letters[idx]

    lines = [ln.strip() for ln in re.split(r"[\r\n]+", t) if ln.strip()]
    for ln in reversed(lines[-5:]):
        m = re.search(line_pat_letter, ln, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = re.search(line_pat_num, ln, flags=re.IGNORECASE)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(letters):
                return letters[idx]

    return None


# ----------------------------
# Prompt Construction
# ----------------------------
SYSTEM_PROMPT = (
    "You are a careful and factual medical assistant. "
    "Use clinical reasoning step by step. Be concise and avoid hallucinations."
)

def build_user_prompt(question: str, options: Dict[str, str], letters: List[str]) -> str:
    lines = [question.strip(), "", "Options:"]
    for L in letters:
        if L in options and str(options[L]).strip():
            lines.append(f"{L}. {str(options[L]).strip()}")
    lines.append("")
    letters_str = "/".join(letters)
    lines.append(
        "Please first provide your medical reasoning step by step, "
        "then give the final choice on a new line in the exact format:"
    )
    lines.append(f"Final Answer: <{letters_str}>")
    return "\n".join(lines)


# ----------------------------
# Dataset Loading and Normalization
# ----------------------------
@dataclass
class QAItem:
    qid: str
    question: str
    options: Dict[str, str]
    gold: str
    dataset: str


def load_medmcqa(split_preference: List[str] = ("validation", "dev", "test")) -> List[QAItem]:
    ds = None
    last_err = None
    for sp in split_preference:
        try:
            ds = load_dataset("openlifescienceai/medmcqa", split=sp)
            break
        except Exception as e:
            last_err = e
    if ds is None:
        raise RuntimeError(f"MedMCQA split not found. Last error: {last_err}")

    items: List[QAItem] = []
    for ex in ds:
        if str(ex.get("choice_type", "single")).lower() != "single":
            continue

        qid = str(ex.get("id", ""))
        q   = str(ex.get("question", "")).strip()
        opa = str(ex.get("opa", "")).strip()
        opb = str(ex.get("opb", "")).strip()
        opc = str(ex.get("opc", "")).strip()
        opd = str(ex.get("opd", "")).strip()
        cop = ex.get("cop", None)

        if not q or cop is None:
            continue

        options = {"A": opa, "B": opb, "C": opc, "D": opd}
        if not all(options.values()):
            continue

        try:
            gold = LETTER_LIST_4[int(cop) - 1]
        except Exception:
            continue

        items.append(QAItem(qid=qid, question=q, options=options, gold=gold, dataset="MedMCQA"))
    return items


def _first_nonempty(d: dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d and str(d[k]).strip():
            return str(d[k]).strip()
    return None


def load_medbullets_op5() -> List[QAItem]:
    ds = None
    for sp in ["test", "validation", "dev", "train", None]:
        try:
            ds = load_dataset("LangAGI-Lab/medbullets_op5", split=sp) if sp else load_dataset("LangAGI-Lab/medbullets_op5")["train"]
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError("Cannot load LangAGI-Lab/medbullets_op5.")

    items: List[QAItem] = []
    for ex in ds:
        ex = dict(ex)

        q = _first_nonempty(ex, ["question", "prompt", "stem", "input", "context"])
        if not q:
            context = _first_nonempty(ex, ["context", "case", "passage", "history"])
            query   = _first_nonempty(ex, ["query", "prompt"])
            q = " ".join([x for x in [context, query] if x]).strip()
        if not q:
            continue

        options: Dict[str, str] = {}
        if "options" in ex and isinstance(ex["options"], (list, tuple)):
            vals = [str(v).strip() for v in ex["options"] if str(v).strip()]
            if len(vals) >= 5:
                for L, v in zip(LETTER_LIST_5, vals[:5]):
                    options[L] = v
        if not options:
            letter_keys = [k for k in ex.keys() if k.upper() in LETTER_LIST_5]
            if letter_keys:
                for L in LETTER_LIST_5:
                    if L in ex and str(ex[L]).strip():
                        options[L] = str(ex[L]).strip()
        if not options:
            opa = _first_nonempty(ex, ["opa", "A"])
            opb = _first_nonempty(ex, ["opb", "B"])
            opc = _first_nonempty(ex, ["opc", "C"])
            opd = _first_nonempty(ex, ["opd", "D"])
            ope = _first_nonempty(ex, ["ope", "E"])
            if all([opa, opb, opc, opd, ope]):
                options = {"A": opa, "B": opb, "C": opc, "D": opd, "E": ope}

        if len(options) < 5:
            continue

        gold: Optional[str] = None
        for k in ["answer", "gold", "label", "correct", "target"]:
            if k in ex:
                val = ex[k]
                if isinstance(val, str) and val.strip().upper() in LETTER_LIST_5:
                    gold = val.strip().upper()
                    break
                if isinstance(val, (int, float)):
                    idx = int(val)
                    if idx in [0,1,2,3,4]:
                        gold = LETTER_LIST_5[idx]
                        break
                    if idx in [1,2,3,4,5]:
                        gold = LETTER_LIST_5[idx-1]
                        break
                if isinstance(val, str):
                    txt = val.strip().lower()
                    for L, opt in options.items():
                        if txt == str(opt).strip().lower():
                            gold = L
                            break
                if gold:
                    break

        if gold is None and "cop" in ex:
            try:
                gold = LETTER_LIST_5[int(ex["cop"]) - 1]
            except Exception:
                pass

        if gold is None or gold not in LETTER_LIST_5:
            continue

        qid = str(ex.get("id", "")) or str(ex.get("qid", "")) or str(hash(q))[:12]
        items.append(QAItem(qid=qid, question=q, options=options, gold=gold, dataset="MedBullets-op5"))

    return items


def load_medqa_usmle(split_preference: List[str] = ("validation", "dev", "test")) -> List[QAItem]:
    ds = None
    src = None
    last_err = None
    candidates = [
        ("openlifescienceai/MedQA-USMLE-4-options-hf", None),
        ("GBaker/MedQA-USMLE-4-options", None),
    ]
    for name, subset in candidates:
        for sp in split_preference:
            try:
                ds = load_dataset(name, split=sp) if subset is None else load_dataset(name, subset, split=sp)
                src = name
                break
            except Exception as e:
                last_err = e
        if ds is not None:
            break
    if ds is None:
        raise RuntimeError(f"MedQA-USMLE split not found. Last error: {last_err}")

    items: List[QAItem] = []
    for ex in ds:
        ex = dict(ex)
        if "sent1" in ex and "label" in ex and any((f"ending{i}" in ex) for i in range(4)):
            q1 = str(ex.get("sent1", "")).strip()
            q2 = str(ex.get("sent2", "")).strip()
            q = (q1 + (" " + q2 if q2 else "")).strip()
            options = {
                "A": str(ex.get("ending0", "")).strip(),
                "B": str(ex.get("ending1", "")).strip(),
                "C": str(ex.get("ending2", "")).strip(),
                "D": str(ex.get("ending3", "")).strip(),
            }
            if not q or not all(options.values()):
                continue
            try:
                gold = LETTER_LIST_4[int(ex["label"])]
            except Exception:
                continue
        else:
            q = _first_nonempty(ex, ["question", "prompt", "stem"])
            od = ex.get("options", None)
            lab = ex.get("label", None)
            if not q or not isinstance(od, dict) or lab is None:
                continue
            options = {k.upper(): str(v).strip() for k, v in od.items() if k and str(v).strip()}
            letters = letters_from_options(options)
            if letters != LETTER_LIST_4:
                continue
            if isinstance(lab, int) and 0 <= lab < 4:
                gold = LETTER_LIST_4[int(lab)]
            elif isinstance(lab, str) and lab.strip().upper() in LETTER_LIST_4:
                gold = lab.strip().upper()
            else:
                continue

        qid = str(ex.get("id", "")) or str(hash(q))[:12]
        items.append(QAItem(qid=qid, question=q, options=options, gold=gold, dataset="MedQA-USMLE"))

    return items


def load_medxpert(subset: str = "Text", split_preference: List[str] = ("test", "dev")) -> List[QAItem]:
    ds = None
    last_err = None
    for sp in split_preference:
        try:
            ds = load_dataset("TsinghuaC3I/MedXpertQA", subset, split=sp)
            break
        except Exception as e:
            last_err = e
    if ds is None:
        raise RuntimeError(f"MedXpertQA subset '{subset}' split not found. Last error: {last_err}")

    dataset_name = f"MedXpert-{subset}"
    items: List[QAItem] = []
    for ex in ds:
        ex = dict(ex)
        q = str(ex.get("question", "")).strip()
        od = ex.get("options", {})
        lab = ex.get("label", None)

        if not q or not isinstance(od, dict) or lab is None:
            continue

        options = {str(k).upper(): str(v).strip() for k, v in od.items() if str(v).strip()}
        letters = letters_from_options(options)
        options = {L: options[L] for L in letters if L in options}

        gold = None
        if isinstance(lab, str) and lab.strip().upper() in letters:
            gold = lab.strip().upper()
        if not gold:
            continue

        qid = str(ex.get("id", "")) or str(hash(q))[:12]
        items.append(QAItem(qid=qid, question=q, options=options, gold=gold, dataset=dataset_name))

    return items


# ----------------------------
# Main Evaluation Flow
# ----------------------------
def build_prompts(items: List[QAItem]) -> List[Tuple[str, str]]:
    prompts = []
    for it in items:
        letters = letters_from_options(it.options)
        user = build_user_prompt(it.question, it.options, letters)
        prompts.append((SYSTEM_PROMPT, user))
    return prompts


def evaluate(
    model_name: str,
    dataset_names: List[str],
    batch_size: int = 16,
    max_new_tokens: int = 1536,
    temperature: float = 0.0,
    top_p: float = 0.0001,
    repetition_penalty: float = 1.0,
    seed: int = 42,
    limit: Optional[int] = None,
    out_dir: str = "./eval_runs",
):
    seed_everything(seed)
    os.makedirs(out_dir, exist_ok=True)

    all_items: List[QAItem] = []
    ds_names_lower = [d.lower() for d in dataset_names]

    if "medmcqa" in ds_names_lower:
        items = load_medmcqa()
        all_items.extend(items)
    if "medbullets_op5" in ds_names_lower:
        items = load_medbullets_op5()
        all_items.extend(items)
    if "medqa" in ds_names_lower or "medqa_usmle" in ds_names_lower:
        items = load_medqa_usmle()
        all_items.extend(items)
    if "medxpert" in ds_names_lower or "medxpert_text" in ds_names_lower:
        items = load_medxpert(subset="Text")
        all_items.extend(items)
    if "medxpert_mm" in ds_names_lower:
        items = load_medxpert(subset="MM")
        all_items.extend(items)

    if not all_items:
        raise RuntimeError(f"No valid datasets loaded from: {dataset_names}")

    if limit:
        all_items = all_items[:limit]

    prompts = build_prompts(all_items)

    vllm = vllmModel(
        model_name=model_name,
        max_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        trust_remote_code=True,
        seed=seed
    )

    gens = vllm.generate(
        prompts,
        batch_size=batch_size,
        show_progress=True,
        progress_desc="Evaluating"
    )
    texts = [g[0] for g in gens]

    preds: List[str] = []
    golds:  List[str] = []
    rows:   List[dict] = []
    correct = 0
    total   = 0

    for it, out_text in zip(all_items, texts):
        letters = letters_from_options(it.options)
        pred = parse_final_answer(out_text, letters)
        preds.append(pred or "")
        golds.append(it.gold)
        ok = (pred == it.gold)
        correct += int(ok)
        total   += 1

        rows.append({
            "id": it.qid,
            "dataset": it.dataset,
            "question": it.question,
            "options": it.options,
            "gold": it.gold,
            "pred": pred,
            "correct": ok,
            "raw_output": out_text,
        })

    by_ds: Dict[str, Dict[str, int]] = {}
    for it, pred in zip(all_items, preds):
        d = it.dataset
        if d not in by_ds:
            by_ds[d] = {"correct": 0, "total": 0}
        by_ds[d]["total"] += 1
        if pred == it.gold:
            by_ds[d]["correct"] += 1

    ts = time.strftime("%Y%m%d_%H%M%S")
    exp_name = safe_filename(f"{model_name}_eval_{ts}")
    out_prefix = os.path.join(out_dir, exp_name)

    with open(out_prefix + ".preds.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "model": model_name,
        "total": total,
        "overall_acc": round(100.0 * correct / max(1, total), 2),
        "per_dataset": {
            d: round(100.0 * v["correct"] / max(1, v["total"]), 2)
            for d, v in by_ds.items()
        }
    }
    with open(out_prefix + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Evaluation Summary ===")
    print(f"Model: {model_name}")
    print(f"Overall: {summary['overall_acc']}%  ({correct}/{total})")
    for d, acc in summary["per_dataset"].items():
        c = by_ds[d]["correct"]
        t = by_ds[d]["total"]
        print(f"- {d}: {acc}%  ({c}/{t})")
    print(f"\nSaved predictions: {out_prefix + '.preds.jsonl'}")
    print(f"Saved summary    : {out_prefix + '.summary.json'}")


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        help="HF path or local directory, e.g. meta-llama/Meta-Llama-3.1-8B-Instruct or ./your_sft_dir")
    parser.add_argument("--datasets", type=str,
                        default="medmcqa,medbullets_op5,medqa,medxpert",
                        help="Comma separated: medmcqa,medbullets_op5,medqa,medxpert(=Text),medxpert_mm")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.0001)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate first N samples for debugging")
    parser.add_argument("--output_dir", type=str, default="./eval_runs")
    args = parser.parse_args()

    ds_list = [x.strip() for x in args.datasets.split(",") if x.strip()]
    evaluate(
        model_name=args.model_name,
        dataset_names=ds_list,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        limit=args.limit,
        out_dir=args.output_dir,
    )

if __name__ == "__main__":
    main()
