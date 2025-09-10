# evaluator.py
import os
import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


# from vllm_models import vllmModel


class MedMCQAEvaluator:
    """
    Evaluator for MedMCQA-style multiple-choice QA using a provided vllmModel.

    - __init__(model, batch_size): store model and batch size
    - evaluate(test_path, out_pred_path, out_metrics_path, max_examples=None):
        * loads records (JSON array or JSONL)
        * builds prompts
        * calls model.generate in batches
        * robustly parses an answer from model text
        * computes accuracy and writes outputs
    """

    def __init__(self, model, batch_size: int = 8):
        self.model = model
        self.batch_size = max(1, int(batch_size))

    # ------------------------ Public API ------------------------ #
    def evaluate(
        self,
        test_path: str,
        out_pred_path: str,
        out_metrics_path: str,
        max_examples: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Run evaluation over the test set and save predictions + metrics."""
        records = self._load_records(test_path)
        if max_examples is not None:
            records = records[: int(max_examples)]

        # Build (system, user) prompts
        prompts: List[Tuple[str, str]] = []
        ids: List[str] = []
        gold_indices: List[int] = []
        meta: List[Tuple[Optional[str], Optional[str]]] = []  # (subject, topic)

        for r in records:
            qid = str(r.get("id", ""))
            question = r["question"]
            options = r["options"]
            # Gold can be 0-3 (preferred) or letter; normalize to index
            gold = self._normalize_gold(r["answer"], len(options))
            system_msg, user_msg = self._build_prompt(question, options)
            prompts.append((system_msg, user_msg))
            ids.append(qid)
            gold_indices.append(gold)
            meta.append((r.get("subject"), r.get("topic")))

        # Batched generation
        outputs: List[str] = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            gen = self.model.generate(
                batch,
                batch_size=self.batch_size,
                show_progress=show_progress,
                progress_desc="MedMCQA Eval",
            )
            # vllmModel.generate returns List[ (text, in_tokens, out_tokens) ]
            outputs.extend([t[0] for t in gen])

        # Parse answers and compute metrics
        pred_indices: List[Optional[int]] = []
        parse_fail = 0
        label_hist = Counter()

        for text, opts in zip(outputs, (r["options"] for r in records)):
            idx = self._parse_answer(text, num_options=len(opts))
            pred_indices.append(idx)
            if idx is None:
                parse_fail += 1
            else:
                label_hist[idx] += 1

        correct_flags = [
            (p is not None) and (p == g) for p, g in zip(pred_indices, gold_indices)
        ]
        total = len(records)
        correct = sum(correct_flags)
        accuracy = correct / total if total > 0 else 0.0

        # Per-subject accuracy
        by_subject_total = Counter()
        by_subject_correct = Counter()
        for (subj, _topic), ok in zip(meta, correct_flags):
            key = subj or "<UNK>"
            by_subject_total[key] += 1
            if ok:
                by_subject_correct[key] += 1
        by_subject_acc = {
            k: (by_subject_correct[k] / v if v > 0 else 0.0)
            for k, v in by_subject_total.items()
        }

        # Save predictions JSONL
        self._write_jsonl(
            out_pred_path,
            (
                {
                    "id": i,
                    "question": r["question"],
                    "options": r["options"],
                    "gold_index": g,
                    "gold_letter": self._index_to_letter(g),
                    "pred_text": out,
                    "pred_index": p,
                    "pred_letter": (self._index_to_letter(p) if p is not None else None),
                    "correct": (p == g),
                    "subject": r.get("subject"),
                    "topic": r.get("topic"),
                }
                for i, r, g, p, out in zip(ids, records, gold_indices, pred_indices, outputs)
            ),
        )

        # Save metrics JSON
        metrics = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "parse_failures": parse_fail,
            "choice_histogram_index": dict(label_hist),
            "choice_histogram_letter": {
                self._index_to_letter(k): v for k, v in label_hist.items()
            },
            "per_subject_accuracy": by_subject_acc,
        }
        self._write_json(out_metrics_path, metrics)
        return metrics

    # ------------------------ Prompting ------------------------ #
    def _build_prompt(self, question: str, options: List[str]) -> Tuple[str, str]:
        """
        Build a simple, strong instruction to force single-letter output.
        We label options as A-D (or more if needed).
        """
        letters = [chr(ord("A") + i) for i in range(len(options))]
        options_block = "\n".join([f"{letters[i]}. {opt}" for i, opt in enumerate(options)])

        system_msg = (

        )

        user_msg = (
            "You are a careful medical exam assistant. "
            "For each multiple-choice question, select the single best option. "
            "STRICT OUTPUT FORMAT: reply with ONLY one capital letter from the given choices "
            "(e.g., A, B, C, or D). Do not add any words, punctuation, or explanation."
            
            f"Question:\n{question}\n\nOptions:\n{options_block}\n\n"
            f"Reply with ONLY one capital letter from {', '.join(letters)}."
        )
        return system_msg, user_msg

    # ------------------------ Parsing ------------------------ #
    _RE_JSON_ANSWER = re.compile(
        r'(?i)"answer"\s*:\s*"?\s*([A-Z]|[0-9])\s*"?'
    )
    _RE_KEYED_LETTER = re.compile(
        r"(?i)(final\s*answer|answer|choice|option|predic(?:tion)?|label)\s*[:\-]?\s*([A-D])"
    )
    _RE_KEYED_NUMBER = re.compile(
        r"(?i)(final\s*answer|answer|choice|option|predic(?:tion)?|label)\s*[:\-]?\s*([1-9]|0)"
    )
    _RE_TAIL_LETTER = re.compile(r"([A-D])")
    _RE_SINGLE_LETTER_LINE = re.compile(r"^\s*([A-D])\s*$")

    def _parse_answer(self, text: str, num_options: int = 4) -> Optional[int]:
        """
        Robustly extract a choice index from model text.
        Priority:
          1) Single-letter last line exactly A-D
          2) JSON-like or keyed 'Answer: C'
          3) Tail fallback: last ~40 chars containing a letter A-D
          4) Numeric forms (1..N), but only when clearly keyed or isolated
        Returns 0-based index or None.
        """
        text = (text or "").strip()
        if not text:
            return None

        letters = [chr(ord("A") + i) for i in range(num_options)]

        # 1) If the last non-empty line is exactly a letter (strongest signal)
        for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
            m = self._RE_SINGLE_LETTER_LINE.match(line.upper())
            if m and m.group(1) in letters:
                return self._letter_to_index(m.group(1))

            # Only check the last non-empty line for keyed numbers like "Answer: 2"
            mk = self._RE_KEYED_NUMBER.search(line)
            if mk:
                num = mk.group(2)
                idx = self._number_to_index(num, num_options)
                if idx is not None:
                    return idx
            break  # only the last non-empty line

        upper = text.upper()

        # 2a) JSON-ish: "answer": "C" or "answer": 2
        mj = self._RE_JSON_ANSWER.search(upper)
        if mj:
            tok = mj.group(1)
            if tok in letters:
                return self._letter_to_index(tok)
            idx = self._number_to_index(tok, num_options)
            if idx is not None:
                return idx

        # 2b) Keyed letter anywhere; take the last occurrence (often "Final answer: C")
        keyed_letters = list(self._RE_KEYED_LETTER.finditer(upper))
        if keyed_letters:
            tok = keyed_letters[-1].group(2)
            if tok in letters:
                return self._letter_to_index(tok)

        # 3) Tail fallback: look in the last 40 characters for a clean letter
        tail = upper[-40:]
        mt = self._RE_TAIL_LETTER.findall(tail)
        if mt:
            # choose the last letter that is within available choices
            for tok in reversed(mt):
                if tok in letters:
                    return self._letter_to_index(tok)

        # 4) As a softer fallback, search a keyed number anywhere
        keyed_numbers = list(self._RE_KEYED_NUMBER.finditer(upper))
        if keyed_numbers:
            num = keyed_numbers[-1].group(2)
            idx = self._number_to_index(num, num_options)
            if idx is not None:
                return idx

        # Give up
        return None

    @staticmethod
    def _letter_to_index(letter: str) -> int:
        return ord(letter.upper()) - ord("A")

    @staticmethod
    def _index_to_letter(idx: Optional[int]) -> Optional[str]:
        if idx is None or idx < 0:
            return None
        return chr(ord("A") + idx)

    @staticmethod
    def _number_to_index(num_str: str, num_options: int) -> Optional[int]:
        """Map '1'..'N' -> 0..N-1; also accept '0'..'N-1' if appears cleanly."""
        if not num_str or not re.fullmatch(r"[0-9]", num_str.strip()):
            return None
        val = int(num_str)
        if 1 <= val <= num_options:
            return val - 1
        if 0 <= val < num_options:
            return val
        return None

    # ------------------------ IO helpers ------------------------ #
    @staticmethod
    def _load_records(path: str) -> List[Dict[str, Any]]:
        """Load JSON array or JSONL into a list of dicts."""
        if path.endswith(".jsonl"):
            out = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
            return out
        else:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text.startswith("["):
                    return json.loads(text)
                # If a .json file is actually JSONL, handle gracefully
                return [json.loads(ln) for ln in text.splitlines() if ln.strip()]

    @staticmethod
    def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_json(path: str, obj: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _normalize_gold(ans_field: Any, num_options: int) -> int:
        """
        Normalize gold answer to 0-based index.
        Accepts:
          - integer index (0..num_options-1)
          - letter 'a'.. ('A'..)
          - 1-based number (1..num_options)
        """
        if isinstance(ans_field, int):
            if 0 <= ans_field < num_options:
                return ans_field
            if 1 <= ans_field <= num_options:
                return ans_field - 1
        if isinstance(ans_field, str):
            s = ans_field.strip()
            # letter?
            if re.fullmatch(r"[A-Za-z]", s):
                return ord(s.upper()) - ord("A")
            # digit?
            if re.fullmatch(r"[0-9]", s):
                val = int(s)
                if 1 <= val <= num_options:
                    return val - 1
                if 0 <= val < num_options:
                    return val
        raise ValueError(f"Unrecognized gold answer format: {ans_field!r}")


# ------------------------ Example usage ------------------------ #

