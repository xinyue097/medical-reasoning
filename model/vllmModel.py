# vllm_models.py
import os
import time
import random
import numpy as np
import torch
import logging
from typing import List, Tuple, Optional, Union, Dict, Any

from transformers import AutoTokenizer
from model.model import Model

logger = logging.getLogger(__name__)

# Optional progress bar: prefer tqdm if available
try:
    from tqdm.auto import tqdm as _tqdm  # works well in terminals and notebooks
except Exception:
    _tqdm = None


class vllmModel(Model):
    """
    vLLM inference interface with GPT-OSS (Transformers pipeline) fallback.

    Supports:
        • Single GPU: gpu_ids="3" or gpu_ids=[3]
        • Multi GPU:  gpu_ids="0,1,2,3" or gpu_ids=[0,1,2,3]
        • Use all visible GPUs by default when no gpu_ids is provided

    New:
        • If the model name contains 'gpt-oss' (case-insensitive), use a
          Hugging Face Transformers text-generation pipeline for inference.
        • Unified inference via the 'generate' method.
        • Token counting currently returns 0 (placeholder for later).
        • Progress bar for both GPT-OSS (pipeline) and vLLM branches.
    """

    def __init__(
        self,
        model_name: str,
        gpu_ids: Union[str, List[int], None] = None,
        max_tokens: int = 33000,
        temperature: float = 0.1,
        seed: Optional[int] = None,
        trust_remote_code: bool = False,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        tensor_parallel_size: Optional[int] = None,
    ):
        super().__init__(model_name, max_tokens, temperature, top_p, repetition_penalty)

        # Routing switch: use HF pipeline when model includes "gpt-oss"
        self._use_gptoss: bool = "gpt-oss" in (getattr(self, "model_name_full", model_name) or "").lower()

        # ---------- 1) GPU visibility ----------
        if gpu_ids is None:
            visible_env = os.getenv("CUDA_VISIBLE_DEVICES")
            self.gpu_ids = visible_env.split(",") if visible_env else []
        else:
            if isinstance(gpu_ids, str):
                self.gpu_ids = [gid.strip() for gid in gpu_ids.split(",") if gid.strip() != ""]
            else:
                self.gpu_ids = [str(g) for g in gpu_ids]
            # Intentionally do not override CUDA_VISIBLE_DEVICES here.

        if tensor_parallel_size is None:
            # Default tensor parallel = number of provided GPUs (>=1)
            self.tensor_parallel_size = max(1, len(self.gpu_ids)) if self.gpu_ids else 1
        else:
            self.tensor_parallel_size = tensor_parallel_size

        # ---------- 2) Seeding ----------
        self.seed = seed or int(time.time() * 1000) % (2**32 - 1)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # ---------- 3) Tokenizer ----------
        self.trust_remote_code = trust_remote_code
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_full, trust_remote_code=self.trust_remote_code
        )

        # ---------- 4) Backends ----------
        self.llm = None
        self.pipe = None

        if self._use_gptoss:
            # ===== GPT-OSS via HF pipeline =====
            # Left padding is generally preferred for decoder-only models when batching.
            self.tokenizer.padding_side = "left"
            if getattr(self.tokenizer, "pad_token_id", None) is None:
                # Fallback to EOS if the tokenizer has no pad token.
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            try:
                from transformers import pipeline as hf_pipeline

                # Build the pipeline. Heavy arguments that affect device placement belong here.
                self.pipe = hf_pipeline(
                    "text-generation",
                    model=self.model_name_full,
                    tokenizer=self.tokenizer,
                    torch_dtype="auto",
                    device_map="auto",
                )
                logger.info(f"Initialized GPT-OSS pipeline for '{self.model_name_full}'.")
            except Exception as e:
                logger.error(f"Failed to initialize HF pipeline for '{self.model_name_full}': {e}")
                raise
        else:
            # ===== vLLM backend (original behavior) =====
            try:
                from vllm import LLM as vLLM
                self.llm = vLLM(
                    model=self.model_name_full,
                    trust_remote_code=True,
                    dtype="bfloat16",
                    max_model_len=12000,
                    tensor_parallel_size=self.tensor_parallel_size,
                    gpu_memory_utilization=0.90,
                    kv_cache_dtype="fp8",
                    enable_chunked_prefill=True,
                    max_num_batched_tokens=80_000,
                    max_num_seqs=32,
                )
                logger.info(
                    f"Loaded '{self.model_name_full}' with tensor_parallel_size={self.tensor_parallel_size} "
                    f"on GPUs: {os.getenv('CUDA_VISIBLE_DEVICES') or 'ALL'}"
                )
            except ImportError:
                logger.error("vLLM not found. Please install it with: pip install vllm")
                raise
            except Exception as e:
                logger.error(f"Error initializing vLLM for '{self.model_name_full}': {e}")
                raise

    # ---------- Progress helpers ----------
    def _make_pbar(self, total: int, desc: str, show: bool):
        """
        Create a progress bar. Use tqdm if available; otherwise a minimal fallback.
        """
        if not show or total <= 0:
            return None

        if _tqdm is not None:
            try:
                return _tqdm(total=total, desc=desc, dynamic_ncols=True, leave=True)
            except Exception:
                pass  # fall through to simple fallback

        # Minimal fallback progress bar if tqdm is not available
        class _SimplePBar:
            def __init__(self, total_, desc_):
                self.total = total_
                self.n = 0
                self.desc = desc_
                print(f"[{self.desc}] 0/{self.total}")

            def update(self, n=1):
                self.n += n
                # Print only every 5% or last batch to avoid flooding the terminal
                pct = int(self.n * 100 / self.total)
                if self.n == self.total or pct % 5 == 0:
                    print(f"[{self.desc}] {self.n}/{self.total} ({pct}%)")

            def close(self):
                if self.n < self.total:
                    print(f"[{self.desc}] {self.n}/{self.total} (closed)")

        return _SimplePBar(total, desc)

    def _iter_batches(self, xs: List[Any], batch_size: int):
        """Yield contiguous chunks of size <= batch_size."""
        for i in range(0, len(xs), batch_size):
            yield i, xs[i : i + batch_size]

    # ---------- 5) Unified inference entrypoint ----------
    def generate(
        self,
        prompts: List[Tuple[str, str]],
        batch_size: int = 1,
        show_progress: bool = True,
        progress_desc: Optional[str] = None,
    ) -> List[Tuple[str, int, int]]:
        """
        Unified inference API.

        Args:
            prompts: List of (system_message, user_message) pairs.
            batch_size: Batch size for processing.
            show_progress: If True, render a terminal progress bar.
            progress_desc: Optional custom description shown in the progress bar.

        Returns:
            A list of tuples (text, input_tokens, output_tokens).
            Token counts are currently set to 0 as placeholders.
        """
        total = len(prompts)
        desc = progress_desc or ("GPT-OSS Inference" if self._use_gptoss else "vLLM Inference")
        pbar = self._make_pbar(total=total, desc=desc, show=show_progress)

        if self._use_gptoss:
            # ---- GPT-OSS / HF pipeline branch with progress bar ----
            all_texts: List[str] = []
            zeros: List[int] = []

            # Convert once to chat-message inputs
            batch_inputs_all = [self._to_chat_messages_pair(sys, usr) for (sys, usr) in prompts]
            pipeline_max_new = min(self.max_tokens, 5120)

            for idx0, batch_inputs in self._iter_batches(batch_inputs_all, max(1, int(batch_size))):
                # Logging for visibility per batch
                batch_id = idx0 // max(1, int(batch_size)) + 1
                num_batches = (total - 1) // max(1, int(batch_size)) + 1
                logger.info(f"[GPT-OSS] Batch {batch_id}/{num_batches} | size={len(batch_inputs)}")

                try:
                    outs = self.pipe(
                        batch_inputs,
                        max_new_tokens=pipeline_max_new,
                        do_sample=False,
                        num_return_sequences=1,
                        truncation=True,
                        return_full_text=False,
                        batch_size=len(batch_inputs),  # internal micro-batching
                    )
                except Exception as e:
                    logger.error(f"Pipeline generation error: {e}")
                    # Fill placeholders for this batch
                    errs = [f"Error: {e}"] * len(batch_inputs)
                    all_texts.extend(errs)
                    zeros.extend([0] * len(errs))
                    if pbar:
                        pbar.update(len(batch_inputs))
                    continue

                # Normalize and collect texts
                # HF pipeline may return list-of-lists in batched mode; flatten conservatively.
                if isinstance(outs, list) and outs and isinstance(outs[0], list):
                    flat = []
                    for item in outs:
                        flat.extend(item)
                    outs = flat

                texts = [self._extract_pipeline_text(one) for one in outs]
                all_texts.extend(texts)
                zeros.extend([0] * len(texts))

                if pbar:
                    pbar.update(len(batch_inputs))

            if pbar:
                pbar.close()
            return list(zip(all_texts, zeros, zeros))

        # ---- vLLM branch (original behavior) with progress bar ----
        try:
            from vllm import SamplingParams
        except Exception as e:
            logger.error(f"Failed to import vLLM SamplingParams: {e}")
            if pbar:
                pbar.close()
            return [("Error: vLLM is not available.", 0, 0) for _ in prompts]

        prompt_texts = self.convert_prompts(prompts)
        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
        )

        responses: List[str] = []
        for idx0, batch in self._iter_batches(prompt_texts, max(1, int(batch_size))):
            batch_id = idx0 // max(1, int(batch_size)) + 1
            num_batches = (total - 1) // max(1, int(batch_size)) + 1
            logger.info(f"[vLLM] Batch {batch_id}/{num_batches} | size={len(batch)}")

            try:
                outs = self.llm.generate(batch, sampling_params)
                responses.extend([o.outputs[0].text for o in outs])
            except Exception as e:
                logger.error(f"Generation error: {e}")
                responses.extend([f"Error: {e}"] * len(batch))

            if pbar:
                pbar.update(len(batch))

        if pbar:
            pbar.close()

        zeros = [0] * len(responses)  # Token counting placeholder
        return list(zip(responses, zeros, zeros))

    # ---------- 6) Prompt construction (used by vLLM branch) ----------
    def convert_prompts(self, prompts: List[Tuple[str, str]]) -> List[str]:
        """
        Convert a list of (system, user) pairs into model-ready prompt texts
        using tokenizer.apply_chat_template.
        """
        prompt_texts: List[str] = []
        for system_msg, user_msg in prompts:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            # Some tokenizers support 'enable_thinking'; guard for compatibility.
            try:
                prompt_texts.append(
                    self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                    )
                )
            except TypeError:
                prompt_texts.append(
                    self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )
        return prompt_texts

    # ---------- 7) Helpers ----------
    @staticmethod
    def _to_chat_messages_pair(system_msg: str, user_msg: str) -> List[Dict[str, str]]:
        """Build a pair of chat messages in the format expected by HF pipeline."""
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    @staticmethod
    def _extract_pipeline_text(one_output: Any) -> str:
        """
        Extract generated text from HF pipeline output.

        Handles common structures:
          1) dict: {'generated_text': '...'}
          2) list[dict]: [{'generated_text': '...'}]
          3) dict: {'generated_text': list[message_dicts]}
          4) list[dict(messages)]: as above
        """
        # Unwrap top-level list if present
        if isinstance(one_output, list):
            if not one_output:
                return ""
            one_output = one_output[0]

        if isinstance(one_output, dict):
            gt = one_output.get("generated_text", "")
            # Case: plain string
            if isinstance(gt, str):
                return gt.strip()
            # Case: list of message dicts; take the last assistant message
            if isinstance(gt, list):
                for msg in reversed(gt):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        return (msg.get("content") or "").strip()
                return str(gt).strip()

        # Fallback: stringify anything else
        return str(one_output).strip()
