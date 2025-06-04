from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import trl

## TODO: Modify this file


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'ioi_code', 'code_format', 'soft_overlong_punishment'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
        max_completion_len (`int`):
            Maximum number of tokens in completion.
        soft_punish_cache (`int`):
            Minimum number of tokens in completion.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        # '(?:python|cpp)'
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash", "cpp"],
        },
    )
    code_eval_test_batch_size: int = field(
        default=1,
        metadata={
            "help": "for each generation, evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases. Useful to avoid overloading the eval server + save time on wrong solutions"
        },
    )
    code_eval_scoring_mode: Literal["pass_fail", "partial", "weighted_sum"] = field(
        default="weighted_sum",
        metadata={"help": "use fraction of passed test cases as reward. If false, use 0/1 scoring."},
    )
    parallel_code_exec_per_proc: int = field(
        default=2,
        metadata={
            "help": "Number of parallel E2B code executions per process. Default of 2 is suitable for the Free Hobby tier of E2B with 8 GPUs used for training."
        },
    )

    dataset_prompt_column: str = field(
        default="prompt",
        metadata={"help": "Column to use as prompts for training."},
    )

    e2b_router_url: Optional[str] = field(
        default=None,
        metadata={"help": "URL for the E2B router. See scripts/e2b_router.py"},
    )

    morph_router_url: Optional[str] = field(
        default=None,
        metadata={"help": "URL for the MorphCloud router. See scripts/morph_router.py"},
    )

    code_provider: Optional[str] = field(
        default="e2b",
        metadata={
            "help": "Provider for code execution. Options: 'e2b', 'local', 'morph'.",
            "choices": ["e2b", "local", "morph"],
        },
    )

    ioi_provider: Optional[str] = field(
        default="piston",
        metadata={
            "help": "Provider for IOI code execution. Options: 'piston', 'morph'.",
            "choices": ["piston", "morph"],
        },
    )

    max_completion_len: int = field(
        default=16384,
        metadata={"help": "Maximum number of characters in completion."},
    )
    soft_punish_cache: int = field(
        default=4096,
        metadata={"help": "Minimum number of characters in completion."},
    )
