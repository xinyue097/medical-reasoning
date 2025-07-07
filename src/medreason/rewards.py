# Licensed under the Apache License, Version 2.0 (the "License");

"""Reward functions for GRPO training."""
# dataset comes from data.py
# Add reward function for multiple choice tasks
# Deleted the code-related reward functions
# Modified get_reward_funcs function

import asyncio
import math
import re
import numpy as np
from typing import Callable, Dict, Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


# This function keeps output 0
def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


# def format_reward(completions, **kwargs):
#     """
#     Reward function that checks if responses follow the required format:
#     <think>...</think>
#     <answer>...</answer>
    
#     Input: completions - List of lists, where each inner list contains dicts with "content" key
#     Output: List of float scores (0.0 to 1.0)
#     """
#     # Extract content from the nested structure: [[{"content": "text"}], [{"content": "text"}], ...]
#     completion_contents = [completion[0]["content"] for completion in completions]
    
#     scores = []
#     for content in completion_contents:
#         content = content.strip()
        
#         # Check for proper think tags (both opening and closing)
#         has_think_open = "<think>" in content.lower()
#         has_think_close = "</think>" in content.lower()
        
#         # Check for proper answer tags (both opening and closing)
#         has_answer_open = "<answer>" in content.lower()
#         has_answer_close = "</answer>" in content.lower()
        
#         # # Debug logging
#         # print(f"DEBUG FORMAT REWARD:")
#         # print(f"  Content preview: {content[:100]}...")
#         # print(f"  Has <think>: {has_think_open}")
#         # print(f"  Has </think>: {has_think_close}")
#         # print(f"  Has <answer>: {has_answer_open}")
#         # print(f"  Has </answer>: {has_answer_close}")
        
#         # Calculate score based on format compliance
#         score = 0.0
        
#         # Reward for complete think section (both tags)
#         if has_think_open and has_think_close:
#             score += 0.5
#         elif has_think_open or has_think_close:  # Partial credit
#             score += 0.2
        
#         # Reward for complete answer section (both tags)
#         if has_answer_open and has_answer_close:
#             score += 0.5
#         elif has_answer_open or has_answer_close:  # Partial credit
#             score += 0.2
        
#         # Bonus for having both sections complete
#         if (has_think_open and has_think_close and has_answer_open and has_answer_close):
#             score = 1.0  # Perfect format gets full score
        
#         print(f"  Format Score: {score}")
#         scores.append(score)
    
#     return scores


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float, language: str = "en"):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://huggingface.co/papers/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    language: Language of the text, defaults to `en`. Used to choose the way to split the text into n-grams.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if language == "en":

        def zipngram(text: str, ngram_size: int):
            words = text.lower().split()
            return zip(*[words[i:] for i in range(ngram_size)]), words

    elif language == "zh":
        from transformers.utils.import_utils import _is_package_available

        if not _is_package_available("jieba"):
            raise ValueError("Please install jieba to use Chinese language")

        def zipngram(text: str, ngram_size: int):
            import jieba

            seg_list = list(jieba.cut(text))
            return zip(*[seg_list[i:] for i in range(ngram_size)]), seg_list

    else:
        raise ValueError(
            f"Word splitting for language `{language}` is not yet implemented. Please implement your own zip-ngram function."
        )

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            ngram_array, words = zipngram(completion, ngram_size)

            if len(words) < ngram_size:
                rewards.append(0.0)
                continue

            for ng in ngram_array:
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def _init_event_loop():
    """Initialize or get the current event loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def get_soft_overlong_punishment(max_completion_len, soft_punish_cache):
    """
    Reward function that penalizes overlong completions. It is used to penalize overlong completions,
    but not to reward shorter completions. Reference: Eq. (13) from the DAPO paper (https://huggingface.co/papers/2503.14476)

    Args:
        max_completion_len: Maximum length of the completion
        soft_punish_cache: Minimum length of the completion. If set to 0, no minimum length is applied.
    """

    def soft_overlong_punishment_reward(completion_ids: list[list[int]], **kwargs) -> list[float]:
        """Reward function that penalizes overlong completions."""
        rewards = []
        for ids in completion_ids:
            completion_length = len(ids)
            if completion_length <= max_completion_len - soft_punish_cache:
                rewards.append(0.0)
            elif max_completion_len - soft_punish_cache < completion_length <= max_completion_len:
                rewards.append((max_completion_len - soft_punish_cache - completion_length) / soft_punish_cache)
            else:
                rewards.append(-1.0)
        return rewards

    return soft_overlong_punishment_reward


#-------------- Multiple Choices specific reward function ----------------

# def multiple_choice_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
#     """Reward function that checks if the completion contains the correct multiple choice answer.
    
#     This function extracts the answer from the completion and compares it with the ground truth.
#     The ground truth uses 0-based indexing where 0=A, 1=B, 2=C, 3=D.
#     Only accepts letter answers (A, B, C, D) from model completions.
#     """
#     # Extract solution from kwargs - GRPO trainer passes it here
#     solution = kwargs.get('solution', kwargs.get('labels', kwargs.get('cop', [])))
    
#     if not solution:
#         print("WARNING: No solution found in kwargs")
#         return [0.0] * len(completions)
    
#     contents = [completion[0]["content"] for completion in completions]
#     rewards = []
    
#     for content, sol in zip(contents, solution):
#         # Convert sol to int if it's a string, handle int64 dtype
#         try:
#             if isinstance(sol, str):
#                 sol_int = int(sol)
#             else:
#                 sol_int = int(sol)  # Handle int64 from pandas
            
#             # Check if the ground truth solution is valid (0, 1, 2, 3)
#             if sol_int in [0, 1, 2, 3]:
#                 # Extract answer from completion - ONLY look for letters A, B, C, D
#                 extracted_answer = None
                
#                 # First try to extract from \\boxed{} format - only letters
#                 boxed_pattern = r'\\boxed\{([A-D])\}'
#                 boxed_match = re.search(boxed_pattern, content, re.IGNORECASE)
#                 if boxed_match:
#                     extracted_answer = boxed_match.group(1).upper().strip()
#                 else:
#                     # Try to extract from <answer> tags - only letters
#                     answer_tag_pattern = r'<answer>\s*([A-D])\s*</answer>'
#                     answer_match = re.search(answer_tag_pattern, content, re.IGNORECASE)
                    
#                     if answer_match:
#                         extracted_answer = answer_match.group(1).upper().strip()
#                     else:
#                         # Try to find patterns like "The answer is X" or "Option X" - only letters
#                         answer_patterns = [
#                             r'(?:the answer is|answer is|answer:|option is|choice is)\s*([A-D])',
#                             r'(?:the answer is|answer is|answer:|option is|choice is)\s*option\s*([A-D])',
#                             r'\b([A-D])\s*(?:is the|is correct|is the answer)',
#                             r'(?:^|\n)\s*([A-D])\s*[.)]?\s*$'
#                         ]
                        
#                         for pattern in answer_patterns:
#                             match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
#                             if match:
#                                 extracted_answer = match.group(1).upper().strip()
#                                 break
                        
#                         if not extracted_answer:
#                             # Last resort: look for isolated letters near the end
#                             final_pattern = r'\b([A-D])\b(?!.*\b[A-D]\b)'
#                             final_match = re.search(final_pattern, content, re.IGNORECASE)
#                             if final_match:
#                                 extracted_answer = final_match.group(1).upper().strip()
                
#                 # Compute binary rewards if answer is extractable, `None` otherwise to skip this example
#                 try:
#                     if extracted_answer and extracted_answer in ['A', 'B', 'C', 'D']:
#                         # Convert ground truth int to letter for comparison
#                         # 0 = A, 1 = B, 2 = C, 3 = D (0-based indexing)
#                         number_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
                        
#                         # Compare extracted letter with converted ground truth
#                         if extracted_answer == number_to_letter[sol_int]:
#                             reward = 1.0
#                         else:
#                             reward = 0.0
#                     else:
#                         reward = 0.0
#                 except Exception as e:
#                     print(f"Multiple choice evaluation failed: {e}, answer: {extracted_answer}, solution: {sol_int}")
#                     reward = None
#             else:
#                 # If the ground truth solution is not valid, we assign `None` to skip this example
#                 reward = None
#                 print("Invalid ground truth solution: ", sol_int)
#         except (ValueError, TypeError) as e:
#             # If conversion to int fails, skip this example
#             reward = None
#             print(f"Failed to convert solution to int: {e}, solution: {sol}")
        
#         rewards.append(reward)
    
#     return rewards


def multiple_choice_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion contains the correct multiple choice answer."""
    
    # Extract solution from kwargs with detailed logging
    solution = kwargs.get('solution', kwargs.get('labels', kwargs.get('cop', [])))
    
    if not solution:
        print("WARNING: No solution found in kwargs")
        print(f"Available kwargs keys: {list(kwargs.keys())}")
        return [0.0] * len(completions)
    
    # # ✅ ADDED: Debug the solution data
    # print(f"DEBUG: Received solution data: {solution[:5]}...")  # First 5 items
    # print(f"DEBUG: Solution data types: {[type(x) for x in solution[:3]]}")
    
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for i, (content, sol) in enumerate(zip(contents, solution)):
        try:
            # Handle different data types
            if hasattr(sol, 'item'):
                sol_int = int(sol.item())
            else:
                sol_int = int(sol)
            
            # ✅ ADDED: More detailed logging for invalid cases
            if sol_int in [0, 1, 2, 3]:
                # Extract answer from model response
                extracted_answer = extract_multiple_choice_answer(content)
                
                # Convert ground truth int to letter for comparison
                number_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
                
                # Compare extracted answer with ground truth
                if extracted_answer == number_to_letter[sol_int]:
                    reward = 1.0
                else:
                    reward = 0.0
            else:
                reward = None
                # print(f"Invalid ground truth solution: {sol_int} (type: {type(sol)}, original: {sol})")
                # print(f"DEBUG: This is item {i} in the batch")
                # print(f"DEBUG: kwargs keys available: {list(kwargs.keys())}")
                # # ✅ ADDED: Check if this is a batch indexing issue
                # if hasattr(kwargs.get('solution', []), '__len__'):
                #     print(f"DEBUG: Solution array length: {len(kwargs.get('solution', []))}")
                
        except (ValueError, TypeError) as e:
            reward = None
            # print(f"Failed to convert solution to int: {e}, solution: {sol} (type: {type(sol)})")
        
        rewards.append(reward)
    
    return rewards

# ✅ ADDED: The missing extraction function
def extract_multiple_choice_answer(content: str) -> str:
    """Extract the multiple choice answer (A, B, C, D) from model response."""
    import re
    
    # Method 1: Look for answer in <answer> tags
    answer_match = re.search(r'<answer>\s*([ABCD])\s*</answer>', content, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    
    # Method 2: Look for "The answer is X" patterns
    answer_match = re.search(r'(?:answer is|answer:|is)\s*([ABCD])', content, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    
    # Method 3: Look for standalone letters at end of response
    answer_match = re.search(r'\b([ABCD])\b(?!.*\b[ABCD]\b)', content, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    
    # Method 4: Look for any A, B, C, D in the response (last resort)
    letters = re.findall(r'\b([ABCD])\b', content, re.IGNORECASE)
    if letters:
        return letters[-1].upper()  # Return the last found letter
    
    # Default: return None if no answer found
    return None


def compute_metrics(eval_preds, tokenizer) -> Dict[str, float]:
    """
    This function is called by the trainer during evaluation.
    It wraps the multiple_choice_reward function to compute overall accuracy.
    """
    # eval_preds is a tuple containing predictions and label_ids
    # For GRPOTrainer, predictions are the generated token IDs
    predictions, label_ids = eval_preds

    # The 'cop' (correct option) column is passed as the label_ids
    # It's important to ensure this is done in grpo.py
    solutions = label_ids

    # Decode the generated sequences into text
    # The -100 labels are for the prompt, we want to decode the full sequence
    predictions[predictions == -100] = tokenizer.pad_token_id
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # The reward function expects a list of lists of dicts
    # We wrap our decoded predictions to match this format for one generation
    wrapped_completions = [[{"content": pred}] for pred in decoded_preds]

    # Call your reward function with kwargs
    rewards = multiple_choice_reward(completions=wrapped_completions, solution=solutions)
    
    # Filter out None values (from examples that were skipped) and calculate accuracy
    valid_rewards = [r for r in rewards if r is not None]
    if len(valid_rewards) == 0:
        return {"eval_accuracy": 0.0, "eval_samples": 0}

    accuracy = np.mean(valid_rewards)
    
    return {
        "eval_accuracy": accuracy,
        "eval_samples": len(valid_rewards) # Report how many examples were successfully evaluated
    }    

#-------------------- Math specific reward function ----------------------
def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://huggingface.co/papers/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward



#------------------------ Final reward functions -----------------------
def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "multiple_choice": multiple_choice_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "tag_count": tag_count_reward,
        "soft_overlong_punishment": get_soft_overlong_punishment(
            max_completion_len=script_args.max_completion_len,
            soft_punish_cache=script_args.soft_punish_cache,
        ),
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs