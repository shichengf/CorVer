import re
import logging
from typing import List
from .reward_utils import save_json_output, get_timestamp

logger = logging.getLogger(__name__)

def format_reward_func(completions: List[str], **kwargs) -> List[float]:
    rewards = []
    log_entries = []
    
    for i, completion in enumerate(completions):
        try:
            # Add <think> prefix to unify format
            if not completion.startswith("<think>"):
                completion = "<think>" + completion
            
            # Regex to check format. Lenient close tags: </think> and </answer> are
            # optional so verbose models (e.g. Mistral-7B that misses ~95% of </answer>)
            # and truncated completions still get format=+1.
            #
            # Anti-hack additions (2026-04-25, Llama-3.2-3B reward hacking discovered):
            #   (A) think content min 30 chars + must contain ≥1 letter (post-check)
            #   (C) think content first non-whitespace must NOT be '<' — prevents the
            #       backtrack exploit where regex captures "</think>" as content when
            #       model writes empty <think> </think> followed by reasoning outside tags.
            regex = r"^<think>\s*([^<\s](?:(?!<think>|<answer>)[\s\S]){29,}?)\s*(?:<\/think>\s*)?<answer>\s*(\S(?:(?!<think>|<answer>)[\s\S])*?)(?:\s*<\/answer>)?\s*$"
            match = re.search(regex, completion, re.DOTALL)

            # Determine reward value
            if match is None or len(match.groups()) != 2:
                rewards.append(-1.0)  # Clear penalty for non-conforming format
                success = False
            else:
                # Post-check: think content must contain at least one letter
                think_content = match.group(1)
                if not re.search(r'[A-Za-z]', think_content):
                    rewards.append(-1.0)
                    success = False
                else:
                    rewards.append(1.0)
                    success = True
                
            # Record detailed information
            log_entries.append({
                "sample_index": i,
                "completion": completion,
                "format_correct": success,
                "reward": rewards[-1],
                "timestamp": get_timestamp()
            })
            
        except Exception as e:
            logger.error(f"Format check error: {str(e)}")
            rewards.append(0.0)  # Return 0 on error
            log_entries.append({
                "sample_index": i,
                "completion": completion,
                "error": str(e),
                "reward": 0.0,
                "timestamp": get_timestamp()
            })
    
    # Save output to JSON file
    save_json_output({"format_checks": log_entries}, "format_reward")
            
    return rewards