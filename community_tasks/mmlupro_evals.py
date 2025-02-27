from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.metrics.metrics import Metrics
from lighteval.metrics import Doc

def mmlu_pro_prompt(line, task_name: str = None):
    """
    Format MMLU-Pro prompts as specified in the original task
    
    Args:
        line: A line from the MMLU-Pro dataset
        task_name: The name of the task
        
    Returns:
        Doc object with properly formatted content
    """
    question = line["question"]
    options = line["options"]
    
    # The dataset actually uses letters (like 'I') instead of numbers for answers
    # Use answer_index if available, otherwise the answer is the letter/string itself
    if "answer_index" in line:
        gold_index = line["answer_index"]
    else:
        # If we only have the answer as a letter (like 'I'), we need to convert it
        answer = line["answer"]
        # If the answer is a letter (A, B, C...), convert it to a zero-indexed number
        if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
            # Convert from letter to index (A=0, B=1, C=2, etc.)
            gold_index = ord(answer.upper()) - ord('A')
        # If the answer is a number as a string ('1', '2', etc.), convert to zero-indexed
        elif isinstance(answer, str) and answer.isdigit():
            gold_index = int(answer) - 1
        # If the answer is already a number, just subtract 1 to make it zero-indexed
        elif isinstance(answer, int):
            gold_index = answer - 1
        else:
            # If we can't determine the index, use 0 as a fallback
            print(f"Warning: Could not determine gold_index for answer: {answer}")
            gold_index = 0
    
    # Create formatted choices for the Doc
    choices = []
    for option in options:
        choices.append(f" {option}")
    
    # Return a Doc object with properly formatted content
    return Doc(
        task_name=task_name,
        query=question,
        choices=choices,
        gold_index=gold_index,
        instruction=""
    )

# Define the MMLU-Pro task using the exact_match metric
MMLU_PRO_TASK = LightevalTaskConfig(
    name="mmlu_pro",
    suite=["custom"],
    prompt_function=mmlu_pro_prompt,
    hf_repo="TIGER-Lab/MMLU-Pro",
    hf_subset="default",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="random",
    generation_size=-1,  # -1 for multiple-choice tasks
    stop_sequence=None,
    metric=[Metrics.exact_match],  # Using the Metrics enum value
    trust_dataset=True
)

# Add to the tasks table
TASKS_TABLE = [MMLU_PRO_TASK]

#lighteval accelerate "pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" "community_tasks|mmlu_pro|0|0" --custom-tasks community_tasks/mmlupro_evals.py
# Define task groups to make the custom suite work
TASKS_GROUPS = {
    "custom": ["mmlu_pro"]
}
