from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.metrics.metrics import Metrics
from lighteval.metrics import Doc

def mmlu_pro_prompt(line, task_name: str = None):
    """
    Format MMLU-Pro prompts for compatibility with lighteval.
    
    Args:
        line: A line from the MMLU-Pro dataset
        task_name: The name of the task
        
    Returns:
        Doc object with properly formatted content
    """
    question = line["question"]
    options = line["options"]
    
    # Determine the correct answer index (0-based)
    if "answer_index" in line:
        answer_idx = line["answer_index"]
    elif isinstance(line.get("answer"), int) and line["answer"] > 0:
        answer_idx = line["answer"] - 1  # Convert 1-based to 0-based
    elif isinstance(line.get("answer"), str) and line["answer"].isdigit():
        answer_idx = int(line["answer"]) - 1
    elif isinstance(line.get("answer"), str) and len(line["answer"]) == 1 and line["answer"].isalpha():
        # Convert letter to 0-based index (A=0, B=1, etc.)
        answer_idx = ord(line["answer"].upper()) - ord('A')
    else:
        answer_idx = 0  # Default to first option
    
    # Format options for the Doc object
    formatted_options = []
    for option in options:
        formatted_options.append(f" {option}")
    
    # Important: Put the instruction at the beginning of the query
    instruction = "Answer with the number of the correct option."
    formatted_query = f"{instruction} {question}"
    
    # Return Doc object
    return Doc(
        task_name=task_name,
        query=formatted_query,
        choices=formatted_options,
        gold_index=answer_idx,
        instruction=instruction  # Include the instruction here as well
    )

# Define the MMLU-Pro task
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
    metric=[Metrics.exact_match],  # Use the enum value, not a string
    trust_dataset=True
)

# Add to the tasks table
TASKS_TABLE = [MMLU_PRO_TASK]

# Define task groups
TASKS_GROUPS = {
    "custom": ["mmlu_pro"]
}
