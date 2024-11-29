import json
import re
from typing import Any, Dict

def evaluate_model(model: Any, test_set_file_path: str) -> Dict[str, Any]:
    """
    Evaluates the finetuned model against the test set and computes evaluation metrics.

    Parameters:
    - model: The finetuned language model to evaluate.
    - test_set_file_path: Path to the test set JSON file.

    Returns:
    - metrics: A dictionary containing evaluation metrics.
    """
    # Load test set
    with open(test_set_file_path, 'r') as f:
        test_set = json.load(f)

    # Initialize evaluation metrics
    total_examples = len(test_set)
    correct_optimal_actions = 0
    correct_template_structures = 0
    total_dynamic_fields = 0
    correct_dynamic_fields = 0
    evaluations = []

    # Regular expression to extract content within brackets []
    bracket_pattern = re.compile(r'\[(.*?)\]')

    # Expected static parts of the template
    static_template = (
        "The game is currently at the stage of [X]. My position is [X]. I am relatively [X]. "
        "My holding is [X]. The board is [X]. My hand currently forms [X]. The current pot size is [X], "
        "and my stack size left is [X]. The stack-to-pot ratio is [X]. "
        "Given these information and the action history, my optimal decision is: X."
    )

    for example in test_set:
        instruction = example['instruction']
        input_text = example.get('input', '')
        ground_truth_output = example['output']

        # Generate model output
        model_output = generate_model_output(model, instruction, input_text)

        # Compare optimal action
        gt_action = extract_optimal_action(ground_truth_output)
        model_action = extract_optimal_action(model_output)
        is_action_correct = gt_action.lower() == model_action.lower()
        if is_action_correct:
            correct_optimal_actions += 1

        # Compare template structure
        is_template_correct = compare_template_structure(static_template, model_output)
        if is_template_correct:
            correct_template_structures += 1

        # Compare dynamic fields
        gt_dynamic_fields = bracket_pattern.findall(ground_truth_output)
        model_dynamic_fields = bracket_pattern.findall(model_output)

        num_dynamic_fields = len(gt_dynamic_fields)
        num_correct_dynamic_fields = sum(
            1 for gt_field, model_field in zip(gt_dynamic_fields, model_dynamic_fields)
            if gt_field.strip().lower() == model_field.strip().lower()
        )

        total_dynamic_fields += num_dynamic_fields
        correct_dynamic_fields += num_correct_dynamic_fields

        # Store evaluation details
        evaluation = {
            'instruction': instruction,
            'input': input_text,
            'ground_truth_output': ground_truth_output,
            'model_output': model_output,
            'is_action_correct': is_action_correct,
            'is_template_correct': is_template_correct,
            'num_dynamic_fields': num_dynamic_fields,
            'num_correct_dynamic_fields': num_correct_dynamic_fields
        }
        evaluations.append(evaluation)

    # Calculate overall metrics
    action_accuracy = correct_optimal_actions / total_examples
    template_accuracy = correct_template_structures / total_examples
    dynamic_field_accuracy = (
        correct_dynamic_fields / total_dynamic_fields if total_dynamic_fields > 0 else 0
    )

    metrics = {
        'total_examples': total_examples,
        'action_accuracy': action_accuracy,
        'template_accuracy': template_accuracy,
        'dynamic_field_accuracy': dynamic_field_accuracy
    }

    # Save evaluations and metrics to a JSON file
    results = {
        'metrics': metrics,
        'evaluations': evaluations
    }
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    return metrics

def generate_model_output(model: Any, instruction: str, input_text: str) -> str:
    """
    Generates output from the model based on the instruction and input text.

    Parameters:
    - model: The finetuned language model.
    - instruction: The instruction text.
    - input_text: Additional input text.

    Returns:
    - The generated output text from the model.
    """
    # Combine instruction and input_text as per model requirements
    prompt = f"{instruction}\n{input_text}".strip()

    # Generate output using the model (implementation depends on your model's API)
    # Replace the following line with your model's actual generation method
    model_output = model.generate(prompt)

    return model_output

def extract_optimal_action(output_text: str) -> str:
    """
    Extracts the optimal action from the output text.

    Parameters:
    - output_text: The text output from which to extract the action.

    Returns:
    - The extracted optimal action as a string.
    """
    # Adjusted regex pattern to capture action and optional number
    match = re.search(r'optimal decision is:\s*([a-zA-Z]+(?:\s+\d+)?)(?:\.|\n|$)', output_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return ''

def compare_template_structure(static_template: str, model_output: str) -> bool:
    """
    Compares the static parts of the template in the model output.

    Parameters:
    - static_template: The expected template with dynamic parts as placeholders.
    - model_output: The model's output text.

    Returns:
    - True if the static parts in the model output match the expected static parts.
    """
    # Replace dynamic placeholders with regex patterns
    dynamic_placeholder_pattern = r'\[X\]|X'
    static_pattern = re.sub(dynamic_placeholder_pattern, '(.*?)', static_template)

    # Escape special regex characters in the static parts
    static_pattern_escaped = ''
    last_end = 0
    for match in re.finditer(r'\(\.\*\?\)', static_pattern):
        # Escape the static text before the dynamic placeholder
        static_text = static_pattern[last_end:match.start()]
        static_pattern_escaped += re.escape(static_text)
        # Add the dynamic placeholder regex back
        static_pattern_escaped += '(.*?)'
        last_end = match.end()
    # Add any remaining static text after the last dynamic placeholder
    static_pattern_escaped += re.escape(static_pattern[last_end:])

    # Compile the regex pattern
    static_regex = re.compile(static_pattern_escaped, re.DOTALL | re.IGNORECASE)

    # Match the model output against the static regex pattern
    match = static_regex.fullmatch(model_output.strip())
    return bool(match)

# Example usage (assuming you have a 'model' object and 'test_set.json' file):
# metrics = evaluate_model(model, 'test_set.json')
# print(metrics)