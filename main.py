import sys
from datetime import datetime
import logging
import torch
from tqdm import tqdm
import numpy as np
import json
from typing import Dict, List, Any
from model import load_model_and_tokenizer
from data import get_dataset


def format_math_prompt(problem: Dict[str, Any]) -> str:
    """Format a math problem into a prompt for the model."""
    return f"""Solve this mathematics problem step by step:
Problem: {problem['problem']}
Let's approach this step by step:"""


def parse_model_response(response: str) -> str:
    """Parse the model's response to extract the final answer."""
    # Split response into lines and look for final answer
    lines = response.split("\n")
    answer = ""

    # Look for lines containing "answer", "therefore", or "final" (case insensitive)
    for line in reversed(lines):
        lower_line = line.lower()
        if any(
            keyword in lower_line for keyword in ["answer:", "therefore,", "final:"]
        ):
            answer = line.split(":")[-1].strip()
            break

    # If no explicit answer found, take the last non-empty line
    if not answer:
        for line in reversed(lines):
            if line.strip():
                answer = line.strip()
                break

    return answer


def evaluate_correctness(predicted_answer: str, ground_truth: str) -> float:
    """
    Evaluate if the predicted answer matches the ground truth.
    Returns 1.0 for correct, 0.0 for incorrect.
    """
    # Clean and normalize both answers
    predicted = predicted_answer.lower().strip()
    truth = ground_truth.lower().strip()

    # Remove common mathematical notation variations
    chars_to_remove = [" ", ",", "$", "\\"]
    for char in chars_to_remove:
        predicted = predicted.replace(char, "")
        truth = truth.replace(char, "")

    # Check for exact match after normalization
    return float(predicted == truth)


def evaluate_model(model, tokenizer, item: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the model on a single item."""
    # Format the prompt
    prompt = format_math_prompt(item)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=2048,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode response
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Parse answer from response
    predicted_answer = parse_model_response(response)

    # Calculate correctness
    correctness = evaluate_correctness(predicted_answer, item["answer"])

    return {
        "problem_id": item.get("unique_id", ""),
        "problem": item["problem"],
        "ground_truth": item["answer"],
        "model_response": response,
        "predicted_answer": predicted_answer,
        "correct": correctness,
    }


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate evaluation metrics from results."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])

    metrics = {
        "accuracy": correct / total if total > 0 else 0,
        "total_evaluated": total,
        "total_correct": correct,
    }

    # Calculate accuracy by problem type if available
    problem_type_metrics = {}
    for result in results:
        if "problem_type" in result:
            ptype = result["problem_type"]
            if ptype not in problem_type_metrics:
                problem_type_metrics[ptype] = {"correct": 0, "total": 0}
            problem_type_metrics[ptype]["total"] += 1
            if result["correct"]:
                problem_type_metrics[ptype]["correct"] += 1

    # Add problem type accuracies to metrics
    for ptype, counts in problem_type_metrics.items():
        metrics[f"accuracy_{ptype}"] = counts["correct"] / counts["total"]

    return metrics


def main():
    logging.info("Starting main script")

    try:
        # Load models and tokenizer
        logging.info("Loading models and tokenizer")
        model_name = "meta-llama/Llama-2-7b-chat"  # Updated to an available model
        model, tokenizer = load_model_and_tokenizer(model_name)
        logging.info("Models loaded successfully")

        # Load dataset
        logging.info("Loading MATH500 dataset")
        dataset = get_dataset("hendrycks/MATH")  # Using the full MATH dataset
        eval_dataset = dataset["test"].select(range(500))  # Taking first 500 examples
        logging.info(f"Dataset size: {len(eval_dataset)} documents")

        # Initialize results list
        results = []

        # Evaluate model on dataset
        for idx, item in enumerate(tqdm(eval_dataset)):
            try:
                result = evaluate_model(model, tokenizer, item)
                results.append(result)

                # Log progress periodically
                if (idx + 1) % 10 == 0:
                    logging.info(f"Processed {idx + 1} problems")

                    # Calculate running metrics
                    running_metrics = calculate_metrics(results)
                    logging.info(f"Running accuracy: {running_metrics['accuracy']:.2%}")

            except Exception as e:
                logging.error(f"Error processing item {idx}: {str(e)}")
                continue

        # Calculate final metrics
        final_metrics = calculate_metrics(results)

        # Save results and metrics
        output = {
            "results": results,
            "metrics": final_metrics,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
        }

        with open("math500_evaluation_results.json", "w") as f:
            json.dump(output, f, indent=2)

        logging.info("Evaluation complete!")
        logging.info(f"Final accuracy: {final_metrics['accuracy']:.2%}")
        logging.info(f"Results saved to math500_evaluation_results.json")

    except Exception as e:
        logging.critical(f"Critical error in main execution: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
