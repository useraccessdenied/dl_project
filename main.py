"""
B-FQG: Bloom's Taxonomy-based Follow-up Question Generation

Recursive Pipeline Implementation
-------------------------------
1. Generate initial questions
2. Score using GriceWise metrics
3. Filter low-quality questions using K-Means clustering/thresholding
4. Augment prompt with high-quality examples
5. Regenerate low-quality questions
"""

from bfqg_generator import BFGQuestionGenerator
from gricewise_evaluator import GriceWiseEvaluator
from pathlib import Path
import json
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

MAX_ITERATIONS = 2
CLUSTERING_MIN_SAMPLES = 5
QUALITY_THRESHOLD = 0.6  # Fallback threshold if clustering not possible


def calculate_aggregate_score(evaluation: dict) -> float:
    """Calculate overall quality score for a question set."""
    if not evaluation:
        return 0.0

    # Average of the aggregate scores of individual questions
    scores = [q_eval.get("aggregate_score", 0.0) for q_eval in evaluation.values()]
    return sum(scores) / len(scores) if scores else 0.0


def format_example_string(results: list) -> str:
    """Format high-quality results into a few-shot example string."""
    examples = []
    for i, res in enumerate(results, 1):
        seed = res["seed_question"]
        followups = res["followups"]

        example = f'Example {i}: Seed question - "{seed}"\n'
        # Sort by level
        sorted_levels = sorted(followups.keys())
        for level in sorted_levels:
            # key is like 'level_2', prompt expects 'Level 2'
            level_num = level.split("_")[1]
            level_name = {
                "2": "Understand",
                "3": "Apply",
                "4": "Analyze",
                "5": "Evaluate",
                "6": "Create",
            }.get(level_num, "Unknown")

            example += f'Level {level_num} ({level_name}): "{followups[level]}"\n'

        examples.append(example)

    return "\n".join(examples)


def main():
    print("=" * 70)
    print("B-FQG: Recursive Pipeline (Bloom's Taxonomy & Grice's Maxims)")
    print("=" * 70)

    # 1. Initialize Models
    print("\n[Step 1] Initializing Models...")
    # Using small models for demo speed; use larger ones for production
    generator = BFGQuestionGenerator(model_name="google/flan-t5-base", use_model=True)
    evaluator = GriceWiseEvaluator()
    print("Models ready.")

    # 2. Load Seeds
    sample_seeds_file = Path("data/sample_seeds.json")
    if sample_seeds_file.exists():
        try:
            with open(sample_seeds_file, "r", encoding="utf-8") as f:
                all_seeds = json.load(f)
        except:
            all_seeds = []

    if not all_seeds:
        all_seeds = [
            "How do I make a phone call?",
            "How do I navigate to an address?",
            "How do I adjust the air conditioning?",
            "How do I play music?",
            "How do I pair a bluetooth device?",
            "How do I check tire pressure?",
            "How do I set the cruise control?",
            "How do I open the sunroof?",
        ]

    print(f"Loaded {len(all_seeds)} seed questions.")

    # State tracking
    current_seeds = all_seeds
    final_results = {}  # Map seed -> result dict

    # 3. Recursive Loop
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n[Iteration {iteration}/{MAX_ITERATIONS}] Processing {len(current_seeds)} seeds...")

        if not current_seeds:
            print("No seeds to process. Stopping.")
            break

        # A. Generate
        results = generator.generate_multiple(current_seeds, show_progress=True)

        # B. Evaluate
        evaluated_results = evaluator.batch_evaluate(results, show_progress=True)

        # C. Score & Store
        iteration_scores = []
        iteration_results = []

        for res in evaluated_results:
            seed = res["seed_question"]
            score = calculate_aggregate_score(res.get("evaluation", {}))
            res["iteration_score"] = score

            final_results[seed] = res  # Update final results
            iteration_scores.append(score)
            iteration_results.append(res)

        # Check convergence or completion
        if iteration == MAX_ITERATIONS:
            print("Max iterations reached.")
            break

        # D. Filter / Cluster
        high_quality_indices = []
        low_quality_indices = []

        scores_np = np.array(iteration_scores).reshape(-1, 1)

        if len(scores_np) >= CLUSTERING_MIN_SAMPLES:
            print("Clustering results into High/Low quality...")
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scores_np)

            # Determine which cluster is "high quality" (higher centroid)
            centers = kmeans.cluster_centers_
            if centers[0] > centers[1]:
                high_label = 0
            else:
                high_label = 1

            high_quality_indices = [i for i, label in enumerate(labels) if label == high_label]
            low_quality_indices = [i for i, label in enumerate(labels) if label != high_label]
        else:
            print(f"Using threshold-based filtering (Threshold: {QUALITY_THRESHOLD})...")
            high_quality_indices = [i for i, s in enumerate(iteration_scores) if s >= QUALITY_THRESHOLD]
            low_quality_indices = [i for i, s in enumerate(iteration_scores) if s < QUALITY_THRESHOLD]

        print(f"High Quality: {len(high_quality_indices)}, Low Quality: {len(low_quality_indices)}")

        if not low_quality_indices:
            print("All questions meet quality standards! Stopping early.")
            break

        if not high_quality_indices:
            print("No high quality examples found to augment prompt. Stopping recursion.")
            break

        # E. Augment Prompt
        # Select top 3 high-quality examples
        top_indices = sorted(high_quality_indices, key=lambda i: iteration_scores[i], reverse=True)[:3]
        top_examples = [iteration_results[i] for i in top_indices]

        new_examples_str = format_example_string(top_examples)
        print("Augmenting prompt with best examples from this iteration.")
        generator.set_few_shot_examples(new_examples_str)

        # F. Prepare for next iteration (Regenerate low quality ones)
        current_seeds = [iteration_results[i]["seed_question"] for i in low_quality_indices]
        print(f"Scheduled {len(current_seeds)} low-scoring questions for regeneration.")

    # 4. Final Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    scores = [res.get("iteration_score", 0.0) for res in final_results.values()]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    print(f"Total Questions Processed: {len(final_results)}")
    print(f"Average Quality Score: {avg_score:.3f}")

    # Save detailed results
    output_file = "bfqg_recursive_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(list(final_results.values()), f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
