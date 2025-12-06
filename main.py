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

MAX_ITERATIONS = 5
CLUSTERING_MIN_SAMPLES = 5
QUALITY_THRESHOLD = 0.6


class RecursivePipeline:
    def __init__(self, generator=None, evaluator=None):
        self.generator = generator or BFGQuestionGenerator(model_name="google/flan-t5-large", use_model=True)
        self.evaluator = evaluator or GriceWiseEvaluator(
            use_chatgpt=True,
            chatgpt_model="gpt-4.1-mini",   # or any other model you have access to
            openai_api_key="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",     # optional; otherwise uses env var
        )

    def calculate_aggregate_score(self, evaluation: dict) -> float:
        """Calculate overall quality score for a question set."""
        if not evaluation:
            return 0.0
        scores = [q_eval.get("aggregate_score", 0.0) for q_eval in evaluation.values()]
        return sum(scores) / len(scores) if scores else 0.0

    def format_example_string(self, results: list) -> str:
        """Format high-quality results into a few-shot example string."""
        examples = []
        for i, res in enumerate(results, 1):
            seed = res["seed_question"]
            followups = res["followups"]
            example = f'Example {i}: Seed question - "{seed}"\n'
            sorted_levels = sorted(followups.keys())
            for level in sorted_levels:
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

    def run(self, seeds: list):
        print("=" * 70)
        print("B-FQG: Recursive Pipeline (Bloom's Taxonomy & Grice's Maxims)")
        print("=" * 70)
        print(f"Processing {len(seeds)} seed questions.")

        current_seeds = seeds
        final_results = {}

        for iteration in range(1, MAX_ITERATIONS + 1):
            print(f"\n[Iteration {iteration}/{MAX_ITERATIONS}] Processing {len(current_seeds)} seeds...")
            if not current_seeds:
                break

            results = self.generator.generate_multiple(current_seeds, show_progress=True)
            evaluated_results = self.evaluator.batch_evaluate(results, show_progress=True)

            iteration_scores = []
            iteration_results = []

            for res in evaluated_results:
                seed = res["seed_question"]
                score = self.calculate_aggregate_score(res.get("evaluation", {}))
                res["iteration_score"] = score
                final_results[seed] = res
                iteration_scores.append(score)
                iteration_results.append(res)

            if iteration == MAX_ITERATIONS:
                break

            high_quality_indices = []
            low_quality_indices = []
            scores_np = np.array(iteration_scores).reshape(-1, 1)

            if len(scores_np) >= CLUSTERING_MIN_SAMPLES:
                print("Clustering results into High/Low quality...")
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                labels = kmeans.fit_predict(scores_np)
                centers = kmeans.cluster_centers_
                high_label = 0 if centers[0] > centers[1] else 1
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

            top_indices = sorted(high_quality_indices, key=lambda i: iteration_scores[i], reverse=True)[:3]
            top_examples = [iteration_results[i] for i in top_indices]
            new_examples_str = self.format_example_string(top_examples)
            print("Augmenting prompt with best examples from this iteration.")
            self.generator.set_few_shot_examples(new_examples_str)

            current_seeds = [iteration_results[i]["seed_question"] for i in low_quality_indices]
            print(f"Scheduled {len(current_seeds)} low-scoring questions for regeneration.")

        return list(final_results.values())


def main():
    print("\n[Step 1] Initializing Models...")
    pipeline = RecursivePipeline()
    print("Models ready.")

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

    results = pipeline.run(all_seeds)

    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    scores = [res.get("iteration_score", 0.0) for res in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    print(f"Total Questions Processed: {len(results)}")
    print(f"Average Quality Score: {avg_score:.3f}")

    output_file = "bfqg_recursive_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
