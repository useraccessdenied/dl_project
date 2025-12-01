"""
B-FQG: Bloom's Taxonomy-based Follow-up Question Generation

Example script demonstrating the B-FQG framework implemented from the ACL 2025 paper:
"From Recall to Creation: Generating Follow-Up Questions Using Bloom's Taxonomy and Grice's Maxims"

This script shows batch processing of seed questions to generate cognitively scaffolded
follow-up questions and evaluate them using the GriceWise framework.
"""

from bfqg_generator import BFGQuestionGenerator
from gricewise_evaluator import GriceWiseEvaluator
from pathlib import Path
import json


def main():
    """Main demonstration function."""
    print("=" * 70)
    print("B-FQG: Bloom's Taxonomy Follow-up Question Generation")
    print("=" * 70)
    print("Implementation of ACL 2025 Industry Track paper")
    print()

    # Initialize models
    print("Loading models...")
    generator = BFGQuestionGenerator()
    evaluator = GriceWiseEvaluator()
    print("OK Models loaded successfully\n")

    # Load sample seed questions
    sample_seeds_file = Path("data/sample_seeds.json")
    if sample_seeds_file.exists():
        try:
            import json

            with open(sample_seeds_file, "r", encoding="utf-8") as f:
                seed_questions = json.load(f)
            print(
                f"Loaded {len(seed_questions)} seed questions from {sample_seeds_file}"
            )
        except Exception as e:
            print(f"Warning: Could not load sample data: {e}")
            seed_questions = []
    else:
        # Fallback sample questions
        seed_questions = [
            "How do I make a phone call?",
            "How do I navigate to an address?",
            "How do I adjust the air conditioning?",
            "How do I play music?",
        ]

    print(f"Processing {len(seed_questions)} seed questions...")
    print("-" * 70)

    # Generate follow-up questions
    results = generator.generate_multiple(seed_questions, show_progress=True)

    # Evaluate questions
    evaluations = evaluator.batch_evaluate(results, show_progress=True)

    # Process and display results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    summary_stats = []
    for i, result in enumerate(evaluations, 1):
        seed = result["seed_question"]
        evaluation = result.get("evaluation", {})

        print(f"\nSeed {i}: {seed}")
        print("-" * (len(seed) + 8))

        if "error" in result:
            print(f"ERROR Error: {result['error']}")
            continue

        # Display generated questions
        followups = result.get("followups", {})
        print("Generated Questions:")
        for level_key, question in followups.items():
            level_num = level_key.split("_")[1]
            level_name = {
                "2": "Understand",
                "3": "Apply",
                "4": "Analyze",
                "5": "Evaluate",
                "6": "Create",
            }.get(level_num, level_key)
            print(f"  L{level_num} ({level_name}): {question}")

        # Display evaluation if available
        if evaluation:
            print("\nEvaluation Scores:")
            for level_key, eval_data in evaluation.items():
                score = eval_data.get("aggregate_score", 0.0)
                level_num = level_key.split("_")[1]
                status = "OK" if score > 0.6 else "WARNING" if score > 0.4 else "FAILED"
                print(f"  L{level_num}: {status} {score:.3f}")

            # Aggregate stats
            scores = [data.get("aggregate_score", 0.0) for data in evaluation.values()]
            if scores:
                avg_score = sum(scores) / len(scores)
                summary_stats.append(avg_score)
                overall = (
                    "Good" if avg_score > 0.6 else "Fair" if avg_score > 0.4 else "Poor"
                )
                print(f"Overall Score: {overall} ({avg_score:.3f})")

    # Overall summary
    if summary_stats:
        total_avg = sum(summary_stats) / len(summary_stats)
        print(f"\n{'=' * 70}")
        print(f"OVERALL STATISTICS")
        print(f"{'=' * 70}")
        print(f"Average Score: {total_avg:.3f}")
        print(f"Questions Generated: {len(seed_questions) * 5}")  # 5 per seed
        print(f"Bloom Levels Covered: 5 (Understand, Apply, Analyze, Evaluate, Create)")
        print("\nFeatures Demonstrated:")
        print("• Cognitive scaffolding using Bloom's Taxonomy")
        print("• Gricean-inspired evaluation (Quality, Quantity, Relation, Manner)")
        print("• LLM-powered question generation with few-shot prompting")
        print("• Automated quality assessment using NLI, embeddings, and heuristics")
    else:
        print("No evaluations completed successfully.")

    # Save results
    output_file = "bfqg_results.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(evaluations, f, indent=2, ensure_ascii=False)
        print(f"\nOK Results saved to {output_file}")
    except Exception as e:
        print(f"\nFAILED Failed to save results: {e}")

    print("\nTo use the interactive REPL, run: uv run python repl.py")
    print("This allows you to generate and evaluate questions interactively.")


if __name__ == "__main__":
    main()
