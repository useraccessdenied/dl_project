#!/usr/bin/env python3
"""
Interactive REPL for Bloom's Taxonomy Follow-up Question Generation

This script provides an interactive command-line interface for generating
and evaluating follow-up questions using the B-FQG and GriceWise frameworks.
"""

import json
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# Import our modules
from bfqg_generator import BFGQuestionGenerator
from gricewise_evaluator import GriceWiseEvaluator


class FollowUpQuestionREPL:
    """
    Interactive REPL for generating and evaluating follow-up questions.
    """

    def __init__(self, save_history: bool = True):
        """
        Initialize the REPL with generators and evaluators.

        Args:
            save_history: Whether to save interaction history
        """
        self.save_history = save_history
        self.history = []
        self.generator = None
        self.evaluator = None

    def initialize_models(self):
        """Initialize the generator and evaluator models."""
        print("Initializing models...")

        try:
            self.generator = BFGQuestionGenerator()
            print("✓ Question generator initialized")
        except Exception as e:
            print("✗ Failed to initialize question generator: {e}")
            return False

        try:
            self.evaluator = GriceWiseEvaluator()
            print("✓ Evaluator initialized")
        except Exception as e:
            print("✗ Failed to initialize evaluator: {e}")
            return False

        print("All models ready!\n")
        return True

    def run(self):
        """Run the main REPL loop."""
        print("=" * 60)
        print("B-FQG Follow-up Question Generator REPL")
        print("=" * 60)
        print("Welcome to the interactive follow-up question generator!")
        print(
            "This tool generates cognitively scaffolded questions using Bloom's Taxonomy"
        )
        print("and evaluates them using Gricean-inspired metrics.\n")

        # Initialize models
        if not self.initialize_models():
            print("Failed to initialize models. Exiting.")
            return

        # Show help
        self.show_help()

        while True:
            try:
                # Get user input
                user_input = input("\n🔍 Enter a seed question or command: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        break
                else:
                    # Generate follow-up questions
                    self.generate_and_evaluate(user_input)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue

        # Save history if enabled
        if self.save_history and self.history:
            self.save_history_to_file()

    def handle_command(self, command: str) -> bool:
        """
        Handle special commands.

        Args:
            command: Command starting with '/'
        Returns:
            True to continue, False to exit
        """
        cmd = command.lower()

        if cmd in ["/help", "/h"]:
            self.show_help()
        elif cmd in ["/examples", "/ex"]:
            self.show_examples()
        elif cmd in ["/history", "/hist"]:
            self.show_history()
        elif cmd in ["/save", "/s"]:
            self.save_last_result()
        elif cmd in ["/batch", "/b"]:
            self.batch_generate()
        elif cmd in ["/quit", "/q", "/exit"]:
            return False
        else:
            print(f"Unknown command: {command}")
            print("Type /help for available commands")

        return True

    def show_help(self):
        """Display help information."""
        print("COMMANDS:")
        print("  /help, /h     - Show this help")
        print("  /examples, /ex - Show example seed questions")
        print("  /history, /hist - Show session history")
        print("  /save, /s     - Save last result to JSON file")
        print("  /batch, /b    - Enter batch mode with multiple seeds")
        print("  /quit, /q     - Exit the REPL")
        print()
        print("USAGE:")
        print("  Enter any seed question (Level 1 - Remember)")
        print("  Example: 'How do I adjust the climate control?'")
        print()
        print("The generator will create 5 follow-up questions:")
        print("  Level 2: Understand - Explain concepts")
        print("  Level 3: Apply - Use information practically")
        print("  Level 4: Analyze - Break down into components")
        print("  Level 5: Evaluate - Make judgments")
        print("  Level 6: Create - Generate new solutions")
        print()
        print("Evaluation metrics:")
        print("  Logical Consistency - How well it follows from context")
        print("  Informativeness - Amount of new information")
        print("  Relevance - Relatedness to conversation")
        print("  Clarity - Ease of understanding")

    def show_examples(self):
        """Show example seed questions."""
        examples = [
            "How do I make a phone call?",
            "How do I navigate to an address?",
            "How do I play music?",
            "How do I adjust the air conditioning?",
            "What is the fuel efficiency of this car?",
            "How do I activate the parking sensors?",
            "When should I check the tire pressure?",
            "How do I answer an incoming call?",
            "Where is the emergency brake?",
            "Can I use voice commands while driving?",
        ]

        print("Example seed questions (Level 1 - Remember):")
        for i, ex in enumerate(examples, 1):
            print(f"  {i}. {ex}")
        print("\nTry any of these or write your own!")

    def show_history(self):
        """Show the history of interactions."""
        if not self.history:
            print("No history yet. Generate some questions first!")
            return

        print(f"Session history ({len(self.history)} entries):")
        for i, entry in enumerate(self.history, 1):
            seed = entry["seed_question"]
            num_questions = len(entry["followups"])
            avg_score = (
                sum(q["aggregate_score"] for q in entry["evaluation"].values())
                / len(entry["evaluation"])
                if entry.get("evaluation")
                else 0.0
            )

            print(f"  {i}. {seed[:50]}{'...' if len(seed) > 50 else ''}")
            print(
                f"     Generated: {num_questions} questions, Avg score: {avg_score:.3f}"
            )
        print("\nUse '/save' to save the last result to a file.")

    def save_last_result(self):
        """Save the last result to a JSON file."""
        if not self.history:
            print("No results to save. Generate some questions first!")
            return

        try:
            filename = f"bfqg_result_{len(self.history)}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.history[-1], f, indent=2, ensure_ascii=False)

            print(f"✓ Saved last result to {filename}")

        except Exception as e:
            print(f"✗ Failed to save: {e}")

    def batch_generate(self):
        """Handle batch generation mode."""
        print("BATCH MODE")
        print("Enter multiple seed questions (one per line)")
        print("Enter an empty line to finish and process:")
        print()

        seeds = []
        while True:
            line = input("  Seed: ").strip()
            if not line:
                break
            seeds.append(line)

        if not seeds:
            print("No seeds entered. Returning to single mode.")
            return

        print(f"\nProcessing {len(seeds)} seed questions...\n")

        # Generate and evaluate
        results = self.generator.generate_multiple(seeds, show_progress=True)
        evaluations = self.evaluator.batch_evaluate(results, show_progress=True)

        # Show summary
        print("\n" + "=" * 60)
        print("BATCH RESULTS SUMMARY")
        print("=" * 60)

        for i, result in enumerate(evaluations, 1):
            seed = result["seed_question"]
            evaluation = result.get("evaluation", {})

            if "error" in result:
                print(f"{i}. {seed}")
                print(f"   ❌ Error: {result['error']}")
                continue

            # Calculate aggregate stats
            scores = []
            for level_data in evaluation.values():
                if isinstance(level_data, dict) and "aggregate_score" in level_data:
                    scores.append(level_data["aggregate_score"])

            if scores:
                avg_score = sum(scores) / len(scores)
                min_score = min(scores)
                max_score = max(scores)

                status = "✓" if avg_score > 0.6 else "⚠"
                print(f"{i}. {seed}")
                print(f"   {status} Generated {len(evaluation)} questions")
                print(
                    f"   Avg: {avg_score:.3f}, Min: {min_score:.3f}, Max: {max_score:.3f}"
                )
            else:
                print(f"{i}. {seed}")
                print("   ❌ No evaluation data")

        # Add to history
        self.history.extend(evaluations)

        print(f"\nBatch processing complete! {len(seeds)} seeds processed.")

    def generate_and_evaluate(self, seed_question: str):
        """Generate follow-up questions and evaluate them."""
        print(f"\n🔧 Processing: {seed_question}")
        print("-" * 60)

        try:
            # Generate questions
            followup_dict = self.generator.generate_followups(seed_question)

            if not followup_dict:
                print("❌ No questions generated. Try a different seed question.")
                return

            print(f"📝 Generated {len(followup_dict)} follow-up questions:\n")

            # Display questions by level
            for level in sorted(
                followup_dict.keys(), key=lambda x: int(x.split("_")[1])
            ):
                level_num = level.split("_")[1]
                question = followup_dict[level]
                level_name = {
                    "2": "Understand",
                    "3": "Apply",
                    "4": "Analyze",
                    "5": "Evaluate",
                    "6": "Create",
                }.get(level_num, level)

                print(f"Level {level_num} ({level_name}):")
                print(f"  {question}")

            print("\n" + "-" * 60)
            print("⚡ Evaluating questions...\n")

            # Evaluate
            evaluation = self.evaluator.evaluate_question_set(
                seed_question, followup_dict, verbose=True
            )

            # Show summary
            print("SUMMARY:")
            total_score = 0
            for level in sorted(evaluation.keys(), key=lambda x: int(x.split("_")[1])):
                score = evaluation[level]["aggregate_score"]
                total_score += score
                status = "🟢" if score > 0.7 else "🟡" if score > 0.5 else "🔴"
                print(f"  Level {level.split('_')[1]}: {status} {score:.3f}")

            avg_score = total_score / len(evaluation) if evaluation else 0.0
            overall_status = (
                "🟢 Excellent"
                if avg_score > 0.75
                else "🟡 Good"
                if avg_score > 0.5
                else "🔴 Needs improvement"
            )
            print(f"\n📊 Overall Score: {overall_status} ({avg_score:.3f})")

            # Save to history
            result = {
                "seed_question": seed_question,
                "followups": followup_dict,
                "evaluation": evaluation,
                "summary": {
                    "average_score": avg_score,
                    "question_count": len(followup_dict),
                    "levels_covered": sorted(followup_dict.keys()),
                },
            }
            self.history.append(result)

        except Exception as e:
            print(f"❌ Error processing question: {e}")
            import traceback

            traceback.print_exc()

    def save_history_to_file(self):
        """Save the entire session history to a file."""
        try:
            filename = "bfqg_session_history.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)

            print(f"Session history saved to {filename}")

        except Exception as e:
            print(f"Failed to save history: {e}")


def main():
    """Main entry point."""
    repl = FollowUpQuestionREPL()
    repl.run()


if __name__ == "__main__":
    main()
