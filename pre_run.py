#!/usr/bin/env python3
"""
Pre-run Setup Script for B-FQG

This script initializes all required resources and data for the B-FQG system,
including downloading models, NLTK resources, and verifying dependencies.
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path


class SetupManager:
    """
    Manages the setup of all required resources for B-FQG.
    """

    def __init__(self):
        self.required_models = [
            "google/flan-t5-base",  # For question generation
            "textattack/roberta-base-MNLI",  # For logical consistency
            "all-MiniLM-L6-v2",  # For sentence embeddings
            "gpt2",  # For informativeness calculation
        ]

        self.required_packages = [
            "torch",
            "transformers",
            "sentence_transformers",
            "nltk",
            "datasets",
            "tqdm",
        ]

        self.nltk_resources = [
            "punkt",
            "punkt_tab",
            "stopwords",
            "wordnet",
            "averaged_perceptron_tagger",
        ]

    def check_imports(self):
        """Check if all required packages can be imported."""
        print("🔍 Checking Python package imports...")

        failed_imports = []
        for package in self.required_packages:
            try:
                importlib.import_module(package)
                print(f"  ✓ {package}")
            except ImportError:
                failed_imports.append(package)
                print(f"  ✗ {package}")

        if failed_imports:
            print(f"\n❌ Missing packages: {', '.join(failed_imports)}")
            print("Run 'uv sync' to install missing dependencies.")
            return False

        print("✓ All packages imported successfully!\n")
        return True

    def download_models(self):
        """Download and cache required models."""
        print("📥 Downloading/caching machine learning models...")

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError:
            print("❌ Cannot download models without required packages.")
            return False

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        for model_name in self.required_models:
            try:
                print(f"  ⬇️  {model_name}")
                if model_name == "all-MiniLM-L6-v2":
                    # Sentence transformer
                    model = SentenceTransformer(model_name)
                elif model_name == "gpt2":
                    # Special handling for GPT-2 (different architecture)
                    from transformers import GPT2Tokenizer, GPT2LMHeadModel

                    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                    model = GPT2LMHeadModel.from_pretrained(model_name)
                elif "flan-t5" in model_name:
                    # T5 model
                    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                else:
                    # Regular transformer
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

                print(f"    ✓ {model_name}")

            except Exception as e:
                print(f"    ❌ Failed to download {model_name}: {e}")
                return False

        print("✓ All models downloaded successfully!\n")
        return True

    def setup_nltk_resources(self):
        """Download required NLTK resources."""
        print("📚 Downloading NLTK resources...")

        try:
            import nltk
        except ImportError:
            print("❌ Cannot download NLTK resources without nltk package.")
            return False

        for resource in self.nltk_resources:
            try:
                print(f"  ⬇️  {resource}")
                nltk.data.find(f"tokenizers/{resource}")
                print(f"    ✓ {resource}")
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                    print(f"    ✓ {resource}")
                except Exception as e:
                    print(f"    ❌ Failed to download {resource}: {e}")
                    return False

        print("✓ All NLTK resources downloaded successfully!\n")
        return True

    def test_components(self):
        """Test that all components can be initialized."""
        print("🧪 Testing component initialization...")

        try:
            # Test BFG generator
            from bfqg_generator import BFGQuestionGenerator

            print("  ✓ BFG Generator can be imported")

            # Test evaluator (but don't initialize models yet)
            from gricewise_evaluator import GriceWiseEvaluator

            print("  ✓ GriceWise Evaluator can be imported")

            # Test REPL imports
            from repl import FollowUpQuestionREPL

            print("  ✓ REPL can be imported")

        except ImportError as e:
            print(f"❌ Import test failed: {e}")
            return False

        print("✓ All components imported successfully!\n")
        return True

    def create_sample_data(self):
        """Create sample data directory and files."""
        print("📁 Setting up sample data...")

        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        # Sample seed questions
        sample_seeds = [
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

        seeds_file = data_dir / "sample_seeds.json"
        try:
            import json

            with open(seeds_file, "w", encoding="utf-8") as f:
                json.dump(sample_seeds, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Created {seeds_file}")
        except Exception as e:
            print(f"  ❌ Failed to create sample data: {e}")
            return False

        print("✓ Sample data created successfully!\n")
        return True

    def main(self):
        """Run the complete setup process."""
        print("=" * 60)
        print("🚀 B-FQG Pre-run Setup")
        print("=" * 60)
        print("This script will initialize all required resources.")
        print("This may take several minutes to download models...\n")

        success = True

        # Step 1: Check imports
        if not self.check_imports():
            success = False

        # Step 2: Download models
        if success and not self.download_models():
            success = False

        # Step 3: Setup NLTK
        if success and not self.setup_nltk_resources():
            success = False

        # Step 4: Test components
        if success and not self.test_components():
            success = False

        # Step 5: Create sample data
        if success and not self.create_sample_data():
            success = False

        # Final status
        print("=" * 60)
        if success:
            print("🎉 Setup completed successfully!")
            print("\nYou can now run:")
            print("  uv run python repl.py     # Interactive REPL")
            print("  uv run python main.py     # Batch processing")
        else:
            print("❌ Setup failed. Please fix the errors above and try again.")
            sys.exit(1)
        print("=" * 60)


def main():
    """Entry point."""
    setup = SetupManager()
    setup.main()


if __name__ == "__main__":
    main()
