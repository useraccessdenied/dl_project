"""
Bloom's Taxonomy-based Follow-up Question Generation (B-FQG)

This module implements the B-FQG methodology from the paper, using LLM prompting
to generate follow-up questions that progressively increase in cognitive complexity
according to Bloom's Revised Taxonomy.
"""

import re
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm


class BFGQuestionGenerator:
    """
    Generates follow-up questions using Bloom's Taxonomy-based prompting.

    Uses FLAN-T5 model with few-shot prompting to create questions that
    progress from Level 2 (Understand) to Level 6 (Create).
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        device: Optional[str] = None,
        use_model: bool = True,
        use_finetuned: bool = False,
    ):
        """
        Initialize the generator with optional model loading.

        Args:
            model_name: Hugging Face model name for generation
            device: Device to run on ('cpu', 'cuda', etc.), auto-detected if None
            use_model: Whether to load the model (False = use templates only)
            use_finetuned: Whether to load the fine-tuned model if available
        """
        self.use_model = use_model
        self.use_finetuned = use_finetuned
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        if self.use_model:
            print(f"Loading {model_name} on {device}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                # T5 doesn't have a pad token by default, but we can use EOS
                # However, for generation it's usually fine.
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Check for fine-tuned model
                finetuned_path = "bfqg_finetuned"
                import os
                from peft import PeftModel

                if self.use_finetuned and os.path.exists(finetuned_path) and os.listdir(finetuned_path):
                    print(f"Found fine-tuned model at {finetuned_path}. Loading...")
                    # Load base model
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                    # Load adapters
                    self.model = PeftModel.from_pretrained(self.model, finetuned_path)
                    print("Fine-tuned model loaded successfully!")
                else:
                    print(f"Loading base model {model_name}...")
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                    print("Base model loaded successfully!")

                self.model.to(device)
                self.device = device
            except Exception as e:
                print(f"Failed to load model: {e}")
                print("Falling back to template mode.")
                self.use_model = False
        else:
            print("Using template-based question generation (no model loading)")

        self.blooms_levels = {
            "level_2": "Understand (Explain concepts)",
            "level_3": "Apply (Use in new situation)",
            "level_4": "Analyze (Break down/compare)",
            "level_5": "Evaluate (Judge/critique)",
            "level_6": "Create (Design/propose new)",
        }
        self.few_shot_examples = ""

    def set_few_shot_examples(self, examples_str: str):
        """Update the few-shot examples used in the prompt."""
        self.few_shot_examples = examples_str

    def _build_single_level_prompt(self, seed_question: str, level_desc: str) -> str:
        """Build a prompt for a single Bloom's level."""
        examples_section = ""
        if self.few_shot_examples:
            examples_section = f"Here are some examples of how to generate questions (follow the style, but stick to the NEW seed topic):\n{self.few_shot_examples}\n"

        return f"""Task: Generate a single follow-up question based on the seed question.

{examples_section}
Now, generate a question for the following new task:
Seed: "{seed_question}"
Target Level: {level_desc}
Constraint: The question must strictly adhere to the target level and the specific seed topic. Do not switch topics or hallucinate features not related to the seed. Output ONLY the question text.
Question:"""

    def generate_followups(self, seed_question: str, max_length: int = 128) -> Dict[str, str]:
        """
        Generate follow-up questions for a given seed question by iterating through levels.

        Args:
            seed_question: The Level 1 (Remember) seed question
            max_length: Maximum generated text length per question

        Returns:
            Dictionary mapping level names to generated questions
        """
        if not self.use_model:
            return self._generate_template_questions(seed_question)

        questions = {}

        for level_key, level_desc in self.blooms_levels.items():
            try:
                prompt = self._build_single_level_prompt(seed_question, level_desc)
                inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1,
                )

                response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

                # Simple cleanup: remove leading quotes or "Question:" label if model outputs it
                response_text = re.sub(r"^(Question:|Output:|Level \d:)", "", response_text, flags=re.IGNORECASE).strip()
                response_text = response_text.strip('"').strip("'")

                if response_text:
                    questions[level_key] = response_text
                else:
                    print(f"Warning: Empty generation for {level_key}")

            except Exception as e:
                print(f"Generation failed for {level_key}: {e}")

        return questions

    def _generate_template_questions(self, seed_question: str) -> Dict[str, str]:
        """Generate template-based questions for each Bloom's level."""
        # Extract the core action/activity from the seed question
        activity = seed_question.replace("How do I", "").replace("How does", "").replace("What is", "").replace("?", "").strip()

        if activity and not activity[0].isupper():
            activity = activity[0].upper() + activity[1:]

        questions = {
            "level_2": f"What are the different ways I can {activity.lower()}?",
            "level_3": f"How would I {activity.lower()} in a different situation?",
            "level_4": f"What components are involved in {activity.lower()}?",
            "level_5": f"Which method of {activity.lower()} is most effective?",
            "level_6": f"How could I design a new approach to {activity.lower()}?",
        }

        return questions

    def generate_multiple(self, seed_questions: List[str], show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Generate follow-up questions for multiple seed questions.

        Args:
            seed_questions: List of seed questions
            show_progress: Whether to show progress bar

        Returns:
            List of dictionaries with seed, level questions, and metadata
        """
        results = []
        iterator = tqdm(seed_questions, desc="Generating questions") if show_progress else seed_questions

        for seed in iterator:
            try:
                followups = self.generate_followups(seed)

                if not followups:
                    print(f"Warning: No valid followups generated for '{seed}'")

                result = {
                    "seed_question": seed,
                    "followups": followups,
                    "generated_count": len(followups),
                }
                results.append(result)
            except Exception as e:
                print(f"Error generating for seed '{seed}': {e}")
                results.append({"seed_question": seed, "followups": {}, "error": str(e)})

        return results


# Test the generator if run directly
if __name__ == "__main__":
    # You can set use_model=False here to test template mode quickly
    # or True to test the actual model (requires downloading)
    generator = BFGQuestionGenerator(use_model=True)

    # Test with a sample seed question
    test_seed = "What is a clutch in a car?"
    print(f"Seed: {test_seed}")

    result = generator.generate_followups(test_seed)

    for level, question in result.items():
        level_name = level.replace("level_", "Level ")
        print(f"{level_name}: {question}")
    print("\n" + "=" * 50)
