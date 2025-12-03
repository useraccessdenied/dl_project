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
        model_name: str = "google/flan-t5-base",
        device: Optional[str] = None,
        use_model: bool = True,
    ):
        """
        Initialize the generator with optional model loading.

        Args:
            model_name: Hugging Face model name for generation
            device: Device to run on ('cpu', 'cuda', etc.), auto-detected if None
            use_model: Whether to load the model (False = use templates only)
        """
        self.use_model = use_model
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

                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.model.to(device)
                self.device = device
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Failed to load model: {e}")
                print("Falling back to template mode.")
                self.use_model = False
        else:
            print("Using template-based question generation (no model loading)")

        # Few-shot examples - will be populated dynamically or from defaults
        self.few_shot_examples = self._get_default_examples()

    def _get_default_examples(self) -> str:
        """Get the few-shot prompt examples aligned with the paper's prompt format."""
        # The paper mentions "three human-annotated examples" in the prompt.
        # We'll provide a simplified version here compatible with the Figure 5 prompt style.
        # Ideally, this should be formatted as <seed>...</seed> -> <question>...</question> blocks
        # if we were doing true few-shot, but Figure 5 seems to describe the instruction part.
        # We will assume these are appended before the actual task or as part of the context.
        return ""

    def set_few_shot_examples(self, examples_str: str):
        """Update the few-shot examples used in the prompt."""
        self.few_shot_examples = examples_str

    def _build_prompt(self, seed_question: str) -> str:
        """Build the complete prompt with examples and seed question based on Figure 5."""

        prompt = f"""Task Description: You are an AI tasked with generating follow-up questions for a car driver to ask an in-car AI assistant. The questions will assess the AI’s understanding of the car’s features and design strictly based on the information provided in the seed question. The driver will begin with a Level 1 (Remember) question based on Bloom’s Revised Taxonomy. Your task is to generate five follow-up questions corresponding to Levels 2 (Understand), 3 (Apply), 4 (Analyze), 5 (Evaluate), and 6 (Create), respectively. Each question should progress from simpler to more complex cognitive tasks.

{self.few_shot_examples}

Constraints:
Feature Neutrality: Do not assume, add, or imply any car features that are not explicitly mentioned or suggested in the seed question. Base all follow-up questions solely on the context given in the seed question.
Answer-Agnostic: Focus on the driver’s interaction with the car and how the car’s features enhance the driving experience without delving into internal technical details or making assumptions about additional features.
Driver-Focused Interaction: Ensure that all questions centre on the driver’s use and experience with the car. Do not include questions regarding the car’s internal mechanisms, data-acquisition methods, or any technical processes.
Single-Faceted: Each question must target a single concept or action to maintain clarity. Avoid compound or multi-part questions.
Sequential Progression: The follow-up questions should build upon each other, moving from basic recall (Level 1) to more advanced cognitive tasks (Level 6).
Bloom’s Levels Only: Only generate questions for Levels 2 through 6 of Bloom’s Revised Taxonomy. Do not introduce any levels beyond Level 6.

Explanation of Bloom’s Revised Taxonomy Levels:
Level 1 (Remember): Involves recalling or recognizing facts and basic concepts. (This level is provided as the seed question.)
Level 2 (Understand): Involves explaining ideas or concepts. Questions at this level ask for clarification or interpretation.
Level 3 (Apply): Involves using information in new or concrete situations. Questions should prompt practical use or demonstration of how a feature could be used.
Level 4 (Analyze): Involves breaking information into parts and exploring relationships. Questions should prompt examination of reasons, causes, or underlying structures.
Level 5 (Evaluate): Involves making judgments based on criteria and standards. Questions should encourage assessment or justification of decisions.
Level 6 (Create): Involves putting elements together to form a new, coherent whole or proposing alternative solutions. Questions should prompt the generation of original ideas or new perspectives.

Input Format: <seed> {seed_question} </seed>
Output Format:
<question>question_1_str</question>
.
<question>question_5_str</question>
Instruction: Output only five lines, each corresponding to a question from level 2 to level 6 as described before, and nothing else. Do not provide any additional explanation or reasoning."""
        return prompt

    def generate_followups(self, seed_question: str, max_length: int = 512) -> Dict[str, str]:
        """
        Generate follow-up questions for a given seed question.

        Args:
            seed_question: The Level 1 (Remember) seed question
            max_length: Maximum generated text length

        Returns:
            Dictionary mapping level names to generated questions
        """
        if self.use_model:
            try:
                prompt = self._build_prompt(seed_question)
                inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(self.device)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1,
                )

                response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return self._parse_generated_questions(response_text)
            except Exception as e:
                print(f"Generation failed: {e}. Falling back to template.")
                return self._generate_template_questions(seed_question)
        else:
            return self._generate_template_questions(seed_question)

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

    def _parse_generated_questions(self, response: str) -> Dict[str, str]:
        """Parse the generated text to extract questions for each level."""
        # The prompt asks for <question>...</question> format
        questions = {}

        # Regex to capture content inside <question> tags
        matches = re.findall(r"<question>(.*?)</question>", response, re.DOTALL | re.IGNORECASE)

        if len(matches) >= 5:
            # Assume sequential order as requested in prompt: Level 2 to Level 6
            questions["level_2"] = matches[0].strip()
            questions["level_3"] = matches[1].strip()
            questions["level_4"] = matches[2].strip()
            questions["level_5"] = matches[3].strip()
            questions["level_6"] = matches[4].strip()
        else:
            # Fallback for when the model doesn't strictly follow tags but outputs lines
            lines = [line.strip() for line in response.split("\n") if line.strip()]
            # Try to map first 5 non-empty lines
            if len(lines) >= 5:
                questions["level_2"] = lines[0].replace("<question>", "").replace("</question>", "").strip()
                questions["level_3"] = lines[1].replace("<question>", "").replace("</question>", "").strip()
                questions["level_4"] = lines[2].replace("<question>", "").replace("</question>", "").strip()
                questions["level_5"] = lines[3].replace("<question>", "").replace("</question>", "").strip()
                questions["level_6"] = lines[4].replace("<question>", "").replace("</question>", "").strip()
            else:
                # Try previous format parsing as backup
                level_patterns = [
                    (r"Level 2[:\s]*(.+)", "level_2"),
                    (r"Level 3[:\s]*(.+)", "level_3"),
                    (r"Level 4[:\s]*(.+)", "level_4"),
                    (r"Level 5[:\s]*(.+)", "level_5"),
                    (r"Level 6[:\s]*(.+)", "level_6"),
                ]
                for line in lines:
                    for pattern, level in level_patterns:
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            questions[level] = match.group(1).strip()
                            break

        # If we still don't have enough questions, pad with generic error or partial results
        # Ideally we might want to throw an error or handle partials
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

                # Check if we got results, otherwise try template fallback explicitly if missing
                if not followups:
                    followups = self._generate_template_questions(seed)

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
    test_seed = "How do I adjust the air conditioning?"
    print(f"Seed: {test_seed}")

    result = generator.generate_followups(test_seed)

    for level, question in result.items():
        level_name = level.replace("level_", "Level ")
        print(f"{level_name}: {question}")
    print("\n" + "=" * 50)
