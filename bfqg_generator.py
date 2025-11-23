"""
Bloom's Taxonomy-based Follow-up Question Generator (B-FQG)

This module implements the B-FQG methodology from the paper, using LLM prompting
to generate follow-up questions that progressively increase in cognitive complexity
according to Bloom's Revised Taxonomy.
"""

import re
from typing import List, Dict, Any, Optional
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm


class BFGQuestionGenerator:
    """
    Generates follow-up questions using Bloom's Taxonomy-based prompting.

    Uses FLAN-T5 model with few-shot prompting to create questions that
    progress from Level 2 (Understand) to Level 6 (Create).
    """

    def __init__(
        self, model_name: str = "google/flan-t5-base", device: Optional[str] = None
    ):
        """
        Initialize the generator with a T5 model.

        Args:
            model_name: Hugging Face model name, default is FLAN-T5 base
            device: Device to run on ('cpu', 'cuda', etc.), auto-detected if None
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading {model_name} on {device}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        self.device = device
        print("Model loaded successfully!")

        # Few-shot examples - can be expanded or loaded from file
        self.few_shot_examples = self._get_default_examples()

    def _get_default_examples(self) -> str:
        """Get the few-shot prompt examples aligned with the paper's prompt format."""
        return """
Example 1: Seed question - "How do I make a phone call?"
Level 2 (Understand): "What are the different ways I can make a call?"
Level 3 (Apply): "How does the call-making process differ from my previous car model?"
Level 4 (Analyze): "What advantages does the car's built-in calling system have over my phone?"
Level 5 (Evaluate): "How does the car's calling system integrate with my phone's contacts and how does it affect call quality?"
Level 6 (Create): "How can I use the call-making feature to improve safety while driving?"

Example 2: Seed question - "How do I play music?"
Level 2 (Understand): "What types of music sources are available?"
Level 3 (Apply): "How do I switch between different music apps?"
Level 4 (Analyze): "What factors affect audio quality in different music sources?"
Level 5 (Evaluate): "Which music source works best for different driving conditions?"
Level 6 (Create): "How can I set up a personalized playlist integration?"
"""

    def _build_prompt(self, seed_question: str) -> str:
        """Build the complete prompt with examples and seed question."""
        prompt = f"""
You are generating follow-up questions for a car driver to ask an in-car AI assistant.
Questions should assess the AI's understanding of car features and increase in cognitive complexity.
Seed question: "{seed_question}"

{
            f'''Few-shot examples:
{self.few_shot_examples}
'''
            if self.few_shot_examples.strip()
            else ""
        }

Generate exactly 5 follow-up questions for Levels 2-6 of Bloom's Taxonomy:
- Level 2 (Understand): Explain ideas or concepts
- Level 3 (Apply): Use information in new situations
- Level 4 (Analyze): Break information into parts and relationships
- Level 5 (Evaluate): Make judgments based on criteria
- Level 6 (Create): Put elements together to form new ideas

Output format:
Level 2: [question]
Level 3: [question]
Level 4: [question]
Level 5: [question]
Level 6: [question]
"""
        return prompt.strip()

    def generate_followups(
        self, seed_question: str, max_length: int = 512
    ) -> Dict[str, str]:
        """
        Generate follow-up questions for a given seed question.

        Args:
            seed_question: The Level 1 (Remember) seed question
            max_length: Maximum generated text length

        Returns:
            Dictionary mapping level names to generated questions
        """
        # Build the prompt
        prompt = self._build_prompt(seed_question)

        # Tokenize and generate
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                do_sample=True,
                temperature=0.7,
            )

        # Decode the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse the response to extract questions
        return self._parse_generated_questions(response)

    def _parse_generated_questions(self, response: str) -> Dict[str, str]:
        """Parse the generated text to extract questions for each level."""
        lines = response.strip().split("\n")
        questions = {}

        level_patterns = [
            (r"Level 2:\s*(.+)", "level_2"),
            (r"Level 3:\s*(.+)", "level_3"),
            (r"Level 4:\s*(.+)", "level_4"),
            (r"Level 5:\s*(.+)", "level_5"),
            (r"Level 6:\s*(.+)", "level_6"),
        ]

        for line in lines:
            line = line.strip()
            for pattern, level in level_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    questions[level] = match.group(1).strip()
                    break

        return questions

    def generate_multiple(
        self, seed_questions: List[str], show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate follow-up questions for multiple seed questions.

        Args:
            seed_questions: List of seed questions
            show_progress: Whether to show progress bar

        Returns:
            List of dictionaries with seed, level questions, and metadata
        """
        results = []
        iterator = (
            tqdm(seed_questions, desc="Generating questions")
            if show_progress
            else seed_questions
        )

        for seed in iterator:
            try:
                followups = self.generate_followups(seed)
                result = {
                    "seed_question": seed,
                    "followups": followups,
                    "generated_count": len(followups),
                }
                results.append(result)
            except Exception as e:
                print(f"Error generating for seed '{seed}': {e}")
                results.append(
                    {"seed_question": seed, "followups": {}, "error": str(e)}
                )

        return results


# Test the generator if run directly
if __name__ == "__main__":
    generator = BFGQuestionGenerator()

    # Test with a sample seed question
    test_seed = "How do I adjust the air conditioning?"
    print(f"Seed: {test_seed}")

    result = generator.generate_followups(test_seed)

    for level, question in result.items():
        level_name = level.replace("level_", "Level ")
        print(f"{level_name}: {question}")
    print("\n" + "=" * 50)

    # Test multiple seeds
    seeds = [
        "How do I make a phone call?",
        "How do I navigate to an address?",
        "How do I play music?",
    ]

    results = generator.generate_multiple(seeds)
    for i, result in enumerate(results):
        print(f"\nSeed {i + 1}: {result['seed_question']}")
        for level, q in result["followups"].items():
            print(f"  {level}: {q}")
