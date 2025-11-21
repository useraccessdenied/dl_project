#!/usr/bin/env python3
"""
T5 (Text-To-Text Transformer) Base Model Testing Script

This script demonstrates various capabilities of the Google T5 base model
including translation, summarization, question-answering, and general text generation.
"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


class T5Tester:
    def __init__(self, model_name="t5-base"):
        """
        Initialize the T5 model and tokenizer.

        Args:
            model_name (str): Name of the T5 model to load (default: "t5-base")
        """
        print(f"Loading {model_name} model and tokenizer...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        print("Model and tokenizer loaded successfully!")

    def generate_text(self, input_text, max_length=50, num_beams=4):
        """
        Generate text from a given input using the T5 model.

        Args:
            input_text (str): The input text with task prefix
            max_length (int): Maximum length of generated text
            num_beams (int): Number of beams for beam search

        Returns:
            str: Generated text
        """
        # Prepare input
        inputs = self.tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True
        )

        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        # Decode and return
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def translate_to_french(self, english_text):
        """Translate English text to French."""
        input_text = f"translate English to French: {english_text}"
        return self.generate_text(input_text)

    def translate_to_german(self, english_text):
        """Translate English text to German."""
        input_text = f"translate English to German: {english_text}"
        return self.generate_text(input_text)

    def translate_to_hindi(self, english_text):
        """Translate English text to Hindi."""
        input_text = f"translate English to Hindi: {english_text}"
        return self.generate_text(input_text)

    def summarize(self, text):
        """Summarize a given text."""
        input_text = f"summarize: {text}"
        return self.generate_text(input_text)

    def answer_question(self, context, question):
        """Answer a question based on given context."""
        input_text = f"question: {question} context: {context}"
        return self.generate_text(input_text, max_length=100)

    def generate_story_prompt(self, prompt):
        """Generate a story continuation from a prompt."""
        input_text = f"generate story: {prompt}"
        return self.generate_text(input_text, max_length=200)


def run_tests():
    """Run comprehensive tests of T5 capabilities."""

    print("=" * 60)
    print("T5 BASE MODEL TESTING SCRIPT")
    print("=" * 60)

    # Initialize T5 tester
    tester = T5Tester()

    print("\n" + "=" * 40)
    print("TEST 1: TRANSLATION (English to French)")
    print("=" * 40)

    test_sentences = [
        "Hello, how are you?",
        "The cat sat on the mat.",
        "I love machine learning.",
    ]

    for sentence in test_sentences:
        translation = tester.translate_to_french(sentence)
        print(f"English: {sentence}")
        print(f"French:  {translation}")
        print("-" * 50)

    print("\n" + "=" * 40)
    print("TEST 2: TRANSLATION (English to German)")
    print("=" * 40)

    for sentence in test_sentences:
        translation = tester.translate_to_german(sentence)
        print(f"English: {sentence}")
        print(f"German:  {translation}")
        print("-" * 50)

    print("\n" + "=" * 40)
    print("TEST 3: TRANSLATION (English to Hindi)")
    print("=" * 40)

    for sentence in test_sentences:
        translation = tester.translate_to_hindi(sentence)
        print(f"English: {sentence}")
        print(f"Hindi:  {translation}")
        print("-" * 50)

    print("\n" + "=" * 40)
    print("TEST 3: TEXT SUMMARIZATION")
    print("=" * 40)

    text_to_summarize = """
    Natural language processing (NLP) is a subfield of artificial intelligence (AI)
    that focuses on the interaction between computers and humans through natural language.
    The ultimate goal of NLP is to read, decipher, understand, and make sense of human language
    in a manner that is valuable. Modern deep learning techniques have revolutionized NLP,
    enabling applications such as machine translation, sentiment analysis, and chatbots.
    """

    summary = tester.summarize(text_to_summarize)
    print(f"Original text: {text_to_summarize.strip()}")
    print(f"Summary: {summary}")
    print("-" * 50)

    print("\n" + "=" * 40)
    print("TEST 4: QUESTION ANSWERING")
    print("=" * 40)

    context = """
    Paris is the capital and most populous city of France. It is located in the north-central
    part of the country along the Seine River. Paris has been one of Europe's major cities
    for more than a thousand years, with its earliest settlement dating back to around 250 BC.
    The city is known for landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum.
    """

    questions = [
        "What is the capital of France?",
        "Where is Paris located?",
        "What famous landmarks does Paris have?",
    ]

    for question in questions:
        answer = tester.answer_question(context, question)
        print(f"Question: {question}")
        print(f"Answer:   {answer}")
        print("-" * 50)

    print("\n" + "=" * 40)
    print("TEST 5: STORY GENERATION")
    print("=" * 40)

    prompts = [
        "Once upon a time, in a magical forest,",
        "In the year 2050, artificial intelligence",
    ]

    for prompt in prompts:
        story = tester.generate_story_prompt(prompt)
        print(f"Prompt: {prompt}")
        print(f"Story:  {story}")
        print("-" * 50)

    print("\n" + "=" * 40)
    print("MODEL INFORMATION")
    print("=" * 40)

    # Display model info
    total_params = sum(p.numel() for p in tester.model.parameters())
    print(f"Model: T5-Base")
    print(f"Total parameters: {total_params:,}")
    model_size_mb = total_params * 4 / 1024 / 1024
    print(f"Model size: {model_size_mb:.2f} MB")

    # Test device (CPU/GPU)
    device = next(tester.model.parameters()).device
    print(f"Device: {device}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
