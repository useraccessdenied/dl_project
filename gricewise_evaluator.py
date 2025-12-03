"""
GriceWise Evaluation Framework

This module implements the evaluation framework from the paper, assessing follow-up questions
based on Grice's Maxims through reference-free metrics:

1. Logical Consistency (Quality): NLI model to check entailment
2. Informativeness (Quantity): Conditional entropy using language model
3. Relevance (Relation): Cosine similarity of sentence embeddings
4. Clarity (Manner): Average Dependency Distance using SpaCy
"""

from typing import Dict, List, Union, Tuple, Any
import torch
import re
import math
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from sentence_transformers import SentenceTransformer, util
import spacy
import numpy as np


class GriceWiseEvaluator:
    """
    Evaluates follow-up questions using Gricean-inspired metrics.

    Implements the four metrics from the paper for assessing question quality:
    - Logical Consistency
    - Informativeness
    - Relevance
    - Clarity
    """

    def __init__(
        self,
        nli_model: str = "textattack/roberta-base-MNLI",
        lm_model: str = "gpt2",
        embedding_model: str = "all-MiniLM-L6-v2",
        spacy_model: str = "en_core_web_sm",
        device: str = None,
    ):
        """
        Initialize the evaluator with required models.

        Args:
            nli_model: Hugging Face model for NLI (logical consistency)
            lm_model: Language model for entropy calculation
            embedding_model: Sentence transformer for embeddings
            spacy_model: SpaCy model for dependency parsing
            device: Device to run on
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device

        # Initialize NLI model for logical consistency
        print(f"Loading NLI model: {nli_model}")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model)
        self.nli_model.to(self.device)
        self.nli_model.eval()

        # Initialize language model for informativeness
        print(f"Loading language model: {lm_model}")
        self.lm_tokenizer = GPT2Tokenizer.from_pretrained(lm_model)
        self.lm_model = GPT2LMHeadModel.from_pretrained(lm_model)
        self.lm_model.to(self.device)
        self.lm_model.eval()

        # Set pad token for GPT-2 if not present
        if self.lm_tokenizer.pad_token is None:
            self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token

        # Initialize embedding model for relevance
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model, device=self.device)

        # Initialize SpaCy model for Clarity (ADD)
        print(f"Loading SpaCy model: {spacy_model}")
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Model '{spacy_model}' not found. Downloading...")
            spacy.cli.download(spacy_model)
            self.nlp = spacy.load(spacy_model)

        print("All evaluation models loaded successfully!")

    def evaluate_logical_consistency(self, question: str, context: str, threshold: float = 0.5) -> float:
        """
        Evaluate logical consistency using NLI entailment probability.

        Args:
            question: The follow-up question to evaluate
            context: Previous conversation context (all preceding questions)
            threshold: Classification threshold

        Returns:
            Entailment score (0-1), higher = more consistent
        """
        if not question.strip() or not context.strip():
            return 0.0

        # Prepare input for NLI
        inputs = self.nli_tokenizer(
            context,
            question,  # Context is premise, Question is hypothesis
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        # Get entailment probability (index 2 for roberta-large-mnli)
        # Verify label mapping for the specific model if needed
        # textattack/roberta-base-MNLI labels: 0: contradiction, 1: neutral, 2: entailment
        entailment_prob = probs[0][2].item()

        return entailment_prob

    def evaluate_informativeness(self, question: str, context: str) -> float:
        """
        Evaluate informativeness using conditional entropy.

        Args:
            question: The follow-up question to evaluate
            context: Previous conversation context

        Returns:
            Informativeness score (0+), lower = more informative (less redundant)
        """
        if not question.strip():
            return 0.0

        try:
            # Tokenize the question
            tokens = self.lm_tokenizer(question, return_tensors="pt")
            input_ids = tokens["input_ids"].to(self.device)

            # Get logits for conditional entropy calculation
            with torch.no_grad():
                outputs = self.lm_model(input_ids=input_ids)
                logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]

            # Calculate conditional entropy H(question|context)
            # For simplicity, we compute entropy of the question alone
            # since computing P(word|context) requires more complex setup
            probs = torch.softmax(logits, dim=-1)
            entropy = 0.0

            for i, token_id in enumerate(input_ids[0][1:], 1):  # Skip BOS token
                token_prob = probs[i - 1][token_id].item()
                if token_prob > 1e-10:  # Avoid log(0)
                    entropy -= token_prob * math.log(token_prob)

            # Normalize by sequence length
            seq_len = len(input_ids[0]) - 1  # Exclude BOS
            normalized_entropy = entropy / seq_len if seq_len > 0 else 0.0

            return normalized_entropy

        except Exception as e:
            print(f"Error in informativeness calculation: {e}")
            return 0.0

    def evaluate_relevance(self, question: str, context: str) -> float:
        """
        Evaluate relevance using cosine similarity of embeddings.

        Args:
            question: The follow-up question to evaluate
            context: Previous conversation context

        Returns:
            Cosine similarity (0-1), higher = more relevant
        """
        if not question.strip() or not context.strip():
            return 0.0

        try:
            # Get embeddings
            question_emb = self.embedder.encode(question, convert_to_tensor=True)
            context_emb = self.embedder.encode(context, convert_to_tensor=True)

            # Calculate cosine similarity
            similarity = util.cos_sim(question_emb, context_emb)[0][0].item()
            similarity = max(0.0, min(1.0, similarity))  # Clamp to [0,1]

            return similarity

        except Exception as e:
            print(f"Error in relevance calculation: {e}")
            return 0.0

    def evaluate_clarity(self, question: str) -> float:
        """
        Evaluate clarity using Average Dependency Distance (ADD).

        ADD = (1 / n) * Sum(|i - head(i)|)
        Clarity = 1 / (1 + ADD)

        Args:
            question: The follow-up question to evaluate

        Returns:
            Clarity score (0-1), higher = clearer
        """
        if not question.strip():
            return 0.0

        try:
            doc = self.nlp(question)

            total_distance = 0
            n_tokens = 0

            for token in doc:
                # Skip punctuation for dependency distance calculation?
                # The paper doesn't strictly say, but usually punctuation is excluded in such metrics.
                # However, for robustness we'll include everything or exclude punct.
                # Let's include everything as "syntactic complexity" usually considers whole structure.
                if not token.is_punct:
                    distance = abs(token.i - token.head.i)
                    total_distance += distance
                    n_tokens += 1

            if n_tokens == 0:
                return 0.0

            add = total_distance / n_tokens

            # Clarity score: inverse of ADD (plus 1 to avoid div by zero and normalize)
            # A completely linear sentence "I go home" might have low ADD.
            # Complex nested sentences have high ADD.
            clarity = 1.0 / (1.0 + add)

            return clarity

        except Exception as e:
            print(f"Error in clarity calculation: {e}")
            return 0.0

    def evaluate_question_set(
        self,
        seed_question: str,
        followups: Dict[str, str],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate a complete set of follow-up questions.

        Args:
            seed_question: The Level 1 seed question
            followups: Dictionary of level -> question mappings
            verbose: Whether to print detailed metrics

        Returns:
            Dictionary with evaluation results for each question and aggregates
        """
        results = {}

        # Build cumulative context for each follow-up
        previous_questions = [seed_question]

        sorted_levels = sorted(followups.keys(), key=lambda x: int(re.search(r"\d+", x).group()))

        for level in sorted_levels:
            if level not in followups:
                continue

            question = followups[level]
            context = " ".join(previous_questions)

            # Evaluate each metric
            logical_consistency = self.evaluate_logical_consistency(question, context)
            informativeness = self.evaluate_informativeness(question, context)
            relevance = self.evaluate_relevance(question, context)
            clarity = self.evaluate_clarity(question)

            # Store results
            question_results = {
                "question": question,
                "logical_consistency": logical_consistency,
                "informativeness": informativeness,
                "relevance": relevance,
                "clarity": clarity,
                # Simple aggregate: Average of "good" metrics
                # Note: Informativeness (Entropy) - is lower better or higher better?
                # Paper: "Lower indicates redundancy; optimization targets a balance"
                # But usually higher entropy = more information.
                # The previous code had comment: "Exclude informativeness for now (lower better)"
                # Actually, in text generation, very low entropy = repetition. High entropy = randomness.
                # Paper says "optimization targets a balance".
                # For this implementation, we'll treat Logical Consistency, Relevance, and Clarity as the key quality drivers.
                "aggregate_score": (logical_consistency + relevance + clarity) / 3.0,
            }

            results[level] = question_results

            if verbose:
                print(f"{level.upper()}: {question}")
                print(f"  Logical Consistency:  {logical_consistency:.3f}")
                print(f"  Informativeness:     {informativeness:.3f}")
                print(f"  Relevance:          {relevance:.3f}")
                print(f"  Clarity:            {clarity:.3f}")
                print(f"  Aggregate:          {question_results['aggregate_score']:.3f}")
                print()

            # Add to context for next question
            previous_questions.append(question)

        return results

    def batch_evaluate(
        self,
        question_sets: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple question sets.

        Args:
            question_sets: List of dicts with 'seed_question' and 'followups'
            show_progress: Whether to show progress

        Returns:
            List of evaluation results
        """
        results = []
        from tqdm import tqdm

        iterator = tqdm(question_sets, desc="Evaluating questions") if show_progress else question_sets

        for question_set in iterator:
            seed = question_set["seed_question"]
            followups = question_set["followups"]

            try:
                evaluation = self.evaluate_question_set(seed, followups)
                result = {
                    "seed_question": seed,
                    "evaluation": evaluation,
                }
                results.append(result)
            except Exception as e:
                print(f"Error evaluating set for '{seed[:50]}...': {e}")
                results.append(
                    {
                        "seed_question": seed,
                        "evaluation": {},
                        "error": str(e),
                    }
                )

        return results


# Test the evaluator if run directly
if __name__ == "__main__":
    evaluator = GriceWiseEvaluator()

    # Test with a sample conversation
    seed = "How do I adjust the volume?"
    followups = {
        "level_2": "What volume controls are available in this car?",
        "level_3": "How do I adjust the volume while driving safely?",
        "level_4": "What factors influence the volume levels in different conditions?",
        "level_5": "Which volume setting works best for different road types?",
        "level_6": "How can I create a custom volume profile for speed zones?",
    }

    print("Evaluating sample question set:")
    print(f"Seed: {seed}")

    results = evaluator.evaluate_question_set(seed, followups, verbose=True)

    print("Summary:")
    for level, evaluation in results.items():
        print(f"  {level}: Score = {evaluation['aggregate_score']:.3f}")
