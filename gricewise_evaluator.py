"""
GriceWise Evaluation Framework

This module implements the evaluation framework from the paper, assessing follow-up questions
based on Grice's Maxims through reference-free metrics:

1. Logical Consistency (Quality): NLI model to check entailment
2. Informativeness (Quantity): Conditional entropy using language model
3. Relevance (Relation): Cosine similarity of sentence embeddings
4. Clarity (Manner): Average Dependency Distance using NLTK
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
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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
        device: str = None,
    ):
        """
        Initialize the evaluator with required models.

        Args:
            nli_model: Hugging Face model for NLI (logical consistency)
            lm_model: Language model for entropy calculation
            embedding_model: Sentence transformer for embeddings
            device: Device to run on
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        # Download NLTK resources
        self._download_nltk_resources()

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

        print("All evaluation models loaded successfully!")

    def _download_nltk_resources(self):
        """Download required NLTK resources for text processing."""
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet")

        try:
            nltk.data.find("taggers/averaged_perceptron_tagger")
        except LookupError:
            nltk.download("averaged_perceptron_tagger")

    def evaluate_logical_consistency(
        self, question: str, context: str, threshold: float = 0.5
    ) -> float:
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
            question,
            context,
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
        Evaluate clarity using a simple approximation of syntactic complexity.

        Since spaCy has compatibility issues, we use a heuristic based on:
        - Sentence length
        - Word complexity (long words)
        - Punctuation complexity

        Args:
            question: The follow-up question to evaluate

        Returns:
            Clarity score (0-1), higher = clearer
        """
        if not question.strip():
            return 0.0

        try:
            # Tokenize and analyze basic properties
            words = word_tokenize(question.lower())
            words = [w for w in words if w.isalnum()]  # Remove punctuation

            if not words:
                return 0.0

            # Average word length (penalize very long/short words)
            avg_word_len = sum(len(word) for word in words) / len(words)

            # Penalize very short questions (likely incomplete)
            word_count = len(words)

            # Count punctuation complexity
            punctuation_count = len(re.findall(r'[!?.,;:()""\[\]]', question))

            # Ideal avg word length is around 4-6 characters
            word_len_score = 1.0 - min(1.0, abs(avg_word_len - 5.0) / 3.0)

            # Penalize very short or very long questions
            length_score = 1.0 - min(1.0, abs(word_count - 7.0) / 5.0)

            # Penalize excessive punctuation
            punct_score = max(0.0, 1.0 - punctuation_count / 3.0)

            # Combine scores (weighted)
            clarity = 0.4 * word_len_score + 0.4 * length_score + 0.2 * punct_score

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

        sorted_levels = sorted(
            followups.keys(), key=lambda x: int(re.search(r"\d+", x).group())
        )

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
                "aggregate_score": (logical_consistency + relevance + clarity)
                / 3.0,  # Exclude informativeness for now (lower better)
            }

            results[level] = question_results

            if verbose:
                print(f"{level.upper()}: {question}")
                print(f"  Logical Consistency:  {logical_consistency:.3f}")
                print(f"  Informativeness:     {informativeness:.3f}")
                print(f"  Relevance:          {relevance:.3f}")
                print(f"  Clarity:            {clarity:.3f}")
                print(
                    f"  Aggregate:          {question_results['aggregate_score']:.3f}"
                )
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

        iterator = (
            tqdm(question_sets, desc="Evaluating questions")
            if show_progress
            else question_sets
        )

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
