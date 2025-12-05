---
base_model: google/flan-t5-large
library_name: peft
tags:
- base_model:adapter:google/flan-t5-large
- lora
- transformers
- bloom-taxonomy
- question-generation
---

# B-FQG Fine-Tuned Model

This model is a fine-tuned version of `google/flan-t5-large` on the **BloomsFQ** dataset. It is designed to generate follow-up questions for in-car assistants that progressively increase in cognitive complexity according to Bloom's Revised Taxonomy.

## Model Details

- **Base Model:** `google/flan-t5-large`
- **Adapter Type:** LoRA (Low-Rank Adaptation)
- **Task:** Question Generation based on Bloom's Taxonomy Levels
- **Language:** English

## Usage

You can use this model with the `peft` and `transformers` libraries.

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load base model and tokenizer
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load fine-tuned adapter
model = PeftModel.from_pretrained(base_model, "path/to/bfqg_finetuned")

# Inference Example
seed_question = "How do I adjust the air conditioning?"
level_desc = "Evaluate (Judge/critique)"

prompt = f"""Task: Generate a single follow-up question for an in-car assistant based on the seed question.

Now, generate a question for the following new task:
Seed: "{seed_question}"
Target Level: {level_desc}
Constraint: The question must strictly adhere to the target level and the specific seed topic. Output ONLY the question text.
Question:"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Details

### Training Data
The model was fine-tuned on the `BloomsFQ.csv` dataset, which contains seed questions mapped to follow-up questions at different Bloom's Taxonomy levels:
- Level 2: Understand
- Level 3: Apply
- Level 4: Analyze
- Level 5: Evaluate
- Level 6: Create

### Training Hyperparameters
- **Epochs:** 5
- **Batch Size:** 4 (Effective batch size 32 via gradient accumulation)
- **Learning Rate:** 1e-3
- **Optimizer:** AdamW
- **LoRA Config:** r=8, lora_alpha=32, dropout=0.1

## Framework Versions
- PEFT
- Transformers
- PyTorch
