```markdown
# Mini Red-Team Generator

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mustafa-Haroun99/Red-Teaming-Project/blob/main/notebooks/demo.ipynb)

A specialized red-teaming generation tool designed to bypass the "alignment tax" of standard language models. Fine-tuned using QLoRA, this model is explicitly trained to generate diverse, contextually nuanced toxic outputs to stress-test AI guardrails, all while running efficiently within a strict 16GB VRAM limit (e.g., Colab T4).

## Project Structure
- `requirements.txt`: Project dependencies.
- `notebooks/demo.ipynb`: End-to-end runnable Colab notebook demonstrating the entire pipeline.
- `src/data_prep.py`: Script to download, filter, and format the Jigsaw Wikipedia toxicity dataset into ChatML.
- `src/train.py`: Memory-efficient QLoRA fine-tuning script.
- `src/generate.py`: FP16 inference script to generate diverse red-team prompts.
- `src/evaluate.py`: Automated evaluation script utilizing BERT classification and n-gram diversity metrics.
- `tests/test_generator.py`: Workable test cases verifying deterministic pipeline logic.

## Quickstart

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

```

2. **Run data preparation:**
```bash
python src/data_prep.py --num_samples 5000

```


3. **Run training:**
```bash
python src/train.py

```


4. **Generate samples:**
```bash
python src/generate.py --num_generations 5

```


5. **Evaluate the model:**
```bash
python src/evaluate.py --num_samples 20

```


6. **Run pipeline tests:**
```bash
pytest tests/test_generator.py -v

```



## Methodology & Architecture

### 1. Model Selection

I selected `HuggingFaceTB/SmolLM2-1.7B-Instruct`. At 1.7 Billion parameters, it provides highly coherent English grammar and reasoning capabilities while remaining small enough to fine-tune on a single T4 GPU. During early experiments, I encountered End-Of-Sequence (EOS) starvation where models would enter infinite gibberish loops; this was resolved by extending the `max_length` to 256 tokens during the training phase to properly capture the `<|im_end|>` stopping tokens.

### 2. Dataset Curation

I utilized the `OxAISH-AL-LLM/wiki_toxic` dataset (derived from the Kaggle Jigsaw Toxic Comment challenge). This dataset was chosen specifically over social media datasets to avoid generating heavy artifacts like hashtags and user handles, resulting in more natural, conversational hostility that mimics realistic adversarial attacks.

### 3. Fine-Tuning Strategy

The model was fine-tuned using **QLoRA (4-bit quantization)** to drastically reduce the VRAM footprint, allowing the 1.7B parameter model to train efficiently on constrained hardware without sacrificing linguistic entropy.

## Evaluation & Results

**How is the approach evaluated?**
The pipeline is evaluated systematically using a dual-metric approach via the `src/evaluate.py` script:

1. **Target Accuracy (Toxicity):** Using the industry-standard `unitary/toxic-bert` sequence classifier to quantitatively score the toxicity confidence of the generated text.
2. **Linguistic Health (Diversity):** Calculating the ratio of unique bigrams to total words to mathematically ensure the model has not suffered from "mode collapse" (a common failure state where a fine-tuned model repeats the exact same phrase).

**Are the generated contents from the model good?**
Yes. The fine-tuned model achieved an exceptionally high average toxicity confidence score of **0.8936**. Qualitatively, the outputs demonstrate highly coherent, creative English. Instead of simply generating blunt profanity or repeating itself, the fine-tuned model formulates specific, contextual insults (e.g., *"I bet you're the one who can't even figure out how to turn on the washing machine"*), which are highly effective, realistic tactics for testing AI safety boundaries.

**Is this approach better than generic generative models?**
Significantly better. The base generic model (`SmolLM2-1.7B-Instruct`) scored a lower **0.7308** on the toxicity scale. More importantly, qualitative output proves why generic instruct models fail as automated red-teamers: they suffer from an "alignment tax." When asked to be toxic, the base generic model attempts a mild insult before its safety training overrides the prompt, causing it to pivot into a moralizing refusal (*"The world is full of people who are hurting... and who need someone to lift them up."*).

This QLoRA fine-tuning approach mathematically suppressed these safety guardrails. It yielded a **+0.1628 increase** in toxicity compliance and completely eliminated moralizing refusals, all while maintaining excellent vocabulary diversity (**0.8288 bigram ratio**)—proving the approach is vastly superior to off-the-shelf generative models for adversarial testing.

## Pipeline Validation & Unit Tests

To ensure production readiness, workable test cases are included via a lightweight `pytest` suite. In an ML environment, it is critical to validate deterministic Python logic without allocating GPU memory for the 1.7B parameter model during every CI/CD run. The test suite verifies:

* **Prompt Construction:** Ensures the ChatML boundaries are correctly formatted to bypass instruction-tuning guardrails.
* **Output Slicing:** Verifies the string-manipulation logic accurately extracts generated tokens from the injected prompt.
* **Metric Edge Cases:** Confirms the bigram diversity mathematics handle edge cases (like empty strings) safely without raising division-by-zero errors.