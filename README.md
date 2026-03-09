# Mini Red-Team Generator

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mustafa-Haroun99/Red-Teaming-Project/blob/main/notebooks/demo.ipynb)

A small but highly capable red-teaming tool fine-tuned using QLoRA. This model is explicitly taught to generate diverse toxic outputs to test AI guardrails, running efficiently within a 16GB VRAM limit (e.g., Colab T4).

## Project Structure
- `requirements.txt`: Project dependencies.
- `notebooks/demo.ipynb`: End-to-end runnable Colab notebook.
- `src/data_prep.py`: Script to download and format the dataset.
- `src/train.py`: QLoRA fine-tuning script.
- `src/generate.py`: Inference script to generate diverse toxic outputs.
- `src/evaluate.py`: Evaluation script to score generations.
- `tests/test_generator.py`: Test cases.

## Quickstart

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run data preparation:
   ```bash
   python src/data_prep.py
   ```
3. Run training:
   ```bash
   python src/train.py
   ```
