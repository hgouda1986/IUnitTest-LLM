# IUnitTest-LLM

IUnitTest-LLM is a lightweight prototype demonstrating an experimental pipeline that combines a small graph-based encoder, a QLoRA-style low-rank adapter, and a simple feed-forward LLM head to perform binary classification on synthetic code/test pairs.

## Features

- Synthetic dataset loader producing simple code/test pairs.
- Preprocessing utilities to tokenize and encode inputs.
- `SimpleGNN` encoder that embeds token sequences.
- `QLoRALayer` low-rank residual adapter.
- `LLMModule` classifier head producing binary predictions.
- Training pipeline with K-Fold cross validation and simple reward-shaped loss.
- Basic evaluation metrics (accuracy, precision, recall, F1).

## Repository structure

- `main.py` — Entry point and training pipeline.
- `data/dataset_loader.py` — Synthetic dataset generator.
- `utils/preprocessing.py` — Preprocessing, vocabulary builder, encoding.
- `utils/evaluation.py` — Evaluation metrics and printing.
- `model/gnn.py` — `SimpleGNN` encoder.
- `model/qlora.py` — `QLoRALayer` adapter.
- `model/llm_module.py` — `LLMModule` classifier head.

## Requirements

- Python 3.8+
- PyTorch
- scikit-learn


## Quick start

1. Clone the repository.
2. Install dependencies and ensure CUDA is available if you want GPU training.
3. Run the training pipeline:


The script runs a synthetic training loop with 10-fold cross validation and prints evaluation metrics at the end.

## Configuration and tuning

- `DEVICE` in `main.py` uses automatic CUDA detection; override as needed.
- Model sizes and hyperparameters are defined in the model classes and the training loop in `main.py`.
- Replace or extend `data/dataset_loader.py` to load real datasets.

## Development notes

- Preprocessing is intentionally minimal; replace with a proper tokenizer for real experiments.
- `QLoRALayer` is a  low-rank adapter; replace with a production LoRA/QLoRA implementation for rigorous research.
- The pipeline is designed to be small and easy to extend for unit test and model-probing experiments.

