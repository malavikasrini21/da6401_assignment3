# Project Structure: Transliteration with and without Attention

This repository is structured into two core implementations:

1. `attn_encoder_decoder` — Encoder-Decoder model **with Attention**
2. `vanilla_encoder_decoder` — Encoder-Decoder model **without Attention**

---

## 📂 Folder Contents

Each folder (`attn_encoder_decoder`, `vanilla_encoder_decoder`) contains the following core scripts:

### 1. `dataload.py`
- Loads the dataset from TSV files.
- train/validation/test splits of Dakshina are handled.
- Tokenizes inputs and targets.
- Returns vocabularies and PyTorch DataLoaders.

### 2. `model.py`
- Defines the encoder-decoder architecture.
- `vanilla_encoder_decoder`: uses basic encoder-decoder without attention.
- supports configurable embedding size, hidden size, learning rate, number of layers, and RNN type (RNN/GRU/LSTM).

### 3. `train.py`
- Handles training with or without Weights & Biases (wandb) sweeps.
- Supports:
  - Hyperparameter configuration
  - Training loop with validation and model checkpointing
- Logs metrics (accuracy, loss) and saves the best model.

### 4. `evaluate.py`
- Loads the best saved model and runs inference on the **test set**.
- Computes:
  - Character-level and word-level accuracy
  - Beam search decoding
- Saves predictions to a `.tsv` or `.csv` file with columns:  
  `source`, `target`, `prediction`, `status`.

- ## python3 evaluate.py --model_path <path_to_bestmodel> --dataset_path <path_to_dataset>

---

## 📁 Output Folder

Each model saves its predictions in a **dedicated output folder**, structured as:
- predictions_attn folder
- └── decoded_outputs_vanilla.tsv ## Contains Source + Ground Truth + predictions + correctness of labels

## To Execute

## python3 train_attn.py --sweep --dataset_path <path_to_the_dataset>
