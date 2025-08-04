# Text Similarity Model Based on Multi-Head Attention

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.13+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

> English | [中文文档](README_zh.md)

## 📋 Project Overview

This project implements a text similarity model based on **Multi-Head Attention Mechanism**. The model effectively captures semantic information in text sequences and calculates similarity scores between two texts.

### 🌟 Key Features

- **Multi-Head Attention Mechanism**: Implements the core component of Transformer architecture, capturing text features from different representation subspaces
- **Flexible Configuration System**: Supports YAML/JSON configuration files for convenient experiment management
- **Complete Training Pipeline**: Includes training, validation, early stopping, checkpoint saving, and more
- **User-Friendly CLI**: Rich command-line parameters for quick experimentation
- **Comprehensive Documentation**: Well-documented code with detailed comments for easy understanding

## 🏗️ Project Structure

```
.
├── configs/                 # Configuration files
│   └── default.yaml        # Default configuration
├── src/                    # Source code
│   ├── configs/           # Configuration management
│   │   ├── __init__.py
│   │   └── config.py      # Configuration classes
│   ├── data/              # Data processing
│   │   ├── __init__.py
│   │   └── dataset.py     # Dataset and data loaders
│   ├── models/            # Model definitions
│   │   ├── __init__.py
│   │   ├── attention.py   # Multi-head attention implementation
│   │   └── similarity_model.py  # Text similarity model
│   └── utils/             # Utilities
│       ├── __init__.py
│       ├── trainer.py     # Training implementation
│       └── utils.py       # General utility functions
├── scripts/               # Scripts
│   └── train.py          # Training script
├── tutorial/             # Tutorials and original code
│   └── mha-lstm/        # Original notebooks and data
├── requirements.txt      # Dependencies
├── README.md            # English documentation
└── README_zh.md         # Chinese documentation
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd <project-directory>

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

The project uses the STSbenchmark dataset. Data files should be placed in the `tutorial/mha-lstm/data/` directory:
- `sts-kaggle-train.csv`: Training data
- `sts-kaggle-test.csv`: Validation/test data

Data format example:
```csv
id,sentence_a,sentence_b,similarity
0,"A kitten is playing with a toy.","A kitten is playing with a blue rope toy.",4.4
1,"A dog is running in a field.","A white and brown dog runs in a field.",2.83
```

### 3. Start Training

#### Using Default Configuration
```bash
python scripts/train.py
```

#### Using Custom Configuration File
```bash
python scripts/train.py --config configs/default.yaml
```

#### Common Command-Line Arguments
```bash
# Adjust batch size and learning rate
python scripts/train.py --batch-size 64 --learning-rate 0.001

# Use pretrained word embeddings
python scripts/train.py --use-pretrained-embeddings --embeddings-name glove.6B.300d

# Specify experiment name and output directory
python scripts/train.py --experiment-name my_experiment --output-dir experiments

# Use GPU for training
python scripts/train.py --device cuda

# Evaluation-only mode
python scripts/train.py --eval-only --resume checkpoints/best_model.pt
```

## 📊 Model Architecture

For detailed architecture diagrams, please refer to the [Architecture Documentation](docs/architecture.md)

### Overall Architecture

```
Input Text Pairs (Text A, Text B)
    ↓
Tokenizer
    ↓
Word Embedding Layer
    ↓
Positional Encoding
    ↓
Multi-Head Attention Layers × N
    ↓
Pooling Layer
    ↓
Output Projection
    ↓
Cosine Similarity
    ↓
Similarity Score
```

### Core Components

#### 1. Multi-Head Attention

Multi-head attention is the core component of the model, computed as follows:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Key Parameters:**
- `embed_dim`: Embedding dimension (default: 256)
- `num_heads`: Number of attention heads (default: 8)
- `dropout`: Dropout probability (default: 0.1)

#### 2. Text Encoder

The encoder transforms input text sequences into fixed-dimensional vector representations:

```
Input Sequence → Word Embedding → Positional Encoding → Multi-Layer Attention → Pooling → Text Representation
```

**Pooling Strategies:**
- `mean`: Average pooling (default)
- `max`: Max pooling
- `cls`: Use [CLS] token representation

#### 3. Similarity Calculation

Cosine similarity is used to calculate the similarity between two text representations:

```python
similarity = cosine_similarity(embedding_a, embedding_b)
```

## 🔧 Configuration

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| embed_dim | int | 256 | Embedding dimension |
| num_heads | int | 8 | Number of attention heads |
| num_layers | int | 2 | Number of encoder layers |
| dropout | float | 0.1 | Dropout probability |
| pooling_strategy | str | mean | Pooling strategy |
| max_seq_len | int | 200 | Maximum sequence length |

### Training Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| batch_size | int | 32 | Batch size |
| learning_rate | float | 1e-4 | Learning rate |
| num_epochs | int | 50 | Number of training epochs |
| optimizer | str | adam | Optimizer type |
| early_stopping | bool | true | Whether to use early stopping |
| patience | int | 10 | Early stopping patience |

### Data Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| train_path | str | - | Training data path |
| val_path | str | - | Validation data path |
| tokenizer | str | basic_english | Tokenizer type |
| normalize_scores | bool | true | Whether to normalize scores |
| score_range | list | [0, 5] | Score range |

## 📈 Experimental Results

### Evaluation Metrics

- **Pearson Correlation**: Measures linear correlation between predictions and ground truth
- **Spearman Correlation**: Measures monotonic correlation between predictions and ground truth
- **MSE (Mean Squared Error)**: Average squared prediction error
- **MAE (Mean Absolute Error)**: Average absolute prediction error

### Training Outputs

The training process generates the following files:
- `output/exp_*/config.yaml`: Experiment configuration
- `output/exp_*/train.log`: Training logs
- `output/exp_*/checkpoints/`: Model checkpoints
- `output/exp_*/history.json`: Training history
- `output/exp_*/report.md`: Experiment report

### Visualization

If TensorBoard is installed, you can visualize training curves:
```bash
tensorboard --logdir runs
```

## 🎯 Application Scenarios

1. **Text Matching**: Determine if two texts express the same meaning
2. **Question Answering**: Match question-answer relevance
3. **Document Retrieval**: Find the most relevant documents based on queries
4. **Duplicate Detection**: Identify duplicate or similar content
5. **Semantic Search**: Search systems based on semantic similarity

## 🔍 Design Patterns

### 1. Factory Pattern
Used in configuration management to create different configuration objects:
```python
config = Config.from_file("config.yaml")  # Create from file
config = Config.from_dict(config_dict)    # Create from dictionary
```

### 2. Strategy Pattern
Pooling strategy implementation uses the strategy pattern to support different pooling methods:
```python
if self.pooling_strategy == 'mean':
    return sequence.mean(dim=1)
elif self.pooling_strategy == 'max':
    return sequence.max(dim=1)[0]
```

### 3. Template Method Pattern
The trainer class defines a training workflow template that can be extended:
```python
def train(self):
    for epoch in range(num_epochs):
        self._train_epoch()
        self._validate()
        self._save_checkpoint()
```

## 💻 Command-Line Interface

### Basic Usage

```bash
python scripts/train.py [OPTIONS]
```

### Main Options

#### Configuration Options
- `--config, -c`: Configuration file path (YAML or JSON)
- `--experiment-name`: Experiment name for identification
- `--seed`: Random seed for reproducibility (default: 42)

#### Data Options
- `--train-path`: Training data file path
- `--val-path`: Validation data file path
- `--test-path`: Test data file path
- `--max-length`: Maximum sequence length

#### Model Options
- `--embed-dim`: Embedding dimension
- `--num-heads`: Number of attention heads
- `--num-layers`: Number of encoder layers
- `--dropout`: Dropout probability
- `--pooling-strategy`: Pooling strategy (mean/max/cls)
- `--use-pretrained-embeddings`: Use pretrained embeddings
- `--embeddings-name`: GloVe embedding name

#### Training Options
- `--batch-size`: Batch size
- `--learning-rate, --lr`: Learning rate
- `--num-epochs`: Number of training epochs
- `--optimizer`: Optimizer type (adam/adamw/sgd)
- `--scheduler`: Learning rate scheduler (cosine/linear/constant)
- `--gradient-clip`: Gradient clipping value
- `--no-early-stopping`: Disable early stopping
- `--patience`: Early stopping patience

#### Device Options
- `--device`: Training device (cuda/cpu/auto)
- `--num-workers`: Number of data loader workers
- `--fp16`: Use mixed precision training

#### Output Options
- `--output-dir`: Output directory
- `--checkpoint-dir`: Checkpoint directory
- `--log-level`: Logging level
- `--use-tensorboard`: Use TensorBoard logging
- `--use-wandb`: Use Weights & Biases logging

#### Other Options
- `--resume`: Resume training from checkpoint
- `--eval-only`: Evaluation mode only
- `--dry-run`: Dry run to show configuration

## 📝 Advanced Usage

### Custom Configuration File

Create a custom configuration file `my_config.yaml`:

```yaml
model:
  embed_dim: 512
  num_heads: 16
  num_layers: 4
  dropout: 0.2

training:
  batch_size: 64
  learning_rate: 0.0002
  num_epochs: 100
  optimizer: adamw

data:
  train_path: path/to/train.csv
  val_path: path/to/val.csv
```

Then train with:
```bash
python scripts/train.py --config my_config.yaml
```

### Experiment Management

Run multiple experiments with different configurations:

```bash
# Experiment 1: Baseline
python scripts/train.py --experiment-name baseline --embed-dim 256 --num-heads 8

# Experiment 2: Larger model
python scripts/train.py --experiment-name large_model --embed-dim 512 --num-heads 16

# Experiment 3: Different pooling
python scripts/train.py --experiment-name max_pooling --pooling-strategy max
```

### Resume Training

Resume from a checkpoint:
```bash
python scripts/train.py --resume output/exp_20240101_120000/checkpoints/best_model.pt
```

### Hyperparameter Tuning

Example of grid search for hyperparameters:

```bash
for lr in 0.0001 0.0005 0.001; do
    for bs in 16 32 64; do
        python scripts/train.py \
            --experiment-name "lr_${lr}_bs_${bs}" \
            --learning-rate $lr \
            --batch-size $bs
    done
done
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the PyTorch team for the deep learning framework
- Thanks to the Hugging Face team for inspiration from the Transformers library
- Thanks to all contributors for their efforts

## 📧 Contact

For questions or suggestions, please:
- Submit an issue on GitHub
- Email: [your-email@example.com]

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{text_similarity_mha,
  title = {Text Similarity Model Based on Multi-Head Attention},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/yourrepository}
}
```

---

**Note**: This project is for educational and research purposes. For commercial use, please ensure compliance with all relevant licenses.
