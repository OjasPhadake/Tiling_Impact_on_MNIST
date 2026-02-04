# MNIST 7x7 Tiling & Separator Experiment

## Purpose
This experiment investigates the impact of spatial fragmentation on a standard Multi-Layer Perceptron (MLP). By decomposing a $28 \times 28$ MNIST image into sixteen $7 \times 7$ tiles and applying a "black border" to each tile, we test how a non-convolutional network handles the loss of global connectivity and the introduction of artificial grid noise.

### The Hypothesis
The "Separator Hypothesis" suggests that by zeroing out the outermost edge of every $7 \times 7$ tile, we provide the model with a structured grid that might assist in feature isolation. However, this comes at an "Information Tax," as approximately **49% of the original pixels** are removed (turning a $7 \times 7$ patch into a $5 \times 5$ active area).



---

## File Structure

The project is organized into modular directories to ensure data integrity and clear result tracking.

```text
mnist_experiment/
├── data/
│   ├── raw/                # Original 28x28 images (10k train / 2k test)
│   └── tiled/              # 7x7 tiled images with 1px black separators
│
├── preprocessing/
│   └── setup_experiment.py # Downloads MNIST and applies tiling transformations
│
├── models/
│   ├── __init__.py
│   ├── mlp_baseline.py     # Architecture: 784 -> 128 -> 64 -> 10 neurons
│   └── mlp_tiled.py     # Architecture: 784 -> 128 -> 64 -> 10 neurons
│
├── results/
│   ├── mlp_raw.pth         # Weights trained on standard images
│   └── mlp_tiled.pth       # Weights trained on 7x7 tiled images
│
├── plots/
│   ├── learning_curves.png # Comparison of training convergence
│   └── comparison.png      # Images correctly classified by Tiled but missed by Raw
│
├── train.py                # Sequentially trains both model variants
└── evaluate.py             # Accuracy testing and visualization suite
