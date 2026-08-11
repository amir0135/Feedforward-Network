# Feedforward Ensemble Network

A PyTorch implementation of a **feedforward ensemble network** with PReLU activations and a bounded sigmoid output head. It is the reference software model for [`ML_SME_FPGA`](https://github.com/amir0135/ML_SME_FPGA-main), where the same architecture is reimplemented in hardware via Synchronous Message Exchange (SME).

Having both implementations lets the FPGA design be validated numerically against a known-good PyTorch baseline.

## Architecture

The network runs `num_networks` parallel sub-networks over a shared input and averages their bounded predictions:

```
x  (batch, 256)
 │
 ├─ W0            linear projection → (batch, 16, 96)
 │
 ├─ PReLU(z)  ──► hz ──► ·Wz ──► sum ──► z
 ├─ PReLU(r)  ──► hr ──► ·Wr ──► sum ──► r
 │
 └─ y = r · (2·σ(z_scale · z) − 1)     bounded to ±max_predict
     ŷ = mean(y over the 16 networks)
```

Two separate PReLU slopes are applied to the same hidden activations, producing a **gate** (`z`) and a **magnitude** (`r`). The sigmoid gate is rescaled to `[-1, 1]`, so each sub-network emits a signed, bounded prediction before averaging.

### Default hyperparameters

| Parameter | Default | Meaning |
|---|---|---|
| `input_size` | 256 | Input feature dimension |
| `hidden_size` | 96 | Hidden units per sub-network |
| `num_networks` | 16 | Parallel sub-networks in the ensemble |
| `max_predict` | 1 | Output clamp bound |

Weights (`W0`, `Wz`, `Wr`, `z_scale`, and the PReLU slopes) are loaded from CSV files in `data/` so the exact same values can be fed to the FPGA implementation.

## Repository layout

```
feedforward_network/
  feedforward_ensemble.py   FeedforwardEnsembleNetwork module
utils/
  data_utils.py             CSV → torch.Tensor helpers
scripts/
  generate_data.py          Creates synthetic weights and training data in data/
  train.py                  MSE training loop (SGD, momentum 0.9)
```

## Getting started

Requires Python 3.9+.

```bash
git clone https://github.com/amir0135/Feedforward-Network.git
cd Feedforward-Network

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Generate the weight and data CSVs, then train:

```bash
python scripts/generate_data.py
python scripts/train.py
```

> Both scripts resolve `data/` relative to the current working directory, so run them from the repository root.

Training prints per-epoch MSE loss for 100 epochs:

```
Epoch 1, Loss: 0.4127...
Epoch 2, Loss: 0.3894...
```

## Using the model directly

```python
import torch
from feedforward_network.feedforward_ensemble import FeedforwardEnsembleNetwork

model = FeedforwardEnsembleNetwork(input_size=256, hidden_size=96, num_networks=16)
x = torch.randn(8, 256)
y = model(x)          # (8,) — each value in [-1, 1]
```

## Tech stack

PyTorch · pandas · NumPy

## License

MIT — see [LICENSE](LICENSE).
