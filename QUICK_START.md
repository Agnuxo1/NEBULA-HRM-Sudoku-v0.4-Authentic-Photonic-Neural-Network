# NEBULA v0.4 - Quick Start Guide

**Equipo NEBULA: Francisco Angulo de Lafuente y Ángel Vega**

---

## 🚀 5-Minute Quick Start

### Step 1: Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pennylane transformers numpy scipy
```

### Step 2: Download and Test
```python
import torch
from NEBULA_UNIFIED_v04 import NEBULAUnifiedModel

# Initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NEBULAUnifiedModel(device=device)

# Test with random sudoku
sudoku = torch.randn(1, 81, device=device)
result = model(sudoku)

print(f"Photonic neural network working! Output shape: {result['main_output'].shape}")
```

### Step 3: Load Pretrained Weights
```python
# Load validated model
model.load_state_dict(torch.load('nebula_photonic_validated_final.pt'))
model.eval()

print("✅ NEBULA v0.4 ready for spatial reasoning!")
```

---

## 💡 Key Features

- **Authentic Photonic Computing**: Real optical physics simulation
- **Quantum Memory**: 4-qubit quantum circuits for information storage
- **Holographic Memory**: Complex interference patterns for associative memory
- **RTX Optimization**: Native GPU acceleration with Tensor Cores

---

## 📊 Expected Results

- **Spatial Reasoning Accuracy**: ~50%
- **Improvement over Random**: +14 percentage points
- **Performance**: 89th percentile
- **Training Time**: ~15 epochs for convergence

---

For complete documentation, see:
- [Technical Details](docs/TECHNICAL_DETAILS.md)
- [Reproducibility Guide](docs/REPRODUCIBILITY_GUIDE.md) 
- [Physics Background](docs/PHYSICS_BACKGROUND.md)

**"Paso a paso, sin prisa, con calma"** - Project NEBULA Philosophy