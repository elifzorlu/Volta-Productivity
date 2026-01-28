# Volta: Productivity Forecasting Model

Volta is a time-series forecasting pipeline designed to model daily productivity scores based on longitudinal behavioral metrics (sleep duration, caffeine intake, screen time).

The core engine is a **PyTorch LSTM (Long Short-Term Memory)** network that captures temporal dependencies in user data. The pipeline is engineered to support hardware-accelerated training via CUDA, allowing for efficient processing of large-scale datasets.

## ⚡ Technical Overview
* **Model Architecture:** 2-Layer LSTM with a fully connected regression head.
* **Input Features:** Multivariate time-series data.
* **Optimization:** Custom training loop with explicit device management (CPU/GPU) to handle memory allocation constraints.
* **Stack:** Python, PyTorch, NumPy, Pandas.
   ```bash
   git clone https://github.com/elifzorlu/Volta-Productivity-Model.git
   cd Volta-Productivity-Model
  ```
Install dependencies:

```bash
pip install -r requirements.txt
 ```
## 🚀 Usage
1. Generate Data
If real behavioral data is unavailable, generate a synthetic dataset for testing/benchmarking:

```bash

python src/generate_data.py# Outputs: data/dummy_productivity.csv
```
2. Train the Model
The training pipeline supports command-line arguments to toggle compute resources.
Run on CPU (Default):
```bash

python src/train.py --epochs 100 --lr 0.01
```
Run on GPU (CUDA): Note: Requires an NVIDIA GPU and compatible Torch drivers.
```bash

python src/train.py --device cuda --epochs 500
```
## **🖥 Hardware Acceleration Logic**
The pipeline implements a "safe-fail" hardware check to verify CUDA availability before attempting tensor allocation on the device.


## Python Snippet from src/train.py
    if device_name == 'cuda' and not torch.cuda.is_available():
     print("⚠️  WARNING: CUDA requested but not available. Fallback to CPU.")
     device = torch.device('cpu')
    else:
    device = torch.device(device_name)
## Future Work / Roadmap
* **C++ Extensions:** Implement custom C++ kernels for faster rolling-window feature extraction.
* **Quantization:** Optimize model weights for on-device inference (mobile target).
* **MPS Support:** Integrate Apple Metal Performance Shaders for local acceleration on macOS.
