---
license: mit
tags:
- lips_model_type:torch_fc
---
# torch_fc_DEFAULT — PyTorch Fully Connected Model

Benchmark1 surrogate model for power grid current flow prediction.  
Trained on the `l2rpn_case14_sandbox` environment to predict line currents (`a_or`, `a_ex`) from grid injections and topology.

**Architecture:** 4 fully connected hidden layers × 300 neurons, ReLU activation, no dropout  
**Framework:** PyTorch  
**Key difference from tf_fc:** Same architecture but implemented in PyTorch. Useful if your pipeline is already PyTorch-based.

---

## Folder contents

```
torch_fc/
├── model_last.pt             ← trained PyTorch weights
├── config.json               ← hyperparameters
├── metadata.json             ← input/output sizes
├── scaler_params.json        ← normalisation statistics
├── losses.json               ← training history
├── torch_fc.ini              ← simulator configuration
└── l2rpn_case14_sandbox.ini  ← benchmark configuration
```

> The weight file uses the `.pt` format (PyTorch state dict), not Keras `.h5`.  
> `model_last.pt` means the weights from the final training epoch (as opposed to a mid-training checkpoint).

---

## Installation

```bash
pip install torch
```

---

## Usage

```python
import pathlib
import numpy as np
from lips.augmented_simulators.torch_models.powergrid import TorchFullyConnected
from lips.dataset.scaler import StandardScaler
from lips.dataset.powergridDataSet import PowerGridDataSet

MODEL_DIR = pathlib.Path("path/to/torch_fc_DEFAULT")

# Instantiate and restore
# name="torch_fc" + sim_config_name="DEFAULT" → model.name = "torch_fc_DEFAULT"
# restore() looks for MODEL_DIR.parent / "torch_fc_DEFAULT"
model = TorchFullyConnected(
    name="torch_fc",
    sim_config_path=MODEL_DIR / "torch_fc.ini",
    sim_config_name="DEFAULT",
    bench_config_path=MODEL_DIR / "l2rpn_case14_sandbox.ini",
    bench_config_name="Benchmark1",
    scaler=StandardScaler,
)
model.restore(MODEL_DIR.parent)

# Prepare input data
N = 100  # number of samples
dataset = PowerGridDataSet(
    name="my_data",
    config_path=MODEL_DIR / "l2rpn_case14_sandbox.ini",
    config_name="Benchmark1",
)
dataset.data = {
    "prod_p":      np.random.rand(N, 6).astype(np.float32),   # MW
    "prod_v":      np.random.rand(N, 6).astype(np.float32),   # per-unit voltage
    "load_p":      np.random.rand(N, 11).astype(np.float32),  # MW
    "load_q":      np.random.rand(N, 11).astype(np.float32),  # MVAr
    "line_status": np.ones((N, 20), dtype=bool),               # True = connected
    "topo_vect":   np.ones((N, 57), dtype=np.int32),           # bus assignments (1 or 2)
    # Dummy targets — required by the data pipeline even at inference time
    "a_or":        np.zeros((N, 20), dtype=np.float32),
    "a_ex":        np.zeros((N, 20), dtype=np.float32),
}

# Run inference
predictions = model.predict(dataset)

print(predictions["a_or"].shape)  # (N, 20) — current at origin end of each line (A)
print(predictions["a_ex"].shape)  # (N, 20) — current at extremity end of each line (A)
```

---

## Input / Output

### Inputs — 111 features total (concatenated in this order)

| Feature | Shape | Unit | Description |
|---|---|---|---|
| `prod_p` | (N, 6) | MW | Active power output per generator |
| `prod_v` | (N, 6) | p.u. | Voltage setpoint per generator |
| `load_p` | (N, 11) | MW | Active power demand per load |
| `load_q` | (N, 11) | MVAr | Reactive power demand per load |
| `line_status` | (N, 20) | bool | Line connection status (1=on, 0=off) |
| `topo_vect` | (N, 57) | int | Bus assignment per element (1 or 2) |

### Outputs — 40 values total

| Feature | Shape | Unit | Description |
|---|---|---|---|
| `a_or` | (N, 20) | A | Current at the origin end of each line |
| `a_ex` | (N, 20) | A | Current at the extremity end of each line |