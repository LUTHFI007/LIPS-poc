---
license: mit
tags:
- lips_model_type:tf_fc
---
# tf_fc_DEFAULT — TensorFlow Fully Connected Model

Benchmark1 surrogate model for power grid current flow prediction.  
Trained on the `l2rpn_case14_sandbox` environment to predict line currents (`a_or`, `a_ex`) from grid injections and topology.

**Architecture:** 4 fully connected hidden layers × 300 neurons, ReLU activation, no dropout  
**Framework:** TensorFlow / Keras

---

## Folder contents

```
tf_fc/
├── model.weights.h5          ← trained Keras weights
├── config.json               ← hyperparameters
├── metadata.json             ← input size (111) and output size (40)
├── scaler_params.json        ← normalisation statistics
├── losses.json               ← training history
├── tf_fc.ini                 ← simulator configuration
└── l2rpn_case14_sandbox.ini  ← benchmark configuration
```

---

## Usage

```python
import pathlib
import numpy as np
from lips.augmented_simulators.tensorflow_models import TfFullyConnected
from lips.dataset.scaler import StandardScaler
from lips.dataset.powergridDataSet import PowerGridDataSet

MODEL_DIR = pathlib.Path("path/to/tf_fc_DEFAULT")

# Instantiate and restore
# name="tf_fc" + sim_config_name="DEFAULT" → model.name = "tf_fc_DEFAULT"
# restore() looks for MODEL_DIR.parent / "tf_fc_DEFAULT"
model = TfFullyConnected(
    name="tf_fc",
    sim_config_path=MODEL_DIR / "tf_fc.ini",
    sim_config_name="DEFAULT",
    bench_config_path=MODEL_DIR / "l2rpn_case14_sandbox.ini",
    bench_config_name="Benchmark1",
    scaler=StandardScaler,
)
model.restore(MODEL_DIR.parent)

# Prepare input data
# dataset.data must contain all keys below with matching shapes
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
# Normalisation and inverse-normalisation are handled internally
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