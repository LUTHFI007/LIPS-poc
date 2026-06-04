---
license: mit
tags:
- lips_model_type:tf_leapnet
---
# tf_leapnet_DEFAULT — TensorFlow LeapNet Model

Benchmark1 surrogate model for power grid current flow prediction.  
Trained on the `l2rpn_case14_sandbox` environment to predict line currents (`a_or`, `a_ex`) from grid injections and topology.

**Architecture:** LeapNet (Latent Encoding of Atypical Perturbations) — encoder + main network + decoder, specialised for handling topology changes via a dedicated tau encoding path  
**Framework:** TensorFlow / Keras  
**Key difference from tf_fc:** LeapNet explicitly separates the topology input (`attr_tau`) from the physical inputs (`attr_x`), making it better at generalising to unseen topologies.

---

## Folder contents

```
tf_leapnet/
├── weights.h5                ← trained Keras weights (legacy naming)
├── config.json               ← hyperparameters
├── metadata.json             ← input/output sizes
├── scaler_params.json        ← normalisation statistics
├── losses.json               ← training history
├── tf_leapnet.ini            ← simulator configuration
└── l2rpn_case14_sandbox.ini  ← benchmark configuration
```

> The weight file is named `weights.h5` (without the `model.` prefix) because this model was saved with an older version of Keras. The restore code handles this automatically.

---

## Installation

```bash
pip install leap-net==0.0.5
```

> `leap-net` is a separate dependency required only for LeapNet. Without it, the import will fail silently with an `ImportError`.

---

## Usage

```python
import pathlib
import numpy as np
from lips.augmented_simulators.tensorflow_models.powergrid import LeapNet
from lips.dataset.scaler import StandardScaler
from lips.dataset.powergridDataSet import PowerGridDataSet

MODEL_DIR = pathlib.Path("path/to/tf_leapnet_DEFAULT")

# Instantiate and restore
# name="tf_leapnet" + sim_config_name="DEFAULT" → model.name = "tf_leapnet_DEFAULT"
# restore() looks for MODEL_DIR.parent / "tf_leapnet_DEFAULT"
model = LeapNet(
    name="tf_leapnet",
    sim_config_path=MODEL_DIR / "tf_leapnet.ini",
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

### Inputs — 111 features total

| Feature | Shape | Unit | Description |
|---|---|---|---|
| `prod_p` | (N, 6) | MW | Active power output per generator |
| `prod_v` | (N, 6) | p.u. | Voltage setpoint per generator |
| `load_p` | (N, 11) | MW | Active power demand per load |
| `load_q` | (N, 11) | MVAr | Reactive power demand per load |
| `line_status` | (N, 20) | bool | Line connection status (1=on, 0=off) |
| `topo_vect` | (N, 57) | int | Bus assignment per element (1 or 2) |

> Internally, LeapNet routes `line_status` and `topo_vect` through a separate tau encoding path. This is handled automatically — your input dict format is the same as tf_fc.

### Outputs — 40 values total

| Feature | Shape | Unit | Description |
|---|---|---|---|
| `a_or` | (N, 20) | A | Current at the origin end of each line |
| `a_ex` | (N, 20) | A | Current at the extremity end of each line |