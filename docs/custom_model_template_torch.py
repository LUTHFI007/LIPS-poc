# =============================================================================
# augmented_simulator.py  —  CUSTOM PYTORCH MODEL TEMPLATE
# =============================================================================
#
# Fill in build_model() with your own architecture, train, and submit.
# Everything else (data loading, preprocessing, the training loop, save/restore)
# is handled for you by LIPS.
#
# IMPORTANT — two different classes are involved here:
#   * TorchFullyConnected  — the MODEL (an nn.Module-style network). You
#                            subclass THIS and override build_model().
#   * TorchSimulator       — the WRAPPER the evaluator runs your model through.
#                            You do NOT subclass it; you pass your model class to
#                            it (see the training script at the bottom).
#
# LIPS version required: lips-benchmark==0.2.7
# HuggingFace model type tag: lips_model_type:custom_torch
# =============================================================================

import torch
from torch import nn

from lips.augmented_simulators.torch_models.fully_connected import (
    TorchFullyConnected,
)
from lips.dataset.scaler import StandardScaler


class MyAugmentedSimulator(TorchFullyConnected):
    """Your custom PyTorch model.

    You ONLY need to implement build_model(). The class name must stay
    `MyAugmentedSimulator` and it MUST subclass `TorchFullyConnected` — the
    evaluator locates your model by looking for that base class and then wraps
    it in a TorchSimulator for you.
    """

    def __init__(self, **kwargs):
        # Nothing custom needed here — just forward everything to the parent.
        super().__init__(**kwargs)

    def build_model(self):
        """Define your network layers and assign them to `self`.

        Variables available to you (all set automatically before this runs —
        DO NOT hardcode them):

          self.input_size   int   number of input features for this benchmark.
          self.output_size  int   number of output targets for this benchmark.
          self.params       dict  parsed from your `.ini` file. Read your
                                   hyperparameters from here, e.g.
                                   self.params["layers"]      -> tuple of widths
                                   self.params["activation"]  -> e.g. "relu"

        Register your layers as attributes/modules on `self` so PyTorch tracks
        their parameters. Return `self`.
        """
        layers     = self.params["layers"]        # e.g. (300, 300, 300, 300)
        activation = self.params.get("activation", "relu")
        act_cls    = {"relu": nn.ReLU, "tanh": nn.Tanh}.get(activation, nn.ReLU)

        # ---- Example architecture: a simple multi-layer dense network. -------
        # Replace this with your own architecture.
        modules = []
        prev = self.input_size
        for width in layers:
            modules.append(nn.Linear(prev, width))
            modules.append(act_cls())
            prev = width
        modules.append(nn.Linear(prev, self.output_size))

        self.layers = nn.Sequential(*modules)
        # ----------------------------------------------------------------------
        return self

    def forward(self, data):
        # Match the input handling of the parent if your data needs reshaping.
        return self.layers(data)

    # -------------------------------------------------------------------------
    # DO NOT OVERRIDE these — the TorchSimulator wrapper implements them and the
    # evaluator relies on the default behaviour:
    #     train()            predict()           restore()
    #     save()             process_dataset()
    # -------------------------------------------------------------------------


# =============================================================================
# Training script — run this to produce the files for your ZIP:
#     python augmented_simulator.py
# =============================================================================
if __name__ == "__main__":
    import pathlib

    from lips.benchmark.powergridBenchmark import PowerGridBenchmark
    from lips.augmented_simulators.torch_simulator import TorchSimulator

    # --- Adjust these paths -----------------------
    BENCHMARK_NAME = "Benchmark1"
    DATASET_ROOT   = "path/to/your/dataset"  # the benchmark dataset that you downloaded from Data Hub
    BENCH_CONFIG   = "benchmark.ini"  # downloaded from the Model Hub — use AS-IS to train your model, do NOT edit/rename or add to your ZIP
    SIM_CONFIG     = "simulator.ini"   # the simulator .ini you downloaded 
    SAVE_DIR       = "saved_model"
    # --------------------------------------------------------------------------

    benchmark = PowerGridBenchmark(
        benchmark_name=BENCHMARK_NAME,
        benchmark_path=DATASET_ROOT,
        load_data_set=True,
        config_path=BENCH_CONFIG,
    )

    # Note: you pass your MODEL class to the TorchSimulator WRAPPER.
    sim = TorchSimulator(
        model=MyAugmentedSimulator,
        sim_config_path=SIM_CONFIG,
        name="my_model",
        bench_config_path=BENCH_CONFIG,
        bench_config_name=BENCHMARK_NAME,
        # StandardScaler normalizes inputs/outputs during training AND makes
        # sim.save() write a scaler_params.json alongside your weights. The
        # evaluator needs that file to de-normalize predictions — WITHOUT it your
        # scores come out enormous. Do NOT remove this argument.
        scaler=StandardScaler,
        log_path=None,
    )

    sim.train(
        train_dataset=benchmark.train_dataset,
        val_dataset=benchmark.val_dataset,
        epochs=sim.params.get("epochs", 10),
    )

    # sim.save() writes the weights + metadata your ZIP needs.
    sim.save(path=SAVE_DIR)

    saved = sorted(p.name for p in pathlib.Path(SAVE_DIR).rglob("*") if p.is_file())
    print(f"\nSaved model to '{SAVE_DIR}/'. Files created:")
    for f in saved:
        print(f"  - {f}")
    print(
        "\nNext: copy this file (augmented_simulator.py) and your adjusted "
        "simulator.ini into that folder, then ZIP the whole folder and upload."
    )
