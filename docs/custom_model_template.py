# =============================================================================
# augmented_simulator.py  —  CUSTOM TENSORFLOW MODEL TEMPLATE
# =============================================================================
#
# Fill in build_model() with your own architecture, train, and submit.
# Everything else (data loading, preprocessing, the training loop, save/restore)
# is handled for you by LIPS through the parent class.
#
# LIPS version required: lips-benchmark==0.2.7
# HuggingFace model type tag: lips_model_type:custom_tf
# =============================================================================

import tensorflow as tf
from tensorflow import keras

from lips.augmented_simulators.tensorflow_models.powergrid.fully_connected import (
    TfFullyConnectedPowerGrid,
)
from lips.dataset.scaler import StandardScaler


class MyAugmentedSimulator(TfFullyConnectedPowerGrid):
    """Your custom TensorFlow simulator.

    You ONLY need to implement build_model(). The class name must stay
    `MyAugmentedSimulator` and it MUST subclass `TfFullyConnectedPowerGrid` —
    the evaluator locates your model by looking for that base class.
    """

    def __init__(self, **kwargs):
        # Nothing custom needed here — just forward everything to the parent.
        super().__init__(**kwargs)

    def build_model(self):
        """Define your network and assign it to `self._model`.

        Variables available to you (all set automatically before this runs —
        DO NOT hardcode them):

          self.input_size   int   number of input features for this benchmark.
          self.output_size  int   number of output targets for this benchmark.
          self.params       dict  parsed from your `.ini` file. Read your
                                   hyperparameters from here, e.g.
                                   self.params["layers"]      -> tuple of widths
                                   self.params["activation"]  -> e.g. "relu"

        You MUST assign a `keras.Model` to `self._model`.
        Do NOT call `self._model.compile()` here — the parent compiles it for
        you using the loss/optimizer from your `.ini`.
        """
        layers     = self.params["layers"]       # e.g. (300, 300, 300, 300)
        activation = self.params.get("activation", "relu")

        # ---- Example architecture: a simple multi-layer dense network. -------
        # Replace the body of this loop with your own architecture.
        inputs = keras.layers.Input(shape=(self.input_size,))
        x = inputs
        for width in layers:
            x = keras.layers.Dense(width, activation=activation)(x)
        outputs = keras.layers.Dense(self.output_size)(x)
        # ----------------------------------------------------------------------

        self._model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        return self._model

    # -------------------------------------------------------------------------
    # DO NOT OVERRIDE these — the parent class implements them and the evaluator
    # relies on the default behaviour:
    #     train()            predict()           restore()
    #     save()             process_dataset()
    # Overriding any of them will break evaluation.
    # -------------------------------------------------------------------------


# =============================================================================
# Training script — run this to produce the files for your ZIP:
#     python augmented_simulator.py
# =============================================================================
if __name__ == "__main__":
    import pathlib

    from lips.benchmark.powergridBenchmark import PowerGridBenchmark

    # --- Adjust these paths to your local LIPS checkout -----------------------
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

    sim = MyAugmentedSimulator(
        name="my_model",
        sim_config_path=SIM_CONFIG,
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
        epochs=sim.params.get("epochs", 5),
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
