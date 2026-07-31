# `openpi` Package Structure

Organize model-specific code by model family and keep shared infrastructure at
the package level.

```text
openpi/
├── models/
│   └── <model_family>/
│       ├── config.py
│       ├── model.py
│       └── *_test.py
├── training/
│   └── <model_family>/
│       ├── config.py
│       ├── data.py
│       ├── checkpoint.py
│       └── *_test.py
├── policies/        # Policy construction and runtime adapters
├── serving/         # Serving infrastructure
└── shared/          # Cross-model utilities and common types
```

Model-specific modules should import through their package path, for example
`openpi.models.arm_value.config` and `openpi.training.arm_value.data`. Shared
code that is used by several model families should remain in the nearest
cross-model package instead of being duplicated inside a model directory.

Current model-family packages include `pi0`, `arm_awbc`, and `arm_value`.
Generic backbones and utilities used by multiple families remain directly
under `openpi.models`.

Repository entrypoints use a matching functional split:

```text
scripts/
├── train/           # Training entrypoints, training tests, and launch scripts
├── tools/           # Dataset, checkpoint, validation, serving, and diagnostic tools
└── docker/          # Container setup and serving assets
```
