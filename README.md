# RxVision

RxVision is a research prototype for studying trustworthy image-classification workflows with pill imagery: dataset preparation, transfer learning, explicit model metadata, confidence-based abstention, and a bounded local API.

> [!CAUTION]
> RxVision is not a medical device, has not been clinically validated, and must not be used to identify medication or make pharmacy, emergency, or treatment decisions. This repository does not include a trained model or claim current accuracy.

## Why this project exists

The project began as a 2020 data-science capstone asking a human-centered question: could computer vision reduce the effort and uncertainty involved in visually distinguishing medications? The original experiment exposed the central difficulty. A model could score well on a controlled validation split and still fail badly on ordinary photographs.

The current repository turns that lesson into an engineering system whose limits are visible:

- a preparation pipeline for externally obtained image collections;
- an auditable summary of the historical NLM reference manifest;
- a PyTorch EfficientNetV2 training path;
- state-dict-only checkpoints with class and preprocessing metadata;
- softmax candidate rankings with an explicit abstention threshold;
- a local API that reports when no model is loaded and rejects unsafe uploads;
- tests covering model contracts, checkpoint validation, inference, training, and API behavior.

## Project status

| Statement | Status |
| --- | --- |
| Reference-manifest analysis | Reproducible summary in [`evidence/dataset-manifest-summary.json`](evidence/dataset-manifest-summary.json) |
| Training and inference pipeline | Implemented and covered by automated tests |
| Published model or benchmark | Not included; bring independently sourced data and evaluate for your research context |
| Intended use | Software engineering and computer-vision research |

See the [model card](docs/MODEL_CARD.md) for intended use and evaluation requirements and [data notes](docs/DATA.md) for provenance and leakage risks.

## System boundary

```text
user-supplied images + reference manifest
                 │
                 ▼
        preparation and audit
                 │
                 ▼
       ImageFolder train / val
                 │
                 ▼
      EfficientNetV2 classifier
                 │
                 ▼
 state dict + classes + transforms
                 │
                 ▼
 candidate ranking ── low confidence ──▶ abstain
```

The API returns candidate labels from a research model. It intentionally does not describe a candidate as an identified medication.

## Quick start

Requires Python 3.10 or newer.

```bash
uv sync --all-extras
uv run pytest
```

Or with standard Python tooling:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,api,data]"
pytest
```

To start the API without a model:

```bash
uvicorn rxvision.inference.api:app --reload
curl http://127.0.0.1:8000/health
```

It will report `not_ready` until `RXVISION_CHECKPOINT` points to a compatible checkpoint.

## Preparing data

No pill images or third-party dataset are distributed here. Obtain any data independently, review the source's current terms, and keep it outside Git.

The historical NLM manifest format is:

```text
NDC|PART|FILE_PATH|TYPE|DRUG_NAME
```

Audit a compatible manifest without copying its contents:

```bash
python scripts/summarize_manifest.py /path/to/directory_of_images.txt
```

Prepare locally obtained images:

```bash
python scripts/prepare_dataset.py \
  --reference-table /path/to/directory_of_images.txt \
  --raw-dir /path/to/raw/images \
  --output-dir data \
  --group-by drug \
  --allow-image-level-split
```

The current preparation script creates image-level splits. That is useful for pipeline development but insufficient for a defensible performance claim when near-duplicate views may cross splits. Use group-aware, independently reviewed evaluation data before reporting model quality.

## Training

The training interface accepts standard `ImageFolder` directories:

```python
from rxvision.training import Trainer, TrainingConfig

trainer, train_loader, validation_loader = Trainer.from_directories(
    "data/train",
    "data/val",
    config=TrainingConfig(epochs=10),
)
history = trainer.fit(train_loader, validation_loader, "artifacts/model.pt")
```

Pretrained weights are opt-in and may be downloaded by torchvision. No model artifact is committed to this repository.

## Repository scope

RxVision focuses on the inspectable path from local data preparation through training, checkpointing, and bounded inference. The repository contains source code, tests, documentation, and aggregate reference-manifest analysis; datasets and trained weights remain external.

Source code is available under the MIT License. Dataset files, model weights, medical claims, and third-party assets are not covered or supplied.
