# Data provenance and handling

## What is included

No real pill images, medication records, patient information, or third-party dataset archives are included. The repository retains only a derived count summary from a historical NLM reference manifest.

## Historical manifest evidence

The archived pipe-delimited manifest contained:

- 133,774 rows;
- 4,864 unique values in the NDC field;
- eight image-type labels before filtering.

These counts describe manifest rows, not unique photographs, medications currently marketed, model classes, or successful downloads. The aggregate record is stored in `evidence/dataset-manifest-summary.json`, and the counting method is reproducible with `scripts/summarize_manifest.py`.

## Source terms

Availability from a government-hosted endpoint does not by itself establish the terms for every file, derivative, label, or downstream use. Anyone obtaining NLM/FDA or other data must review the current source documentation and terms independently. The MIT License in this repository covers this project's source code only.

## Privacy

The historical pill-image collection did not require patient information for this experiment. Do not add prescriptions, patient images, account identifiers, access tokens, or protected health information to the repository, issues, model artifacts, or telemetry.

## Evaluation leakage

Pill-image archives can include multiple views, resolutions, crops, or processed derivatives of the same physical reference. Random image-level splitting can place near-duplicates in training and evaluation sets and materially inflate performance. The included preparation script is suitable for pipeline development; it is not presented as a benchmark builder. A performance study must deduplicate and use group-aware splits reviewed for the specific research question.

## Label semantics

An NDC identifies a listed drug product or package context; it is not proof of FDA approval, present marketing status, or clinical equivalence. The [FDA's NDC Directory guidance](https://www.fda.gov/drugs/drug-approvals-and-databases/national-drug-code-directory) explicitly warns that assignment of an NDC does not denote approval. A visual class label must not be converted into a medical conclusion without independent authoritative verification.
