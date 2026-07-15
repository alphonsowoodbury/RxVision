# RxVision model card

## Status

No trained model is distributed with this repository. The code defines an experimental EfficientNetV2-S classification path and the metadata contract required to interpret a checkpoint.

## Intended use

- learning and portfolio review;
- controlled computer-vision experiments;
- research into dataset shift, confidence thresholds, abstention, and human review;
- testing data and model interfaces without making medical claims.

## Out-of-scope use

- identifying a pill for ingestion;
- pharmacy dispensing verification;
- emergency, poison-control, diagnostic, or treatment decisions;
- counterfeit or contamination detection;
- unsupervised use by patients or clinicians;
- any representation of FDA clearance, HIPAA compliance, or clinical validation.

## Metrics

There are no current performance metrics. The 2020 capstone reported 93% validation accuracy on a controlled split and roughly 50% on informal real-world photographs. The original checkpoint, exact split, calibration protocol, and independent test set are not available here, so those observations are historical context—not reproducible results or evidence of present performance.

## Required evaluation before any expanded use

1. Create a provenance-reviewed dataset with explicit reuse rights.
2. Deduplicate near-identical views before splitting.
3. Split by an appropriate physical or product grouping to prevent leakage.
4. Reserve an independently collected, untouched test set.
5. Report top-1 and top-k accuracy, per-class recall, calibration error, coverage at abstention thresholds, subgroup behavior, and failure examples.
6. Evaluate distribution shift across cameras, lighting, backgrounds, occlusion, damage, packaging, and visually similar products.
7. Require domain, regulatory, privacy, security, accessibility, and human-factors review.

## Known limitations

- A class score is not proof of identity.
- Softmax confidence can be overconfident under distribution shift.
- Multiple medications can look alike, and one medication can look different by manufacturer or dose.
- Reference imagery may not represent consumer photographs.
- NDC listings and image labels may be incomplete, outdated, or unsuitable for the intended task.
- An abstention threshold reduces coverage but does not create clinical validity.
