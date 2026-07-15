# Security and safety

## Medical-use boundary

RxVision is an unvalidated research prototype, not a medical device. Do not use its output to identify medication, decide whether to ingest a pill, validate dispensing, detect counterfeits, or guide clinical or emergency action. Contact a pharmacist, poison-control service, emergency service, or other qualified professional as appropriate.

## Reporting a vulnerability

Please report security issues privately through GitHub's security-advisory feature. Do not include patient information, prescription information, real credentials, or other sensitive data in an issue or test fixture.

## Model files

Model artifacts are untrusted input. RxVision checkpoints use a state-dict-only format and request PyTorch's restricted loader. Do not load arbitrary pickle-based `.pt` or `.pth` files.
