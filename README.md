# Basketball Scoresheet Parser

[![Tests](https://github.com/renato-lucindo/basketball-scoresheet-parser/actions/workflows/tests.yml/badge.svg)](https://github.com/renato-lucindo/basketball-scoresheet-parser/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Basketball Scoresheet Parser extracts structured data from basketball scoresheets. It was initially developed around the FECABA format and provides validation and review workflows for scanned documents.

The project helps developers, analysts, and basketball organizations transform scoresheets into structured data.

The project prioritizes correctness over raw automation: uncertain observations should remain reviewable instead of becoming silently accepted data. See [Project Goals](docs/project-goals.md) for the product direction, v1.0 quality gates, and scope boundaries.

## Features

- Team and player extraction
- Participation and starting lineup detection
- Score event extraction by period
- Team and individual foul analysis
- Handwriting recognition experiments
- Dataset and review workflows
- Confidence-based validation

The [human review workflow](docs/review-workflow.md) explains how candidates move through Label Studio and return as hash-validated reusable data.

## Technology Stack

- Python 3.11+
- Pydantic
- OpenCV / PyMuPDF (optional)
- PyTorch (optional)
- Pytest

## Architecture

The pipeline is organized into normalization, extraction, recognition, validation, and structured output stages.

The [structured output contract](docs/output-schema.md) documents decision states, team results, and the machine-readable core-field completeness inventory.

## Getting Started

Install development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Run tests:

```powershell
python -m pytest -q
```

Analyze a FECABA scoresheet:

```powershell
scoresheet-parser analyze game.pdf `
  --handwriting-model-dir models/handwriting `
  --output result.json
```

Automatic roster recognition currently emits review candidates because M1 found no acceptance threshold that meets the quality gate. Use `--roster-a` and `--roster-b` to provide reviewed rosters. The parser marks unavailable or contradictory core fields as `unresolved` or `review`. See the [output contract](docs/output-schema.md) for optional written-score context and result semantics.

## Project Structure

- `src/sumula_reader/`: application code
- `tests/`: automated tests
- `datasets/`: datasets and evaluation assets
- `models/`: local model checkpoints
- `docs/`: developer documentation

## Roadmap

- Build a trusted real-data evaluation set
- Complete the end-to-end FECABA parser
- Meet the defined reliability and automation quality gates
- Validate generalization to unseen documents and writers
- Make human review efficient and reproducible
- Release FECABA v1.0 before expanding to additional scoresheet formats

Detailed milestones and scope boundaries are maintained in [Project Goals](docs/project-goals.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License.
