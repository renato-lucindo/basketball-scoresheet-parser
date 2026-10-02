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

## Technology Stack

- Python 3.11+
- Pydantic
- OpenCV / PyMuPDF (optional)
- PyTorch (optional)
- Pytest

## Architecture

The pipeline is organized into normalization, extraction, recognition, validation, and structured output stages.

## Getting Started

Install development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Run tests:

```powershell
python -m pytest -q
```

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
