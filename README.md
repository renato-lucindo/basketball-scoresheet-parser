# Basketball Scoresheet Parser

[![Tests](https://github.com/renato-lucindo/basketball-scoresheet-parser/actions/workflows/tests.yml/badge.svg)](https://github.com/renato-lucindo/basketball-scoresheet-parser/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Basketball Scoresheet Parser extracts structured data from basketball scoresheets. It was initially developed around the FECABA format and provides validation and review workflows for scanned documents.

The project helps developers, analysts, and basketball organizations transform scoresheets into structured data.

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

- Support more scoresheet formats
- Improve recognition models
- Expand evaluation datasets
- Improve contributor tooling

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License.
