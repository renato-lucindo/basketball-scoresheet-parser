# Changelog

All notable changes to this project are documented in this file.

## [1.0.0] - 2026-10-03

### Added

- End-to-end FECABA image and PDF parsing with structured JSON output.
- Explicit accepted, review, and unresolved states for the complete core game record.
- Participation, starter, scoring-event, period-score, final-score, individual-foul, and team-foul extraction paths.
- Reviewed roster and written-score overrides with reconciliation warnings.
- Reproducible dataset audit, held-out evaluation, writer isolation, and model fingerprinting.
- Component, end-to-end, capture-condition, and unseen-writer reliability reports.
- Hash-validated Label Studio review and second-review workflow.
- Open-source governance, security, contribution, architecture, development, and release documentation.

### Quality policy

- FECABA v1.0 uses `review_all` because no global threshold satisfies the accepted-error and automation quality gate on the defined held-out evidence.
- Automatically accepted recognition decisions: 0.
- Accepted error rate: 0%.
- Review rate: 100%.

[1.0.0]: https://github.com/renato-lucindo/basketball-scoresheet-parser/releases/tag/v1.0.0
