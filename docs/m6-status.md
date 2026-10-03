# M6 Status

Status snapshot: 2026-10-03.

M6 — FECABA v1.0 is **complete**.

## Release contract

- Package version: `1.0.0`.
- Structured output schema: `1.0`.
- Supported inputs: single-page FECABA PDF, PNG, JPG, and JPEG sources.
- Public API and CLI: documented in the README and output contract.
- Release evidence: [FECABA v1.0 Release Record](release-v1.0.md).
- Release notes: [`CHANGELOG.md`](../CHANGELOG.md).
- Release tag: `v1.0.0`.

## Quality evidence

The release is bound to corpus SHA-256 `92f9e8d1124a2839b2029bc4e55d66309800629916c38e07d9031f2e4a726ba7` and model fingerprints embedded in the committed reports.

The global acceptance policy is `review_all`. Held-out and generalization reports contain zero automatically accepted decisions, zero accepted errors, 0% automation, and 100% review. This satisfies the maximum 1% accepted-error gate without overstating recognition quality.

## Validation

- Full working-copy suite: 121 passed, 3 subtests passed.
- Clean local clone with a new virtual environment and `.[dev,vision,ml,jev]`: 121 passed, 3 subtests passed.
- GitHub Actions on the release PR: passed.
- Package import reports version `1.0.0`.
- `scoresheet-parser schema` reports schema version `1.0`.
- `git diff --check`: passed.

## Exit-criterion evidence

- All v1.0 quality gates and M1-M5 milestone criteria have committed evidence.
- A clean checkout installs and passes the full suite.
- The public output contract and supported inputs are documented.
- The release record identifies the corpus, metrics, model checkpoints, and quality policy.
- The merge commit is tagged `v1.0.0` after the release PR is integrated.

Future work belongs to the post-v1.0 roadmap and must preserve the same evidence and review guarantees.
