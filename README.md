# EMG Data + ML Workspace

This workspace contains raw recordings, generated datasets, training runs, and ML tooling.

## Categories
- `(<NN>)<Name>/`, `Example_subject/`: raw subject trial recordings (source data)
- `dataset_*`, `runs_*`, `_cache_emg/`: generated data and training outputs
- `EMG_opener.py`, `plots_*.py`, `ml/`: processing, export, labeling, and model code
- `docs/`: project/spec/setup documentation
- `reports/figures/`, `reports/tables/`: analysis outputs and summaries
- `tools/`: maintenance utilities (for example file audits)
- `tmp/argparse_artifacts/`: accidental CLI output artifacts

## Important paths
- Pipeline spec: `docs/PROJECT_SPEC.md`
- Organization plan: `docs/ORGANIZATION_PLAN.md`
- Claude Code setup: `docs/SETUP_CLAUDE_CODE_VSCODE.md`

## Why some files may not open
A large fraction of files are Dropbox cloud placeholders (`dataless` flag on macOS).
They appear in Finder/VS Code but have no local bytes yet, so previews fail until downloaded.

Audit placeholders:
```bash
./tools/audit_dataless_files.sh . | head -n 50
```

## Notes
- Current repo has `.git/index.lock`; clear it before normal git operations.
