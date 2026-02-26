# Project Organization Plan

## Phase 1 (applied, non-breaking)
- Add root `README.md` with folder roles.
- Add `.vscode/settings.json` to declutter Explorer/Search.
- Keep current script paths unchanged to avoid import/runtime breakage.

## Phase 2 (safe cleanup, optional)
- Remove stray root files that look like accidental CLI outputs:
  - `--out`, `--noise-out`, `--subject`, `--subjects`, `--trial`, `--window-sec`
- Move ad-hoc analysis images from root into `reports/figures/`.
- Move one-off CSV summaries into `reports/tables/`.

## Phase 3 (deeper refactor, optional)
- Convert core scripts into a package (e.g. `src/emg_pipeline/`).
- Update imports in `ml/*.py` from root-file imports to package imports.
- Add CLI entry points (`python -m emg_pipeline...`) and tests.

## Guardrails
- Do not move raw subject folders.
- Do not rename dataset/run folders until scripts are updated.
- Validate key scripts after each refactor step.
