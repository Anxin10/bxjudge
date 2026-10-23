# Codex Port and Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the valuable pieces of `codex/eval-reliability` and `codex/recorded-calibration` onto `feat/local-judge` by hand, without merging those branches, and close the data-protection and evaluation-strictness gaps the comparison found.

**Architecture:** This is incremental hardening of existing modules. Each task cites the Codex commit that holds reference code (`git show <sha>`); the port must fit our current interfaces, not Codex's. Both Codex branches stay untouched.

**Tech Stack:** Python 3.12, OpenCV, Streamlit, pytest, uv. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-23-auto-calibration-design.md` (auto calibration) and `docs/spec.md` (judge, §11 success criteria). The comparison reports `.superpowers/sdd/2026-10-23-auto-calibration/branch-compare-eval-reliability.md` and `branch-compare-recorded-calibration.md` list every behaviour this plan ports. The user approved batches A, B, C and D on 2026-10-24.

## Decisions (made for the user; recorded as rulings in the plan's ledger)

- **R66 (replaces R61's pass rule; user chose batch C):** the speed criterion passes only if EVERY real video that was measured has `processing (wall) ≤ playback duration`. This is the worst-video rule, and it uses the measured per-video wall time. A video without a per-video wall measurement makes speed `unverified`, which never counts as a pass. The pooled batch wall ratio and the median ratio remain as diagnostics.
- **R67 (replaces R63's extrapolation; batch C):** the trade-off block no longer extrapolates `auto_accuracy` / `final_accuracy` over unreviewed rounds. "Overall accuracy" is shown only when every real round has truth. It always shows a conservative lower bound: correct reviewed rounds ÷ all real rounds, so unknown truth counts as error. Automation rate (`1 − review_ratio`) and speed stay.
- **R68 (batch B, data protection):** when a previously judged video's re-analysis fails transiently (`decode_error`, ffprobe/probe failure, `analysis_error`), its existing rows are kept untouched and its processed record is left for retry. This is NOT a placeholder replacement. It keeps T4-3, so vetoed rows survive a failed rerun. A deliberate `recalibrate` (selection returns None because calibration was removed/changed) still follows spec §6.3b. `videos.json[video]["analysis_status"]` ∈ {`ok`, `recalibrate`, `decode_error`, `analysis_error`} records the latest attempt explicitly. `evaluate` uses it instead of inferring from `calibration: null`.
- **R69 (batch B, audit freeze):** `bxjudge audit-sample` writes `audit_sample.json` with seed, n and the exact (video, round) population and selection. Re-marking with a different population requires `--replace`. Re-analysis is still allowed, because calibration correctness wins. `eval` checks the recorded selection against `rounds.json`. If any sampled row is missing or was replaced, the audit estimate is `INVALIDATED (resample)`: unverified, never a pass, and never computed from a silently shrunk sample. `read_results` rejects audit marks that have no `audit_sample.json`.

## Global Constraints

- Never merge, rebase, check out or modify `codex/eval-reliability`, `codex/recorded-calibration` or their worktrees. One worktree is stuck mid-merge; leave it alone. Read Codex code only through `git show <sha>:<path>`.
- Never stage the user's modified `.gitignore` or these untracked files: `.agents/ AGENTS.md CLAUDE.md .claude.code-workspace antigravity.db .graftignore .graphifyignore graphify-out/ src/graphify-out/`.
- All thresholds and policy live in `config/*.yaml` (R40). There are no compatibility fallbacks (project rulings).
- Keep earlier rulings intact: T2-1, T3-1, T4-1/T4-3/T4-4/T4-5, F-1, F-2 (auto-calibration ledger), and R62/R64/R65 (local-judge ledger), except where R66–R69 explicitly replace them.
- Suite stays green (`uv run pytest -q`, 938 at start). `uvx ruff check src tests` stays at the 24 baseline.

## Review Focus

- Results dirs written before this plan, which have no `analysis_status` and no `audit_sample.json`. Status is unknown, so they must fail loudly or show as unverified, never pass silently (Tasks 4, 5).
- A rerun after a veto when decoding fails: the vetoed rows must still be there (Task 4).
- Speed with one slow outlier video and all others fast: speed must FAIL (Task 6).
- `--out` equal to `--calib`, or nested either way: refused before anything is written (Task 1).
- Ctrl+C during `bxjudge run`: the batch exits 130 with `cancelled`, and processed records written so far stay valid (Task 2).

---

### Task 1: Snapshot and output-path validation

**Files:** Modify `src/bxjudge/calibration_store.py`, `src/bxjudge/cli.py`, `src/bxjudge/pipeline.py`, `src/bxjudge/output.py`. Tests: `tests/test_calibration_store.py`, `tests/test_cli_autocalib.py`, `tests/test_pipeline.py`, `tests/test_output.py`.
**Reference:** codex d6bae8d (snapshot shape check), 55a4233 + 17b8428 (`validate_output_paths`).

**Interfaces:**
- `load_snapshot` raises `ValueError("<path>: malformed calibration snapshot (...)")` for a wrong top-level shape, missing or ill-typed keys, unknown `kind`, a polygon that is not a list of 2-number points, or a missing ref image. Every caller already maps `ValueError` to the CLI `error:` path, or to the per-video analysis error.
- `validate_output_paths(out_dir: Path, calib_dir: Path) -> None` lives in `calibration_store.py`. It raises `ValueError` if the resolved `out_dir == calib_dir`, if either is inside the other, or if `out_dir` is a file. Both `run_batch` and `bxjudge autocalib` call it before writing anything.
- `bxjudge autocalib --truth` PASS requires `error < max_boundary_error_px`, strictly. Spec §6.6 says 小於.
- `output.read_rounds` raises `ValueError("<path>: rounds.json must be a list of rows")` for any other JSON type.

- [ ] **Step 1: Failing tests.**
  - `{"snapshot": []}` and a snapshot with polygon `[[1]]` → `ValueError` mentioning "malformed".
  - `run_batch` and `autocalib` with `--out calib` → `error:` exit 2 (CLI) / `ValueError` (API), and `calib/` unchanged.
  - Error exactly 8.0 → FAIL.
  - `rounds.json` = `{}` → `ValueError`.
- [ ] **Step 2: Run them; they fail. Implement. Run focused tests, then the full suite and ruff.**
- [ ] **Step 3: Commit** `fix: validate calibration snapshots, output paths and rounds.json shape`.

### Task 2: Operability (Ollama scripts, CLI, preflight, docs sync)

**Files:**
- Create: `scripts/setup_ollama.py`, `scripts/serve_ollama.sh` (mode 755, LF), `tests/test_ollama_setup.py`.
- Modify: `src/bxjudge/cli.py`, `src/bxjudge/vlm.py`, `README.md`, `docs/superpowers/specs/2026-10-23-auto-calibration-design.md` (§6.3b items 3–4), `docs/superpowers/plans/2026-10-23-auto-calibration.md` (Task 3 text).
- Test: `tests/test_vlm.py`, a new `tests/test_operability.py` (port only what fits our API).

**Reference:** codex 95073db (scripts, CLI, operability tests), 4f4ec3b (preflight version check), 17b8428 (the spec/plan T4-3 wording; see `branch-compare-recorded-calibration.md` #23 for the exact edits).

**Interfaces:**
- `setup_ollama.py`:
  - Downloads the pinned Ollama 0.40.7 archive to a temp path.
  - Verifies size and SHA256 (`7f082e625b4c334df84ed1558fe282d6d8091fe3545790aef30b3a9c50d44d41`) before extracting.
  - Extracts with path-traversal protection into `weights/ollama-runtime`.
  - Refuses to overwrite an existing runtime unless `--force`.
  - Never pulls models.
- `serve_ollama.sh` runs Ollama in the foreground with `OLLAMA_HOST=127.0.0.1:11435`, `OLLAMA_MODELS=$PWD/weights/ollama` and `OLLAMA_NO_CLOUD=1`. The README server section uses both scripts.
- CLI:
  - `--debug` is accepted before or after the subcommand.
  - `KeyboardInterrupt` → stderr `cancelled`, exit 130.
  - `RuntimeError` is still not caught (keep ours).
- VLM preflight treats an empty or missing `version` in the `/api/version` reply as unreachable, with the reason in the note.
- The spec/plan text now states T4-3: a veto alone never deletes rows, and reanalysis happens only when a non-vetoed snapshot becomes available.

- [ ] **Step 1: Failing tests.**
  - Setup script: a checksum mismatch → no extraction; a traversal entry in the archive → refused; an existing runtime without `--force` → refused. Use a local fake archive; no network.
  - `bxjudge run ... --debug` works.
  - Ctrl+C → exit 130.
  - Preflight `{"version": ""}` → unreachable.
- [ ] **Step 2: Run them; they fail. Implement. Run the full suite and ruff.**
- [ ] **Step 3: Commit** `feat: ollama setup/serve scripts, CLI --debug anywhere and Ctrl+C, strict preflight, T4-3 docs`.

### Task 3: Review UI marks vetoed rounds

**Files:** Modify `src/bxjudge/review_app.py`, `src/bxjudge/review_ui.py`. Test: `tests/test_review_ui.py`.

**Interfaces:**
- The round list labels every round whose `calibration_status` is `vetoed` with the prefix `⛔ 校準被否決`.
- The sidebar filter gains `校準被否決`.
- The `需覆核` filter includes vetoed unreviewed rounds, consistent with Task 4 of the auto-calibration plan.

- [ ] **Step 1: Failing AppTest tests:**
  - the list label of a vetoed round;
  - the filter selects exactly the vetoed rounds;
  - the 需覆核 filter includes them.
- [ ] **Step 2: Implement, then run the tests.**
- [ ] **Step 3: Commit** `feat: surface vetoed rounds in the review list and filters`.

### Task 4: Keep rows on transient re-analysis failure; explicit analysis_status (R68)

**Files:** Modify `src/bxjudge/pipeline.py`, `src/bxjudge/output.py`, `src/bxjudge/evaluate.py`. Test: `tests/test_pipeline.py`, `tests/test_evaluate.py`.
**Reference:** codex 17b8428 (`analysis_status`, keep vetoed rows on a failed rerun); see `branch-compare-recorded-calibration.md` #29/#30.

**Interfaces:**
- `record_video_status(out_dir: Path, video: str, status: str) -> None` lives in `output.py`. It writes `videos.json[video]["analysis_status"]`, and only the four values above are allowed.
- In `run_batch`, for a video that already has non-placeholder rows, a `decode_error` / probe failure / `analysis_error` result does NOT go through `write_results`. The existing rows stay, the status is recorded, and the processed record is not updated.
- `evaluate` counts recalibrate / decode / analysis failures per video from `analysis_status`. A results dir whose videos lack `analysis_status` → a calibration section with `unverified (no analysis_status; rerun bxjudge run)` and no recalibrate count inferred from `calibration: null`.

- [ ] **Step 1: Failing tests.**
  - Judge, then rerun with a forced decode failure: the rows are byte-identical (including `audit_sample` and `calibration_id`), status is `decode_error`, and the video is retried next run.
  - The same with a vetoed group (T4-3 preserved).
  - A forced `analysis_error` behaves the same way.
  - A first-ever decode failure still writes the placeholder.
  - Eval labels decode vs recalibrate correctly.
  - An old dir without status → unverified.
- [ ] **Step 2: Implement, then run the focused tests and the full suite.**
- [ ] **Step 3: Commit** `fix: transient re-analysis failures keep existing rows; explicit analysis_status`.

### Task 5: Audit freeze (R69)

**Files:** Modify `src/bxjudge/corrections.py` (`mark_audit_sample`), `src/bxjudge/output.py` (`read_results`), `src/bxjudge/evaluate.py`, `src/bxjudge/cli.py`. Test: `tests/test_audit.py`.
**Reference:** codex 4f4ec3b (`audit_sample.json`, population/selection checks, 5 freeze tests).

**Interfaces:**
- `mark_audit_sample(out_dir, n, seed, *, replace: bool = False)` writes `<out>/audit_sample.json` = `{"seed", "n", "population": [[video, round], ...], "selection": [[video, round], ...], "at"}`. When the file already exists and the current population differs, the call raises `ValueError` unless `replace=True`.
- CLI `audit-sample --replace`.
- `read_results`, or the eval loader, raises `ValueError` when rows carry `audit_sample: true` but `audit_sample.json` is missing.
- `evaluate`: if any selected key is missing from `rounds.json`, or its row is no longer marked, the unflagged-accuracy status is `INVALIDATED (resample)`, which is never a PASS, and the report says to rerun `audit-sample --replace`.

- [ ] **Step 1: Failing tests:**
  - sample → rerun replaces a sampled row → eval `INVALIDATED`;
  - re-mark without `--replace` after the population changed → error;
  - with `--replace` → a new sample;
  - marks without the file → error;
  - an unchanged sample → the R64 gate behaves as before.
- [ ] **Step 2: Implement, then run the focused tests and the full suite.**
- [ ] **Step 3: Commit** `feat: freeze audit samples and invalidate estimates when sampled rows change`.

### Task 6: Worst-video speed gate (R66)

**Files:** Modify `src/bxjudge/evaluate.py`, `src/bxjudge/pipeline.py` / `src/bxjudge/output.py` (record per-video wall), `config/evaluation.yaml` if a key is needed. Test: `tests/test_evaluate.py`, a new `tests/test_observed_timing.py`.
**Reference:** codex 5d37757 (observed per-video wall time), 7fff24d (invocation deadlines).

**Interfaces:**
- `record_video_timing` additionally stores `wall_s`: the observed per-video wall time, from the first decode of that video through its write, including that video's selection pass.
- The speed pass/fail uses `max(wall_s / duration_s)` over real measured videos and must be ≤ 1.0. A real video missing `wall_s` → speed `unverified`.
- The report keeps the pooled batch wall ratio and the median ratio as diagnostics, and names the slowest video.

- [ ] **Step 1: Failing tests:**
  - 9 fast videos plus 1 at a ratio of 1.3 → FAIL, and the slowest video is named;
  - all ≤ 1 → PASS;
  - one video without `wall_s` → unverified;
  - `run_batch` records `wall_s` for each video.
- [ ] **Step 2: Implement, then run the focused tests and the full suite.**
- [ ] **Step 3: Commit** `feat: speed passes only when every measured video finishes within its playback time`.

### Task 7: Trade-off block without extrapolation (R67)

**Files:** Modify `src/bxjudge/evaluate.py`. Test: `tests/test_evaluate.py`.
**Reference:** codex 4f4ec3b (trade-off rework).

**Interfaces:**
- Remove the `auto_accuracy` / `final_accuracy` extrapolation from Metrics and the report.
- Add `accuracy_lower_bound: float` = reviewed rounds whose original finish and winner are correct ÷ all real rounds.
- Add `overall_accuracy: float | None`, shown only when every real round is reviewed; otherwise `unverified (truth coverage x/y)`.
- Automation rate and speed stay.
- Update the CLI summary, the report table and the README description.

- [ ] **Step 1: Failing tests:**
  - 10 real rounds, 4 reviewed, 3 of them correct → lower bound 0.3, overall unverified (4/10);
  - all reviewed → overall equals the exact accuracy;
  - the report has no extrapolated rows.
- [ ] **Step 2: Implement, then run the focused tests and the full suite.**
- [ ] **Step 3: Commit** `feat: trade-off block reports a conservative lower bound instead of extrapolated accuracy`.

### Task 8: Alignment weakness as strict xfail tests

**Files:** Modify `tests/test_autocalib.py`. Create `scripts/check_autocalib_real.py`.
**Reference:** codex 17b8428 (ghost and anisotropic test cases only; do NOT port its sector check).

**Interfaces:**
- Add the ghost (displaced-structure, 25 px) and anisotropic (15–16 px) synthetic cases as `@pytest.mark.xfail(strict=True, reason="known: accepted misaligned; see final-review I1")`. Each asserts that the alignment is rejected, or that its error is below 8 px.
- `scripts/check_autocalib_real.py <video> --calib calib` prints the score, the estimated transform and the preview path for one real clip. It is used to document 20261001_203029 (0.615) in the report.

- [ ] **Step 1: Add the tests and confirm they XFAIL (not pass, not error). Run the script on 20261001_203029 and put the output in the report.**
- [ ] **Step 2: Commit** `test: pin known auto-alignment weaknesses as strict xfail cases`.
