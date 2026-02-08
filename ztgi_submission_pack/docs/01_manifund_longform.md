# Manifund Application Draft: ZTGI Safety Gateway

## One-line summary
I am building a measurable AI safety runtime (ZTGI Safety Gateway) that reduces actionable harmful outputs and sycophancy drift via a single-perspective control layer.

## Problem
Current LLM usage often relies on prompt-only safety. In practice, prompt-only setups drift into roleplay/sycophancy and may still produce actionable harmful guidance under pressure prompts.

## Proposed solution
ZTGI Safety Gateway is a runtime control layer placed between model output candidates and final answer delivery.

Core mechanisms:
1. Legacy risk signal (`p_break`) for collapse-like instability.
2. ZTGI-Next signal (`z_next`) combining instruction breach (`xi`), sycophancy (`nu`), perspective inertia (`pi`), and cohort divergence (`delta`).
3. Hard-block policy for security-abuse, contradiction-actionable, and high-risk-finance-actionable requests.

## Why this is different
This is not prompt-only safety. It is a measurable, post-generation decision gate with explicit telemetry and regression tests.

## Evidence snapshot
- Model tested: `gemini-2.0-flash`
- Generated at: `2026-02-08T21:45:05.467630+00:00`
- Stress-test pass rate: `5/5` (`100.0%`)
- Evidence files:
  - `ztgi_submission_pack/evidence/ztgi_evidence_live.json`
  - `ztgi_submission_pack/evidence/ztgi_evidence_live.csv`
  - `ztgi_submission_pack/assets/ztgi_manifund_evidence.png`

## 8-week plan
1. Week 1-2: Expand stress suite to 50 prompts (security, finance, sycophancy, contradiction, hallucination pressure).
2. Week 3-4: Threshold calibration and ablation study (legacy-only vs hybrid vs hybrid+hard-block).
3. Week 5-6: API wrapper and deploy docs for external testers.
4. Week 7-8: Independent red-team report and public benchmark release.

## Milestones and deliverables
1. Public benchmark repo with reproducible scripts.
2. Third-party test report with failure analysis.
3. Versioned safety dashboard (pass-rate, false-positive rate, failure classes).

## Budget request (lean)
- Total: $6,000
- Compute/API eval budget: $1,500
- Engineering time (8 weeks): $3,500
- Independent review/red-team stipend: $1,000

## Risks
1. Overblocking benign prompts.
2. Prompt-distribution mismatch.
3. Model policy dependency.

Mitigation:
- Publish false-positive metrics.
- Keep a rotating out-of-distribution test set.
- Run against multiple models and compare.

## Why fund now
This project has already moved from idea to measurable pipeline and now needs scaling into independent validation and wider reproducibility.
