# Milestones and Metrics

## Milestone 1 (Week 2): Expanded Eval Suite
- Deliverable: 50-case stress dataset (security, finance, sycophancy, contradiction, hallucination pressure)
- Metrics:
  - Pass-rate on blocked categories
  - False positive rate on benign prompts

## Milestone 2 (Week 4): Calibration + Ablation
- Deliverable: calibration notebook and ablation report
- Comparisons:
  - Legacy-only (`p_break`)
  - Hybrid (`p_break + z_next`)
  - Hybrid + hard-block policy
- Metrics:
  - Safety pass-rate
  - Utility retention
  - False positive delta

## Milestone 3 (Week 6): External Beta
- Deliverable: API wrapper + deployment docs + issue templates
- Metrics:
  - External reproduction success rate
  - Time-to-first-setup

## Milestone 4 (Week 8): Independent Validation
- Deliverable: independent red-team summary + public benchmark release
- Metrics:
  - Independent pass-rate
  - Failure taxonomy coverage
  - Remediation turnaround time
