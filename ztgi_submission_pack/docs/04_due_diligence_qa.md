# Due Diligence Q&A (Ready Answers)

## Q1: Why is this not just prompt engineering?
ZTGI is a runtime gate after candidate generation. It computes risk telemetry (`p_break`, `z_next`) and can hard-block actionable outputs even when the base model proposes them.

## Q2: What is new here?
The novelty is not a claim of AGI. The novelty is a measurable control layer for:
1. Instruction-breach detection (`xi`)
2. Sycophancy detection (`nu`)
3. Contradiction and abuse hard-blocking
4. Single-perspective arbitration across candidate responses

## Q3: How do you avoid overblocking?
We track false positives as a first-class metric and run threshold calibration + ablation in weeks 3-4.

## Q4: Is this model-dependent?
Partly yes. That is why the evaluation plan includes multi-model comparison and policy-independence analysis.

## Q5: What is the strongest current evidence?
Live stress suite currently reports 5/5 passes on high-risk scenarios with explicit hard-block reasons in telemetry.

## Q6: Main project risk?
Over-conservative behavior on benign prompts. Mitigation: measured threshold tuning and external tester feedback loop.
