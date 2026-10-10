# ResearchLens Scientific Model Registry

## 1. Scientific Governance & Human Approval

The ResearchLens Model Registry governs models discovered or trained during research projects. Unlike typical industrial registries that auto-promote based on arbitrary score thresholds, **ResearchLens strictly enforces human-in-the-loop scientific review**.

```
Experiment Execution
        │
        ▼
Completed Run & Validated Results
        │
        ▼
Artifact Hash Calculated (SHA-256)
        │
        ▼
Candidate Registration (STAGE: CANDIDATE)
        │
        ├────────────────────────────┐
        ▼                            ▼
Human Reviewer Approval      Unapproved Promotion
        │                            │
        ▼                            ▼
STAGE: VALIDATED                  REJECTED
        │
        ▼
Human Approver Approval
        │
        ▼
STAGE: PRODUCTION
        │
        ▼
STAGE: ARCHIVED
```

---

## 2. Model Lifecycle Stages

| Stage | Definition | Prerequisites |
|---|---|---|
| `NONE` | Unregistered artifact | N/A |
| `CANDIDATE` | Registered candidate model | Run status `COMPLETED`, `results.json` valid, artifact SHA-256 computed. |
| `VALIDATED` | Verified by a human peer reviewer | Formal review notes, non-empty reviewer ID with `REVIEWER`+ permissions. |
| `PRODUCTION` | Approved for primary research pipeline | Must currently be in `VALIDATED` stage; direct skip from `CANDIDATE` to `PRODUCTION` is prohibited. |
| `ARCHIVED` | Retired model | Retains full immutable version history. |

---

## 3. Strict Model Registration Rules

A model can only be registered if:
1. The source run exists and has status `COMPLETED`.
2. A valid artifact exists and its SHA-256 hash has been verified.
3. The specified model version does not collide with an existing version (non-destructive versioning).
4. An empirical evaluation metric and dataset are documented.

---

## 4. Scientific Comparability vs. "Best Model"

MLflow does not decide scientific superiority. When comparing models in the ResearchLens UI or service layer:
- We never declare a model "universally best".
- We report: **"Highest observed metric among selected comparable runs."**
- Cross-paper and baseline comparability rules (Phase 6 and Phase 11) remain authoritative.
