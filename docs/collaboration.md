# Collaboration in ResearchLens

## 1. Overview
ResearchLens Phase 11 introduces a multi-researcher collaboration model allowing research labs, peer review groups, and distributed study teams to work synchronously within a shared workspace.

## 2. Workspace Membership & Roles
Workspaces support multiple collaborators managed via `WorkspaceMember`:

| Role | Permissions |
|:---|:---|
| **OWNER** | Full administrative rights: manage members, edit workspace settings, delete workspace, full access to all research artifacts. |
| **RESEARCHER** | Full research rights: add papers, bookmark evidence, create questions, hypotheses, experiment plans, import results, draft findings. |
| **REVIEWER** | Read, comment, and review rights: view all items, comment on discussions, approve/reject hypotheses, experiments, and findings. |
| **VIEWER** | Read-only access across all workspace components. |

## 3. Contextual Comments & Discussions
- **`ResearchComment`**: Comments can be attached directly to any target entity:
  - `PAPER`: Methodological critiques, relevance notes.
  - `EVIDENCE`: Questions on quotes, limitation discussions.
  - `QUESTION`: Refinements, scope adjustments.
  - `HYPOTHESIS`: Theoretical considerations, variable justifications.
  - `EXPERIMENT`: Hyperparameter suggestions, protocol reviews.
  - `RESULT`: Metric anomalies, evaluation split questions.
  - `RESEARCH_PLAN`: Milestone scheduling, budget feedback.
- **`DiscussionThread`**: Dedicated multi-turn discussions with titles, timestamps, and threaded messages.

## 4. Shared Evidence Annotations
- Evidence bookmarks are shared across the workspace rather than duplicated per user.
- Bookmarks track `created_by`, `visibility` (`WORKSPACE`, `PRIVATE`), and dynamic `comments_count`.

## 5. Team Activity Timeline & Security Auditing
- **Activity Feed**: Visible in the workspace dashboard showing recent events across all collaborators.
- **Audit Logging**: Every create, update, delete, import, review, and export action is permanently logged as an `AuditEvent`. Credential tokens and secrets are stripped prior to storage.
