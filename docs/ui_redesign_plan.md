# ResearchLens UI/UX Redesign Plan

## 1. Vision & Design Principles

ResearchLens is redesigned as an enterprise-grade scientific intelligence and automated research platform. The user interface prioritizes clarity, scientific credibility, evidence provenance, and seamless workflow progression over decorative gimmicks.

### Core Design Principles:
1. **Editorial Scientific Hierarchy**: High typographic clarity, structured headings, readable tabular numbers, and disciplined whitespace.
2. **Strict Evidence Provenance**: Every extracted finding, Q&A citation, and candidate gap displays explicit source page, section header, and verbatim text.
3. **Structured Navigation**: Replace 13 sprawling horizontal tabs with a logical 6-pillar research workflow.
4. **State & Action Transparency**: Clear, unambiguous loading, empty, success, warning, and approval-gate states.
5. **Accessibility & Contrast**: Full compliance with WCAG 2.1 AA standards; visible focus rings for keyboard navigation.
6. **Zero Regression**: 100% preservation of all existing models, pipelines, services, and security controls.

---

## 2. Design System Architecture (`app/styles/custom.css`)

### Color Palette Tokens:
- **Background Root**: `#0A0F1D` (Deep Cosmic Obsidian)
- **Surface Primary**: `#111827` (Rich Slate Surface)
- **Surface Elevated**: `#1F2937` (Card & Panel Background)
- **Surface Highlight**: `#374151` (Hover & Active States)
- **Border Subtle**: `#2D3748` / `rgba(255, 255, 255, 0.08)`
- **Border Focus**: `#3B82F6` (Accessible focus ring outline: 2px solid)
- **Text Primary**: `#F9FAFB` (High Contrast Neutral)
- **Text Secondary**: `#9CA3AF` (Readable Muted Neutral)
- **Text Tertiary**: `#6B7280` (Subtle Captions & Footnotes)
- **Brand Scientific Cobalt**: `#2563EB` / `#3B82F6`
- **Verified Emerald**: `#10B981` (Confirmed findings, successful runs)
- **Pending Amber**: `#F59E0B` (Approval gates, open hypotheses)
- **Critical Rose**: `#EF4444` (Risks, timeouts, failed executions)
- **Insight Cyan**: `#06B6D4` (Key phrases, algorithmic entities)

### Typography & Spacing:
- **Font Stack**: `'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`
- **Base Font Size**: `15px` / Line Height: `1.6` for long-form research text
- **Code & Identifiers**: `'JetBrains Mono', 'Fira Code', Menlo, monospace`

---

## 3. Navigation Hierarchy Modernization

Consolidate the 13 flat top tabs into 6 cohesive research operational pillars:

```
┌────────────────────────────────────────────────────────────────────────┐
│  ResearchLens App Header: Workspace Status · Health Indicator · Auth    │
├─────────┬──────────────────────────────────────────────────────────────┤
│ Sidebar │ Main Content Viewport                                        │
│ Nav     │                                                              │
│  [1] ───┼─► 1. Research Dashboard (Active Workspace Overview & Stats)  │
│  [2] ───┼─► 2. Paper Analysis & Q&A (Upload, Sections, Intel, Gaps, QA)│
│  [3] ───┼─► 3. Literature Intelligence (Multi-Paper, Matrix, Monitor)  │
│  [4] ───┼─► 4. Collaborative Workspace (Team, Collections, Hypotheses) │
│  [5] ───┼─► 5. Experiments & MLOps (Sandboxed Execution, MLflow Infra) │
│  [6] ───┼─► 6. Research Automation (13-Stage Workflow & Human Review)  │
└─────────┴──────────────────────────────────────────────────────────────┘
```

Within each operational pillar, relevant sub-capabilities are organized with clean, focused sub-navigation tabs to maintain contextual continuity.

---

## 4. Prioritized Implementation Steps

- [x] **Step 1**: Complete codebase audit and capture test baseline.
- [ ] **Step 2**: Rebuild [app/styles/custom.css](file:///c:/Users/msiva/Music/ResearchLens/app/styles/custom.css) with the comprehensive scientific design system and tokens.
- [ ] **Step 3**: Create reusable UI helpers in [app/components/ui_elements.py](file:///c:/Users/msiva/Music/ResearchLens/app/components/ui_elements.py) (status badges, metric cards, citation panels, empty states).
- [ ] **Step 4**: Implement [app/components/dashboard_view.py](file:///c:/Users/msiva/Music/ResearchLens/app/components/dashboard_view.py) computing real metrics from `WorkspaceStore` and `HealthCheck`.
- [ ] **Step 5**: Modernize [app/components/upload.py](file:///c:/Users/msiva/Music/ResearchLens/app/components/upload.py) and [app/components/summary_view.py](file:///c:/Users/msiva/Music/ResearchLens/app/components/summary_view.py).
- [ ] **Step 6**: Modernize [app/components/qa_view.py](file:///c:/Users/msiva/Music/ResearchLens/app/components/qa_view.py) and [app/components/metadata_view.py](file:///c:/Users/msiva/Music/ResearchLens/app/components/metadata_view.py).
- [ ] **Step 7**: Modernize [app/streamlit_app.py](file:///c:/Users/msiva/Music/ResearchLens/app/streamlit_app.py) with the 6-pillar application shell, breadcrumbs, and workspace status bar.
- [ ] **Step 8**: Run full test verification (`compileall`, `ruff`, `pytest`) and ensure 100% passing tests with zero regressions.
- [ ] **Step 9**: Document component inventory in `docs/ui_component_inventory.md` and verification report in `docs/ui_verification_report.md`.
