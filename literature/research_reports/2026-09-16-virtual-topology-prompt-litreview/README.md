# Virtual topology prompt review and design decision

Completed 2026-09-16. Five literature threads, a verification pass, a synthesis, then an adversarial
review and a feasibility audit; the architecture decision is v2 of the specification below. The
architecture is specified and **gated**, not implemented or trained.

Read in this order:

1. [Verification and corrections](phase3_verification.md): theorem scope, source checks, probe
   limitations and prior test exposure. This overrides overstatements in the thread reports.
2. [Synthesis](phase4_synthesis.md): evidence, rejected alternatives and the first decision.
3. [Adversarial review](phase5_adversarial_review.md): returned "do not proceed as specified", with
   seven ranked objections and a one-GPU-hour ceiling test.
4. [Feasibility audit](phase5_feasibility_audit.md): returned "buildable with named changes"; the
   structural stream is the dominant cost, and five spec assertions about this codebase were false.
5. [Final specification, v2](../../../docs/superpowers/specs/2026-09-16-motif-graph-grit-prompt-design.md):
   the decision, the nine amendments and the gate. Sections 0.1–0.3 are the decision record.

Original inputs, retained as provenance:

| Report | Subject |
|---|---|
| [Brief](00_brief.md) | Original scope and five assignments |
| [A](phase2_bibliography_A.md) | Graph prompting and L3-PPI |
| [B](phase2_bibliography_B.md) | PPI structural templates |
| [C](phase2_bibliography_C.md) | Graph transformers and pair readouts |
| [D](phase2_bibliography_D.md) | Cold-start structure transfer and coarsening |
| [E](phase2_bibliography_E.md) | Discriminative structure learning |
| [Dataset probes](../../../docs/tmp/template_discriminativeness/README.md) | Own-graph measurements; read with the verification corrections |

The five bibliography entry counts overlap. Their extraction is not a meta-analysis and does not prove
a novelty claim. Cached PDFs are local, gitignored assets; authoritative paper links remain in the
bibliographies and the synthesis. No new model result or held-out improvement is claimed anywhere in
this folder, and the test universe was already explored by the probes, which is disclosed in the
specification's evidence limits.
