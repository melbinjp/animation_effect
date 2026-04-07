---
name: "Consolidate Branches And Productionize Site"
description: "Use when consolidating repo branches into main, deleting merged branches safely, and turning a web project into a production-grade institutional website"
argument-hint: "Optional: target branch, brand brief, must-keep features, deployment constraints"
agent: "agent"
model: "GPT-5 (copilot)"
---
Consolidate this repository into a production-ready mainline and upgrade the website to an institutional standard.

Use the current workspace as the source of truth, especially [index.html](../../index.html), [style.css](../../style.css), [script.js](../../script.js), and assets under [vendor](../../vendor).

Inputs to honor:
- User goal or brand brief: ${input:Optional brand brief or institutional direction}
- Target branch: ${input:Target branch to consolidate into, default main}
- Must-keep features: ${input:Features that must survive consolidation and redesign}
- Constraints: ${input:Hosting, browser, accessibility, or deployment constraints}

Complete the task in this order:

1. Inspect the repository state.
- Enumerate local and remote branches.
- Identify what each branch contributes relative to the target branch.
- Detect whether the target branch already exists; if it does not, create it from the most suitable base after explaining the choice.

2. Build a safe consolidation plan.
- Prefer merge or cherry-pick strategies that preserve working features.
- Call out conflicts, risk areas, and uncertain branch intent before applying destructive actions.
- Do not delete any branch until its changes are confirmed as preserved in the target branch.

3. Consolidate features into the target branch.
- Merge or port meaningful work from other branches.
- Resolve conflicts carefully and keep the strongest implementation when branches overlap.
- Preserve existing functionality unless the user explicitly asks to remove it.

4. Upgrade the site to a production-grade institutional webpage.
- Improve information architecture, content hierarchy, semantic HTML, accessibility, responsiveness, and performance.
- Replace placeholder or hobby-grade presentation with a credible institutional visual system.
- Keep the design intentional and polished rather than generic.
- Preserve or refine distinctive effects only when they support the final experience.

5. Harden implementation quality.
- Eliminate brittle code paths.
- Tighten CSS structure and JS behavior.
- Ensure the page works on desktop and mobile.
- Verify there are no obvious regressions.

6. Clean up branches only after verification.
- Delete only branches that are fully merged or explicitly superseded.
- Keep any branch that still contains unresolved or intentionally deferred work, and explain why.

7. Report the result clearly.
- Summarize which branches were consolidated.
- List any branches deleted and any intentionally retained.
- Summarize website improvements.
- Call out residual risks, follow-up work, or anything that still needs user confirmation.

Working rules:
- Act autonomously where safe, but pause before irreversible git actions if branch intent is ambiguous.
- Prefer root-cause fixes over cosmetic patches.
- Keep edits focused and production-oriented.
- Validate the final state with available checks or manual inspection steps.