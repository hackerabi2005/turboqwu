# Rules of Engagement (kvpipe)

1. Work phase by phase. Start phase N+1 only after phase N's Done-when checks pass. After every task, log what ran, the command, the result file and the status in PLAN_STATE.md so a fresh session can resume.
2. Stop and ask the owner at checkpoints CP0 to CP4. Never guess the owner's workload, accuracy tolerance or GPU budget.
3. Never state a number you did not measure. Label unmeasured values as expected and name the source from section 3 of PLAN.md.
4. Pin everything: model and dataset revisions, third-party git commits, package versions (lockfile) and seeds.
5. Before any download over 5 GB or GPU job over 30 minutes, state the estimate and get confirmation. Run a small pilot first and extrapolate from it.
6. Do not enable trust_remote_code for models the owner has not named. The official TriAttention calibrator uses it.
7. Never edit third-party repos in place. Keep patches in third_party/patches/ with a written reason.
8. Depend only on the projects in section 3 of PLAN.md. Projects in section 9 are not dependencies.
9. Every result file records engine and version, model and revision, weight dtype, effective KV dtype, compression settings, git SHA, seed and timestamp.
10. Make small commits, each with tests. If P0.2 finds no NVIDIA GPU, stop and ask whether to use a remote GPU machine or to build only Phases 1, 3 and 7 against the stub engine.
