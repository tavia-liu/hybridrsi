# rod_8mm_final (imported from assembly-nist cap/clean/autoloop; NOT the HybridRSI configuration)

Kept under its original name so it is not mistaken for a HybridRSI run. This is the earlier code-as-policy result on the
8 mm rod (R8), produced 2026-11-08 by assembly-nist cap/auto/loop.py at the API level "final". Its inputs are richer than
anything a HybridRSI policy gets:

| | rod_8mm_final (this run) | HybridRSI autoloop (e.g. runs/autoloop/peg_4mm_auto) |
|---|---|---|
| task | R8, 8 mm round rod | P4, 4 mm rectangular peg (0.15-0.3 mm radial clearance) |
| perception | api.camera() with the part instance mask + depth, api.part_model() (CAD), api.locate_part() (FoundationPose tracker -> part pose) | two RGB-D cameras, part orange / fixture blue; no mask, no CAD, no pose |
| starts | board starts from the frozen eval bank (data/banks_board) | board starts from seeds; held-out = seeds 0..50 |
| horizon | 40 s (1000 control steps) | 6000 control steps (240 s) |
| held-out protocol | first 100 bank starts | 51 seeds |
| held-out result | **28/100 (28%)**, mean 32 s on successes (eval_r25.json) | 0/51 for peg_4mm_auto |
| debug set at the end | 3/5 | 2/5 |

Files: BRIEF.md (what the coder was told), plan.md + stages/ + policy.py (the final program), rounds/ (per-round programs),
findings.md / diagnosis.md (the coder's own account), log.jsonl (one record per round), starts.json, state.json,
eval_r25.json (per-episode held-out outcomes). The full run with attempts/, experiments/ and the eval videos (840 MB) is on
engr2 under hybridrsi/runs/autoloop/rod_8mm_final/ (not in git).
