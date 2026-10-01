# Review-added analysis plan (fixed before the new solves)

Purpose: answer the pasted review with hydrogen-storage design evidence, not
algorithm complexity or claims of hydrogen-exclusive mathematics. This plan
does not replace or tune the frozen primary results. New analyses are explicitly
review-added scenario diagnostics, not observed clusters or preregistered tests.

| Block | Factor changed | Fixed settings | New cases | Endpoint and interpretation |
|---|---|---|---:|---|
| Wind/PV composition | Design-period wind energy share 0.25 or 0.75 instead of 0.50 | Three previously declared anchors; 2015/2019/2023; overbuild 1.5; 2006–2014 sizing; original process, product and storage rules | 36 (2 shares × 3 anchors × 3 years × rigid/60% package) | Continuous one-unit service capacity and paired gain. Parametric weather-derived configurations, not new observed facilities. Range overlap cannot establish sample representativeness. |
| Reference working volume | Medium or large Hystories volume instead of small | Same nine anchor–year inputs, unchanged capacities/process/buffer; smallest whole-volume ceiling covering rigid free inventory; common ceiling within each pair; injection rating W/2 | 36 (2 volumes × 9 pairs × 2 process cases) | Minimum directional rating and paired relief. Separates volume-convention dependence; does not calibrate new wells or compressor costs. |
| Grid boundary | G/Gcrit = 0, 0.60 or 1.00, where Gcrit = alpha + gamma*eta*Emax | Huelva anchor; three declared years; original renewable/ely sizing; rigid/60% package; no cavern rate bounds | 18 | Free inventory, feasible/infeasible status, combined-stock identity residual and same-electrolysis rigid reconstruction electricity requirement. Identity is algebraic at every G; the relief bound requires a feasible reconstruction, not necessarily the conservative sufficient Gcrit. |

Total: 90 new full-hourly configurations. No GPU or paid API is needed. CPU time
is measured by a pilot, with resume/checkpointing and original-matrix residual
checks. Result files and logs should be well below 100 MB; all writes are on D.
Existing public raw inputs on F are read-only. New source snapshots, if needed,
use a new read-only directory under F. No probability is assigned to the
purposive cases. Report all outcomes and every admitted case, including grid
infeasibilities and ties; do not choose settings from the resulting benefit.

Acceptance checks: frozen model hash unchanged; exact expected unique keys;
optimal status or an explicitly certified infeasible LP; original matrix and
variable-bound residual below 1e-5; matched capacities and working volumes;
stock-identity residual checked independently. Existing 405 primary cases,
27 original frontier, 27 original common-volume and 90 original threshold
cases remain separately reported. These new diagnostics do not create a census
or a substitute for observed process/geological capability.
