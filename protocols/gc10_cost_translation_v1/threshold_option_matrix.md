# Threshold and Design Option Matrix

Protocol version: `gc10-cost-translation-v1-candidate`

This matrix provides defensible options; it does not make the final decisions. Choosing after viewing Phase A arm results would invalidate the prospective cost-endpoint claim.

| Decision | Option | Advantage | Risk | Current evidence | Why not choose after outcomes |
|---|---|---|---|---|---|
| D01 Annotator count | 2 | Feasible for thesis; complete paired annotation of 233 images | Bounded paired study only; weak human-population inference | 466 Phase A tasks | A smaller/larger panel cannot be chosen to suppress observed disagreement |
| D01 Annotator count | 4 | Better annotator-variation estimate and interaction visibility | Roughly doubles human workload | 932 Phase A tasks | Adding annotators after seeing an unfavorable result is sequential outcome chasing unless predeclared |
| D02 \(T\) | Documented external operating-session duration | Highest industrial interpretability | External rule may not exist or may yield sparse labels | Repository currently has no operational session length | Arm outcomes must not determine a convenient time horizon |
| D02 \(T\) | Stage 0 common-support calibration from the same predeclared 20-image sequence, rounded down to 5 minutes | Strategy-blind and executable | Calibration remains tool/annotator specific | Permitted only before Phase A | Choosing a time where curves separate maximally overstates utility |
| D03 Idle cutoff | 30 s | Aggressively excludes distraction | May remove legitimate visual inspection | No current event log | Different cutoffs can reverse time ranking |
| D03 Idle cutoff | 60 s | Moderate candidate | Still arbitrary without Stage 0 | No current event log | Must use repeatability/obvious-idle evidence, not arm difference |
| D03 Idle cutoff | 90 or 100 s | Close to CVAT-style event-gap convention | Can retain short distractions | CVAT documentation uses 100 s in its working-time calculation; local UI is not CVAT | Adopting it only when favorable is not independent |
| D04 Burden materiality | Absolute seconds/minutes per q20 batch | Direct operational interpretation | Depends on session/tool speed | No human-time data | Absolute cutoff can be moved around the observed difference |
| D04 Burden materiality | Relative percentage of Random q20 time | Easier across annotators | Ratio unstable when times vary | No human-time data | Favorable denominator selection is a post-hoc degree of freedom |
| D04 Burden materiality | Externally supplied labor-saving requirement | Strongest economic claim | Requires stakeholder documentation | Not present in repository | External requirement must predate results |
| D05 Quality NI | Zero tolerated decrease in missed-defect/rare recall | Strong safety posture | May be underpowered and strict | Industrial defects make omissions consequential | A looser margin after seeing rare misses rescues an unsafe arm |
| D05 Quality NI | Small predeclared absolute margin justified by audit policy | Balances measurement noise and safety | Requires an external or supervisor rationale | No current tolerance | Margin is part of the claim, not a tuning knob |
| D06 Phase B minimum mAP | Preserve legacy +0.010 mean mAP50–95 gate | Temporal continuity and least outcome-dependent | May be modest in practical terms | Existing frozen GC10 detector protocol | Lowering after known earlier outcomes is clear outcome chasing |
| D06 Phase B minimum mAP | Higher externally justified practical effect | Stronger utility claim | More likely STOP | No external requirement documented | Must be set before new learner outcomes |
| D07 Rare safety | Preserve legacy rare macro AP +0.020 plus class guards; add class 10 guard prospectively | Strict and continuous with prior gate | Difficult to pass | Prior GC10 rare macro AP was negative; this cannot justify relaxing | A relaxed rule would directly rescue a known failure mode |
| D07 Rare safety | Predeclared non-inferiority for rare AP/recall with external safety rationale | May better match operations | Could be perceived as weakening unless justified independently | No external safety margin exists | Margin chosen from observed rare results is invalid |
| D08 Adjudication | Both annotators label all; blinded third decision on disagreements | Strong provenance | Requires a third qualified person and extra time | No adjudicator assigned | Reviewing only an unfavorable arm biases cost |
| D08 Adjudication | Deterministic consensus rule plus blinded disagreement-only review | Lower workload | Consensus algorithm may hide ambiguity | Requires exact rule before labels | Rule choice can alter both quality and learner labels |
| D08 Adjudication | One primary annotator; second annotation used only as quality audit | Simpler learner label provenance | Annotator choice can dominate result | Two-annotator minimum still required | Selecting the better annotator after outcomes is prohibited |
| D09 Human participation | Formal IRB approval | Strongest institutional assurance | Time and administration | Requirement unknown | Ethics scope cannot be decided from result desirability |
| D09 Human participation | Formal exemption/not-human-subjects determination | May be sufficient for internal task study | Must come from institution, not researcher | UNVERIFIED | Self-declaration after data collection is not adequate |
| D09 Human participation | Do not collect human data | Zero participation risk | Study stops before Phase A | Always available | A legitimate STOP option must remain |

## Recommendation hierarchy without final selection

1. Prefer 4 annotators if resources permit; 2 is the thesis minimum and must be described as bounded.
2. Prefer an external operational \(T\); otherwise use only a strategy-blind Stage 0 calibration rule.
3. Preserve or strengthen earlier rare-safety rules unless an independent external policy justifies a different margin.
4. Include adjudication time in cost under every option.
5. Obtain an institutional determination before recruiting or timing anyone.
