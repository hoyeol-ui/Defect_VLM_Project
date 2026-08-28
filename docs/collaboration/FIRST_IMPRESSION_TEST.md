# First-Impression Test

Scenario: a collaborator opens `al-selector` today and has no prior explanation from the author.

Verdict: **COLLABORATOR ONBOARDING PASS (15/15)**

| Question | Where a newcomer gets the answer | Result |
|---|---|---|
| 1. What is the purpose? | README “What this project studies”; Collaborator Brief | PASS |
| 2. What is the research question? | opening paragraph and brief | PASS |
| 3. Why did the direction change? | README and `RESEARCH_HISTORY.md` | PASS |
| 4. Why Random and Frozen DINO? | brief and controlled protocol | PASS |
| 5. Which datasets were used? | `DATASETS.md` separates controlled/historical use | PASS |
| 6. What are their original sources? | source/download/citation/license rows in `DATASETS.md` | PASS, unknowns visibly flagged |
| 7. What is 440? | README and protocol derivation/table | PASS |
| 8. Why is statistical n not 440? | README, brief, protocol | PASS |
| 9. What is the primary endpoint? | protocol AULC/estimand section | PASS |
| 10. What are the final results? | README table and `FINAL_RESULT_BRIEF.md` | PASS |
| 11. Where is core code? | README and `CODE_INDEX.md` | PASS |
| 12. How are results reproduced? | `REPRODUCIBILITY.md` and verification script | PASS |
| 13. How do historical and controlled evidence differ? | README evidence layers and history reading rule | PASS |
| 14. What may be claimed? | final brief and claim matrix | PASS |
| 15. What must not be claimed? | claim matrix global forbidden claims | PASS |

## Simulated first path

Following the README's seven-item sequence gives a question-first overview, exact data terms, the design, the result, canonical code, and claim limits before exposing the reader to the large historical tree. Table aliases in `results/thesis_final/` make raw-to-summary tracing explicit, while `THESIS_FILE_INDEX.md` provides a repository-level route.

## Known friction that does not change PASS

The inherited repository remains large and contains many historical scripts. Some frozen execution scripts use site-local paths and full training cannot run without external datasets/weights. These limitations are stated before reproduction, and `CODE_INDEX.md` prevents legacy files from appearing equally authoritative.
