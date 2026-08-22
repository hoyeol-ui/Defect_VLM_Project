# CPU Preflight Results v1.1

- Result: **22 PASS / 0 FAIL**
- Original v1 gates: 14/14 PASS
- Added v1.1 source/lock gates: 8/8 PASS
- Actual final authorization artifact: ABSENT
- Actual final registry access: HARD FAIL

The added tests cover development access, unauthorized real-final failure,
incomplete 439-cell rejection, wrong protocol SHA rejection, a correct in-memory
440-cell authorization fixture, locked-final-only primary AULC, development
metric rejection, and missing-final-point `INCOMPLETE_NOT_ESTIMATED` behavior.

The in-memory fixture did not create a real authorization file and did not read
the final split.

