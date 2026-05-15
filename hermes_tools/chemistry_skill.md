---
name: lead-optimization
description: Iteratively optimize drug candidates using ADMET scoring. Use when asked to improve a molecule's CNS penetration, solubility, drug-likeness, or other pharmacokinetic properties.
---

## Overview
You have three local tools for medicinal chemistry (no external API needed):
1. `validate_smiles(smiles)` — validate SMILES, get canonical form
2. `analyze_molecule(smiles)` — full ADMET profile (MW, cLogP, TPSA, QED, BBB, CNS MPO, solubility, alerts)
3. `compare_candidates(smiles_list, labels)` — side-by-side comparison

## Strategy
1. Validate the starting SMILES first
2. Analyze it to get baseline scores
3. Identify which property is furthest from the goal
4. Propose ONE structural change with clear chemical reasoning
5. Validate → analyze the new candidate
6. Compare every 2 rounds
7. Stop when goal met or no improvement across 2 consecutive rounds

## Key Structure–Property Rules
- **CNS/BBB**: HBD ≤ 1, TPSA < 90 Å², cLogP 1–4, MW < 400
- **Solubility**: lower cLogP, add polar groups (OH, NH), shorten alkyl chains
- **Oral absorption**: TPSA ≤ 140, rotatable bonds ≤ 10 (Veber rules)
- **Drug-likeness (QED)**: MW 200–450, LogP 0–5, fewer rings and rotatable bonds
- **Bioisosteres**: -CONH₂→-CN, -COOH→-tetrazole, -OH→-F, phenyl→pyridine
- **Remove alerts**: avoid quinones, catechols, Michael acceptors, PAINS motifs

## Score Interpretation
| Score | Range | Good target |
|-------|-------|-------------|
| QED | 0–1 | > 0.7 |
| CNS MPO | 0–5 | > 4.0 |
| BBB probability | 0–1 | > 0.7 for CNS drugs |
| logS (solubility) | negative | > −3 (moderately soluble) |
| Fail-fast score | 0–10 | < 3 (low risk) |
| Synthetic accessibility | 1–10 | < 5 (synthesizable) |

## Decision Guidance
- **Kill**: fail-fast > 7, or ≥ 3 PAINS alerts, or MW > 600
- **Optimize**: salvageable with targeted changes
- **Progress**: meets drug-likeness criteria, no showstopper alerts

## Example Workflow
```
User: optimize CC(C)NCC(O)COc1ccc(CC(N)=O)cc1 for brain penetration

1. validate_smiles("CC(C)NCC(O)COc1ccc(CC(N)=O)cc1")
2. analyze_molecule("<canonical>")  → baseline: TPSA 92, HBD 3, BBB 42%
3. Reasoning: remove amide (HBD −1), methylate NH (HBD −1)
4. validate_smiles("<new SMILES>")
5. analyze_molecule("<new SMILES>") → TPSA 72, HBD 1, BBB 71% ✓
6. compare_candidates([old, new], ["baseline", "round1"])
```
