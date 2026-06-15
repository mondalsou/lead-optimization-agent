---
name: lead-optimization
description: Iteratively optimize drug candidates using ADMET scoring. Use when asked to improve a molecule's CNS penetration, solubility, drug-likeness, or other pharmacokinetic properties.
---

## CRITICAL: Always use tools — never reason without calling them

You have three local tools. **Do not describe what you would do — actually call the tools and report the numbers.**

1. `validate_smiles(smiles)` — ALWAYS call this first before any analysis
2. `analyze_molecule(smiles)` — full ADMET profile (MW, cLogP, TPSA, QED, BBB, CNS MPO, solubility, alerts)
3. `compare_candidates(smiles_list, labels)` — side-by-side comparison

## Mandatory workflow (follow exactly, every round)
1. `validate_smiles(starting_smiles)` → get canonical form
2. `analyze_molecule(canonical_smiles)` → read actual scores
3. Identify ONE property furthest from goal (from the numbers, not intuition)
4. Propose ONE structural change with SAR reasoning
5. `validate_smiles(new_smiles)` → confirm valid
6. `analyze_molecule(new_smiles)` → compare actual scores
7. After every 2 rounds: `compare_candidates([smiles1, smiles2, ...], [labels])`
8. Repeat from step 3 until goal met or no improvement for 2 rounds

**Never skip tool calls. Never propose a molecule without validating and scoring it.**

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
