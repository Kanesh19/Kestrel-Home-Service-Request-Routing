# Kestrel Home - Evidence that the router works
## Expected hidden-test score
**Expected match: ~95-97%; planning midpoint ~95.6%.** The strongest evidence is the time-based May-Jun 2026 holdout: **95.48% accuracy** and **95.27% macro-F1** on 1,438 requests. A separate stratified random 20% holdout was **95.57% accuracy** and **95.35% macro-F1**. The hidden test is July-Sep 2026, so the time holdout is the more relevant estimate.
## Validation method
- The label used for training is the historical `team_label`, which exactly matches `resolution_log.first_team` in all 10,822 training rows.
- The policy says the 15 Jan 2026 team changes were name changes only, so `Installations` was normalized to `Installs & Demo` and `Consumables` to `Filters & Consumables`.
- Features: cleaned request text plus product, warranty, channel and source tokens; word TF-IDF 1-2 grams + character TF-IDF 3-5 grams; LinearSVC C=2.0.
- No paid API, external data, or future timestamps were used to train the model.
## Time-based holdout performance
| Team | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Billing | 96.8% | 98.1% | 97.4% | 213 |
| Filters & Consumables | 93.0% | 95.7% | 94.3% | 208 |
| Installs & Demo | 94.2% | 97.3% | 95.7% | 149 |
| Product Advice | 93.5% | 96.0% | 94.7% | 150 |
| Repairs | 97.8% | 94.5% | 96.1% | 421 |
| Returns & Replacement | 95.1% | 92.8% | 93.9% | 166 |
| Warranty Claims | 94.7% | 94.7% | 94.7% | 131 |

## What it gets wrong
The May-Jun holdout missed **1,272 / 1,438 (4.52%)**. The largest confusion pairs are around ambiguous service requests (especially Repairs vs Filters & Consumables) and mixed-intent messages that contain both fault/usage/payment/installation language. The source data also contains **599 repeated request-text values**, including **23 repeated texts with conflicting team labels**; this creates unavoidable ambiguity for a text-first router.
## Historical cost context
Over the 18-month history there were **2,696 requests with at least one transfer (24.9%)** and **3,902 transfer events**. At the policy rates that is **Rs 1,190,110** of transfer handling plus **Rs 700,960** of extra-contact cost, or **Rs 1,891,070 over 18 months / ~Rs 1,260,713 annualized**. Separately, the vendor licence is **Rs 3.2 lakh/year**. These are historical cost exposure figures, not guaranteed future savings.
## Current routing labels vs final resolution
`team_label` matches `first_team` 100% and matches `final_team` 77.17%. That is expected: agents transferred requests that landed in the wrong queue. The submission therefore treats `team_label/first_team` as the correct historical routing label, because the brief asks the replacement to match routing labels at creation rather than reproduce post-resolution ownership.
## Acceptance view
The model clears the stated 90% target with a buffer of roughly 5.5 percentage points on the most relevant time-based holdout. I would still launch with a monitored cutover and a manual review path for low-confidence cases.
