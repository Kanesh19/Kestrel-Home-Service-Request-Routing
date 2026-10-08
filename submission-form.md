# Kestrel Home - Task 2 V2 Submission Form

## Decision
Do not renew the vendor routing bot. Deploy the local replacement and switch the vendor bot off after a short monitored cutover.

## Expected hidden-test score
**Expected match: 95-97%; planning midpoint: ~95.6%.**

Why: the most relevant time-based holdout (May-Jun 2026, immediately before the July-Sep 2026 unlabelled test period) scored **95.48% accuracy** and **95.27% macro-F1**. A separate temporal 20% holdout scored **95.70%** and the stratified random 20% holdout scored **95.57%**.

## What was trained
A local word + character TF-IDF classifier with LinearSVC (C=2.0). Text was cleaned and augmented with product, warranty-status, channel and source tokens.

## Label decision
`team_label` exactly matches `first_team` in the supplied resolution log for all 10,822 training rows, while `final_team` is different when agents transfer a request. I treated `team_label/first_team` as the historical routing label because the brief asks for routing-label match at creation. The policy says the 15 Jan 2026 changes were name changes only, so the model normalizes:

- `Installations` -> `Installs & Demo`
- `Consumables` -> `Filters & Consumables`

This produces the seven current queues used in the output.

## Evidence
- 10,822 labelled training requests.
- 2,178 unlabelled test requests.
- May-Jun 2026 time holdout: 95.48% accuracy, 95.27% macro-F1.
- Historical transfers: 2,696 requests with >=1 transfer (24.91%); 3,902 transfer events.
- Policy-rate historical routing cost exposure: Rs 18,91,070 over 18 months (~Rs 12.61 lakh annualized), plus the Rs 3.2 lakh/year vendor licence. These are exposure figures, not guaranteed future savings.

## How often it fails
The May-Jun 2026 holdout missed **65 of 1,438 requests (4.52%)**. Errors are concentrated in ambiguous/multi-intent messages and noisy historical labels, especially Repairs vs Filters & Consumables. There are 599 repeated request-text values; 89 repeated texts have conflicting labels.

## What was tried
**Kept:** word + character TF-IDF + LinearSVC.

**Discarded:** logistic regression on the same feature recipe because validation was slightly lower (~94.9-95.0%); keyword-only policy rules because they reached only ~66.9% on the full labelled set and were brittle on multi-intent/noisy requests; paid LLM/API routing because it adds key/cost/runtime dependency without a demonstrated accuracy benefit for this dataset.

## AI tools
ChatGPT was used as a coding/review assistant during development. The shipped service does not call an AI API and requires **Rs 0 of external API spend** at runtime.

## Working service
Start from the README with Python 3.10+; install `requirements.txt`; run `uvicorn api:app --host 127.0.0.1 --port 8000`; open `http://127.0.0.1:8000/`. The browser screen calls `POST /route`.

## Source-pack gaps / assumptions
The supplied `README.txt`, `email-thread.txt`, and no `submission-form.md` template were empty/missing. I therefore made the explicit routing and data-handling decisions above from the populated CSVs, `teams.csv`, and `ops-policy.pdf`, and recorded them here rather than inventing missing facts.

## Data handling
No raw customer-request CSVs are included in the runnable package. Keep the engagement data and this pack inside the approved engagement team; do not publish it to public repositories.

## Deliverables
- `predictions.csv`
- `model.joblib`
- `api.py`, `routing.py`, `train_model.py`
- `README.md` / `README.txt`
- `evidence/validation_report.md`, metrics and confusion matrix
- `memo_to_ritu.pdf` and `memo_to_ritu.md`
- `screen_recording.mp4`
- this completed `submission-form.md`
