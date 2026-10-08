# Kestrel Home - Service-Request Routing (Variant B)

A small, local, no-paid-API service that routes one Kestrel service request to one of the seven current team queues and gives employee-readable reasons.

## Decision

Do not renew the Rs 3.2 lakh/year vendor routing bot. The replacement is a word+character TF-IDF classifier (LinearSVC) trained on the labelled service-request export, with the two team-name changes normalized according to the operations policy.

Expected hidden-test match: **about 95-97%**, with a midpoint expectation of **~95.6%**. The model scored 95.48% on a time-based May-Jun 2026 holdout and 95.80% on a later 20% temporal holdout; random 20% validation was 95.57%.

## Run from a clean machine

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

The submission already contains `model.joblib`, so no training data or paid API key is needed to start the service.

### Start the API

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

The single application endpoint is `POST http://127.0.0.1:8000/route`.

Example:

```bash
curl -X POST http://127.0.0.1:8000/route ^
  -H "Content-Type: application/json" ^
  -d "{\"request_id\":\"SR-DEMO\",\"channel\":\"chat\",\"product_family\":\"Water Purifier\",\"warranty_status\":\"in_warranty\",\"request_text\":\"display of purifier gone blank and not turning on\",\"source\":\"crm\"}"
```

On macOS/Linux, use `\\` instead of `^` for line continuation, or put the JSON on one line.

### Open the screen

After starting the API, open **http://127.0.0.1:8000/** in a browser. The browser screen calls the same `POST /route` endpoint; it does not contain a second copy of the routing logic.

## Optional retraining

Raw Kestrel data is intentionally **not** included in the runtime package because the operations policy says customer/operational data must not be published or shared beyond the engagement team. To retrain, place the labelled export at `data/train.csv` and run:

```bash
python train_model.py --train data/train.csv --out model.joblib
```

The training script maps the legacy names `Installations -> Installs & Demo` and `Consumables -> Filters & Consumables` because the policy says responsibilities did not change, only the team names.

## What was tried

1. Word + character TF-IDF + LinearSVC: kept. It was the strongest simple local model and cleared the 90% target on every validation split used.
2. Logistic regression on the same features: discarded for the final service because it was consistently slightly lower (~94.9-95.0% in the tested splits).
3. Keyword-only policy rules: discarded as the primary router. The rules were good at finding explicit cues but missed/overrode many ambiguous multi-intent records; whole-dataset accuracy was ~66.9%.
4. Paid LLM/API routing: not used. It was unnecessary for this dataset, adds a key dependency, and the brief explicitly says a small service that runs is preferable.

## Data-quality / ambiguity notes

- The supplied `README.txt` and `email-thread.txt` were empty. Decisions were therefore made from the populated CSVs, `teams.csv`, and `ops-policy.pdf` and are written down in `submission-form.md`.
- `team_label` in `train (2).csv` exactly matches `resolution_log.first_team`; it does **not** always match `final_team`. For training the routing model, `team_label/first_team` was treated as the requested historical routing label, because the brief asks to match routing labels rather than final resolution ownership.
- There are 599 repeated request texts and 89 repeated texts with conflicting labels. This is a real ceiling on deterministic text-only accuracy and is called out in the evidence report.
- All timestamps in the request exports are IST. Resolution-log event times from legacy are UTC according to policy; the evidence report does not use those event timestamps to train the model.

## Data handling

Keep this submission pack inside the engagement team. Do not upload customer/operational source data to public repositories.
