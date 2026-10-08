from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from scipy.sparse import hstack

MODEL_PATH = Path(__file__).with_name("model.joblib")
RENAME = {"Installations": "Installs & Demo", "Consumables": "Filters & Consumables"}
ACTIVE_TEAMS = ["Billing", "Filters & Consumables", "Installs & Demo", "Product Advice", "Repairs", "Returns & Replacement", "Warranty Claims"]

POLICY_EXPLANATIONS = {
    "Billing": "Billing is used only when the payment itself is the problem (for example invoice/GST, double charge, refund, EMI conversion or a payment coupon issue). A mention that the customer already paid is not enough.",
    "Filters & Consumables": "Filters & Consumables handles filters, candles, membranes, jars, brushes, blades and AMC/spare kits; these are not routed as product faults unless the request is actually about a fault.",
    "Installs & Demo": "Installs & Demo handles new-product installation, demonstrations and wall-mounting visits.",
    "Product Advice": "Product Advice handles pre- and post-purchase usage questions when no product fault is reported.",
    "Repairs": "Repairs handles product faults, breakdowns, error codes, unusual noise and leaks that need a technician.",
    "Returns & Replacement": "Returns & Replacement handles damaged, wrong or incomplete deliveries plus returns and exchanges within the return window.",
    "Warranty Claims": "Warranty Claims handles warranty/Shiled registration, coverage questions, certificates and claim status.",
}

# These are explanation signals, not hard routing rules; the trained model remains the decision maker.
REASON_PATTERNS = {
    "Billing": [
        (r"emi conversion", "EMI conversion"), (r"double charge", "double charge"),
        (r"gst", "GST"), (r"invoice", "invoice"), (r"refund", "refund"),
        (r"coupon", "coupon/payment issue"), (r"payment (?:failed|issue|problem|reversed)", "payment problem"),
    ],
    "Filters & Consumables": [
        (r"filter", "filter"), (r"candle", "candle"), (r"membrane", "membrane"),
        (r"jar", "jar"), (r"brush", "brush"), (r"blade", "blade"),
        (r"amc", "AMC kit"), (r"consumable", "consumables"), (r"spare", "spare"),
    ],
    "Installs & Demo": [
        (r"install", "installation"), (r"installer", "installer"), (r"wall mount", "wall-mounting"),
        (r"demo", "demo"), (r"technician", "technician visit"),
    ],
    "Product Advice": [
        (r"which .* right", "product-selection question"), (r"best settings", "settings question"),
        (r"recipe", "recipe/usage question"), (r"power consumption", "power-usage question"),
        (r"how to", "how-to question"), (r"safe", "safety/usage question"),
        (r"run on inverter", "usage/compatibility question"),
    ],
    "Repairs": [
        (r"error code", "error code"), (r"not turning on", "won't turn on"),
        (r"stopped working", "stopped working"), (r"not working", "not working"),
        (r"gone blank", "display gone blank"), (r"leak", "leak"), (r"noise", "noise"),
        (r"burnt smell", "burnt smell"), (r"tripping", "electrical tripping"),
        (r"fault", "fault"), (r"overheat", "overheating"),
    ],
    "Returns & Replacement": [
        (r"return", "return"), (r"exchange", "exchange"), (r"damaged", "damaged delivery"),
        (r"wrong .*deliver", "wrong delivery"), (r"scratched", "scratched delivery"),
        (r"missing (?:parts|item)", "missing parts/item"), (r"box was open", "opened box"),
    ],
    "Warranty Claims": [
        (r"warranty", "warranty"), (r"shield", "Kestrel Shield"), (r"claim", "claim"),
        (r"coverage", "coverage"), (r"certificate", "warranty certificate"), (r"register", "registration"),
    ],
}


def clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    for enc in ("latin1", "cp1252"):
        try:
            fixed = text.encode(enc).decode("utf-8")
            if fixed != text:
                text = fixed
                break
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class Router:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}. Run train_model.py first.")
        self.artifact: dict[str, Any] = joblib.load(model_path)
        self.word = self.artifact["word_vectorizer"]
        self.char = self.artifact["char_vectorizer"]
        self.clf = self.artifact["classifier"]
        self.teams = self.artifact["active_teams"]

    def _model_text(self, record: dict[str, Any]) -> str:
        frame = pd.DataFrame([record])
        if "request_text" not in frame.columns:
            frame["request_text"] = ""
        for col in ("product_family", "warranty_status", "channel", "source"):
            if col not in frame.columns:
                frame[col] = ""
        text = frame["request_text"].map(clean_text)
        product = frame["product_family"].fillna("").map(clean_text).str.replace(" ", "_", regex=False)
        warranty = frame["warranty_status"].fillna("").map(clean_text).str.replace(" ", "_", regex=False)
        channel = frame["channel"].fillna("").map(clean_text)
        source = frame["source"].fillna("").map(clean_text)
        return (
            text.iloc[0] + " product_" + product.iloc[0] +
            " warranty_" + warranty.iloc[0] +
            " channel_" + channel.iloc[0] +
            " source_" + source.iloc[0]
        ).strip()

    def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        if not str(record.get("request_text", "")).strip():
            raise ValueError("request_text is required")
        model_text = self._model_text(record)
        x = hstack([self.word.transform([model_text]), self.char.transform([model_text])]).tocsr()
        scores = self.clf.decision_function(x)[0]
        idx = int(scores.argmax())
        team = str(self.clf.classes_[idx])
        # Relative score, deliberately not labelled as a calibrated probability.
        shifted = scores - scores.max()
        exp = __import__("numpy").exp(shifted)
        confidence = float(exp[idx] / exp.sum())
        clean = clean_text(record["request_text"])
        matched = []
        for pat, label in REASON_PATTERNS.get(team, []):
            if re.search(pat, clean):
                matched.append(label)
            if len(matched) >= 3:
                break

        reasons = []
        if matched:
            joined = ", ".join(dict.fromkeys(matched))
            reasons.append(f"The message contains routing cues for {team}: {joined}.")
        else:
            # Use model evidence from the word features when there is no plain-language cue.
            try:
                word_x = self.word.transform([model_text])
                feature_names = self.word.get_feature_names_out()
                coefs = self.clf.coef_
                class_idx = list(self.clf.classes_).index(team)
                row = word_x.toarray()[0]
                if coefs.shape[0] == 1:
                    weights = coefs[0] * row if team == self.clf.classes_[1] else -coefs[0] * row
                else:
                    weights = coefs[class_idx] * row
                top = sorted(((weights[i], feature_names[i]) for i in row.nonzero()[0]), reverse=True)[:3]
                terms = [t for s, t in top if s > 0][:3]
                if terms:
                    reasons.append("The routing model found the strongest matching message cues: " + ", ".join(terms) + ".")
                else:
                    reasons.append("No single policy phrase dominates this message; the route is based on the closest learned pattern from historical requests.")
            except Exception:
                reasons.append("No single policy phrase dominates this message; the route is based on the closest learned pattern from historical requests.")
        reasons.append(POLICY_EXPLANATIONS[team])
        if confidence < 0.55:
            reasons.append("Manual review is recommended because the model's relative confidence is low for this request.")
        return {
            "team": team,
            "confidence": round(confidence, 3),
            "confidence_note": "relative model confidence; not a calibrated probability",
            "reasons": reasons,
            "model": self.artifact["model"],
        }
