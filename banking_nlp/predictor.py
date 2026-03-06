import re
import unicodedata
from pathlib import Path
from threading import Lock
from typing import Any

import spacy

MODEL_DIR = Path(__file__).resolve().parent / "models"
INTENT_MODEL_PATH = MODEL_DIR / "intent_textcat_fr"
SLOTS_DIR = MODEL_DIR / "slots"

DEFAULT_SCORE_THRESHOLD = 0.60
DEFAULT_MARGIN_THRESHOLD = 0.15

_INIT_LOCK = Lock()
_INITIALIZED = False
_intent_nlp: Any | None = None
_slot_models: dict[str, Any] = {}

KNOWN_BENEFICIAIRES = {
    "jean",
    "marc",
    "kodjo",
    "paul",
    "marie",
    "sophie",
    "yao",
    "akoua",
    "maman",
    "papa",
    "frere",
    "soeur",
    "ami",
    "epargne",
}


def normalize(text: str) -> str:
    text = text or ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact(text_norm: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text_norm)


def _load_spacy(path: Path):
    if not path.exists():
        return None
    return spacy.load(path)


def warmup() -> None:
    global _INITIALIZED, _intent_nlp
    if _INITIALIZED:
        return
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        _intent_nlp = _load_spacy(INTENT_MODEL_PATH)
        if _intent_nlp is None:
            raise FileNotFoundError(f"Intent model not found: {INTENT_MODEL_PATH}")

        for slot_name in [
            "compte_textcat_fr",
            "hist_time_textcat_fr",
            "hist_type_textcat_fr",
            "virement_ner_fr",
        ]:
            model = _load_spacy(SLOTS_DIR / slot_name)
            if model is not None:
                _slot_models[slot_name] = model
        _INITIALIZED = True


def _parse_amount(text_norm: str) -> str | None:
    m = re.search(r"\b((?:\d{1,3}(?:[ \.,]\d{3})+)|\d+)\s*(mille|milles)\b", text_norm)
    if m:
        base = int(re.sub(r"[^\d]", "", m.group(1)))
        return str(base * 1000)
    m = re.search(r"\b(\d+)\s*(million|millions)\b", text_norm)
    if m:
        return str(int(m.group(1)) * 1_000_000)
    m = re.search(r"\b((?:\d{1,3}(?:[ \.,]\d{3})+)|\d+)\b", text_norm)
    if m:
        return re.sub(r"[^\d]", "", m.group(1))
    return None


def _canonicalize_beneficiaire(value: str | None) -> str | None:
    if value is None:
        return None
    v = normalize(value)
    flat = compact(v)
    if any(
        stem in flat
        for stem in [
            "epargne",
            "epagne",
            "epane",
            "epagn",
            "economi",
            "livret",
            "cagnotte",
            "basdelaine",
            "decote",
        ]
    ):
        return "epargne"
    return v


def _parse_beneficiaire(text_norm: str) -> str | None:
    flat = compact(text_norm)
    if any(
        stem in flat
        for stem in [
            "epargne",
            "epagne",
            "epane",
            "epagn",
            "economi",
            "livret",
            "cagnotte",
            "basdelaine",
            "decote",
        ]
    ):
        return "epargne"

    for name in sorted(KNOWN_BENEFICIAIRES, key=len, reverse=True):
        if name == "epargne":
            continue
        if re.search(rf"\b{re.escape(name)}\b", text_norm):
            return name

    m = re.search(r"\b(?:a|vers|pour|au profit de|en faveur de)\s+([a-z][a-z\- ]{1,40})\b", text_norm)
    if m:
        candidate = m.group(1).strip()
        candidate = re.sub(r"\b(ce soir|maintenant|rapidement|svp|merci)\b", "", candidate).strip()
        if candidate and candidate not in {"mon compte", "mon compte courant", "mon compte cheque"}:
            return candidate
    return None


def _extract_virement_entities(text: str) -> dict[str, Any]:
    text_norm = normalize(text)
    beneficiaire = None
    montant = None
    ner = _slot_models.get("virement_ner_fr")
    if ner is not None:
        doc = ner(text)
        for ent in doc.ents:
            if ent.label_ == "BENEFICIAIRE" and not beneficiaire:
                beneficiaire = ent.text.strip().lower()
            elif ent.label_ == "MONTANT" and not montant:
                montant = re.sub(r"[^\d]", "", ent.text)
    if not beneficiaire:
        beneficiaire = _parse_beneficiaire(text_norm)
    if not montant:
        montant = _parse_amount(text_norm)
    return {
        "beneficiaire": _canonicalize_beneficiaire(beneficiaire),
        "montant": montant,
    }


def _extract_historique_filtre_entities(text: str) -> dict[str, Any]:
    type_value = "all"
    time_value = None
    type_model = _slot_models.get("hist_type_textcat_fr")
    time_model = _slot_models.get("hist_time_textcat_fr")
    if type_model is not None:
        d = type_model(text)
        type_value = max(d.cats, key=d.cats.get)
    if time_model is not None:
        d = time_model(text)
        time_value = max(d.cats, key=d.cats.get)
    return {"type": type_value, "time": time_value}


def _extract_compte_entities(text: str) -> dict[str, Any]:
    model = _slot_models.get("compte_textcat_fr")
    if model is not None:
        d = model(text)
        compte = max(d.cats, key=d.cats.get)
        return {"compte": compte}
    text_norm = normalize(text)
    flat = compact(text_norm)
    if any(stem in flat for stem in ["epargne", "epagne", "epagn", "econom", "livret", "cagnotte"]):
        return {"compte": "epargne"}
    if any(stem in flat for stem in ["courant", "cheque", "depens", "quotidien", "paiement"]):
        return {"compte": "courant"}
    return {"compte": "all"}


def _apply_intent_overrides(text_norm: str, predicted: str) -> str:
    flat = compact(text_norm)
    has_greeting = any(stem in flat for stem in ["bonjour", "bonsoir", "salut", "hello", "coucou"])
    has_help = any(
        stem in flat
        for stem in [
            "aide",
            "help",
            "assistance",
            "support",
            "quescequetusaisfaire",
            "quepeuxtufaire",
            "quesaitufaire",
            "fonctionnalite",
            "fonctionnalites",
            "commentutiliser",
            "expliquemoi",
            "mexpliquer",
        ]
    )
    has_cancel = any(stem in flat for stem in ["annule", "annuler", "stop", "laissetomber", "oublie", "arrete", "nenvoie"])
    has_virement = any(
        stem in flat
        for stem in ["virement", "transfer", "transfert", "virer", "envoy", "envoi", "credit", "epargn", "economis", "securis", "placer"]
    )
    has_hist = any(stem in flat for stem in ["historique", "operation", "transaction", "mouvement", "releve"])
    has_time = any(
        stem in flat for stem in ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche", "hier", "aujourdhui", "semaine", "mois", "annee"]
    )
    has_balance = any(stem in flat for stem in ["solde", "disponible", "combien", "argenttotal", "compte"])

    if has_cancel:
        return "ANNULATION"
    if has_virement:
        return "VIREMENT_INIT"
    if has_hist and has_time:
        return "HISTORIQUE_FILTRE"
    if has_balance and not has_hist:
        return "CONSULTER_SOLDE"
    if has_hist:
        return "HISTORIQUE_SIMPLE"
    if has_help:
        return "aide"
    if has_greeting:
        return "salutation"
    return predicted


def _predict_intent(
    text: str,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> dict[str, Any]:
    warmup()
    text_norm = normalize(text)
    doc = _intent_nlp(text_norm)
    ranked = sorted(doc.cats.items(), key=lambda x: x[1], reverse=True)
    top1_label, top1_score = ranked[0]
    top2_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top1_score - top2_score

    intent = _apply_intent_overrides(text_norm, top1_label)
    fallback = top1_score < score_threshold or margin < margin_threshold
    if fallback and intent not in {"VIREMENT_INIT", "ANNULATION", "CONSULTER_SOLDE", "HISTORIQUE_SIMPLE", "HISTORIQUE_FILTRE", "aide", "salutation"}:
        intent = "INCOMPRIS"

    return {
        "intent": intent,
        "score": float(top1_score),
        "margin": float(margin),
        "fallback": fallback,
        "top_labels": ranked[:3],
    }


def predict(
    text: str,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> dict[str, Any]:
    result = _predict_intent(text, score_threshold=score_threshold, margin_threshold=margin_threshold)
    intent = result["intent"]
    entities: dict[str, Any]
    if intent == "VIREMENT_INIT":
        entities = _extract_virement_entities(text)
    elif intent == "HISTORIQUE_FILTRE":
        entities = _extract_historique_filtre_entities(text)
    elif intent == "CONSULTER_SOLDE":
        entities = _extract_compte_entities(text)
    else:
        entities = {}

    return {
        "intention": intent,
        "entités": entities,
        "score": round(result["score"], 4),
        "fallback": result["fallback"],
    }
