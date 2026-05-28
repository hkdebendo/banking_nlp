import json
import re
import unicodedata
from pathlib import Path
from threading import Lock
from typing import Any

import spacy

MODEL_DIR = Path(__file__).resolve().parent / "models"
RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
INTENT_MODEL_PATH = MODEL_DIR / "intent_textcat_fr"
SLOTS_DIR = MODEL_DIR / "slots"
KNOWN_BENEFICIAIRES_PATH = RESOURCES_DIR / "known_beneficiaires.json"
ENTITIES_KEY = "entités"
DEFAULT_SCORE_THRESHOLD = 0.60
DEFAULT_MARGIN_THRESHOLD = 0.15

LEADING_BENEFICIARY_TOKENS = {
    "mon",
    "ma",
    "mes",
    "ton",
    "ta",
    "tes",
    "son",
    "sa",
    "ses",
    "notre",
    "nos",
    "votre",
    "vos",
    "leur",
    "leurs",
    "le",
    "la",
    "les",
    "l",
    "un",
    "une",
    "ce",
    "cet",
    "cette",
    "ces",
}

TRAILING_BENEFICIARY_TOKENS = {
    "ce",
    "soir",
    "maintenant",
    "rapidement",
    "svp",
    "stp",
    "merci",
    "silteplait",
    "silvousplait",
}

INVALID_BENEFICIARY_TOKENS = {
    "franc",
    "francs",
    "fcfa",
    "xof",
    "cfa",
    "euro",
    "euros",
    "mille",
    "milles",
    "million",
    "millions",
}

INVALID_BENEFICIARY_ACTION_TOKENS = {
    "argent",
    "crediter",
    "envoie",
    "envoyer",
    "envoyez",
    "effectuer",
    "faire",
    "lancer",
    "transferer",
    "transfert",
    "virer",
    "virement",
}

OUT_OF_SCOPE_SIGNALS = {
    "aeroport",
    "anglais",
    "basket",
    "bluetooth",
    "burger",
    "canada",
    "casque",
    "chocolat",
    "chauddehors",
    "centreville",
    "dissertation",
    "devoir",
    "dakar",
    "dormir",
    "exercice",
    "exercices",
    "fievre",
    "film",
    "films",
    "garde",
    "gmail",
    "googlemaps",
    "grippe",
    "hotel",
    "imprimante",
    "intelligenceartificielle",
    "itineraire",
    "lasagne",
    "lasagnes",
    "maps",
    "maths",
    "meteo",
    "musique",
    "netflix",
    "pizza",
    "pleuvoir",
    "psg",
    "pythagore",
    "premierleague",
    "pc",
    "recette",
    "regime",
    "restaurant",
    "realmadrid",
    "revolution",
    "rome",
    "serie",
    "score",
    "stationservice",
    "tete",
    "temps",
    "telephone",
    "virus",
    "visa",
    "windows",
    "yopougon",
}

MONTH_TOKENS = {
    "janvier",
    "fevrier",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "aout",
    "septembre",
    "octobre",
    "novembre",
    "decembre",
}

DAY_TOKENS = {
    "lundi",
    "mardi",
    "mercredi",
    "jeudi",
    "vendredi",
    "samedi",
    "dimanche",
}

_INIT_LOCK = Lock()
_INITIALIZED = False
_intent_nlp: Any | None = None
_slot_models_cache: dict[str, Any] = {}
_known_beneficiaires: set[str] = set()


def normalize(text: str) -> str:
    text = text or ""
    text = text.replace("œ", "oe").replace("Œ", "OE").replace("æ", "ae").replace("Æ", "AE")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("’", "'").replace("â€™", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_intent(text: str) -> str:
    text = normalize(text)
    replacements = {
        "econiliser": "economiser",
        "econimiser": "economiser",
        "economisser": "economiser",
        "envoye": "envoyer",
        "milles": "mille",
        "qu es": "ques",
        "qu est": "quest",
    }
    for src, dst in replacements.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text)
    return text


def compact(text_norm: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text_norm)


def normalize_digits(text: str) -> str:
    return re.sub(r"[^\d]", "", text or "")


def normalize_beneficiary_token(token: str) -> str:
    token = normalize(token).replace("'", "")
    if len(token) <= 5:
        token = re.sub(r"(.)\1+$", r"\1", token)
    return token


def normalize_beneficiary_text(text: str) -> str:
    text = text or ""
    text = text.replace("œ", "oe").replace("Œ", "OE").replace("æ", "ae").replace("Æ", "AE")
    text = text.replace("Å“", "oe").replace("Å’", "OE").replace("Ã¦", "ae").replace("Ã†", "AE")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("â€™", "'").replace("Ã¢â‚¬â„¢", "'")
    text = re.sub(r"[^a-z0-9' -]+", " ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_spacy(path: Path):
    if not path.exists():
        return None
    return spacy.load(path)


def clean_beneficiary_candidate(value: str | None) -> str | None:
    if not value:
        return None

    candidate = normalize_beneficiary_text(value)
    candidate = re.sub(
        r"\b(?:ce soir|maintenant|rapidement|svp|stp|merci|s'il te plait|s'il vous plait|silteplait|silvousplait)\b",
        " ",
        candidate,
    )

    changed = True
    while candidate and changed:
        previous = candidate
        candidate = re.sub(r"^(?:sur|dans)\s+(?:le|la|l')?\s*compte\s+de\s+", "", candidate)
        candidate = re.sub(r"^(?:au profit de|en faveur de|a|vers|pour|chez)\s+", "", candidate)

        tokens = candidate.split()
        while tokens and normalize_beneficiary_token(tokens[0]) in LEADING_BENEFICIARY_TOKENS:
            tokens = tokens[1:]
        while tokens and normalize_beneficiary_token(tokens[-1]) in TRAILING_BENEFICIARY_TOKENS:
            tokens = tokens[:-1]
        while tokens and normalize_beneficiary_token(tokens[-1]) in INVALID_BENEFICIARY_TOKENS:
            tokens = tokens[:-1]

        candidate = " ".join(tokens).strip(" '")
        changed = candidate != previous

    if any(token in INVALID_BENEFICIARY_ACTION_TOKENS for token in candidate.split()):
        return None
    if normalize_beneficiary_token(candidate) in INVALID_BENEFICIARY_TOKENS:
        return None
    return candidate or None


def resolve_known_beneficiaire(value: str | None, known_beneficiaires: set[str]) -> str | None:
    if not value:
        return None
    value_flat = compact(value)
    for name in sorted(known_beneficiaires, key=len, reverse=True):
        if compact(name) == value_flat:
            return name
    return value


def load_known_beneficiaires() -> set[str]:
    if not KNOWN_BENEFICIAIRES_PATH.exists():
        return set()
    with KNOWN_BENEFICIAIRES_PATH.open("r", encoding="utf-8-sig") as f:
        values = json.load(f)
    return {canonicalize_beneficiaire(value) for value in values if canonicalize_beneficiaire(value)}


def warmup() -> None:
    global _INITIALIZED, _intent_nlp, _known_beneficiaires
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
                _slot_models_cache[slot_name] = model

        _known_beneficiaires = load_known_beneficiaires()
        _INITIALIZED = True


def load_slot_model(name: str):
    warmup()
    return _slot_models_cache.get(name)


def has_time_reference(text_norm: str) -> bool:
    flat = compact(text_norm)
    tokens = set(text_norm.split())
    if any(token in tokens for token in DAY_TOKENS | MONTH_TOKENS):
        return True
    if any(token in tokens for token in [
        "hier",
        "aujourdhui",
        "demain",
        "jour",
        "matin",
        "soir",
        "semaine",
        "mois",
        "annee",
        "l'annee",
        "an",
    ]):
        return True
    if any(token in flat for token in [
        "dujour",
        "cematin",
        "cesoir",
        "weekend",
        "weekenddernier",
        "semainepassee",
        "moispasse",
        "anpasse",
        "anneepassee",
        "debutdelasemaine",
        "debutdumois",
    ]):
        return True
    if re.search(r"\b20\d{2}\b", text_norm):
        return True
    return False


def has_greeting_signal(text_norm: str) -> bool:
    flat = compact(text_norm)
    tokens = set(text_norm.split())
    return (
        any(stem in flat for stem in ["bonjour", "bonsoir", "salut", "hello", "coucou", "wesh", "bjr", "bsr", "slt", "allo"])
        or any(token in tokens for token in ["hey", "yo", "cc"])
    )


def has_social_greeting_signal(text_norm: str) -> bool:
    flat = compact(text_norm)
    return has_greeting_signal(text_norm) and any(
        stem in flat
        for stem in [
            "cava",
            "caroule",
            "commenttuvas",
            "commentcava",
            "bienoubien",
        ]
    )


def has_help_signal(text_norm: str) -> bool:
    flat = compact(text_norm)
    tokens = set(text_norm.split())
    if any(token in tokens for token in ["aide", "help", "assistance", "support"]):
        return True
    if any(stem in flat for stem in ["aider", "guide", "expliquemoi", "mexpliquer", "commentutiliser", "fonctionnalite", "fonctionnalites"]):
        return True
    if any(stem in flat for stem in ["quepeuxtufaire", "quesaitufaire", "quesaistufaire", "quesavezvousfaire", "cequetupeuxfaire", "cequetusaisfaire"]):
        return True
    if "commande" in flat and any(stem in flat for stem in ["disponible", "utiliser", "comprend", "possible"]):
        return True
    return False


def has_guidance_signal(text_norm: str) -> bool:
    flat = compact(text_norm)
    tokens = set(text_norm.split())
    return (
        has_help_signal(text_norm)
        or any(token in tokens for token in ["comment", "aide", "assistance", "guide", "support"])
        or any(stem in flat for stem in [
            "aider",
            "aidemoi",
            "besoindaide",
            "besoindassistance",
            "bloque",
            "commandedisponible",
            "commandesdisponibles",
            "commentcamarche",
            "commentjefais",
            "coupdemain",
            "expliquemoi",
            "galere",
            "guidemoi",
            "perdu",
        ])
    )


def has_explicit_out_of_scope_frame(text_norm: str) -> bool:
    flat = compact(text_norm)
    return any(
        stem in flat
        for stem in [
            "rienavoiraveclabanque",
            "horssujetbancaire",
            "cenestpasunedemandebancaire",
            "jeneparlepasdebanque",
        ]
    )


def has_out_of_scope_signal(text_norm: str) -> bool:
    flat = compact(text_norm)
    return (
        has_explicit_out_of_scope_frame(text_norm)
        or "programmetele" in flat
        or any(signal in flat for signal in OUT_OF_SCOPE_SIGNALS)
    )


def has_banking_domain_signal(text_norm: str) -> bool:
    flat = compact(text_norm)
    return any(
        stem in flat
        for stem in [
            "banque",
            "bancaire",
            "compte",
            "solde",
            "epargne",
            "livret",
            "courant",
            "virement",
            "transfert",
            "transaction",
            "operation",
            "historique",
            "releve",
            "paiement",
            "debit",
            "credit",
            "argent",
            "franc",
            "fcfa",
            "xof",
            "beneficiaire",
            "application",
            "service",
        ]
    )


def has_transfer_action_signal(text_norm: str) -> bool:
    if re.search(r"\bepargne\s+\d", text_norm):
        return True
    return any(
        re.search(pattern, text_norm)
        for pattern in [
            r"\b(?:envoyer|envoie|envoyez|envoyons)\b",
            r"\b(?:transferer|transfere|transfert|transferez|transfer)\b",
            r"\b(?:virer|vire|virez)\b",
            r"\b(?:crediter|credite|creditez)\b",
            r"\b(?:placer|place)\b",
            r"\b(?:mettre|mets|mettez)\b",
            r"\b(?:alimenter|alimente|alimentez)\b",
            r"\b(?:economiser|economise|epargner|securiser|securise)\b",
            r"\b(?:faire|fais|lancer|effectuer|passer)\s+un\s+virement\b",
            r"\b(?:faire|fais|lancer|effectuer|passer)\s+un\s+transfert\b",
        ]
    )


def has_balance_signal(text_norm: str) -> bool:
    flat = compact(text_norm)
    has_account_target = any(
        stem in flat
        for stem in [
            "solde",
            "compte",
            "epargne",
            "epagne",
            "livret",
            "economies",
            "courant",
            "avoir",
            "situationglobale",
            "montanttotal",
            "totaldisponible",
            "argenttotal",
        ]
    )
    has_query = any(stem in flat for stem in ["combien", "reste", "ouenest", "quelest", "cestquoi", "montremoi", "affiche", "donnemoi", "voir", "consulter"])
    return has_account_target and has_query


def has_history_subject_signal(text_norm: str) -> bool:
    flat = compact(text_norm)
    if any(
        stem in flat
        for stem in [
            "historique",
            "transaction",
            "transactions",
            "operation",
            "operations",
            "mouvement",
            "mouvements",
            "releve",
            "activite",
            "activites",
            "depense",
            "depenses",
            "achat",
            "achats",
            "paiement",
            "paiements",
            "credit",
            "credits",
            "rentree",
            "rentrees",
            "salaire",
            "salaires",
            "remboursement",
            "remboursements",
            "versement",
            "versements",
            "retrait",
            "retraits",
            "sortiedargent",
            "sortiesdargent",
        ]
    ):
        return True
    if "virement" in flat and any(stem in flat for stem in ["recu", "recus", "envoye", "envoyes"]):
        return True
    return False


def has_history_recent_signal(text_norm: str) -> bool:
    flat = compact(text_norm)
    return (
        any(stem in flat for stem in ["recent", "recemment", "dernier", "derniere", "passe"])
        or "ilyaeuquoi" in flat
        or "cequiestpasse" in flat
        or "revoir" in flat
        or "bouge" in flat
    )


def parse_amount(text_norm: str) -> str | None:
    m = re.search(r"\b((?:\d{1,3}(?:[ \.,]\d{3})+)|\d+)\s*(mille|milles)\b", text_norm)
    if m:
        base = int(re.sub(r"[^\d]", "", m.group(1)))
        return str(base * 1000)

    m = re.search(r"\b(\d+)\s*(million|millions)\b", text_norm)
    if m:
        return str(int(m.group(1)) * 1_000_000)
    if re.search(r"\bun\s+millions?\b", text_norm):
        return "1000000"

    m = re.search(r"\b((?:\d{1,3}(?:[ \.,]\d{3})+)|\d+)\b", text_norm)
    if m:
        return re.sub(r"[^\d]", "", m.group(1))
    return None


def parse_beneficiaire(text_norm: str, known_beneficiaires: set[str]) -> str | None:
    flat = compact(text_norm)
    if any(
        stem in flat
        for stem in [
            "epargne",
            "epagne",
            "epane",
            "epagn",
            "economi",
            "conomi",
            "economis",
            "epargn",
            "livret",
            "cagnotte",
            "basdelaine",
            "mettredecote",
            "decote",
        ]
    ):
        return "epargne"

    for name in sorted(known_beneficiaires, key=len, reverse=True):
        if name == "epargne":
            continue
        if re.search(rf"\b{re.escape(name)}\b", text_norm):
            return name

    extraction_patterns = [
        r"\b(?:sur|dans)\s+(?:le|la|l')?\s*compte\s+de\s+(?P<candidate>[a-z][a-z\- ']{1,50})\b",
        r"\b(?:a|vers|pour|chez|au profit de|en faveur de)\s+(?P<candidate>[a-z][a-z\- ']{1,50})\b",
    ]
    for pattern in extraction_patterns:
        m = re.search(pattern, text_norm)
        if not m:
            continue
        candidate = clean_beneficiary_candidate(m.group("candidate"))
        candidate = canonicalize_beneficiaire(candidate)
        if candidate and candidate not in {"compte", "compte courant", "compte cheque"}:
            return resolve_known_beneficiaire(candidate, known_beneficiaires)

    m = re.search(r"\b(?:envoy[a-z]*|transf[a-z]*|virer|virement|credit[a-z]*)\s+\d+\s+([a-z][a-z\-]{1,30})\b", text_norm)
    if m:
        candidate = canonicalize_beneficiaire(clean_beneficiary_candidate(m.group(1).strip()))
        if candidate and normalize_beneficiary_token(candidate) not in INVALID_BENEFICIARY_TOKENS:
            return candidate

    return None


def canonicalize_beneficiaire(value: str | None) -> str | None:
    if value is None:
        return None
    value = clean_beneficiary_candidate(value)
    if not value:
        return None
    flat = compact(value)
    if any(stem in flat for stem in ["epargne", "epagne", "epane", "epagn", "economi", "conomi", "livret", "cagnotte", "basdelaine", "mettredecote", "decote"]):
        return "epargne"
    return value


def extract_virement_entities(text: str, known_beneficiaires: set[str]) -> dict[str, Any]:
    ner = load_slot_model("virement_ner_fr")
    text_norm = normalize(text)
    beneficiaire = None
    montant = None

    if ner is not None:
        doc = ner(text)
        for ent in doc.ents:
            if ent.label_ == "BENEFICIAIRE" and not beneficiaire:
                beneficiaire = ent.text.strip().lower()
            if ent.label_ == "MONTANT" and not montant:
                montant = normalize_digits(ent.text)

    parsed_amount = parse_amount(text_norm)
    if parsed_amount:
        montant = parsed_amount

    if not beneficiaire:
        beneficiaire = parse_beneficiaire(text_norm, known_beneficiaires)

    return {
        "beneficiaire": resolve_known_beneficiaire(canonicalize_beneficiaire(beneficiaire), known_beneficiaires),
        "montant": montant or None,
    }


def extract_historique_filtre_entities(text: str) -> dict[str, Any]:
    type_model = load_slot_model("hist_type_textcat_fr")
    time_model = load_slot_model("hist_time_textcat_fr")

    type_value = "all"
    time_value = None

    if type_model is not None:
        d = type_model(text)
        type_value = max(d.cats, key=d.cats.get)
    if time_model is not None:
        d = time_model(text)
        time_value = max(d.cats, key=d.cats.get)

    return {"type": type_value, "time": time_value}


def infer_compte_rule(text: str) -> str | None:
    flat = compact(normalize(text))
    if any(stem in flat for stem in ["epargne", "epagne", "epagn", "econom", "livret", "cagnotte", "basdelaine", "mettredecote"]):
        return "epargne"
    if any(stem in flat for stem in ["courant", "cheque", "principal", "quotidien", "touslesjours", "depense", "depenses", "paiement"]):
        return "courant"
    if "compte" in flat:
        return "all"
    return None


def extract_consulter_solde_entities(text: str) -> dict[str, Any]:
    rule_compte = infer_compte_rule(text)
    if rule_compte is not None:
        return {"compte": rule_compte}

    model = load_slot_model("compte_textcat_fr")
    if model is not None:
        d = model(text)
        compte = max(d.cats, key=d.cats.get)
    else:
        flat = compact(normalize(text))
        if any(stem in flat for stem in ["epargne", "epagne", "epagn", "econom", "livret", "cagnotte", "basdelaine"]):
            compte = "epargne"
        elif any(stem in flat for stem in ["courant", "cheque", "depens", "quotidien", "paiement"]):
            compte = "courant"
        else:
            compte = "all"
    return {"compte": compte}


def apply_intent_overrides(text_norm: str, predicted: str) -> str:
    flat = compact(text_norm)
    has_greeting = has_greeting_signal(text_norm)
    has_social_greeting = has_social_greeting_signal(text_norm)
    has_help = has_help_signal(text_norm)
    has_guidance = has_guidance_signal(text_norm)
    has_out_of_scope = has_out_of_scope_signal(text_norm)
    has_banking_domain = has_banking_domain_signal(text_norm)
    has_cancel = any(
        stem in flat
        for stem in [
            "annule",
            "annuler",
            "annulation",
            "stop",
            "laissetomber",
            "oublie",
            "arrete",
            "neveuxplus",
            "nenvoie",
            "netransferepas",
            "cancel",
        ]
    )
    has_virement_action = has_transfer_action_signal(text_norm)
    has_balance = has_balance_signal(text_norm)
    has_hist_subject = has_history_subject_signal(text_norm)
    has_hist_recent = has_history_recent_signal(text_norm)
    has_hist_filter = has_time_reference(text_norm) and (
        has_hist_subject or ("virement" in flat and any(stem in flat for stem in ["recu", "recus", "envoye", "envoyes"]))
    )
    has_hist_simple = has_hist_subject or has_hist_recent

    if has_explicit_out_of_scope_frame(text_norm):
        return "INCOMPRIS"
    if has_out_of_scope and not (has_virement_action or has_balance or has_hist_subject or has_banking_domain):
        return "INCOMPRIS"

    if has_social_greeting and not (has_banking_domain or has_out_of_scope):
        return "salutation"
    if has_guidance and has_banking_domain and not has_out_of_scope:
        return "aide"
    if has_cancel:
        return "ANNULATION"
    if has_hist_filter:
        return "HISTORIQUE_FILTRE"
    if has_balance and not has_hist_subject and not has_virement_action:
        return "CONSULTER_SOLDE"
    if has_hist_simple and not has_virement_action:
        return "HISTORIQUE_SIMPLE"
    if has_virement_action:
        return "VIREMENT_INIT"
    if (has_help or has_guidance) and not has_out_of_scope:
        return "aide"
    if has_greeting and not has_out_of_scope:
        return "salutation"
    return predicted


def has_strong_rule_signal(text_norm: str, intent: str) -> bool:
    flat = compact(text_norm)
    if intent == "ANNULATION":
        return any(stem in flat for stem in ["annule", "annuler", "annulation", "stop", "laissetomber", "oublie", "arrete", "nenvoie"])
    if intent == "salutation":
        return has_greeting_signal(text_norm) and not has_out_of_scope_signal(text_norm)
    if intent == "aide":
        return (has_help_signal(text_norm) or has_guidance_signal(text_norm)) and not has_out_of_scope_signal(text_norm)
    if intent == "VIREMENT_INIT":
        has_action = has_transfer_action_signal(text_norm)
        return has_action
    if intent == "HISTORIQUE_FILTRE":
        return has_history_subject_signal(text_norm) and has_time_reference(text_norm)
    if intent == "HISTORIQUE_SIMPLE":
        return has_history_subject_signal(text_norm) or has_history_recent_signal(text_norm)
    if intent == "CONSULTER_SOLDE":
        return has_balance_signal(text_norm)
    return False


def predict_with_confidence(
    text: str,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> dict[str, Any]:
    warmup()
    prepared = normalize_for_intent(text)
    flat = compact(prepared)
    doc = _intent_nlp(prepared)
    ranked = sorted(doc.cats.items(), key=lambda x: x[1], reverse=True)
    top1_label, top1_score = ranked[0]
    top2_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top1_score - top2_score

    intent = apply_intent_overrides(prepared, top1_label)
    fallback = False
    low_signal = any(stem in flat for stem in ["hmm", "euh", "reflech", "passur", "jesaispas", "jenecomprendspas"])
    strong_rule = has_strong_rule_signal(prepared, intent)
    if (top1_score < score_threshold or margin < margin_threshold or low_signal) and not strong_rule:
        intent = "INCOMPRIS"
        fallback = True

    return {
        "intent": intent,
        "score": float(top1_score),
        "margin": float(margin),
        "fallback": fallback,
        "top_labels": ranked[:3],
    }


def build_output(
    text: str,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> dict[str, Any]:
    pred = predict_with_confidence(text, score_threshold=score_threshold, margin_threshold=margin_threshold)
    intention = pred["intent"]

    if intention == "VIREMENT_INIT":
        entites = extract_virement_entities(text, _known_beneficiaires)
    elif intention == "HISTORIQUE_FILTRE":
        entites = extract_historique_filtre_entities(text)
    elif intention == "CONSULTER_SOLDE":
        entites = extract_consulter_solde_entities(text)
    else:
        entites = {}

    return {
        "intention": intention,
        ENTITIES_KEY: entites,
        "score": round(pred["score"], 4),
        "fallback": pred["fallback"],
    }


def predict(
    text: str,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> dict[str, Any]:
    return build_output(text, score_threshold=score_threshold, margin_threshold=margin_threshold)
