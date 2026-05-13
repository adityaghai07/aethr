"""
Medical guardrail reward plugin for Vithos.

Rewards responses that observe and contextualize lab data calmly.
Penalizes (and flags for hard-block) responses that:
  - Diagnose ("you have X", "this indicates Y disease")
  - Alarm ("DANGER", "critical", "emergency")
  - Prescribe ("take X mg", "start/stop your medication")
  - Give legalistic disclaimers (the robotic AI-voice version)

The goal: a response like
  "Your LDL is 142 mg/dL — above the 130 threshold and rising across your
   last 3 tests. Worth mentioning at your next checkup."
scores high. A response like
  "DANGER: Your LDL is critically elevated. You must see a doctor immediately."
gets a hard violation and is blocked before it reaches the user.

This plugin is separate from the general reward stack so you can:
  - Tune its weight independently
  - Disable it entirely for non-medical deployments
  - Replace its patterns without touching any other file
"""
import re
import logging
from rewards.registry import RewardPlugin, RewardResult, register

logger = logging.getLogger(__name__)


# ── Pattern banks ─────────────────────────────────────────────────────────────
# Each list is a (pattern, explanation) tuple so violations are explainable in logs.

_DIAGNOSIS_PATTERNS: list[tuple[str, str]] = [
    (r"\byou (?:have|are suffering from|are diagnosed with)\b", "diagnosis statement"),
    (r"\bthis (?:indicates?|suggests?|confirms?|means? you have)\b", "diagnosis implication"),
    (r"\byou (?:likely|probably|definitely) have\b", "diagnosis speculation"),
    (r"\byou(?:'re| are) (?:diabetic|hypertensive|anemic|pre-?diabetic)\b", "labeling diagnosis"),
    (r"\bdiagnosis (?:is|seems to be)\b", "explicit diagnosis"),
]

_ALARM_PATTERNS: list[tuple[str, str]] = [
    (r"\b(?:DANGER|CRITICAL|URGENT|EMERGENCY|ALERT)\b", "alarm capitalized keyword"),
    (r"\bcritically (?:elevated|low|high|abnormal)\b", "alarm adjective"),
    (r"\bimmediate(?:ly)? (?:see|consult|visit|call)\b", "urgency command"),
    (r"\bvery (?:dangerous|alarming|serious|concerning)\b", "alarm intensifier"),
    (r"\blife.?threatening\b", "life-threat language"),
    (r"\bseek (?:emergency|immediate) (?:care|attention|help)\b", "emergency directive"),
]

_TREATMENT_PATTERNS: list[tuple[str, str]] = [
    (r"\b(?:take|start|stop|discontinue|increase|decrease|adjust) (?:your )?(?:medication|medicine|drug|dose|dosage|prescription)\b", "treatment instruction"),
    (r"\byou should (?:take|start|stop|avoid)\b", "treatment recommendation"),
    (r"\bI (?:recommend|suggest|advise) (?:taking|starting|stopping)\b", "treatment advice"),
    (r"\bprescri(?:be|ption)\b", "prescription language"),
    (r"\bdos(?:e|age) (?:of|should be)\b", "dosage instruction"),
]

_REDIRECT_PATTERNS: list[tuple[str, str]] = [
    (r"\b(?:worth mentioning|worth discussing|worth bringing up) (?:at|to|with) your (?:doctor|GP|physician|specialist|next checkup)\b", "good redirect"),
    (r"\byour (?:doctor|GP|physician|specialist|healthcare provider) (?:would|can|should|is best)\b", "good defer"),
    (r"\bsomething to (?:go over|discuss|talk about) with\b", "good redirect variant"),
    (r"\bbring (?:this|it) up (?:at|with|to)\b", "gentle redirect"),
]

_GOOD_DISCLAIMER_PATTERNS: list[tuple[str, str]] = [
    (r"\bthese are observations from your data\b", "human disclaimer — exact"),
    (r"\bnot a diagnosis\b", "human disclaimer — component"),
    (r"\byour (?:doctor|GP|physician) has the final word\b", "doctor deference"),
    (r"\bI(?:'m| am) just (?:observing|noting|tracking|contextualizing)\b", "observer framing"),
]

_BAD_DISCLAIMER_PATTERNS: list[tuple[str, str]] = [
    (r"\bI am not a (?:doctor|medical professional|physician|licensed)\b", "legalistic AI disclaimer"),
    (r"\bthis (?:is|does) not (?:constitute|replace) (?:medical )?(?:advice|diagnosis|treatment)\b", "legalistic boilerplate"),
    (r"\bconsult (?:a|your) (?:qualified )?(?:medical professional|healthcare provider|physician) before\b", "legalistic consult"),
    (r"\bfor informational purposes only\b", "legalese"),
]

_CLINICAL_QUERY_PATTERNS: list[tuple[str, str]] = [
    (r"\b(?:symptom|diagnos|medic|treatment|medication|drug|prescri|dose|dosage|side effect)\b", "clinical query"),
    (r"\bdo I have\b", "diagnosis query"),
    (r"\bwhat (?:disease|condition|illness|disorder)\b", "disease query"),
    (r"\bshould I (?:take|start|stop|avoid)\b", "treatment query"),
]


def _match_any(text: str, patterns: list[tuple[str, str]]) -> list[str]:
    """Return descriptions of all patterns that match."""
    text_lower = text.lower()
    return [
        desc for pat, desc in patterns
        if re.search(pat, text_lower, re.IGNORECASE)
    ]


class MedicalGuardrailPlugin(RewardPlugin):
    """
    Enforces Vithos's medical observation guardrails.

    Scoring logic:
      +0.30  — uses calm, contextualizing language
      +0.20  — human-sounding disclaimer present
      +0.20  — proper doctor redirect when query is clinical
      −0.40  — uses legalistic AI boilerplate (sounds robotic)
      −0.60  — alarming language detected  (violated=True)
      −1.00  — diagnosis statement detected (violated=True → hard block)
      −1.00  — treatment recommendation detected (violated=True → hard block)

    Hard violations (violated=True) cause the bot to discard the response
    and regenerate, so they never reach the user.
    """
    name = "medical_guardrails"
    weight = 0.40   # Higher weight than general for medical deployments

    async def score(self, prompt: str, response: str, history: list[dict]) -> RewardResult:
        score = 0.0
        details: dict = {}
        violated = False

        # ── Hard violations ───────────────────────────────────────────────────

        diagnosis_hits = _match_any(response, _DIAGNOSIS_PATTERNS)
        if diagnosis_hits:
            score -= 1.0
            violated = True
            details["diagnosis_violations"] = diagnosis_hits
            logger.warning(f"Diagnosis language detected: {diagnosis_hits}")

        treatment_hits = _match_any(response, _TREATMENT_PATTERNS)
        if treatment_hits:
            score -= 1.0
            violated = True
            details["treatment_violations"] = treatment_hits
            logger.warning(f"Treatment recommendation detected: {treatment_hits}")

        alarm_hits = _match_any(response, _ALARM_PATTERNS)
        if alarm_hits:
            score -= 0.60
            violated = True
            details["alarm_violations"] = alarm_hits
            logger.warning(f"Alarming language detected: {alarm_hits}")

        # ── Soft penalties ────────────────────────────────────────────────────

        bad_disclaimer_hits = _match_any(response, _BAD_DISCLAIMER_PATTERNS)
        if bad_disclaimer_hits:
            score -= 0.40
            details["legalistic_disclaimer"] = bad_disclaimer_hits

        # ── Positive signals ──────────────────────────────────────────────────

        good_disclaimer_hits = _match_any(response, _GOOD_DISCLAIMER_PATTERNS)
        if good_disclaimer_hits:
            score += 0.20
            details["good_disclaimer"] = good_disclaimer_hits

        redirect_hits = _match_any(response, _REDIRECT_PATTERNS)
        # Only reward redirect if the prompt was actually clinical
        is_clinical_query = bool(_match_any(prompt, _CLINICAL_QUERY_PATTERNS))
        if redirect_hits and is_clinical_query:
            score += 0.20
            details["redirect_present"] = redirect_hits
        elif is_clinical_query and not redirect_hits:
            score -= 0.20
            details["redirect_missing"] = "clinical query without doctor redirect"

        # Calm contextualizing language (words that observe without alarming)
        calm_words = ["trending", "above", "below", "within range", "worth mentioning",
                      "noted", "tracking", "across your last", "compared to"]
        calm_hits = [w for w in calm_words if w in response.lower()]
        if calm_hits:
            score += 0.30
            details["calm_language"] = calm_hits

        # Clamp to [-1, 1]
        score = max(-1.0, min(1.0, score))

        return RewardResult(score=score, details=details, violated=violated)


register(MedicalGuardrailPlugin())
