"""
Compute per-stakeholder reputation scores and reactions for each unique event
by leveraging stakeholder profile prompts in stakeholders/ and OpenAI.

Outputs: unique_events_output/stakeholder_reactions.json
Structure:
{
  "generated_at": iso8601,
  "stakeholders": ["CEO", "customers", "employees", "investor", ...],
  "events": [
     {
       "event_key": "<name>|<date>",
       "event_name": str,
       "event_date": str,
       "reactions": {
           "CEO": {"reputation_score": 1-5, "reaction_label": str, "rationale": str},
           "customers": { ... },
           ...
       }
     }, ...
  ]
}
"""
import os
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
import asyncio

from enum import Enum
from pydantic import BaseModel, Field, conint, confloat
import instructor
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

UNIQUE_EVENTS_FILE = 'unique_events_output/unique_events_chatgpt_v2.json'
STAKEHOLDERS_DIR = 'stakeholders'
OUTPUT_FILE = 'unique_events_output/stakeholder_reactions.json'
EVENT_FLAGS_FILE = 'unique_events_output/event_flags.json'

# Default stakeholders (extensible): map filename prefix to display key
# Files are named like: stakeholder-CEO-Alan-Joyce.md, stakeholder-customers.md, etc.
# We'll derive the key automatically but keep a preferred map for canonical names.
PREFERRED_KEYS = {
    'ceo': 'CEO',
    'customers': 'Customers',
    'employees': 'Employees',
    'investor': 'Investors',
}

# Allowed reaction labels (global enum values) per stakeholder_specifications.md
ALLOWED_REACTION_LABELS = [
    "Do Nothing",
    "Share Opinion with Others",
    "Switch to Competitor",
    "Legal/Regulatory Action",
    "Shareholder Activism",
    "Devalue Shares",
    "Industrial Action",
]

def extract_allowed_actions(profile_text: str) -> List[str]:
    """Parse allowed actions from a stakeholder profile markdown.
    Looks for a section starting with 'Allowed Actions' or 'Actions' followed by bullet points or lines.
    If not found, returns the full global list.
    """
    lines = [l.strip() for l in (profile_text or '').splitlines()]
    allowed: List[str] = []
    capture = False
    for ln in lines:
        low = ln.lower()
        if low.startswith('allowed actions') or low == 'actions:' or low.startswith('actions'):
            capture = True
            continue
        if capture:
            if not ln:
                break
            # bullets like '- Do Nothing' or '* Do Nothing'
            if ln.startswith(('-', '*', '•')):
                val = ln.lstrip('-*• ').strip()
                if val in ALLOWED_REACTION_LABELS:
                    allowed.append(val)
            else:
                # plain line until blank
                val = ln.strip()
                if val in ALLOWED_REACTION_LABELS:
                    allowed.append(val)
    return allowed or ALLOWED_REACTION_LABELS

def extract_profile_constraints(profile_text: str) -> str:
    """Extract constraint hints from profile to include in prompt.
    - Pulls explicit 'Trigger Conditions' block if present (bullet list or lines until blank).
    - Adds generic timeframe constraint if detected.
    """
    text = profile_text or ''
    lines = [l.rstrip() for l in text.splitlines()]
    constraints: List[str] = []

    # Extract Trigger Conditions block
    trig_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith('trigger conditions'):
            trig_idx = i + 1
            break
    if trig_idx is not None:
        for j in range(trig_idx, len(lines)):
            cur = lines[j].strip()
            if cur == '':
                break
            if cur.startswith(('-', '*', '•')):
                constraints.append(cur.lstrip('-*• ').strip())
            else:
                constraints.append(cur)

    return '\n'.join(constraints)

# Note: We no longer infer labels in code; the model must decide based on the profile.

def extract_disallowed_contexts(profile_text: str) -> List[Dict[str, str]]:
    """Parse 'Disallowed in Contexts' rules.
    Expected format (bulleted lines):
    - <rule text containing one of the allowed action labels and a context description>
    We'll extract the first matching action label and keep the rest of the line as 'context'.
    Returns list of dicts: { 'action': <label>, 'context': <lowercased context text> }
    """
    rules: List[Dict[str, str]] = []
    if not profile_text:
        return rules
    lines = [l.rstrip() for l in profile_text.splitlines()]
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith('disallowed in contexts'):
            start = i + 1
            break
    if start is None:
        return rules
    for j in range(start, len(lines)):
        cur = lines[j].strip()
        if cur == '':
            break
        if cur.startswith(('-', '*', '•')):
            text = cur.lstrip('-*• ').strip()
        else:
            text = cur
        # find the earliest-mentioned action label in text (case-insensitive)
        lower_text = text.lower()
        earliest_idx = None
        matched_action = None
        for lbl in ALLOWED_REACTION_LABELS:
            idx = lower_text.find(lbl.lower())
            if idx != -1 and (earliest_idx is None or idx < earliest_idx):
                earliest_idx = idx
                matched_action = lbl
        if matched_action:
            # remove action phrase from context loosely
            ctx = text
            rules.append({'action': matched_action, 'context': ctx.lower()})
    return rules

class ReactionType(str, Enum):
    DoNothing = "Do Nothing"
    ShareOpinion = "Share Opinion with Others"
    SwitchToCompetitor = "Switch to Competitor"
    LegalOrRegulatoryAction = "Legal/Regulatory Action"
    ShareholderActivism = "Shareholder Activism"
    DevalueShares = "Devalue Shares"
    IndustrialAction = "Industrial Action"


class StakeholderReaction(BaseModel):
    reputation_score: conint(ge=-5, le=5) = Field(..., description="Delta score in [-5,5]")
    reaction_label: ReactionType
    reaction_probability: confloat(ge=0.0, le=1.0)
    rationale: str = Field(..., max_length=200)


class EventClassification(BaseModel):
    is_sector_wide_exogenous: bool
    is_material_to_qantas: bool
    is_safety_event: bool
    is_customer_service_event: bool
    is_corporate_governance_event: bool
    is_labour_event: bool


def adjust_event_flags(ev: Dict, flags: EventClassification) -> EventClassification:
    """Deterministically adjust event flags based on clear governance/financial cues.
    If the event clearly indicates a regulatory/court penalty/fine/settlement or shareholder/governance action
    and the primary entity is Qantas, then mark as corporate governance and material.
    """
    name = (ev.get('event_name') or '').lower()
    cats = [str(c).lower() for c in (ev.get('event_categories') or [])]
    primary = (ev.get('primary_entity') or '').lower()
    governance_keywords = [
        'penalty', 'fine', 'civil penalty', 'settlement', 'court', 'lawsuit', 'class action',
        'remuneration', 'say on pay', 'remuneration report', 'bonus', 'executive pay', 'shareholder vote',
        'accc', 'asic', 'regulator', 'regulatory'
    ]
    cat_governance_signals = {'governance', 'legal', 'regulatory', 'compliance'}
    has_kw = any(k in name for k in governance_keywords)
    has_cat = any(c in cat_governance_signals for c in cats)
    is_qantas = primary in ['qantas', 'qantas airways'] or ('qantas' in name)
    if is_qantas and (has_kw or has_cat):
        flags.is_corporate_governance_event = True
        flags.is_material_to_qantas = True
    return flags

def load_stakeholder_profiles() -> Dict[str, str]:
    profiles: Dict[str, str] = {}
    if not os.path.isdir(STAKEHOLDERS_DIR):
        return profiles
    for p in Path(STAKEHOLDERS_DIR).glob('stakeholder-*.md'):
        content = p.read_text(encoding='utf-8')
        # Derive key from filename
        # Example: stakeholder-CEO-Alan-Joyce.md -> ceo
        #          stakeholder-customers.md -> customers
        parts = p.stem.split('-')  # ['stakeholder', 'CEO', 'Alan', 'Joyce'] or ['stakeholder', 'customers']
        if len(parts) >= 2:
            raw = parts[1].lower()
            key = PREFERRED_KEYS.get(raw, parts[1].title())
            profiles[key] = content
    return profiles


def load_events() -> List[Dict]:
    if not os.path.exists(UNIQUE_EVENTS_FILE):
        raise FileNotFoundError(f"Unique events file not found: {UNIQUE_EVENTS_FILE}")
    with open(UNIQUE_EVENTS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Expecting a list of event dicts
        return data


def event_key(name: str, date: str) -> str:
    base = f"{name}|{date}"
    # Ensure stable short key
    h = hashlib.md5(base.encode('utf-8')).hexdigest()[:8]
    return f"{base}|{h}"


def build_event_summary(ev: Dict) -> str:
    cats = ', '.join([c for c in ev.get('event_categories', []) if str(c).lower() != 'reputation'])
    stakes = ', '.join(ev.get('stakeholders', []) or [])
    resp = ', '.join(ev.get('response_strategies', []) or [])
    summary = (
        f"Event: {ev.get('event_name', 'Unknown')}\n"
        f"Date: {ev.get('event_date', '')}\n"
        f"Primary entity: {ev.get('primary_entity', 'Unknown')}\n"
        f"Categories: {cats or 'N/A'}\n"
        f"Stakeholders mentioned: {stakes or 'N/A'}\n"
        f"Qantas response strategies observed: {resp or 'N/A'}\n"
        f"Mean damage score: {ev.get('mean_damage_score', 0)}\n"
        f"Mean response score: {ev.get('mean_response_score', 0)}\n"
        f"Number of articles: {ev.get('num_articles', 0)}\n"
    )
    return summary


def load_event_flags_cache() -> Dict[str, Dict[str, bool]]:
    if not os.path.exists(EVENT_FLAGS_FILE):
        return {}
    try:
        with open(EVENT_FLAGS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_event_flags_cache(cache: Dict[str, Dict[str, bool]]):
    os.makedirs(os.path.dirname(EVENT_FLAGS_FILE), exist_ok=True)
    with open(EVENT_FLAGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def compute_materiality_hint(ev: Dict) -> Tuple[bool, str]:
    """Heuristic hint to help the model decide materiality correctly.
    Not a hard rule, but used to guide and (if needed) validate outputs.
    """
    dmg = float(ev.get('mean_damage_score') or 0)
    arts = int(ev.get('num_articles') or 0)
    name = (ev.get('event_name') or '').lower()
    cats = [str(c).lower() for c in (ev.get('event_categories') or [])]
    primary = (ev.get('primary_entity') or '').lower()

    # Likely immaterial if low damage, few articles, not a Qantas-specific operational/safety/governance incident
    likely_immaterial = (dmg <= 2.0 and arts < 15)
    if any(k in name for k in ["ranking", "tops", "award", "accolade", "list", "wins"]):
        likely_immaterial = True
    if any(k in name for k in ["data breach", "privacy", "crm"]) and arts < 30 and dmg <= 2.5:
        # customer data issues often immaterial to investors/employees actions
        likely_immaterial = True
    if 'safety' in cats and 'incident' in cats and primary in ['qantas', 'qantas airways']:
        # Qantas-specific safety incident: potentially material
        likely_immaterial = False
    # Governance/remuneration controversies are material for investors
    if any(k in name for k in [
        'remuneration', 'exec pay', 'executive pay', 'bonus', 'bonuses', 'pay vote', 'rem report', 'rem report', 'agm pay', 'say on pay'
    ]) or 'governance' in cats:
        if primary in ['qantas', 'qantas airways']:
            likely_immaterial = False

    reason = f"dmg={dmg}, articles={arts}, primary='{primary}', categories={cats}"
    return likely_immaterial, reason


def classify_event(client: OpenAI, ev: Dict) -> EventClassification:
    """Classify the event once using the LLM. Cached per event in the run."""
    schema = json.dumps(EventClassification.model_json_schema(), ensure_ascii=False)
    prompt = (
        "Classify the following event about Qantas with boolean flags. Return strict JSON only.\n"
        "Required keys: is_sector_wide_exogenous, is_material_to_qantas, is_safety_event, is_customer_service_event, is_corporate_governance_event, is_labour_event.\n\n"
        f"JSON Schema:\n{schema}\n\n"
        f"Event Context\n------------\n{build_event_summary(ev)}\n\n"
        "Notes:\n"
        "- Sector-wide/exogenous means constraints affecting all airlines similarly (e.g., regional airspace closure, broad regulation).\n"
        "- Corporate governance includes remuneration/bonus controversies, say-on-pay, remuneration report.\n"
        "- Labour includes bargaining, union action, outsourcing, rosters, hours, pay.\n"
        "- Safety event is Qantas-attributable operational safety incident; generic rankings are not safety incidents.\n"
        "Output JSON only."
    )
    msg = [
        {"role": "system", "content": "You are a precise event classifier. Answer with strict JSON matching the schema."},
        {"role": "user", "content": prompt},
    ]
    result: EventClassification = client.chat.completions.create(
        model="gpt-4o",
        response_model=EventClassification,
        messages=msg,
        temperature=0.0,
    )
    return result


def get_client() -> OpenAI:
    load_dotenv()
    # Patch OpenAI client for structured extraction via instructor
    return instructor.patch(OpenAI())


def save_output(events: List[Dict]):
    """Persist the final reactions file to OUTPUT_FILE.
    Structure: { generated_at, stakeholders, events }
    """
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    # Stakeholder names inferred from reaction keys of first event, else from profiles discovered in stakeholders/
    stakeholders: List[str] = []
    if events and isinstance(events, list):
        first = events[0]
        if isinstance(first, dict) and 'reactions' in first and isinstance(first['reactions'], dict):
            stakeholders = list(first['reactions'].keys())
    payload = {
        'generated_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'stakeholders': stakeholders,
        'events': events,
    }
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def score_event_for_stakeholder(client: OpenAI, stakeholder_name: str, profile_prompt: str, ev: Dict, ev_flags: EventClassification) -> Dict:
    system_prompt = profile_prompt.strip()
    allowed = extract_allowed_actions(profile_prompt)
    constraints_hint = extract_profile_constraints(profile_prompt)
    disallowed_rules = extract_disallowed_contexts(profile_prompt)
    # Include explicit JSON schema to reduce retries
    json_schema = json.dumps(StakeholderReaction.model_json_schema(), ensure_ascii=False)
    hint_immaterial, hint_reason = compute_materiality_hint(ev)
    base_user_prompt = (
        "You are asked to assess a single historical event about Qantas.\n"
        "Return a strict JSON object with keys: reputation_score (integer delta from -5..5), reaction_label (enum), reaction_probability (0..1 float), rationale (≤ 30 words).\n"
        "Definition: reputation_score is a DELTA for this stakeholder caused by this event: -5=extremely negative reaction, 0=neutral, +5=extremely positive reaction.\n"
        "reaction_label MUST be exactly one of: ['Do Nothing','Share Opinion with Others','Switch to Competitor','Legal/Regulatory Action','Shareholder Activism','Devalue Shares','Industrial Action'].\n"
        f"For this stakeholder ('{stakeholder_name}'), ONLY choose from this subset: {allowed}.\n"
        f"Profile Constraints (follow strictly if applicable):\n{constraints_hint}\n\n"
        "Materiality Guidance (apply BEFORE choosing action):\n"
        "- First assess whether the event is company-specific and materially impacts Qantas (routes, safety incidents attributable to Qantas, costs, revenue, licence to operate).\n"
        "- If the event is a sector-wide or exogenous constraint affecting all airlines similarly (e.g., regional airspace closure, broad regulatory change) and the impact on Qantas is immaterial, then choose 'Do Nothing' and set reputation_score = 0 (neutral).\n"
        "- Only choose switching/activism/industrial actions when there is clear stakeholder-specific justification tied to material Qantas impact consistent with the stakeholder profile.\n"
        "- For competitor accolades/rankings (e.g., another airline topping a safety list), treat as immaterial unless there is specific, verified evidence of a Qantas safety issue in the event context; default to 'Do Nothing' with reputation_score = 0.\n"
        "- For Qantas governance/executive remuneration controversies (e.g., bonus disputes, say-on-pay, remuneration report issues), treat as material for investors and consider Shareholder Activism per investor profile triggers.\n"
        "- Avoid overreacting to news that lacks direct, material Qantas impact.\n\n"
        f"Event Classification (deterministic; do not contradict):\n"
        f"- is_sector_wide_exogenous = {ev_flags.is_sector_wide_exogenous}\n"
        f"- is_material_to_qantas = {ev_flags.is_material_to_qantas}\n"
        f"- is_safety_event = {ev_flags.is_safety_event}\n"
        f"- is_customer_service_event = {ev_flags.is_customer_service_event}\n"
        f"- is_corporate_governance_event = {ev_flags.is_corporate_governance_event}\n"
        f"- is_labour_event = {ev_flags.is_labour_event}\n\n"
    )
    # Conditional investor guidance
    if stakeholder_name == 'Investors':
        base_user_prompt += (
            "Investor Guidance: Only react with Shareholder Activism or Devalue Shares when the event is a corporate governance/financial issue. "
            "For operational/service/safety/labour items without governance signals, choose a neutral action (Do Nothing/Share Opinion with Others) "
            "with a small delta in [-1,0,1].\n\n"
        )
    base_user_prompt += (
        f"Materiality Hint (heuristic): This event appears likely immaterial = {str(hint_immaterial)} based on {hint_reason}. If immaterial, set is_material_to_qantas=false, reputation_score=0, and choose a neutral action (Do Nothing/Share Opinion with Others).\n\n"
        "Disallowed in Contexts (from profile; do not violate):\n"
        f"{json.dumps(disallowed_rules, ensure_ascii=False)}\n\n"
        "reaction_probability is your estimated probability that THIS stakeholder takes the action indicated by reaction_label (between 0 and 1). Decide independently for this stakeholder (do not reuse another stakeholder's outputs).\n\n"
        "JSON Schema (conform exactly):\n"
        f"{json_schema}\n\n"
        f"Event Context\n------------\n{build_event_summary(ev)}\n\n"
        "Output JSON only."
    )

    def attempt(prompt_suffix: str = "") -> StakeholderReaction:
        return client.chat.completions.create(
            model="gpt-4o",
            response_model=StakeholderReaction,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": base_user_prompt + ("\n" + prompt_suffix if prompt_suffix else "")},
            ],
            temperature=0.0,
        )

    retries = 5
    last_error = ""
    for i in range(retries):
        try:
            retry_suffix = "STRICT VALIDATION: Ensure reputation_score is an integer in [-5,5]; reaction_label is in the allowed subset; reaction_probability is a float in [0,1]."
            if hint_immaterial:
                retry_suffix += " For this event, treat as immaterial unless specific Qantas impacts are present: set is_material_to_qantas=false; choose a neutral action (Do Nothing or Share Opinion with Others); keep reputation_score in [-1,0,1] (prefer 0)."
            parsed = attempt(retry_suffix if i > 0 else "")
            # Enforce stakeholder-specific subset
            label_text = parsed.reaction_label.value
            if label_text not in allowed:
                raise ValueError(f"Label '{label_text}' not in allowed subset {allowed}")
            # Enforce immaterial neutrality: if event is not material to Qantas, only neutral actions allowed and score must be small (−1..1)
            if not bool(ev_flags.is_material_to_qantas):
                if label_text not in ["Do Nothing", "Share Opinion with Others"]:
                    raise ValueError("Immaterial event chosen with non-neutral action; violates materiality guidance")
                if int(parsed.reputation_score) not in (-1, 0, 1):
                    raise ValueError("Immaterial event must have a small neutral delta in [-1,0,1] (prefer 0)")
            # Enforce disallowed-in-contexts rules
            ev_ctx = build_event_summary(ev).lower()
            for rule in disallowed_rules:
                act = rule.get('action')
                ctx_sub = (rule.get('context') or '').lower()
                sector_flag = 'sector-wide' in ctx_sub or 'exogenous' in ctx_sub or 'all airlines' in ctx_sub
                matches_context = (ctx_sub and (ctx_sub in ev_ctx)) or (sector_flag and bool(ev_flags.is_sector_wide_exogenous))
                if label_text == act and matches_context:
                    raise ValueError(f"Action '{label_text}' is disallowed under context rule: {ctx_sub}")
            # Investor guardrail: Only allow activism/valuation actions when governance is involved.
            if stakeholder_name == 'Investors':
                if not bool(ev_flags.is_corporate_governance_event):
                    # Operational/service/safety/labour without governance => neutral-only, small delta
                    if label_text not in ["Do Nothing", "Share Opinion with Others"]:
                        raise ValueError("Investors: activism/valuation actions require corporate governance/financial issues. Choose a neutral action.")
                    if int(parsed.reputation_score) not in (-1, 0, 1):
                        raise ValueError("Investors: neutral-only small delta [-1,0,1] for non-governance events.")
            # Note: Do not hard-fail on heuristic disagreements; enforcement is driven by event-level flags (ev_flags)

            return {
                "reputation_score": int(parsed.reputation_score),
                "reaction_label": label_text,
                "reaction_probability": round(float(parsed.reaction_probability), 3),
                "rationale": parsed.rationale.strip(),
            }
        except Exception as e:
            last_error = str(e)
            continue

    # After retries, HARD FAIL: raise with context
    raise ValueError(
        f"Validation failed after {retries} attempts for stakeholder '{stakeholder_name}' on event '{ev.get('event_name','Unknown')}'. "
        f"Last error: {last_error}"
    )


def main():
    async def run_all():
        client = get_client()
        profiles = load_stakeholder_profiles()
        allowed_stakeholders = list(profiles.keys())
        events = load_events()
        out_records = []

        loop = asyncio.get_event_loop()

        def make_event_record(ev, flags: EventClassification):
            ek = ev.get('event_key') or event_key(ev.get('event_name', ''), ev.get('event_date', ''))
            return {
                'event_key': ek,
                'event_name': ev['event_name'],
                'event_date': ev['event_date'],
                'event_flags': {
                    'is_sector_wide_exogenous': bool(flags.is_sector_wide_exogenous),
                    'is_material_to_qantas': bool(flags.is_material_to_qantas),
                    'is_safety_event': bool(flags.is_safety_event),
                    'is_customer_service_event': bool(flags.is_customer_service_event),
                    'is_corporate_governance_event': bool(flags.is_corporate_governance_event),
                    'is_labour_event': bool(flags.is_labour_event),
                },
                'reactions': {}
            }

        # Load cache once
        flags_cache = load_event_flags_cache()

        async def score_for_event(ev):
            # Classify event once, using cache when available
            ek = ev.get('event_key') or event_key(ev.get('event_name', ''), ev.get('event_date', ''))
            cached = flags_cache.get(ek)
            if cached:
                ev_flags = EventClassification(**cached)
            else:
                ev_flags = classify_event(client, ev)
            # Deterministic adjustment for governance/financial cues
            ev_flags = adjust_event_flags(ev, ev_flags)
            # Persist adjusted flags to cache (overwrites stale cache if needed)
            flags_cache[ek] = {
                'is_sector_wide_exogenous': bool(ev_flags.is_sector_wide_exogenous),
                'is_material_to_qantas': bool(ev_flags.is_material_to_qantas),
                'is_safety_event': bool(ev_flags.is_safety_event),
                'is_customer_service_event': bool(ev_flags.is_customer_service_event),
                'is_corporate_governance_event': bool(ev_flags.is_corporate_governance_event),
                'is_labour_event': bool(ev_flags.is_labour_event),
            }
            # Run stakeholder scoring concurrently for this event
            tasks = []
            for sk in allowed_stakeholders:
                profile = profiles[sk]
                # Use thread executor for blocking network call
                tasks.append(loop.run_in_executor(
                    None,
                    score_event_for_stakeholder,
                    client, sk, profile, ev, ev_flags
                ))
            results = await asyncio.gather(*tasks, return_exceptions=False)
            rec = make_event_record(ev, ev_flags)
            for sk, res in zip(allowed_stakeholders, results):
                rec['reactions'][sk] = res
            return rec

        for ev in tqdm(events, desc='Scoring events'):
            rec = await score_for_event(ev)
            out_records.append(rec)

        # Persist caches and output
        save_event_flags_cache(flags_cache)
        save_output(out_records)

    asyncio.run(run_all())


if __name__ == '__main__':
    main()
