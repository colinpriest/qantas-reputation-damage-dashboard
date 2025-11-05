"""
LLM-based Factor Extraction and Explainable Feature Engineering for Stock-Move Explanations

Overview
--------
This library implements an LLM-centric pipeline inspired by ideas in
"LLMFactor: Extracting Profitable Factors through Prompts for Explainable Stock Movement Prediction".

Given a pandas.DataFrame with columns:
    - text: str                # free-text explanation for the move
    - y: float | int           # target (e.g., abnormal return, or +1/-1 direction)
    - (optional) id/date/...   # any metadata you wish to carry through

The pipeline:
1) Extract fine-grained, finance-relevant factors from each explanation using an LLM prompt.
2) For each factor, determine direction (tailwind/headwind), centrality (primary/secondary), and causal status.
3) Canonicalize factors to a controlled taxonomy and compute per-factor features (presence, direction, centrality-weighted).
4) Produce an explainable feature matrix suitable for tree-based models (e.g., RandomForest).

The design supports pluggable LLM backends (OpenAI by default). No secrets are
hardcoded; pass keys via environment variables or constructor arguments.

Usage (minimal):
----------------
>>> import pandas as pd
>>> from llm_factors import FactorPipeline, OpenAIBackend
>>> df = pd.DataFrame({
...     'text': [
...         'Shares rose after analyst upgrade; oil prices fell; strong post-COVID demand.',
...         'Stock fell as jet fuel spiked and unions threatened strike during wage talks.'
...     ],
...     'y': [1, -1]
... })
>>> backend = OpenAIBackend(model="gpt-4o-mini")  # or gpt-4.1, etc.
>>> pipe = FactorPipeline(llm=backend)
>>> features, records = pipe.transform(df, text_col='text')
>>> features.head()

See `if __name__ == "__main__":` for a runnable demo skeleton.

Dependencies
------------
- pandas, numpy, scikit-learn, pydantic (for schema validation)

"""
from __future__ import annotations

import json
import os
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

# Optional: only required if you use OpenAI backend
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover
    OPENAI_AVAILABLE = False

# ------------------------------
# Logging
# ------------------------------
logger = logging.getLogger("llm_factors")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ------------------------------
# Factor Taxonomy & Canonicalization
# ------------------------------
CANONICAL_FACTORS: Dict[str, List[str]] = {
    # canonical_name: list of synonyms/keywords (lowercase)
    "fuel_prices": ["fuel", "jet fuel", "oil", "brent", "energy prices", "fuel cost", "oil price"],
    "labor_union": ["union", "strike", "industrial action", "wage", "pilot union", "cabin crew"],
    "route_network": ["route", "capacity", "load factor", "network", "frequencies", "slot"],
    "regulatory": ["regulator", "regulatory", "accc", "acma", "apra", "casa", "approval", "block", "fine"],
    "covid": ["covid", "pandemic", "reopening", "lockdown", "travel restrictions"],
    "executive_comp": ["executive comp", "ceo pay", "bonus", "remuneration", "exec compensation"],
    "competition": ["competitor", "competition", "virgin", "rex", "airline rivalry", "price war"],
    "technical_flow": ["ex-dividend", "technical", "profit-taking", "rebalancing", "index inclusion", "ftse", "asx"],
    "reputation": ["reputation", "brand", "customer service", "scandal", "shonky", "trust"],
    "agm_vote": ["agm", "remuneration vote", "two-strikes", "shareholder meeting"],
    "analyst_action": ["analyst", "broker", "downgrade", "upgrade", "target price", "recommendation"],
    "earnings_guidance": ["earnings", "guidance", "profit", "ebit", "revenue", "forecast"],
    "operations_safety": ["safety", "incident", "engine", "maintenance", "delay", "cancellation"],
    "partnership_mna": ["code share", "partnership", "alliance", "merger", "acquisition", "joint venture"],
    "macro_fx": ["macro", "inflation", "interest rate", "fx", "australian dollar", "aud"],
}

CANON_TO_HUMAN_LABEL: Dict[str, str] = {
    k: k.replace("_", " ").title() for k in CANONICAL_FACTORS.keys()
}

# ------------------------------
# Pydantic Schemas for LLM Outputs
# ------------------------------
class FactorItem(BaseModel):
    name: str = Field(..., description="Raw factor label emitted by the LLM (free text)")
    canonical: Optional[str] = Field(
        None, description="Canonical factor key if recognized (e.g., 'fuel_prices')"
    )
    direction: Literal["positive", "negative", "neutral"] = Field(
        ..., description="Direction of influence on the stock price"
    )
    centrality: Literal["primary", "secondary", "peripheral"] = Field(
        ..., description="How central this factor is to the explanation"
    )
    causal: bool = Field(
        ..., description="True if the text presents this as a causal reason for the move"
    )
    confidence: float = Field(
        0.5, ge=0.0, le=1.0, description="LLM confidence or plausibility score"
    )
    evidence: Optional[str] = Field(
        None, description="Short quote or span from the text supporting this factor"
    )

    @field_validator("name")
    @classmethod
    def strip_name(cls, v: str) -> str:
        return v.strip()

class ExtractionRecord(BaseModel):
    text: str
    factors: List[FactorItem]

# ------------------------------
# LLM Backend Abstraction
# ------------------------------
class LLMBackend:
    """Abstract base for LLM calls."""
    def __init__(self, temperature: float = 0.0, max_tokens: int = 800, seed: Optional[int] = None):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
        raise NotImplementedError

class OpenAIBackend(LLMBackend):
    """OpenAI Chat Completions backend (GPT-4.x, GPT-4o-mini, etc.)."""
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. `pip install openai`.")
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.organization = organization or os.getenv("OPENAI_ORG")
        if self.api_key is None:
            logger.warning("OPENAI_API_KEY not set; calls will fail until provided.")
        self.client = OpenAI(api_key=self.api_key, organization=self.organization)

    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
        use_model = model or self.model
        # Using Chat Completions for broad compatibility
        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(
                    model=use_model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=self.seed,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # pragma: no cover
                wait = 2 ** attempt
                logger.warning(f"OpenAI error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
        raise RuntimeError("OpenAI chat failed after retries.")

# ------------------------------
# Prompt Template & JSON Parsing Helpers
# ------------------------------
EXTRACTION_SYSTEM_PROMPT = (
    "You extract EXPLAINABLE stock-movement FACTORS from short news-like texts. "
    "Return strict JSON only, validating against the provided schema."
)

EXTRACTION_USER_PROMPT_TEMPLATE = (
    """
You will read a short explanation of a stock's daily move. Identify discrete FACTORS in that text that plausibly drove the move.
For each factor, provide:
- name: concise label (e.g., "fuel prices", "analyst downgrade", "union strike")
- direction: positive | negative | neutral (with respect to the stock)
- centrality: primary | secondary | peripheral (how central in this text)
- causal: true | false (is the factor framed as a cause, not background)
- confidence: 0.0..1.0 (plausibility)
- evidence: short supporting quote or phrase from the text

Output STRICT JSON: {{"factors": [{{...}}]}}

Text:
{text}

Schema reminder:
{{"type":"object","properties":{{"factors":{{"type":"array","items":{{"type":"object","properties":{{"name":{{"type":"string"}},"direction":{{"type":"string","enum":["positive","negative","neutral"]}},"centrality":{{"type":"string","enum":["primary","secondary","peripheral"]}},"causal":{{"type":"boolean"}},"confidence":{{"type":"number"}},"evidence":{{"type":["string","null"]}}}},"required":["name","direction","centrality","causal","confidence"]}}}}},"required":["factors"]}}
    """
).strip()

JSON_BLOCK_RE = re.compile(r"```(?:json)?\n(.*?)\n```", re.DOTALL)


def extract_json_from_text(s: str) -> str:
    """Best-effort: extract JSON from a response. Prefer fenced blocks; else raw JSON."""
    m = JSON_BLOCK_RE.search(s)
    if m:
        return m.group(1).strip()
    # fallback: try raw json in the message
    s = s.strip()
    # Trim any pre/post text heuristically
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    raise ValueError("No JSON found in LLM output")

# ------------------------------
# Canonicalization Utilities
# ------------------------------

def canonicalize_factor(raw: str) -> Optional[str]:
    r = raw.strip().lower()
    # exact match first
    if r in CANONICAL_FACTORS:
        return r
    # synonym lookup
    for canon, syns in CANONICAL_FACTORS.items():
        if r == canon:
            return canon
        for s in syns:
            if s in r or r in s:
                return canon
    # simple stemming heuristics
    r2 = r.replace("prices", "price").replace("costs", "cost")
    for canon, syns in CANONICAL_FACTORS.items():
        if canon in r2:
            return canon
        for s in syns:
            if s in r2:
                return canon
    return None

# ------------------------------
# Core Extractor
# ------------------------------
class FactorExtractor:
    def __init__(self, llm: LLMBackend):
        self.llm = llm

    def extract_from_text(self, text: str) -> ExtractionRecord:
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": EXTRACTION_USER_PROMPT_TEMPLATE.format(text=text)},
        ]
        raw = self.llm.chat(messages)
        payload = extract_json_from_text(raw)
        data = json.loads(payload)
        try:
            record = ExtractionRecord(text=text, **data)
        except Exception as ve:
            # Try a softer parse: coerce booleans/strings
            factors = []
            for f in data.get("factors", []):
                factors.append(
                    {
                        "name": str(f.get("name", "")).strip(),
                        "direction": str(f.get("direction", "neutral")).lower(),
                        "centrality": str(f.get("centrality", "secondary")).lower(),
                        "causal": bool(f.get("causal", False)),
                        "confidence": float(f.get("confidence", 0.5)),
                        "evidence": f.get("evidence"),
                    }
                )
            record = ExtractionRecord(text=text, factors=[FactorItem(**f) for f in factors])
        # canonicalize
        canon_items: List[FactorItem] = []
        for item in record.factors:
            canon = canonicalize_factor(item.name)
            canon_items.append(FactorItem(**{**item.model_dump(), "canonical": canon}))
        return ExtractionRecord(text=text, factors=canon_items)

# ------------------------------
# Feature Engineering
# ------------------------------
@dataclass
class FeatureConfig:
    use_direction_channels: bool = True
    use_centrality_weight: bool = True
    include_neutral: bool = False  # rarely useful; set True if desired


class FeatureEngineer:
    def __init__(self, config: FeatureConfig = FeatureConfig()):
        self.config = config
        # Stable, ordered list of canonical keys for matrix columns
        self.canonical_keys: List[str] = list(CANONICAL_FACTORS.keys())

    def factor_to_weight(self, item: FactorItem) -> float:
        if not self.config.use_centrality_weight:
            return 1.0
        return {"primary": 1.0, "secondary": 0.6, "peripheral": 0.3}.get(item.centrality, 0.5)

    def direction_channels(self) -> List[str]:
        dirs = ["pos", "neg"] + (["neu"] if self.config.include_neutral else [])
        return dirs

    def build_columns(self) -> List[str]:
        cols: List[str] = []
        if self.config.use_direction_channels:
            for k in self.canonical_keys:
                for d in self.direction_channels():
                    cols.append(f"{k}__{d}")
        else:
            cols = [f"{k}" for k in self.canonical_keys]
        # causality intensity channels
        cols += [f"{k}__causal" for k in self.canonical_keys]
        return cols

    def row_features(self, record: ExtractionRecord) -> Dict[str, float]:
        row = {c: 0.0 for c in self.build_columns()}
        for item in record.factors:
            if item.canonical is None:
                continue
            w = self.factor_to_weight(item) * float(item.confidence)
            if self.config.use_direction_channels:
                ch = {
                    "positive": "pos",
                    "negative": "neg",
                    "neutral": "neu",
                }.get(item.direction, "neu")
                if ch == "neu" and not self.config.include_neutral:
                    ch = None
                if ch:
                    key = f"{item.canonical}__{ch}"
                    row[key] = row.get(key, 0.0) + w
            else:
                key = item.canonical
                row[key] = row.get(key, 0.0) + w * ({"positive": 1, "negative": -1, "neutral": 0}.get(item.direction, 0))
            if item.causal:
                row[f"{item.canonical}__causal"] += w
        return row

    def transform(self, records: List[ExtractionRecord]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame(columns=self.build_columns())
        rows = [self.row_features(r) for r in records]
        return pd.DataFrame(rows)[self.build_columns()].fillna(0.0)

# ------------------------------
# Full Pipeline
# ------------------------------
class FactorPipeline:
    def __init__(self, llm: LLMBackend, feature_config: FeatureConfig = FeatureConfig()):
        self.extractor = FactorExtractor(llm)
        self.fe = FeatureEngineer(config=feature_config)

    def transform(self, df: pd.DataFrame, text_col: str = "text") -> Tuple[pd.DataFrame, List[ExtractionRecord]]:
        records: List[ExtractionRecord] = []
        for i, text in enumerate(df[text_col].astype(str).tolist()):
            try:
                rec = self.extractor.extract_from_text(text)
            except Exception as e:
                logger.error(f"Extraction failed at row {i}: {e}")
                rec = ExtractionRecord(text=text, factors=[])
            records.append(rec)
        X = self.fe.transform(records)
        return X, records

# ------------------------------
# Utilities: Pretty Reporting
# ------------------------------

def summarize_records(records: List[ExtractionRecord]) -> pd.DataFrame:
    rows = []
    for ridx, rec in enumerate(records):
        for f in rec.factors:
            rows.append(
                {
                    "row": ridx,
                    "raw_name": f.name,
                    "canonical": f.canonical,
                    "direction": f.direction,
                    "centrality": f.centrality,
                    "causal": f.causal,
                    "confidence": f.confidence,
                    "evidence": f.evidence,
                }
            )
    return pd.DataFrame(rows)

# ------------------------------
# Demo
# ------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Demo LLM-based factor pipeline")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    df_demo = pd.DataFrame(
        {
            "text": [
                "Shares rose after analyst upgrade; oil prices fell; strong post-COVID demand.",
                "Stock fell as jet fuel spiked and unions threatened strike during wage talks.",
                "Qantas gained on regulatory approval of a new partnership and upbeat guidance.",
            ],
            "y": [1, -1, 1],
        }
    )

    backend = OpenAIBackend(model=args.model, temperature=0.0)
    pipe = FactorPipeline(llm=backend)

    X, recs = pipe.transform(df_demo, text_col="text")
    print("\nExtracted features:\n", X.head())
    print("\nExtraction summary:\n", summarize_records(recs))

