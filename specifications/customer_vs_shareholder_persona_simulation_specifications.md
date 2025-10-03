# LLM Persona Simulation Design: Customer vs Shareholder Analysis

## Executive Summary

This document provides detailed specifications for LLM-based persona simulation systems designed to predict stakeholder reactions to reputation events. The system uses GPT-4o with structured prompts, n-shot learning examples, and sophisticated prompt engineering to simulate authentic stakeholder decision-making patterns. The design emphasizes behavioral realism through carefully crafted constraints, trigger conditions, and action-selection heuristics.

## Table of Contents

1. [Core Architecture](#core-architecture)
2. [Customer Persona Design](#customer-persona-design)
3. [Shareholder Persona Design](#shareholder-persona-design)
4. [N-Shot Learning Methodology](#n-shot-learning-methodology)
5. [Prompt Engineering Framework](#prompt-engineering-framework)
6. [RAGLite Algorithm Integration](#raglite-algorithm-integration)
7. [Behavioral Calibration & Bias Correction](#behavioral-calibration--bias-correction)
8. [Industry Superannuation Fund Customization](#industry-superannuation-fund-customization)
9. [Implementation Guidelines](#implementation-guidelines)

---

## Core Architecture

### System Overview

The persona simulation system leverages a multi-layered architecture combining:

- **Structured Prompting**: Domain-specific system prompts with embedded behavioral heuristics
- **N-Shot Learning**: Curated examples demonstrating authentic stakeholder decision patterns
- **Constraint Enforcement**: Rule-based validation preventing unrealistic reactions
- **Event Classification**: Pre-processing to identify event materiality and stakeholder relevance
- **Response Validation**: Post-processing to ensure outputs align with stakeholder capabilities

### Technical Stack

```python
# Core components from stakeholder_reactions.py:
class StakeholderReaction(BaseModel):
    reputation_score: conint(ge=-5, le=5)     # Delta impact score
    reaction_label: ReactionType              # Enumerated action
    reaction_probability: confloat(ge=0.0, le=1.0)  # Likelihood estimate
    rationale: str = Field(max_length=200)    # Concise reasoning
```

### Event Classification Pipeline

Events undergo deterministic classification before persona analysis:

```python
class EventClassification(BaseModel):
    is_sector_wide_exogenous: bool           # Affects all airlines equally
    is_material_to_qantas: bool              # Direct operational/financial impact
    is_safety_event: bool                    # Qantas-attributable safety incident
    is_customer_service_event: bool          # Service delivery impact
    is_corporate_governance_event: bool      # Governance/executive issues
    is_labour_event: bool                    # Industrial relations
```

---

## Customer Persona Design

### Behavioral Model Foundation

The customer persona simulates Australian airline passengers using pre-2020 behavioral patterns. The design incorporates:

**Core Behavioral Heuristics:**
- Price sensitivity thresholds (20-30% leisure, 15% SME)
- Reliability memory (2-3 poor experiences trigger trial switching)
- Status gravity (Gold/Platinum retention logic)
- Brand safety signals (safety accolades drive defensive loyalty)

**Decision Architecture:**
```
Event → Persona Context → Apply Heuristics → Select Action → Validate Constraints
```

### Stakeholder-Specific Constraints

**Allowed Actions:**
- Do Nothing
- Share Opinion with Others
- Switch to Competitor
- Legal/Regulatory Action

**Trigger Conditions:**
- Switch only with clear service impact or value gap (≥20-30% price differential, ≥2-3 disruptions)
- Legal/Regulatory only after airline complaint timeframes (ACA after ~20 business days)
- Prefer neutral actions for internal disputes without service impact

**Disallowed Contexts:**
- Switch for sector-wide exogenous disruptions
- Switch for internal labour disputes unless flights disrupted
- Legal action without exhausting airline complaint processes

### Knowledge Base (Pre-2020)

The customer persona incorporates fixed historical knowledge:

```markdown
• Complaint escalation: Airline Customer Advocate (ACA) handles unresolved complaints
• Reliability tracking: BITRE publishes monthly OTP data
• FF program changes: 2019 "biggest overhaul" with more reward seats
• Pride moments: Perth-London non-stop (2018), Wi-Fi rollout (2017)
• Social backlash triggers: Fleet grounding (2011), #QantasLuxury backfire
```

---

## Shareholder Persona Design

### HESTA Case Study Architecture

The HESTA superannuation fund persona demonstrates sophisticated institutional investor behavior:

**Core Philosophy:**
- Fiduciary-first decision making
- ESG integration with financial materiality
- Active ownership escalation ladder
- Long-term value optimization

**Decision Framework:**
```
Event → ESG/Financial Materiality Assessment → Policy Precedent Check → Escalation Level → Action Selection
```

### Escalation Ladder

1. **Quiet Engagement** → Private dialogue with management
2. **Public Engagement** → Coalition building, public statements
3. **Voting Opposition** → Vote against directors/remuneration
4. **Shareholder Activism** → File/co-file resolutions
5. **Divestment** → Exit position when engagement fails

### Stakeholder-Specific Rules

**Allowed Actions:**
- Do Nothing
- Share Opinion with Others
- Shareholder Activism
- Devalue Shares

**Trigger Conditions:**
- Shareholder Activism only for company-specific governance/culture/safety concerns with financial materiality
- Devalue Shares only for material, persistent earnings impairment from Qantas decisions
- Neutral actions for macro/sector shocks without Qantas-specific mismanagement

**Policy Integration:**
```markdown
• Tobacco exclusion (2013): Portfolio-wide prohibition
• Human rights escalation: Divestment precedent (Transfield 2015)
• Gender diversity: Vote against all-male boards
• Climate screening: New thermal coal asset exclusions
```

---

## N-Shot Learning Methodology

### Customer N-Shot Design Principles

The customer persona uses 10 carefully crafted examples demonstrating behavioral anchors:

**Example Categories:**
1. **Positive Brand Moments** (Perth-London milestone flight)
2. **Loyalty Economics** (Gold traveller fare premium tolerance)
3. **Price-Driven Switching** (Family 30% savings threshold)
4. **Social Media Triggers** (Tone-deaf campaigns)
5. **Reliability Frustration** (3 delays in 6 weeks)
6. **Status Competitions** (Competitor fast-track offers)
7. **Regulatory Escalation** (ACA complaint process)
8. **Loyalty Program Sentiment** (FF program improvements)
9. **Service Recovery** (Proactive rebooking defense)
10. **Operational Complaints** (Baggage and points issues)

### Bias Correction Through "No Action" Examples

**Problem Identified:** Initial testing revealed unrealistic negative bias - the LLM over-predicted customer reactions when historical data showed most airline customers don't react unless directly affected.

**Solution Implemented:** Additional "Do Nothing" examples were strategically added to demonstrate:

- Competitor accolades don't trigger switching without Qantas service degradation
- Industry-wide issues don't motivate individual airline switches
- Minor operational incidents don't generate social media storms
- Awards and rankings typically generate neutral responses

**Example Addition:**
```markdown
Shot 11 — Competitor recognition doesn't trigger action
User: "Virgin Australia wins 'Best Domestic Airline' award - do customers react?"
Customers:
Persona & context: Regular Qantas travellers reading industry news
Decision: Do Nothing
Why: Competitor accolades don't affect personal Qantas experience; no service impact
Fallback: Only react if concurrent Qantas service degradation evident
```

### Shareholder N-Shot Design Principles

HESTA examples demonstrate institutional sophistication:

**Example Categories:**
1. **Board Diversity** (All-male board voting opposition)
2. **Executive Remuneration** (Pay-performance misalignment)
3. **Human Rights** (Contractor abuse divestment precedent)
4. **Exclusion Policy** (Tobacco exposure management)
5. **Climate Policy** (New thermal coal opposition)
6. **Safety Governance** (Culture and oversight failures)

**Materiality Focus:** Each example explicitly connects ESG issues to financial risk/return implications.

---

## Prompt Engineering Framework

### System Prompt Structure

**Customer Persona Template:**
```
SYSTEM PROMPT — "Simulate Australian Qantas Customers (Pre-2020)"
├── Identity & Scope Definition
├── Fixed Knowledge Base (Pre-2020 facts with citations)
├── Behavioral Heuristics & Thresholds
├── Response Format Instructions
├── N-Shot Examples (10 behavioral anchors)
├── Allowed Actions (enumerated list)
├── Trigger Conditions (positive logic rules)
└── Disallowed Contexts (negative constraints)
```

**Shareholder Persona Template:**
```
SYSTEM PROMPT — "Simulate HESTA (pre-2020)"
├── Identity & Fiduciary Purpose
├── Core Beliefs & Policies (with direct quotes)
├── Decision Heuristics (escalation framework)
├── Output Style Requirements
├── N-Shot Examples (institutional precedents)
├── Response Template (structured format)
├── Allowed Actions (investor-specific)
├── Trigger Conditions (materiality requirements)
└── Disallowed Contexts (scope limitations)
```

### User Prompt Engineering

**Event Context Injection:**
- Event classification flags (deterministic inputs)
- Materiality guidance (heuristic hints)
- Stakeholder-specific guardrails
- Constraint violation warnings

**Validation Loop Integration:**
```python
def score_event_for_stakeholder(client, stakeholder_name, profile_prompt, ev, ev_flags):
    # Multi-layer validation:
    # 1. Allowed action subset enforcement
    # 2. Materiality constraint checking
    # 3. Disallowed context rule application
    # 4. Stakeholder-specific guardrails
    # 5. Heuristic disagreement flagging
```

### LLM Instruction Optimization

**Key Instruction Elements:**
1. **Example Usage Clarity**: "Use these examples as behavioral anchors for decision patterns, not literal templates"
2. **Overthinking Prevention**: "Apply the heuristics directly without extensive elaboration beyond the examples"
3. **Consistency Enforcement**: "Maintain alignment with n-shot decision logic rather than generating novel reasoning"

**Anti-Drift Mechanisms:**
```markdown
• Explicit probability anchoring: "reaction_probability between 0 and 1"
• Scope limiting: "Using only information available before 31 Dec 2019"
• Format enforcement: "Output JSON only"
• Validation warnings: "Ensure reputation_score is integer in [-5,5]"
```

---

## RAGLite Algorithm Integration

### Knowledge Retrieval Architecture

The system implements a lightweight retrieval-augmented generation approach:

**Static Knowledge Bases:**
- Historical event precedents (2011-2019)
- Regulatory framework facts (ACA, BITRE, ACSI)
- Stakeholder policy documents (HESTA exclusions, voting guidelines)

**Dynamic Context Injection:**
```python
def build_event_summary(ev: Dict) -> str:
    return f"""
    Event: {ev.get('event_name')}
    Date: {ev.get('event_date')}
    Primary entity: {ev.get('primary_entity')}
    Categories: {ev.get('event_categories')}
    Stakeholders mentioned: {ev.get('stakeholders')}
    Response strategies: {ev.get('response_strategies')}
    Damage score: {ev.get('mean_damage_score')}
    Response score: {ev.get('mean_response_score')}
    Article count: {ev.get('num_articles')}
    """
```

**Materiality Heuristics:**
```python
def compute_materiality_hint(ev: Dict) -> Tuple[bool, str]:
    dmg = float(ev.get('mean_damage_score', 0))
    arts = int(ev.get('num_articles', 0))

    # Low materiality signals
    likely_immaterial = (dmg <= 2.0 and arts < 15)

    # High materiality overrides
    if 'safety' in categories and 'incident' in categories:
        likely_immaterial = False

    return likely_immaterial, f"dmg={dmg}, articles={arts}"
```

### Information Architecture Benefits

**Stakeholder-Specific Knowledge:**
- Customers: Historical complaint patterns, loyalty program changes, service precedents
- Shareholders: ESG policy evolution, voting precedents, engagement outcomes

**Temporal Consistency:**
- Knowledge cutoff enforcement (pre-2020)
- Historical fact verification
- Precedent-based reasoning chains

---

## Behavioral Calibration & Bias Correction

### Identified Bias Pattern

**Original Problem:** GPT-4o exhibited systematic negative bias, predicting customer reactions to events that historically generated minimal response.

**Root Cause Analysis:**
1. Training data bias toward newsworthy (negative) events
2. Lack of "normal" non-reaction examples in training
3. Overestimation of stakeholder engagement levels
4. Insufficient understanding of customer attention thresholds

### Correction Methodology

**Approach 1: Negative Example Augmentation**
Added explicit "Do Nothing" examples for:
- Competitor achievements without Qantas impact
- Industry-wide regulatory changes
- Minor operational incidents
- Routine business announcements

**Approach 2: Probability Anchoring**
```python
# Materiality-based probability constraints
if not ev_flags.is_material_to_qantas:
    if label_text not in ["Do Nothing", "Share Opinion with Others"]:
        raise ValueError("Immaterial event requires neutral action")
    if reputation_score not in (-1, 0, 1):
        raise ValueError("Immaterial event requires small neutral delta")
```

**Approach 3: Baseline Calibration**
Established realistic reaction thresholds:
- 70-80% "Do Nothing" for routine industry news
- 15-20% "Share Opinion" for moderate relevance
- 5-10% active responses for high-impact events only

### Validation Framework

**Historical Benchmarking:**
- 2011 fleet grounding: High reaction rates validated
- 2018 Perth-London launch: Positive sentiment confirmed
- Routine OTP reports: Minimal reaction verified

**Cross-Stakeholder Consistency:**
- Customers: Action-oriented responses
- Shareholders: Governance-focused activism
- Employees: Industrial relations sensitivity

---

## Industry Superannuation Fund Customization

### Generic Framework Adaptation

**Base Template Modifications for Industry Super Funds:**

1. **Member Demographics Integration**
   ```markdown
   • Member base: Healthcare and community services workers
   • Sector exposure: Healthcare infrastructure, aged care, community services
   • Values alignment: Social impact, healthcare outcomes, worker welfare
   ```

2. **Investment Philosophy Customization**
   ```markdown
   • Active ownership approach: Engagement before divestment
   • ESG integration: Material risk assessment framework
   • Long-term focus: 20-30 year member retirement horizons
   • Coalition participation: ACSI, IGCC, PRI membership
   ```

3. **Sector-Specific Considerations**
   ```markdown
   • Healthcare worker interests: Occupational health and safety standards
   • Community service values: Social licence to operate emphasis
   • Public sector alignment: Government policy coordination
   ```

### Australian Industry Super Fund with Shareholder Activism History

**Enhanced Persona Development:**

**Step 1: Historical Precedent Integration**
```markdown
Activism History:
• Executive remuneration campaigns (2015-2019)
• Climate resolutions co-filing (2017-2018)
• Board diversity initiatives (2016-ongoing)
• Fossil fuel divestment advocacy (2018-2019)
• Tax transparency campaigns (2017-2018)
```

**Step 2: Marketing Materials Analysis**
```markdown
Ethical Investment Positioning:
• "Investing for Impact" framework
• ESG-integrated investment across all options
• Climate change as material financial risk
• Human rights due diligence processes
• Active ownership as fiduciary duty
```

**Step 3: Decision Framework Customization**
```python
def enhanced_superannuation_heuristics():
    return {
        'engagement_threshold': 'Material ESG risk OR member interest alignment',
        'activism_triggers': [
            'Executive pay misalignment > 2 years',
            'Climate strategy gaps vs Paris goals',
            'Board diversity below industry median',
            'Safety culture failures affecting workers'
        ],
        'divestment_criteria': [
            'Engagement failure after 18-24 months',
            'Fundamental business model misalignment',
            'Excluded sector exposure (tobacco, weapons)'
        ]
    }
```

**Step 4: N-Shot Example Enhancement**
```markdown
Shot 1 — Climate disclosure advocacy
Context: Company lacks Paris-aligned transition plan despite material exposure
Materiality: Transition risk threatens long-term portfolio returns
Precedent: Previous climate resolutions filed 2017-2018 per marketing materials
Action: Co-file climate transition resolution; engage with company and peers
Rationale: Consistent with "Investing for Impact" framework and fiduciary duty

Shot 2 — Executive remuneration challenge
Context: CEO pay increased 40% while TSR negative over 3-year period
Materiality: Pay-performance misalignment indicates governance failures
Precedent: Historical remuneration campaigns per ethical investment positioning
Action: Vote AGAINST remuneration report; issue public statement; engage other institutions
Rationale: Aligns with member interests and sustainable long-term value creation
```

### Implementation Customization Guide

**Profile Development Process:**
1. **Research Phase**: Analyze 3-5 years of voting records, public statements, and policy documents
2. **Precedent Mapping**: Identify 8-10 historical activism examples across different issue categories
3. **Constraint Definition**: Establish materiality thresholds and engagement timelines
4. **N-Shot Creation**: Develop 6-8 examples demonstrating decision patterns
5. **Validation Testing**: Test against known historical decisions for consistency

**Key Differentiation Factors:**
- **Member demographics**: Professional vs industry vs public sector
- **Asset size**: Large funds have more activism capacity
- **Governance structure**: Board composition affects risk tolerance
- **Investment philosophy**: Active vs passive ownership approaches

---

## Implementation Guidelines

### Technical Implementation

**Core Infrastructure:**
```python
# stakeholder_reactions.py integration points
class EnhancedStakeholderSimulation:
    def __init__(self, profile_path: str, constraints_config: Dict):
        self.profile = load_stakeholder_profile(profile_path)
        self.constraints = extract_constraints(self.profile)
        self.n_shot_examples = parse_n_shot_examples(self.profile)

    def simulate_reaction(self, event: Dict, flags: EventClassification):
        # Apply bias correction
        materiality_hint = compute_materiality_hint(event)

        # Generate response with validation
        response = score_event_for_stakeholder(
            client=self.client,
            stakeholder_name=self.name,
            profile_prompt=self.profile,
            ev=event,
            ev_flags=flags
        )

        # Validate against constraints
        return self.validate_response(response, flags)
```

**Quality Assurance Framework:**
1. **Constraint Validation**: Automated checking of trigger conditions and disallowed contexts
2. **Historical Benchmarking**: Comparison against known stakeholder responses to historical events
3. **Cross-Stakeholder Consistency**: Ensuring reaction patterns align with stakeholder capabilities
4. **Bias Monitoring**: Tracking reaction distributions for systematic biases

### Deployment Considerations

**Scalability Requirements:**
- Concurrent processing for multiple stakeholders
- Event classification caching for efficiency
- Response validation with retry logic
- Error handling and graceful degradation

**Monitoring and Evaluation:**
```python
def monitor_simulation_quality():
    metrics = {
        'reaction_distribution': track_action_frequencies(),
        'materiality_alignment': validate_materiality_responses(),
        'constraint_violations': count_rule_violations(),
        'historical_consistency': benchmark_known_responses()
    }
    return metrics
```

**Continuous Improvement:**
- Regular n-shot example updates based on new stakeholder precedents
- Constraint refinement based on validation failures
- Bias detection and correction through response analysis
- Performance optimization through caching and parallel processing

---

## Conclusion

This LLM persona simulation framework demonstrates sophisticated stakeholder modeling through:

1. **Behavioral Realism**: N-shot examples and constraint systems that capture authentic decision patterns
2. **Bias Correction**: Systematic addressing of LLM over-reaction tendencies through negative examples
3. **Stakeholder Specificity**: Customized prompt engineering for different stakeholder types and individual organizations
4. **Technical Robustness**: Validation frameworks and quality assurance processes ensuring reliable outputs

The customer vs shareholder analysis reveals fundamentally different decision architectures - customers focus on service value propositions while institutional investors emphasize governance and long-term financial materiality. The framework successfully captures these distinctions through stakeholder-specific constraints, action sets, and n-shot examples.

The system's ability to model complex institutional investors like industry superannuation funds with shareholder activism histories demonstrates the framework's flexibility and sophistication. By incorporating historical precedents, policy documents, and marketing positioning, the system can generate realistic simulations of nuanced institutional behavior.

Future enhancements should focus on dynamic learning from new stakeholder precedents, expanded constraint sophistication, and improved bias detection mechanisms to maintain simulation quality over time.