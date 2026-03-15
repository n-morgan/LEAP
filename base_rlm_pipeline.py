"""
DSPy ClimatePolicy integration for RLM.

Uses RLM to recursively process long climate policy PDFs, then structures
the raw extraction output through DSPy signatures into validated
ClimatePolicy objects.

Single-document pipeline only — no batch/multi-document logic here.
"""

import json
import os
import re
from typing import List, Literal, Optional

import dspy
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

dspy.configure(lm=dspy.LM('openai/gpt-5.2'))
# ---------------------------------------------------------------------------
# Schema — identical to the notebook's ClimatePolicy
# ---------------------------------------------------------------------------


class ClimatePolicy(BaseModel):
    """Complete climate policy with extraction, validation, and classification."""

    # EXTRACTION FIELDS
    policy_commitment: str = Field(
        description="A concise, self-contained summary rewritten as a clear commitment statement"
    )
    source_quote: str = Field(
        description="Verbatim text from source document (2-3 sentences, no paraphrasing)"
    )

    # SOUNDNESS INDICATORS
    has_measurable_target: str = Field(
        description="'Yes' or 'No'. If Yes, details like '40% reduction by 2030'"
    )
    has_deadline: str = Field(
        description="'Yes' or 'No'. If Yes, details like 'by 2030'"
    )
    has_legal_mandate: str = Field(
        description="'Yes' or 'No'. If Yes, details like 'regulatory requirement', 'ordinance'"
    )
    has_geographic_scope: str = Field(
        description="'Yes' or 'No'. If Yes, details like 'citywide', 'national', 'district-level'"
    )

    # VALIDATION RESULTS
    is_actionable: bool = Field(
        description="True if policy is valid and actionable, False if vague/performative"
    )
    soundness_confidence: float = Field(
        description="Confidence in soundness assessment (0.0-1.0)"
    )
    validation_reasoning: str = Field(
        description="Why this policy is/isn't sound based on protocol criteria"
    )
    weak_language_detected: str = Field(
        description="Vague words like 'promote', 'explore', 'encourage' found in policy"
    )
    strong_language_detected: str = Field(
        description="Action words like 'mandate', 'install', 'phase-out', specific numbers"
    )

    # CLASSIFICATION FIELDS
    climate_strategy_type: str = Field(
        description="Primary category: Mitigation, Adaptation, Resource Efficiency, or Nature-Based Solutions"
    )
    additional_strategy_types: List[str] = Field(
        default_factory=list,
        description="Other applicable categories if policy serves multiple purposes",
    )
    causal_mechanism: str = Field(
        description="How this policy achieves climate impact (e.g., 'Climate hazard → ↓ exposure')"
    )
    policy_instruments: List[str] = Field(
        default_factory=list,
        description="Specific mechanisms: renewable mandates, building codes, wetland restoration, etc.",
    )
    classification_signals: List[str] = Field(
        default_factory=list,
        description="Key phrases that determined classification: 'flood defenses', 'renewables', etc.",
    )
    classification_confidence: float = Field(
        description="Confidence in classification (0.0-1.0)"
    )
    co_benefits_identified: List[str] = Field(
        default_factory=list,
        description="Secondary benefits not used for primary classification",
    )

    # METADATA
    extraction_notes: str = Field(
        default="",
        description="Any flags, edge cases, or ambiguities during extraction",
    )
    classification_notes: str = Field(
        default="",
        description="Edge cases or ambiguities during classification, if any",
    )


class DocumentLocation(BaseModel):
    """Geographic context for the policy document."""

    country: str
    state_or_province: Optional[str] = None
    city: Optional[str] = None


# ---------------------------------------------------------------------------
# DSPy Signatures — mirror the notebook but kept self-contained
# ---------------------------------------------------------------------------


class PolicyValidationSignature(dspy.Signature):
    """Evaluate whether a climate policy is VALID (actionable/measurable) or
    NON-SOUND (vague/performative).

    A VALID POLICY contains at least one of:
    1. Quantifiable targets (numbers, dates, units)
    2. Binding mechanisms (law, regulation)
    3. Spatial specificity (exact location/scope)
    4. Technological shift (transition from one state to another)

    A NON-SOUND POLICY exhibits:
    1. Process without outcome (meetings, dialogue)
    2. Aspirational language without a roadmap
    3. Lack of baseline
    4. Administrative maintenance rebranded as climate action

    Be balanced — if a policy shows STRONG soundness in at least one
    dimension, mark it VALID even if other dimensions are weaker.
    """

    policy_statement: str = dspy.InputField(desc="The climate policy statement to evaluate")
    verbatim_text: str = dspy.InputField(desc="Original verbatim text from the source document")
    has_quantifiable_target: str = dspy.InputField(desc="Whether policy includes quantifiable targets")
    has_timeline: str = dspy.InputField(desc="Whether policy includes specific timeline/deadline")
    has_binding_mechanism: str = dspy.InputField(desc="Whether policy has legal/regulatory backing")
    has_spatial_specificity: str = dspy.InputField(desc="Whether policy identifies specific location/scope")

    validation_result: Literal["VALID", "NON-SOUND"] = dspy.OutputField(
        desc="VALID (actionable/measurable) or NON-SOUND (vague/performative)"
    )
    confidence_score: float = dspy.OutputField(desc="Confidence in validation (0.0-1.0)")
    reasoning: str = dspy.OutputField(desc="Explanation referencing specific validation criteria")
    weak_signals: str = dspy.OutputField(desc="Red-flag keywords detected")
    strong_signals: str = dspy.OutputField(desc="Valid action markers detected")
    final_verdict: bool = dspy.OutputField(desc="Is this policy likely sound?")


class PolicyClassificationSignature(dspy.Signature):
    """Classify a climate policy into one or more categories.

    Categories:
    1. Mitigation — acts on the climate system (emissions, carbon sinks)
    2. Adaptation — responds to climate impacts (resilience, risk reduction)
    3. Resource Efficiency — optimises resource use (energy/water/material efficiency)
    4. Nature-Based Solutions — uses ecosystems as infrastructure

    Rules:
    - Classify by primary mechanism, not secondary co-benefits.
    - Multi-label is allowed but a primary must be identifiable.
    """

    policy_statement: str = dspy.InputField(desc="The climate policy statement to classify")
    verbatim_text: str = dspy.InputField(desc="Original verbatim text from source document")

    primary_category: str = dspy.OutputField(
        desc="Primary: Mitigation | Adaptation | Resource Efficiency | Nature-Based Solutions"
    )
    secondary_categories: str = dspy.OutputField(
        desc="Comma-separated additional categories, or 'None'"
    )
    primary_causal_pathway: str = dspy.OutputField(desc="How the policy achieves climate impact")
    key_indicators: str = dspy.OutputField(desc="Phrases that signalled this classification")
    policy_mechanisms: str = dspy.OutputField(desc="Specific mechanisms used")
    classification_reasoning: str = dspy.OutputField(desc="Step-by-step reasoning")
    confidence_score: float = dspy.OutputField(desc="Confidence (0.0-1.0)")
    edge_case_notes: str = dspy.OutputField(desc="Edge case considerations or 'None'")
    co_benefits: str = dspy.OutputField(desc="Secondary benefits, or 'None'")


# ---------------------------------------------------------------------------
# DSPy Modules
# ---------------------------------------------------------------------------


class PolicyValidator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.validate = dspy.ChainOfThought(PolicyValidationSignature)

    def forward(self, policy_data: dict):
        return self.validate(
            policy_statement=policy_data.get("policy_statement", ""),
            verbatim_text=policy_data.get("verbatim_text", ""),
            has_quantifiable_target=policy_data.get("has_quantifiable_target", "Unknown"),
            has_timeline=policy_data.get("has_timeline", "Unknown"),
            has_binding_mechanism=policy_data.get("has_binding_mechanism", "Unknown"),
            has_spatial_specificity=policy_data.get("has_spatial_specificity", "Unknown"),
        )


class PolicyClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(PolicyClassificationSignature)

    def forward(self, policy_data: dict):
        return self.classify(
            policy_statement=policy_data.get("policy_statement", ""),
            verbatim_text=policy_data.get("verbatim_text", ""),
        )


# ---------------------------------------------------------------------------
# Document ingestion (with caching)
# ---------------------------------------------------------------------------

CACHE_DIR = "./cache/"

_MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdown", ".mkd", ".txt"}


def _cache_path(file_path: str) -> str:
    basename = os.path.splitext(os.path.basename(file_path))[0]
    return os.path.join(CACHE_DIR, f"{basename}.txt")


def parse_document(file_path: str, use_cache: bool = True) -> str:
    """Read a document and return markdown text.

    - .md / .markdown / .txt files are read directly (no conversion).
    - PDFs are converted via Docling, with results cached to disk.
    """
    ext = os.path.splitext(file_path)[1].lower()

    # Markdown or plain-text — just read as-is, no conversion needed
    if ext in _MARKDOWN_EXTENSIONS:
        with open(file_path, "r") as f:
            return f.read()

    # For PDFs (and anything else Docling supports), use cache
    cache = _cache_path(file_path)
    if use_cache and os.path.exists(cache):
        with open(cache, "r") as f:
            return f.read()

    converter = DocumentConverter()
    result = converter.convert(file_path)
    md = result.document.export_to_markdown()

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache, "w") as f:
            f.write(md)
    return md


# ---------------------------------------------------------------------------
# RLM custom system prompt — instructs the root LM to extract policies
# as structured JSON that downstream DSPy modules can consume.
# ---------------------------------------------------------------------------

CLIMATE_RLM_SYSTEM_PROMPT = """You are tasked with extracting climate policies from a long policy document.


YOUR TASK:
Extract every climate policy from the document. For each policy produce a JSON object with these fields:
- "policy_statement": concise commitment summary
- "verbatim_text": 2-3 sentence verbatim quote from the document
- "has_quantifiable_target": "Yes — <detail>" or "No"
- "has_timeline": "Yes — <detail>" or "No"
- "has_binding_mechanism": "Yes — <detail>" or "No"
- "has_spatial_specificity": "Yes — <detail>" or "No"
- "extraction_rationale": why this qualifies as a policy

WHAT IS A POLICY:
A stated commitment by a governing body to achieve a defined outcome through deliberate action, resource allocation, or regulatory change. It must have at least one of: quantifiable target, binding mechanism, specific intervention, or resource allocation.

DO NOT EXTRACT: background information, current conditions, vague aspirations with no anchor, process statements.

STRATEGY:
1. Chunk the document intelligently (by section headers or by character length).
2. Collect all extracted policies into a single Python list of dicts.
3. Deduplicate if needed.
4. Store the final list as `all_policies` and call FINAL_VAR(all_policies).

OUTPUT FORMAT:
Your final answer MUST be a valid JSON list of policy objects (as described above). Use FINAL_VAR on a variable containing this JSON string.



IMPORTANT: When done, provide FINAL_VAR(all_policies) where all_policies is a JSON string of extracted policies.
"""


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def _parse_rlm_policies(raw: str) -> list[dict]:
    """Best-effort parse of the RLM's JSON output into a list of dicts."""
    # Try direct parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array in the raw string
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Last resort: find individual JSON objects
    objects = re.findall(r"\{[^{}]+\}", raw)
    results = []
    for obj in objects:
        try:
            results.append(json.loads(obj))
        except json.JSONDecodeError:
            continue
    return results


def extract_policies_rlm(
    document_markdown: str,
    expert_knowledge: str | None = None,
    backend: str = "openai",
    model_name: str = "gpt-5.2",
    sub_model_name: str = "gpt-5-mini",
    max_iterations: int = 30,
    log_dir: str = "./logs",
) -> list[dict]:
    """
    Use RLM to recursively extract raw policy dicts from a long document.

    Returns a list of dicts with keys matching the RLM prompt schema
    (policy_statement, verbatim_text, has_quantifiable_target, etc.).
    """
    logger = RLMLogger(log_dir=log_dir)

    # Build context payload for RLM
    context: dict | str
    if expert_knowledge:
        context = {
            "document": document_markdown,
            "extraction_criteria": expert_knowledge,
        }
    else:
        context = document_markdown

    rlm = RLM(
        backend=backend,
        backend_kwargs={
            "model_name": model_name,
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        other_backends=[backend],
        other_backend_kwargs=[
            {"model_name": sub_model_name, "api_key": os.getenv("OPENAI_API_KEY")},
        ],
        environment="local",
        environment_kwargs={},
        max_depth=1,
        max_iterations=max_iterations,
        custom_system_prompt=CLIMATE_RLM_SYSTEM_PROMPT,
        logger=logger,
        verbose=True,
    )

    result = rlm.completion(
        prompt=context,
        root_prompt="Extract all climate policies from the document as a JSON list.",
    )

    return _parse_rlm_policies(result.response)




def validate_and_classify(
    raw_policies: list[dict],
) -> tuple[list[ClimatePolicy], list[dict]]:
    """
    Run each raw policy dict through DSPy validation and classification,
    returning fully structured ClimatePolicy objects AND rejected policies.
    
    Returns:
        (validated_policies, rejected_policies)
    """
    validator = PolicyValidator()
    classifier = PolicyClassifier()

    validated: list[ClimatePolicy] = []
    rejected: list[dict] = []

    for policy_data in raw_policies:
        # --- Validation ---
        try:
            val = validator.forward(policy_data)
            validation_succeeded = True
        except Exception as e:
            print(f"[WARN] Validation failed for policy: {e}")
            rejected.append({
                **policy_data,
                "rejection_reason": f"Validation error: {str(e)}",
                "validation_result": "ERROR"
            })
            continue

        # Check if policy passed validation
        if val.validation_result != "VALID" or not val.final_verdict:
            rejected.append({
                **policy_data,
                "rejection_reason": val.reasoning,
                "validation_result": val.validation_result,
                "confidence_score": float(val.confidence_score),
                "weak_signals": val.weak_signals,
                "strong_signals": val.strong_signals,
            })
            continue

        # --- Classification ---
        try:
            cls = classifier.forward(policy_data)
        except Exception as e:
            print(f"[WARN] Classification failed for policy: {e}")
            rejected.append({
                **policy_data,
                "rejection_reason": f"Classification error: {str(e)}",
                "validation_result": val.validation_result,
                "passed_validation": True,
            })
            continue

        # Parse comma-separated fields
        def _split(text: str) -> list[str]:
            return [s.strip() for s in text.split(",") if s.strip() and s.strip().lower() != "none"]

        climate_policy = ClimatePolicy(
            # Extraction
            policy_commitment=policy_data.get("policy_statement", ""),
            source_quote=policy_data.get("verbatim_text", ""),
            has_measurable_target=policy_data.get("has_quantifiable_target", "Unknown"),
            has_deadline=policy_data.get("has_timeline", "Unknown"),
            has_legal_mandate=policy_data.get("has_binding_mechanism", "Unknown"),
            has_geographic_scope=policy_data.get("has_spatial_specificity", "Unknown"),
            extraction_notes=policy_data.get("extraction_rationale", ""),
            # Validation
            is_actionable=True,
            soundness_confidence=float(val.confidence_score),
            validation_reasoning=val.reasoning,
            weak_language_detected=val.weak_signals,
            strong_language_detected=val.strong_signals,
            # Classification
            climate_strategy_type=cls.primary_category,
            additional_strategy_types=_split(cls.secondary_categories),
            causal_mechanism=cls.primary_causal_pathway,
            policy_instruments=_split(cls.policy_mechanisms),
            classification_signals=_split(cls.key_indicators),
            classification_confidence=float(cls.confidence_score),
            co_benefits_identified=_split(cls.co_benefits),
            classification_notes=cls.edge_case_notes,
        )
        validated.append(climate_policy)

    return validated, rejected



def process_document(
    pdf_path: str,
    country: str,
    state_or_province: str | None = None,
    city: str | None = None,
    expert_knowledge_path: str | None = None,
    output_path: str | None = None,
    backend: str = "openai",
    model_name: str = "gpt-5.2",
    sub_model_name: str = "gpt-5-mini",
    max_iterations: int = 30,
) -> list[ClimatePolicy]:
    """
    End-to-end single-document pipeline:
      PDF → markdown → RLM extraction → DSPy validation & classification → ClimatePolicy list.
    """
    location = DocumentLocation(
        country=country, state_or_province=state_or_province, city=city
    )
    location_label = city or state_or_province or country

    print(f"{'=' * 60}")
    print(f"Processing: {pdf_path}")
    print(f"Location:   {location_label}, {country}")
    print(f"{'=' * 60}")

    # 1. PDF → Markdown
    print("\n[1/4] Converting PDF to markdown...")
    document_md = parse_document(pdf_path)
    print(f"  {len(document_md):,} characters")

    # 2. Optional expert knowledge
    expert_knowledge = None
    if expert_knowledge_path and os.path.exists(expert_knowledge_path):
        print("[1.5/4] Loading expert extraction criteria...")
        expert_knowledge = parse_document(expert_knowledge_path)

    # 3. RLM extraction
    print("\n[2/4] Running RLM extraction...")
    raw_policies = extract_policies_rlm(
        document_markdown=document_md,
        expert_knowledge=expert_knowledge,
        backend=backend,
        model_name=model_name,
        sub_model_name=sub_model_name,
        max_iterations=max_iterations,
    )
    print(f"  {len(raw_policies)} raw policies extracted")

    # 4. DSPy validation + classification
    print("\n[3/4] Validating and classifying with DSPy...")
    validated_policies, rejected_policies = validate_and_classify(raw_policies)
    print(f"  {len(validated_policies)} policies passed validation")
    print(f"  {len(rejected_policies)} policies rejected")

    # 5. Save output
    if output_path:
        print(f"\n[4/4] Saving to {output_path}")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Save validated policies
        with open(output_path, "w") as f:
            json.dump([p.model_dump() for p in validated_policies], f, indent=2)
        
        # Save rejected policies to separate file
        rejected_path = output_path.replace(".json", "_rejected.json")
        with open(rejected_path, "w") as f:
            json.dump(rejected_policies, f, indent=2)
        
        print(f"  Validated policies saved to: {output_path}")
        print(f"  Rejected policies saved to: {rejected_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"  Total extracted:    {len(raw_policies)}")
    print(f"  Passed validation:  {len(validated_policies)}")
    print(f"  Rejected:           {len(rejected_policies)}")
    if validated_policies:
        cats = {}
        for p in validated_policies:
            cats[p.climate_strategy_type] = cats.get(p.climate_strategy_type, 0) + 1
        print("  Category distribution:")
        for cat, count in sorted(cats.items(), key=lambda x: x[1], reverse=True):
            print(f"    {cat}: {count}")
    print(f"{'=' * 60}\n")

    return validated_policies


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__": 


    policies = process_document(
        pdf_path="./chicago.md",
        country="United States",
        state_or_province="Illinois",
        city="Chicago",
        model_name="gpt-5.2",
        sub_model_name="gpt-5-mini",
        expert_knowledge_path="./RLM_proc_instr.pdf",
        output_path="./output/seattle_policies.json",
    )
    # import argparse

    # parser = argparse.ArgumentParser(
    #     description="Extract climate policies from a PDF using RLM + DSPy"
    # )
    # parser.add_argument("pdf", help="Path to the climate policy PDF")
    # parser.add_argument("--country", required=True)
    # parser.add_argument("--state", default=None)
    # parser.add_argument("--city", default=None)
    # parser.add_argument("--expert-knowledge", default=None, help="Path to expert criteria PDF")
    # parser.add_argument("--output", "-o", default=None, help="Output JSON path")
    # parser.add_argument("--backend", default="openai")
    # parser.add_argument("--model", default="gpt-5.2")
    # parser.add_argument("--sub-model", default="gpt-5-mini")
    # parser.add_argument("--max-iterations", type=int, default=30)
    # parser.add_argument("--dspy-model", default="openai/gpt-4o-mini", help="DSPy LM model string")
    # args = parser.parse_args()

    # # Configure DSPy LM
    # dspy.configure(lm=dspy.LM(model=args.dspy_model, api_key=os.getenv("OPENAI_API_KEY")))

    # policies = process_document(
    #     pdf_path=args.pdf,
    #     country=args.country,
    #     state_or_province=args.state,
    #     city=args.city,
    #     expert_knowledge_path=args.expert_knowledge,
    #     output_path=args.output,
    #     backend=args.backend,
    #     model_name=args.model,
    #     sub_model_name=args.sub_model,
    #     max_iterations=args.max_iterations,
    # )

    print(f"Extracted {len(policies)} validated policies.")


