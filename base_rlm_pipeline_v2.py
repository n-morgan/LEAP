"""
base_rlm_pipeline_v2.py

Consolidated LEAP pipeline. Base RLM handles extraction AND classification in a
single recursive pass. DSPy PolicyValidator is kept as a separate step so output
quality can be measured independently — consistent with the comparison against
Ziyad's GENIUS structured DSPy pipeline.

Architecture:
    PDF/MD  →  markdown
            →  RLM (extract + classify in one call)
            →  DSPy PolicyValidator (separate, quality measurement only)
            →  JSON output  +  RLMLogger trace file (sibling to output)

Output per document:
    {stem}_policies.json        — validated ClimatePolicy records
    {stem}_rejected.json        — rejected records with validator reasoning
    {stem}_trace.json           — RLMLogger thought tree from the extraction pass
"""

import glob
import json
import os
import re
import tempfile
from typing import List, Literal, Optional

import dspy
import pandas as pd
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

dspy.configure(lm=dspy.LM("openai/gpt-5", cache=False))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class ClimatePolicy(BaseModel):
    """
    Lean climate policy record.

    Extraction and classification are resolved in a single RLM pass.
    Validation fields (is_actionable, validation_confidence, validation_reasoning)
    are populated afterwards by the DSPy PolicyValidator and are None until then.
    """

    # -- Hierarchy --
    role: Literal["individual", "sub", "parent"] = Field(
        description=(
            "'individual': standalone policy, no parent-child relationship. "
            "'sub': specific action listed under a named parent initiative "
            "(e.g. lettered/numbered list items). "
            "'parent': umbrella initiative that has sub-policies listed beneath it."
        )
    )
    parent_statement: Optional[str] = Field(
        default=None,
        description=(
            "For 'sub' role only: the name or summary of the parent initiative "
            "exactly as written in the document. Null for 'individual' and 'parent'."
        ),
    )

    # -- Core extraction --
    policy_statement: str = Field(
        description=(
            "Concise, self-contained summary of the policy commitment. "
            "Must be directly supported by source_quote."
        )
    )
    source_quote: str = Field(
        description=(
            "Verbatim text from the source document (2-3 sentences). "
            "No paraphrasing. Used for grounding verification."
        )
    )
    section_header: str = Field(
        description=(
            "Section or subsection heading under which this policy appears, "
            "copied verbatim from the document."
        )
    )
    extraction_rationale: str = Field(
        description=(
            "Why this qualifies as a policy. "
            "Note any vagueness, edge cases, or ambiguities."
        )
    )

    # -- Classification (resolved within the RLM call) --
    primary_category: Literal[
        "Mitigation", "Adaptation", "Resource Efficiency", "Nature-Based Solutions"
    ] = Field(
        description=(
            "Primary climate strategy category determined by dominant causal mechanism. "
            "Mitigation: acts on the climate system (GHG emissions, carbon sinks). "
            "Adaptation: responds to climate impacts (resilience, risk reduction). "
            "Resource Efficiency: optimises resource use (energy, water, materials). "
            "Nature-Based Solutions: uses ecosystems as climate infrastructure."
        )
    )

    # -- Validation (filled post-hoc by DSPy PolicyValidator) --
    is_actionable: Optional[bool] = Field(
        default=None,
        description=(
            "True if the policy passed soundness validation. "
            "Populated by DSPy PolicyValidator after extraction."
        ),
    )
    validation_confidence: Optional[float] = Field(
        default=None,
        description="Validator confidence score (0.0-1.0).",
    )
    validation_reasoning: Optional[str] = Field(
        default=None,
        description="Step-by-step validator reasoning. Populated by DSPy PolicyValidator.",
    )


class DocumentMetadata(BaseModel):
    """Geographic context for the policy document."""

    country: str
    state_or_province: Optional[str] = None
    city: Optional[str] = None


# ---------------------------------------------------------------------------
# RLM system prompt — extraction + classification in one call
# ---------------------------------------------------------------------------

CLIMATE_RLM_SYSTEM_PROMPT = """You are a climate policy analyst. Your task is to extract and classify every climate policy from a long policy document.

For each policy, produce a JSON object with these fields:

EXTRACTION FIELDS:
- "role": "individual" | "sub" | "parent"
    • "individual": standalone policy, no parent-child relationship
    • "sub": specific action listed under a named parent initiative (e.g. lettered or numbered list items A., B., C.)
    • "parent": umbrella initiative/program that has sub-actions listed beneath it
- "parent_statement": the name or summary of the parent initiative for "sub" role only. Use null for "individual" and "parent".
- "policy_statement": concise, self-contained commitment summary. Must be supported by source_quote.
- "source_quote": verbatim 2-3 sentence quote from the document. No paraphrasing.
- "section_header": the section or subsection heading this policy appears under, copied from the document.
- "extraction_rationale": why this qualifies as a policy; note any vagueness or edge cases.

CLASSIFICATION FIELD — assign alongside extraction:
- "primary_category": one of "Mitigation" | "Adaptation" | "Resource Efficiency" | "Nature-Based Solutions"
    • Mitigation: acts on the climate system (GHG emissions reductions, carbon sinks, renewable energy)
    • Adaptation: responds to climate impacts (resilience, risk reduction, hazard management, cooling)
    • Resource Efficiency: optimises resource use (energy efficiency, water conservation, material use, transport)
    • Nature-Based Solutions: uses ecosystems as climate infrastructure (urban greening, wetlands, reforestation)
    Classify by the primary causal mechanism, not secondary co-benefits.

WHAT COUNTS AS A POLICY:
A stated commitment by a governing body to achieve a defined outcome through deliberate action, resource allocation, or regulatory change. Must contain at least one of:
1. Quantifiable target (numbers, units, deadlines)
2. Binding mechanism (ordinance, mandate, regulatory code change)
3. Specific named intervention (program, technology, action)
4. Resource allocation (committed funding)

DO NOT EXTRACT: background information, current conditions, vague aspirations with no concrete anchor, process descriptions without commitments.

HIERARCHY RULES:
- When you see a named initiative followed by lettered or numbered sub-items, extract the initiative header as "parent" AND each sub-item as "sub" with parent_statement pointing to that initiative name.
- If a policy stands alone with no sub-items beneath it, it is "individual".
- When in doubt between "parent" and "individual", prefer "individual" unless there are clear sub-items.


OUTPUT FORMAT:
Your final answer MUST be a valid JSON list of policy objects.
"""


# ---------------------------------------------------------------------------
# DSPy Validator — kept separate for output quality measurement
# ---------------------------------------------------------------------------


class PolicyValidationSignature(dspy.Signature):
    """
    Evaluate whether a climate policy is VALID (actionable/measurable) or
    NON-SOUND (vague/performative).

    VALID requires at least one of:
    1. Quantifiable target (numbers, units, deadlines)
    2. Binding mechanism (law, ordinance, regulation)
    3. Specific named intervention with clear deliverable
    4. Committed resource allocation with measurable outcome

    NON-SOUND when:
    1. Process without outcome (meetings, dialogue, studies)
    2. Aspirational language with no roadmap or anchor
    3. Administrative maintenance rebranded as climate action
    4. Vague verbs (promote, explore, encourage) with no hard target

    Be balanced: mark VALID if there is strong soundness in at least one dimension.
    """

    policy_statement: str = dspy.InputField(desc="Concise policy commitment to evaluate")
    source_quote: str = dspy.InputField(desc="Verbatim source text for grounding")
    extraction_rationale: str = dspy.InputField(desc="Extractor's initial reasoning")

    validation_result: Literal["VALID", "NON-SOUND"] = dspy.OutputField(
        desc="VALID or NON-SOUND"
    )
    confidence_score: float = dspy.OutputField(desc="Confidence in the verdict (0.0-1.0)")
    reasoning: str = dspy.OutputField(
        desc="Step-by-step justification citing specific evidence from source_quote"
    )
    final_verdict: bool = dspy.OutputField(
        desc="True only if validation_result is VALID and confidence_score >= 0.7"
    )


class PolicyValidator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.validate = dspy.ChainOfThought(PolicyValidationSignature)

    def forward(self, policy_data: dict):
        return self.validate(
            policy_statement=policy_data.get("policy_statement", ""),
            source_quote=policy_data.get("source_quote", ""),
            extraction_rationale=policy_data.get("extraction_rationale", ""),
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
    """
    Return document content as a markdown string.
    Markdown/text files are read directly. PDFs go through Docling (result cached).
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in _MARKDOWN_EXTENSIONS:
        with open(file_path, "r") as f:
            return f.read()

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
# RLM extraction + classification
# ---------------------------------------------------------------------------


def _parse_rlm_output(raw) -> list[dict]:
    """
    Best-effort parse of the RLM's final JSON output into a list of dicts.
    Handles direct JSON, markdown code fences, Python literals, and
    individual object fragments as a last resort.
    """
    import ast

    # RLM may resolve FINAL_VAR to the actual Python object
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for v in raw.values():
            if isinstance(v, list):
                return v
        return [raw]

    stripped = re.sub(r"```(?:json)?\s*", "", raw).strip()

    for candidate in (stripped, raw):
        # Direct JSON parse
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        return v
                return [parsed]
        except json.JSONDecodeError:
            pass

        # Python literal (handles str(list) output from SUBMIT)
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        # JSON array anywhere in the string
        match = re.search(r"\[.*\]", candidate, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

    # Last resort: individual JSON objects
    results = []
    for obj in re.findall(r"\{[^{}]+\}", raw):
        try:
            results.append(json.loads(obj))
        except json.JSONDecodeError:
            continue
    return results


def _collect_trace(log_dir: str) -> dict:
    """
    Read all JSON files written by RLMLogger into log_dir and return
    them as a single dict keyed by filename stem.
    Falls back to raw text for non-JSON files.
    """
    trace: dict = {}
    if not os.path.isdir(log_dir):
        return trace

    for path in sorted(glob.glob(os.path.join(log_dir, "**", "*"), recursive=True)):
        if not os.path.isfile(path):
            continue
        rel = os.path.relpath(path, log_dir)
        try:
            with open(path, "r") as f:
                content = f.read()
            try:
                trace[rel] = json.loads(content)
            except json.JSONDecodeError:
                trace[rel] = content
        except Exception:
            trace[rel] = f"<unreadable: {path}>"

    return trace


def extract_and_classify_rlm(
    document_markdown: str,
    expert_knowledge: str | None = None,
    trace_output_path: str | None = None,
    backend: str = "openai",
    model_name: str = "gpt-5",
    sub_model_name: str = "gpt-5-nano",
    max_iterations: int = 50,
) -> list[dict]:
    """
    Use base RLM to extract and classify policies from a document in a single pass.

    The RLM system prompt instructs the model to produce a JSON list where each
    object includes both extraction fields AND primary_category.

    If trace_output_path is provided, the RLMLogger thought tree is collected and
    saved to that path after the run.

    Returns a list of raw policy dicts.
    """
    # Build context payload
    if expert_knowledge:
        context = f"DOCUMENT:\n{document_markdown}\n\nEXTRACTION CRITERIA:\n{expert_knowledge}"
    else:
        context = document_markdown

    # Use a dedicated temp dir for this run's logs so we can collect them cleanly
    log_dir = tempfile.mkdtemp(prefix="rlm_trace_")
    logger = RLMLogger(log_dir=log_dir)

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
        root_prompt="Extract and classify all climate policies from the document as a JSON list.",
    )

    # Collect and save trace alongside output
    if trace_output_path:
        trace = _collect_trace(log_dir)
        trace["_rlm_response"] = result.response
        os.makedirs(os.path.dirname(trace_output_path) or ".", exist_ok=True)
        with open(trace_output_path, "w") as f:
            json.dump(trace, f, indent=2)
        print(f"  Trace saved: {trace_output_path}")

    return _parse_rlm_output(result.response)


# ---------------------------------------------------------------------------
# DSPy validation pass
# ---------------------------------------------------------------------------


def validate_policies(
    raw_policies: list[dict],
) -> tuple[list[ClimatePolicy], list[dict]]:
    """
    Run each raw policy dict through DSPy PolicyValidator.

    Returns (validated, rejected).
    Validated records are ClimatePolicy objects with is_actionable/validation_*
    fields populated. Rejected records are plain dicts with a rejection_reason key.
    """
    validator = PolicyValidator()
    validated: list[ClimatePolicy] = []
    rejected: list[dict] = []

    for raw in raw_policies:
        try:
            val = validator.forward(raw)
        except Exception as e:
            print(f"  [WARN] Validator error: {e}")
            rejected.append({**raw, "rejection_reason": f"Validator error: {e}"})
            continue

        # Normalise role/parent_statement consistency
        role = raw.get("role", "individual")
        parent_stmt = raw.get("parent_statement") or None
        if parent_stmt and role == "individual":
            role = "sub"

        # Build ClimatePolicy regardless of validation result so we capture
        # the confidence/reasoning for every record (useful for quality analysis)
        try:
            policy = ClimatePolicy(
                role=role,
                parent_statement=parent_stmt,
                policy_statement=raw.get("policy_statement", ""),
                source_quote=raw.get("source_quote", raw.get("verbatim_text", "")),
                section_header=raw.get("section_header", ""),
                extraction_rationale=raw.get("extraction_rationale", ""),
                primary_category=raw.get("primary_category", "Mitigation"),
                is_actionable=bool(val.final_verdict),
                validation_confidence=float(val.confidence_score),
                validation_reasoning=val.reasoning,
            )
        except Exception as e:
            print(f"  [WARN] ClimatePolicy construction failed: {e}")
            rejected.append({
                **raw,
                "rejection_reason": f"Schema construction error: {e}",
                "validation_result": val.validation_result,
            })
            continue

        if val.validation_result == "VALID" and val.final_verdict:
            validated.append(policy)
        else:
            rejected.append({
                **policy.model_dump(),
                "rejection_reason": val.reasoning,
                "validation_result": val.validation_result,
            })

    return validated, rejected


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _derive_output_paths(output_path: str) -> tuple[str, str, str]:
    """Return (policies_path, rejected_path, trace_path) from a base output path."""
    stem = output_path.replace(".json", "")
    return (
        f"{stem}_policies.json",
        f"{stem}_rejected.json",
        f"{stem}_trace.json",
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_document(
    pdf_path: str,
    country: str,
    state_or_province: str | None = None,
    city: str | None = None,
    expert_knowledge_path: str | None = None,
    output_path: str | None = None,
    output_format: Literal["json", "csv", "both"] = "json",
    backend: str = "openai",
    model_name: str = "gpt-5",
    sub_model_name: str = "gpt-5-nano",
    max_iterations: int = 50,
) -> list[ClimatePolicy]:
    """
    End-to-end single-document pipeline:
      PDF/MD → markdown → RLM (extract + classify) → DSPy Validator → output

    output_format controls what gets written when output_path is provided:
      "json"  — {output_path}_policies.json, {output_path}_rejected.json  (default)
      "csv"   — {output_path}_policies.csv,  {output_path}_rejected.csv
      "both"  — all four files above

    Always writes: {output_path}_trace.json (RLMLogger thought tree)
    """
    metadata = DocumentMetadata(
        country=country, state_or_province=state_or_province, city=city
    )
    location_label = city or state_or_province or country

    print(f"{'=' * 60}")
    print(f"Processing : {pdf_path}")
    print(f"Location   : {location_label}, {country}")
    print(f"{'=' * 60}")

    # 1. Ingest document
    print("\n[1/3] Converting document to markdown...")
    document_md = parse_document(pdf_path)
    print(f"  {len(document_md):,} characters")

    # 2. Optional expert knowledge
    expert_knowledge = None
    if expert_knowledge_path and os.path.exists(expert_knowledge_path):
        print("[1.5/3] Loading expert extraction criteria...")
        expert_knowledge = parse_document(expert_knowledge_path)

    # 3. RLM extraction + classification
    print("\n[2/3] Running RLM extraction + classification...")
    policies_path, rejected_path, trace_path = (
        _derive_output_paths(output_path) if output_path else (None, None, None)
    )
    raw_policies = extract_and_classify_rlm(
        document_markdown=document_md,
        expert_knowledge=expert_knowledge,
        trace_output_path=trace_path,
        backend=backend,
        model_name=model_name,
        sub_model_name=sub_model_name,
        max_iterations=max_iterations,
    )
    print(f"  {len(raw_policies)} raw policies extracted")

    # 4. DSPy validation
    print("\n[3/3] Validating with DSPy PolicyValidator...")
    validated, rejected = validate_policies(raw_policies)
    print(f"  {len(validated)} passed  |  {len(rejected)} rejected")

    # 5. Save outputs
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        validated_dicts = [p.model_dump() for p in validated]

        if output_format in ("json", "both"):
            with open(policies_path, "w") as f:
                json.dump(validated_dicts, f, indent=2)
            with open(rejected_path, "w") as f:
                json.dump(rejected, f, indent=2)
            print(f"\n  Policies JSON : {policies_path}")
            print(f"  Rejected JSON : {rejected_path}")

        if output_format in ("csv", "both"):
            validated_csv = policies_path.replace(".json", ".csv")
            rejected_csv = rejected_path.replace(".json", ".csv")
            pd.DataFrame(validated_dicts).to_csv(validated_csv, index=False)
            pd.DataFrame(rejected).to_csv(rejected_csv, index=False)
            print(f"  Policies CSV  : {validated_csv}")
            print(f"  Rejected CSV  : {rejected_csv}")

        print(f"  Trace         : {trace_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Total extracted : {len(raw_policies)}")
    print(f"  Validated       : {len(validated)}")
    print(f"  Rejected        : {len(rejected)}")
    if validated:
        dist: dict[str, int] = {}
        for p in validated:
            dist[p.primary_category] = dist.get(p.primary_category, 0) + 1
        print("  Category distribution:")
        for cat, count in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {count}")
    print(f"{'=' * 60}\n")

    return validated


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # policies = process_document(
    #     pdf_path="./../GENIUS/docs/cities/LV.md",
    #     country="United States",
    #     state_or_province="Nevada",
    #     city="Las Vegas",
    #     expert_knowledge_path="./RLM_proc_instr.pdf",
    #     output_path="./output/LasVegas",
    #     output_format="csv",   # "json" | "csv" | "both"
    #     model_name="gpt-5.4",
    #     sub_model_name="gpt-5.4",
    #     max_iterations=50,
    # )

    policies = process_document(
        pdf_path="./../GENIUS/docs/cities/seattle_markdown.md",
        country="United States",
        state_or_province="Washington",
        city="Seattle",
        expert_knowledge_path="./RLM_proc_instr.pdf",
        output_path="./output/Seattle",
        output_format="csv",   # "json" | "csv" | "both"
        model_name="gpt-5.4",
        sub_model_name="gpt-5.4",
        max_iterations=50,
    )

    print(f"Done. {len(policies)} validated policies.")

