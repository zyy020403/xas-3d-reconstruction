#!/usr/bin/env python
"""
step2.0_audit_dtype.py
Exp6 Phase 2 SA2 — bf16 dtype hardcoding audit.

Rationale (EXP6_PHASE2_SA2_HANDOFF.md §2.1 + EXP4_ERRATA_2026-04-28.md §2):
    Exp4 forward path had 3 hardcoded fp32 sites (SinusoidalTimeEmbeddings /
    t_per_atom / cspnet) that broke GPU bf16 with mat1/mat2 dtype mismatch
    on PT 2.4.1. SA1's smoke ran fp32 only (ERRATA_2 §3.1 — not bf16-audited).
    This script audits Exp6 shared/ for the same class of bug before SA2's
    bf16 sanity launch.

Scope (handoff §2.1 step 2):
    Scan all forward() methods of nn.Module subclasses in shared/*.py for:
        P1. torch.arange(...)            without dtype= / .to(...)
        P2. torch.zeros|ones|empty|full(...) without dtype= / .to(...)
        P3. torch.tensor([...])          literal (defaults fp32)
        P4. .float() / .double()         explicit cast
        P5. dtype=torch.float32 / float64 hardcoded literal

Approach:
    1. AST-parse each .py — confirms syntactic validity (cheap import smoke).
    2. AST-locate every `def forward(...)` inside a `class ... (... Module ...)`.
    3. Regex-grep ONLY within those forward bodies (not __init__, not utils).
    4. Also do whole-file pass with "outside-forward" tag — buffers in __init__
       can still mismatch under model.to(bf16) if not registered properly.
    5. Each hit: file:line + 3-line context + severity + manual-review note.

Output: stdout machine-readable hit count + sectioned report.
Exit code: 0 always (audit is informational, decisions belong to SA2/MA1).

Usage (run from Exp6 project root):
    /home/tcat/conda_envs/mlff/bin/python step2/step2.0_audit_dtype.py shared/ \\
        | tee step2/step2.0_audit_log.txt
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# pattern table
# ---------------------------------------------------------------------------

@dataclass
class Pattern:
    label: str
    severity: str  # "high" | "medium"
    regex: re.Pattern
    note: str
    needs_dtype_check: bool  # if True, suppress hit when match contains dtype= or .to(

PATTERNS: list[Pattern] = [
    Pattern(
        label="P1_arange_no_dtype",
        severity="high",
        regex=re.compile(r"torch\.arange\s*\("),
        note=(
            "torch.arange defaults to fp32. If output cats/adds with bf16 weights "
            "(e.g. position embedding sin/cos), mat1/mat2 dtype mismatch will fire. "
            "Fix: torch.arange(...).to(reference_tensor.dtype) or pass dtype=x.dtype."
        ),
        needs_dtype_check=True,
    ),
    Pattern(
        label="P2_alloc_no_dtype",
        severity="medium",
        regex=re.compile(r"torch\.(zeros|ones|empty|full|zeros_like|ones_like)\s*\("),
        note=(
            "Default fp32 alloc; matters only if tensor flows into compute graph. "
            "*_like variants inherit from input — usually safe. "
            "Bare zeros/ones in forward() are suspect."
        ),
        needs_dtype_check=True,
    ),
    Pattern(
        label="P3_tensor_literal",
        severity="medium",
        regex=re.compile(r"torch\.tensor\s*\(\s*\["),
        note=(
            "Literal float list defaults to fp32. If used as buffer or constant "
            "in forward, may mismatch bf16 model. Check whether it's registered "
            "as buffer (bf16-following) or created fresh per call (fp32 pinned)."
        ),
        needs_dtype_check=True,
    ),
    Pattern(
        label="P4_explicit_cast_float",
        severity="high",
        regex=re.compile(r"\.\s*(float|double)\s*\(\s*\)"),
        note=(
            "Explicit fp32/fp64 cast will fight bf16 autocast. "
            "If intentional (e.g. .float() before scipy call), record reason. "
            "Otherwise replace with .to(reference.dtype)."
        ),
        needs_dtype_check=False,
    ),
    Pattern(
        label="P5_dtype_pin_fp32",
        severity="high",
        regex=re.compile(r"dtype\s*=\s*torch\.(float32|float64|float)\b"),
        note=(
            "Hard-pinned fp32/fp64 dtype literal. Will fight bf16 autocast. "
            "Use dtype=reference.dtype unless there's a numerical-stability reason."
        ),
        needs_dtype_check=False,
    ),
]

# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

@dataclass
class ForwardSpan:
    file: Path
    class_name: str
    start_line: int  # 1-indexed inclusive
    end_line: int    # 1-indexed inclusive

def is_module_subclass(class_node: ast.ClassDef) -> bool:
    """Cheap heuristic — class inherits from anything ending in 'Module'."""
    for base in class_node.bases:
        if isinstance(base, ast.Attribute) and base.attr.endswith("Module"):
            return True
        if isinstance(base, ast.Name) and base.id.endswith("Module"):
            return True
    return False

def collect_forward_spans(path: Path) -> tuple[list[ForwardSpan], str | None]:
    """Returns (spans, ast_error_or_None). ast_error means file is not syntactically valid."""
    text = path.read_text()
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as e:
        return [], f"SyntaxError at {path}:{e.lineno}: {e.msg}"

    spans: list[ForwardSpan] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and is_module_subclass(node):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    spans.append(ForwardSpan(
                        file=path,
                        class_name=node.name,
                        start_line=item.lineno,
                        end_line=item.end_lineno or item.lineno,
                    ))
    return spans, None

# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------

def extract_full_call(text: str, match_start: int) -> str:
    """Walk balanced parens from match_start to find the full call expression.
    Returns the substring; falls back to one line if balance fails."""
    # find the opening paren after match_start
    paren_idx = text.find("(", match_start)
    if paren_idx == -1:
        return text[match_start:match_start + 80]
    depth = 0
    i = paren_idx
    while i < len(text):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[match_start:i + 1]
        i += 1
    return text[match_start:match_start + 200]  # unbalanced; cap

def has_dtype_or_to(call_text: str) -> bool:
    """Does the call contain a dtype= kwarg or chained .to(...)?"""
    if "dtype=" in call_text:
        return True
    # .to( appearing in matched call (chained) — may be the wrong tensor though;
    # but cheap heuristic, low false-suppression risk for an audit.
    if ".to(" in call_text:
        return True
    return False

@dataclass
class Hit:
    file: Path
    line: int
    label: str
    severity: str
    in_forward: bool
    forward_class: str | None
    match_text: str
    full_call: str
    context: str
    note: str
    review_flag: bool  # True = "manual review needed"

def line_of(text: str, char_index: int) -> int:
    return text[:char_index].count("\n") + 1

def context_lines(lines: list[str], target_line: int) -> str:
    start = max(0, target_line - 2)
    end = min(len(lines), target_line + 1)
    return "\n".join(f"  {i + 1:4d}: {lines[i]}" for i in range(start, end))

def scan_file(path: Path, forward_spans: list[ForwardSpan]) -> list[Hit]:
    text = path.read_text()
    lines = text.splitlines()
    hits: list[Hit] = []

    fwd_in_this_file = [s for s in forward_spans if s.file == path]

    def in_forward_at(line_no: int) -> tuple[bool, str | None]:
        for s in fwd_in_this_file:
            if s.start_line <= line_no <= s.end_line:
                return True, s.class_name
        return False, None

    for pat in PATTERNS:
        for m in pat.regex.finditer(text):
            line_no = line_of(text, m.start())
            full_call = extract_full_call(text, m.start())

            if pat.needs_dtype_check and has_dtype_or_to(full_call):
                continue  # legitimate

            in_fwd, fwd_cls = in_forward_at(line_no)

            review = pat.severity == "medium" or (
                # high in __init__ for buffer registration is usually OK; flag review
                pat.severity == "high" and not in_fwd
            )

            hits.append(Hit(
                file=path,
                line=line_no,
                label=pat.label,
                severity=pat.severity,
                in_forward=in_fwd,
                forward_class=fwd_cls,
                match_text=m.group(0).strip(),
                full_call=full_call.replace("\n", " \\n ")[:120],
                context=context_lines(lines, line_no),
                note=pat.note,
                review_flag=review,
            ))

    return hits

# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def print_header():
    print("=" * 72)
    print("Exp6 Step 2.0 — bf16 dtype hardcoding audit")
    print("Handoff §2.1 spec — patterns P1..P5, AST forward-method scoping.")
    print("=" * 72)

def print_file_smoke(path: Path, ast_err: str | None, n_forwards: int):
    if ast_err:
        print(f"  [AST FAIL] {path.name}: {ast_err}")
    else:
        print(f"  [ok]       {path.name}  ({n_forwards} forward method(s))")

def report_hits(hits: list[Hit]) -> None:
    if not hits:
        print("\n(no hits)")
        return

    by_sev = {"high": [], "medium": []}
    for h in hits:
        by_sev[h.severity].append(h)

    print()
    print(f"Hits: {len(hits)}  [high: {len(by_sev['high'])}, medium: {len(by_sev['medium'])}]")

    # also count in-forward vs outside
    in_fwd = sum(1 for h in hits if h.in_forward)
    print(f"  in-forward: {in_fwd}, outside-forward (init/utils): {len(hits) - in_fwd}")

    for sev in ("high", "medium"):
        if not by_sev[sev]:
            continue
        print()
        print("=" * 72)
        print(f"{sev.upper()} severity hits")
        print("=" * 72)
        for h in by_sev[sev]:
            scope = (
                f"forward of {h.forward_class}" if h.in_forward
                else "OUTSIDE forward (init/utils — manual review needed)"
            )
            print()
            print(f"[{h.label}] {h.file.name}:{h.line}  ({scope})")
            print(f"  match:    {h.match_text}")
            print(f"  call:     {h.full_call}")
            print(f"  note:     {h.note}")
            if h.review_flag:
                print(f"  STATUS:   SA2 manual review needed")
            print(f"  context:")
            print(h.context)

def print_decision(hits: list[Hit]) -> None:
    print()
    print("=" * 72)
    print("DECISION GUIDE (handoff §2.1 + §10)")
    print("=" * 72)

    high_in_fwd = [h for h in hits if h.severity == "high" and h.in_forward]
    high_out = [h for h in hits if h.severity == "high" and not h.in_forward]
    med = [h for h in hits if h.severity == "medium"]

    if not hits:
        print("CLEAN — bf16 path safe. Step 2.1 sanity may run bf16 directly.")
        return

    if not high_in_fwd and not high_out:
        print(f"Only medium hits ({len(med)}) — likely false positives in")
        print("buffer-registration sites. SA2 should manually inspect each, then")
        print("decide bf16 (preferred) vs fp32 (deferred).")
        return

    locked_files = {
        "transformer.py", "matcher.py", "criterion.py", "detr_xas.py",
        "spectrum_tokenizer.py", "eval_metrics.py",
    }
    in_locked = [h for h in (high_in_fwd + high_out) if h.file.name in locked_files]

    print(f"High-severity hits: {len(high_in_fwd)} in-forward, {len(high_out)} outside-forward.")
    if in_locked:
        print()
        print(f"⚠️  {len(in_locked)} hit(s) live in SA1-LOCKED files:")
        for h in in_locked:
            print(f"    {h.file.name}:{h.line}  ({h.label})")
        print("→ Per handoff §10 row 1 + row 6: PUSH MA1. Do NOT self-edit.")
    else:
        print("All high hits are outside SA1-locked files.")
        print("→ SA2 may patch (dtype-follow-input wrapper) IF scope is wrapper-only.")
        print("→ If scope crosses into transformer/matcher/criterion/detr_xas/")
        print("  spectrum_tokenizer/eval_metrics: PUSH MA1 (handoff §10 row 1).")
    print()
    print("Until decision: Step 2.1 sanity defaults to fp32 (safe fallback).")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shared_dir", help="Path to Exp6 shared/ directory")
    args = parser.parse_args()

    root = Path(args.shared_dir)
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    py_files = sorted(p for p in root.glob("*.py") if not p.name.startswith("__"))
    if not py_files:
        print(f"ERROR: no .py files in {root} (excluding __*).", file=sys.stderr)
        sys.exit(2)

    print_header()
    print(f"shared dir: {root.resolve()}")
    print(f"files scanned ({len(py_files)}):")

    all_spans: list[ForwardSpan] = []
    any_ast_err = False
    for p in py_files:
        spans, err = collect_forward_spans(p)
        all_spans.extend(spans)
        print_file_smoke(p, err, len(spans))
        if err:
            any_ast_err = True

    if any_ast_err:
        print("\nAST fail in at least one file — fix before audit decisions are reliable.")
        # continue anyway; remaining files still scanned

    print()
    print(f"Forward methods found across all files: {len(all_spans)}")
    for s in all_spans:
        print(f"  {s.file.name}:{s.start_line}-{s.end_line}  class {s.class_name}")

    all_hits: list[Hit] = []
    for p in py_files:
        all_hits.extend(scan_file(p, all_spans))

    report_hits(all_hits)
    print_decision(all_hits)

    # always exit 0 — audit is informational, decision belongs to SA2/MA1
    sys.exit(0)

if __name__ == "__main__":
    main()
