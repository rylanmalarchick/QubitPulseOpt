#!/usr/bin/env python3
"""
Power of 10 Compliance Checker for Quantum Control Project

This script analyzes Python source code for compliance with the adapted
NASA/JPL Power of 10 rules for safety-critical code.

Rules checked:
1. Simple Control Flow - No recursion, <3 nesting levels
2. Bounded Loops - All loops have explicit upper bounds
3. No Dynamic Allocation After Init - Pre-allocate arrays
4. Function Length ≤60 Lines
5. Assertion Density ≥2/function
6. Minimal Scope - Local variables, explicit data flow
7. Check Return Values - Validate all inputs/outputs
8. Minimal Metaprogramming - Avoid exec/eval
9. Restricted Indirection - Flat data structures
10. Zero Warnings - Static analysis passes

Usage:
    python power_of_10_checker.py [path] [--json] [--verbose]
"""

import ast
import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


@dataclass
class ViolationReport:
    """Single violation instance."""

    rule: int
    severity: str  # 'error', 'warning', 'info'
    line: int
    column: int
    message: str
    context: str = ""


@dataclass
class ModuleReport:
    """Compliance report for a single module."""

    module_path: str
    lines_of_code: int = 0
    functions: int = 0
    classes: int = 0
    violations: List[ViolationReport] = field(default_factory=list)
    rule_scores: Dict[int, bool] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    def add_violation(
        self,
        rule: int,
        severity: str,
        line: int,
        column: int,
        message: str,
        context: str = "",
    ):
        """Add a violation to the report."""
        self.violations.append(
            ViolationReport(
                rule=rule,
                severity=severity,
                line=line,
                column=column,
                message=message,
                context=context,
            )
        )

    def compliance_score(self) -> float:
        """Calculate overall compliance score (0-100)."""
        if not self.rule_scores:
            return 0.0
        compliant = sum(1 for v in self.rule_scores.values() if v)
        return (compliant / len(self.rule_scores)) * 100


@dataclass
class ProjectReport:
    """Aggregate compliance report for entire project."""

    modules: Dict[str, ModuleReport] = field(default_factory=dict)
    summary: Dict[str, int] = field(
        default_factory=lambda: {
            "total_modules": 0,
            "total_functions": 0,
            "total_violations": 0,
            "errors": 0,
            "warnings": 0,
            "info": 0,
        }
    )

    def add_module(self, report: ModuleReport):
        """Add a module report."""
        self.modules[report.module_path] = report
        self.summary["total_modules"] += 1
        self.summary["total_functions"] += report.functions
        self.summary["total_violations"] += len(report.violations)
        for v in report.violations:
            key = v.severity + "s"
            if key in self.summary:
                self.summary[key] += 1

    def overall_score(self) -> float:
        """Calculate project-wide compliance score."""
        if not self.modules:
            return 0.0
        scores = [m.compliance_score() for m in self.modules.values()]
        return sum(scores) / len(scores)


class PowerOf10Checker(ast.NodeVisitor):
    """AST visitor that checks Power of 10 compliance."""

    def __init__(self, source_code: str, filepath: str):
        self.source_code = source_code
        self.filepath = filepath
        self.lines = source_code.split("\n")
        self.report = ModuleReport(module_path=filepath)

        # Analysis state
        self.current_function: Optional[str] = None
        self.function_calls: Dict[str, Set[str]] = defaultdict(set)
        self.function_defs: Set[str] = set()
        self.nesting_stack: List[ast.AST] = []
        self.assertions_per_function: Dict[str, int] = defaultdict(int)
        self.function_lines: Dict[str, Tuple[int, int]] = {}
        self.loop_bounds: Dict[int, bool] = {}  # line -> has_bound

    def check(self) -> ModuleReport:
        """Run all compliance checks."""
        try:
            tree = ast.parse(self.source_code, filename=self.filepath)
            self.report.lines_of_code = len(self.lines)

            # First pass: collect definitions and basic metrics
            self.visit(tree)

            # Second pass: rule-specific checks
            self._check_recursion()
            self._check_function_assertions()
            self._finalize_scores()

        except SyntaxError as e:
            self.report.add_violation(
                10, "error", e.lineno or 0, e.offset or 0, f"Syntax error: {e.msg}", ""
            )

        return self.report

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definition."""
        self.report.functions += 1
        self.function_defs.add(node.name)

        prev_function = self.current_function
        self.current_function = node.name

        # Calculate function length
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        func_length = end_line - start_line + 1
        self.function_lines[node.name] = (start_line, end_line)

        # Rule 4: Function length check
        if func_length > 60:
            context = self._get_context(start_line)
            self.report.add_violation(
                4,
                "warning",
                start_line,
                node.col_offset,
                f"Function '{node.name}' has {func_length} lines (max 60)",
                context,
            )

        # Rule 5: Check for parameter validation
        has_param_check = False
        if node.args.args:
            # Look for assertions/checks in first few statements
            for stmt in node.body[:5]:
                if isinstance(stmt, ast.Assert):
                    has_param_check = True
                    break
                if isinstance(stmt, ast.If):
                    has_param_check = True
                    break

        # Visit function body
        self.generic_visit(node)

        self.current_function = prev_function

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definition."""
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definition."""
        self.report.classes += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Visit function call."""
        if self.current_function:
            # Track function calls for recursion detection
            if isinstance(node.func, ast.Name):
                self.function_calls[self.current_function].add(node.func.id)

            # Rule 8: Check for exec/eval
            if isinstance(node.func, ast.Name):
                if node.func.id in ("exec", "eval"):
                    self.report.add_violation(
                        8,
                        "error",
                        node.lineno,
                        node.col_offset,
                        f"Metaprogramming forbidden: '{node.func.id}' not allowed",
                        self._get_context(node.lineno),
                    )

        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert):
        """Visit assertion."""
        if self.current_function:
            self.assertions_per_function[self.current_function] += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        """Visit for loop."""
        self._check_nesting_depth(node)
        self._check_loop_bound(node)
        self.nesting_stack.append(node)
        self.generic_visit(node)
        self.nesting_stack.pop()

    def visit_While(self, node: ast.While):
        """Visit while loop."""
        self._check_nesting_depth(node)
        self._check_while_bound(node)
        self.nesting_stack.append(node)
        self.generic_visit(node)
        self.nesting_stack.pop()

    def visit_If(self, node: ast.If):
        """Visit if statement."""
        self._check_nesting_depth(node)
        self.nesting_stack.append(node)
        self.generic_visit(node)
        self.nesting_stack.pop()

    def visit_With(self, node: ast.With):
        """Visit with statement."""
        self._check_nesting_depth(node)
        self.nesting_stack.append(node)
        self.generic_visit(node)
        self.nesting_stack.pop()

    def visit_Try(self, node: ast.Try):
        """Visit try statement."""
        self._check_nesting_depth(node)
        self.nesting_stack.append(node)
        self.generic_visit(node)
        self.nesting_stack.pop()

    def _check_nesting_depth(self, node: ast.AST):
        """Rule 1: Check nesting depth."""
        depth = sum(
            1
            for n in self.nesting_stack
            if isinstance(n, (ast.For, ast.While, ast.If, ast.With, ast.Try))
        )
        if depth >= 3:
            self.report.add_violation(
                1,
                "warning",
                node.lineno,
                node.col_offset,
                f"Nesting depth {depth + 1} exceeds limit of 3",
                self._get_context(node.lineno),
            )

    def _check_loop_bound(self, node: ast.For):
        """Rule 2: Check for loop has explicit bound."""
        has_bound = False

        # Check if iterating over range() with constant
        if isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                # range() call - check if has constant or known bound
                if node.iter.args:
                    arg = node.iter.args[-1]  # Last arg is the upper bound
                    if isinstance(arg, (ast.Constant, ast.Num)):
                        has_bound = True
                    elif isinstance(arg, ast.Name):
                        # Named constant - consider it bounded
                        has_bound = True

        # Check for enumerate, zip, etc. over known-length iterables
        elif isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Name):
                if node.iter.func.id in ("enumerate", "zip"):
                    has_bound = True

        # Check for iteration over list/tuple/array attributes
        elif isinstance(node.iter, ast.Attribute):
            has_bound = True  # Assume object attributes are bounded

        elif isinstance(node.iter, ast.Name):
            has_bound = True  # Named iterables assumed bounded

        self.loop_bounds[node.lineno] = has_bound

        if not has_bound:
            self.report.add_violation(
                2,
                "info",
                node.lineno,
                node.col_offset,
                "Loop bound not statically verifiable",
                self._get_context(node.lineno),
            )

    def _check_while_bound(self, node: ast.While):
        """Rule 2: Check while loop has bound."""
        # While loops are harder to verify - flag for manual review
        has_explicit_bound = False

        # Look for counter-based while loops
        if isinstance(node.test, ast.Compare):
            has_explicit_bound = True

        self.loop_bounds[node.lineno] = has_explicit_bound

        if not has_explicit_bound:
            self.report.add_violation(
                2,
                "warning",
                node.lineno,
                node.col_offset,
                "While loop bound not statically verifiable - requires manual review",
                self._get_context(node.lineno),
            )

    def _check_recursion(self):
        """Rule 1: Check for direct or indirect recursion."""
        # Direct recursion
        for func, calls in self.function_calls.items():
            if func in calls:
                if func in self.function_lines:
                    line, _ = self.function_lines[func]
                    self.report.add_violation(
                        1,
                        "error",
                        line,
                        0,
                        f"Direct recursion detected in function '{func}'",
                        self._get_context(line),
                    )

        # Indirect recursion (simple cycle detection)
        visited = set()

        def has_cycle(func: str, path: Set[str]) -> bool:
            if func in path:
                return True
            if func in visited:
                return False
            visited.add(func)
            path.add(func)
            for called in self.function_calls.get(func, []):
                if called in self.function_defs and has_cycle(called, path):
                    return True
            path.remove(func)
            return False

        for func in self.function_defs:
            if has_cycle(func, set()):
                if func in self.function_lines:
                    line, _ = self.function_lines[func]
                    self.report.add_violation(
                        1,
                        "error",
                        line,
                        0,
                        f"Indirect recursion detected in call chain involving '{func}'",
                        self._get_context(line),
                    )

    def _check_function_assertions(self):
        """Rule 5: Check assertion density."""
        for func, count in self.assertions_per_function.items():
            if count < 2:
                if func in self.function_lines:
                    line, _ = self.function_lines[func]
                    self.report.add_violation(
                        5,
                        "info",
                        line,
                        0,
                        f"Function '{func}' has {count} assertions (recommended ≥2)",
                        self._get_context(line),
                    )

    def _finalize_scores(self):
        """Calculate per-rule compliance scores."""
        # Rule 1: Control flow
        rule1_violations = [v for v in self.report.violations if v.rule == 1]
        self.report.rule_scores[1] = len(rule1_violations) == 0

        # Rule 2: Loop bounds
        rule2_violations = [v for v in self.report.violations if v.rule == 2]
        self.report.rule_scores[2] = (
            len([v for v in rule2_violations if v.severity == "error"]) == 0
        )

        # Rule 3: Memory allocation (requires manual review)
        self.report.rule_scores[3] = True  # Default pass, requires manual check

        # Rule 4: Function length
        rule4_violations = [v for v in self.report.violations if v.rule == 4]
        self.report.rule_scores[4] = len(rule4_violations) == 0

        # Rule 5: Assertions
        rule5_violations = [v for v in self.report.violations if v.rule == 5]
        # Pass if >80% of functions meet density
        if self.report.functions > 0:
            compliance_rate = 1 - (len(rule5_violations) / self.report.functions)
            self.report.rule_scores[5] = compliance_rate >= 0.8
        else:
            self.report.rule_scores[5] = True

        # Rule 6: Scope (requires manual review)
        self.report.rule_scores[6] = True

        # Rule 7: Return value checks (requires manual review)
        self.report.rule_scores[7] = True

        # Rule 8: Metaprogramming
        rule8_violations = [v for v in self.report.violations if v.rule == 8]
        self.report.rule_scores[8] = len(rule8_violations) == 0

        # Rule 9: Indirection (requires manual review)
        self.report.rule_scores[9] = True

        # Rule 10: Warnings (checked by external tools)
        self.report.rule_scores[10] = True

        # Store metrics
        self.report.metrics["avg_function_length"] = (
            sum(end - start + 1 for start, end in self.function_lines.values())
            / len(self.function_lines)
            if self.function_lines
            else 0
        )
        self.report.metrics["assertion_density"] = (
            sum(self.assertions_per_function.values()) / self.report.functions
            if self.report.functions > 0
            else 0
        )
        self.report.metrics["bounded_loops"] = (
            sum(1 for bounded in self.loop_bounds.values() if bounded)
            / len(self.loop_bounds)
            if self.loop_bounds
            else 1.0
        )

    def _get_context(self, line: int, context_lines: int = 1) -> str:
        """Get source code context around a line."""
        start = max(0, line - context_lines - 1)
        end = min(len(self.lines), line + context_lines)
        context = self.lines[start:end]
        return "\n".join(context)


def check_file(filepath: Path, verbose: bool = False) -> ModuleReport:
    """Check a single Python file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

        checker = PowerOf10Checker(source, str(filepath))
        report = checker.check()

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Module: {filepath}")
            print(
                f"LOC: {report.lines_of_code}, Functions: {report.functions}, "
                f"Classes: {report.classes}"
            )
            print(f"Compliance Score: {report.compliance_score():.1f}%")
            print(f"Violations: {len(report.violations)}")

            if report.violations:
                for v in report.violations[:10]:  # Show first 10
                    print(
                        f"  [{v.severity.upper()}] Rule {v.rule} @ L{v.line}: {v.message}"
                    )
                if len(report.violations) > 10:
                    print(f"  ... and {len(report.violations) - 10} more")

        return report

    except Exception as e:
        print(f"Error checking {filepath}: {e}", file=sys.stderr)
        return ModuleReport(module_path=str(filepath))


def check_directory(dirpath: Path, verbose: bool = False) -> ProjectReport:
    """Recursively check all Python files in a directory."""
    project_report = ProjectReport()

    python_files = list(dirpath.rglob("*.py"))

    if verbose:
        print(f"Found {len(python_files)} Python files in {dirpath}")

    for filepath in python_files:
        # Skip __pycache__ and other build artifacts
        if "__pycache__" in str(filepath) or ".tox" in str(filepath):
            continue

        report = check_file(filepath, verbose=verbose)
        project_report.add_module(report)

    return project_report


def print_summary(report: ProjectReport):
    """Print compliance summary."""
    print("\n" + "=" * 80)
    print("POWER OF 10 COMPLIANCE REPORT")
    print("=" * 80)
    print(f"\nProject Compliance Score: {report.overall_score():.1f}%")
    print(f"\nModules Analyzed: {report.summary['total_modules']}")
    print(f"Total Functions: {report.summary['total_functions']}")
    print(f"Total Violations: {report.summary['total_violations']}")
    print(f"  - Errors: {report.summary['errors']}")
    print(f"  - Warnings: {report.summary['warnings']}")
    print(f"  - Info: {report.summary['info']}")

    print("\n" + "-" * 80)
    print("TOP 10 MODULES BY VIOLATION COUNT")
    print("-" * 80)

    sorted_modules = sorted(
        report.modules.values(), key=lambda m: len(m.violations), reverse=True
    )[:10]

    for i, module in enumerate(sorted_modules, 1):
        score = module.compliance_score()
        print(f"{i}. {module.module_path}")
        print(
            f"   Score: {score:.1f}% | Violations: {len(module.violations)} | "
            f"Functions: {module.functions}"
        )

    print("\n" + "-" * 80)
    print("VIOLATIONS BY RULE")
    print("-" * 80)

    violations_by_rule = defaultdict(int)
    for module in report.modules.values():
        for v in module.violations:
            violations_by_rule[v.rule] += 1

    rule_names = {
        1: "Simple Control Flow",
        2: "Bounded Loops",
        3: "No Dynamic Allocation",
        4: "Function Length ≤60",
        5: "Assertion Density ≥2",
        6: "Minimal Scope",
        7: "Check Return Values",
        8: "Minimal Metaprogramming",
        9: "Restricted Indirection",
        10: "Zero Warnings",
    }

    for rule in sorted(violations_by_rule.keys()):
        count = violations_by_rule[rule]
        print(f"Rule {rule:2d} ({rule_names[rule]}): {count} violations")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check Python code for Power of 10 compliance"
    )
    parser.add_argument(
        "path", nargs="?", default="src", help="Path to check (file or directory)"
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", help="Output file for JSON report")

    args = parser.parse_args()

    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path {path} does not exist", file=sys.stderr)
        sys.exit(1)

    if path.is_file():
        report = check_file(path, verbose=args.verbose)
        project_report = ProjectReport()
        project_report.add_module(report)
    else:
        project_report = check_directory(path, verbose=args.verbose)

    if args.json:
        # Convert to JSON-serializable format
        output = {
            "overall_score": project_report.overall_score(),
            "summary": project_report.summary,
            "modules": {
                path: {
                    "lines_of_code": mod.lines_of_code,
                    "functions": mod.functions,
                    "classes": mod.classes,
                    "compliance_score": mod.compliance_score(),
                    "rule_scores": mod.rule_scores,
                    "metrics": mod.metrics,
                    "violations": [
                        {
                            "rule": v.rule,
                            "severity": v.severity,
                            "line": v.line,
                            "column": v.column,
                            "message": v.message,
                        }
                        for v in mod.violations
                    ],
                }
                for path, mod in project_report.modules.items()
            },
        }

        json_str = json.dumps(output, indent=2)

        if args.output:
            with open(args.output, "w") as f:
                f.write(json_str)
            print(f"Report written to {args.output}")
        else:
            print(json_str)
    else:
        print_summary(project_report)

    # Exit code based on compliance
    if project_report.summary["errors"] > 0:
        sys.exit(1)
    elif project_report.summary["warnings"] > 10:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
