#!/usr/bin/env python3
"""
Engine Validation Script

Validates all engine.yaml files against the JSON schema and performs
cross-file consistency checks.

Usage:
    python scripts/validate_engines.py           # Validate all engines
    python scripts/validate_engines.py --verbose # Show detailed output
    python scripts/validate_engines.py --fix     # Show suggested fixes

Exit codes:
    0 - All validations passed
    1 - Validation errors found
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import yaml

# Try to import jsonschema, provide helpful error if missing
try:
    from jsonschema import Draft7Validator
except ImportError:
    print("[FAIL] jsonschema package not installed")
    print("       Install with: pip install jsonschema")
    sys.exit(1)


# =============================================================================
# Constants
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
SCHEMA_PATH = SCRIPT_DIR / "engine-schema.json"

# Engine type to directory mapping
ENGINE_TYPE_DIRS = {
    "tts": "tts",
    "stt": "stt",
    "text": "text_processing",
    "audio": "audio_analysis",
}

# Reverse mapping for validation
DIR_TO_ENGINE_TYPE = {v: k for k, v in ENGINE_TYPE_DIRS.items()}


# =============================================================================
# Validation Results
# =============================================================================

class ValidationResult:
    """Holds validation results for a single engine."""

    def __init__(self, engine_path: Path):
        self.engine_path = engine_path
        self.engine_name = engine_path.parent.name
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.fixes: list[str] = []

    def add_error(self, msg: str, fix: Optional[str] = None):
        self.errors.append(msg)
        if fix:
            self.fixes.append(fix)

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0


# =============================================================================
# Schema Validation
# =============================================================================

def load_schema() -> dict:
    """Load the JSON schema."""
    if not SCHEMA_PATH.exists():
        print(f"[FAIL] Schema not found: {SCHEMA_PATH}")
        sys.exit(1)

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_schema(data: dict, schema: dict, result: ValidationResult):
    """Validate engine.yaml against JSON schema."""
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)

    for error in errors:
        path = ".".join(str(p) for p in error.path) if error.path else "(root)"
        result.add_error(f"Schema: {path} - {error.message}")


# =============================================================================
# Cross-File Validation
# =============================================================================

def validate_name_matches_directory(data: dict, result: ValidationResult):
    """Check that 'name' matches the directory name."""
    yaml_name = data.get("name", "")
    dir_name = result.engine_name

    if yaml_name != dir_name:
        result.add_error(
            f"name mismatch: '{yaml_name}' != directory '{dir_name}'",
            f"Change name: \"{yaml_name}\" to name: \"{dir_name}\""
        )


def validate_engine_type_matches_parent(data: dict, result: ValidationResult):
    """Check that 'engine_type' matches the parent directory."""
    engine_type = data.get("engine_type", "")
    parent_dir = result.engine_path.parent.parent.name

    expected_type = DIR_TO_ENGINE_TYPE.get(parent_dir)

    if expected_type and engine_type != expected_type:
        result.add_error(
            f"engine_type mismatch: '{engine_type}' but in {parent_dir}/ (expected '{expected_type}')",
            f"Change engine_type: \"{engine_type}\" to engine_type: \"{expected_type}\""
        )


def validate_default_model_exists(data: dict, result: ValidationResult):
    """Check that default_model exists in models array."""
    default_model = data.get("default_model")
    models = data.get("models", [])

    if not default_model:
        return  # Optional field

    model_names = [m.get("name") for m in models if m.get("name")]

    if default_model not in model_names:
        result.add_error(
            f"default_model '{default_model}' not found in models array",
            f"Add model '{default_model}' to models array or change default_model"
        )


def validate_models_not_empty(data: dict, result: ValidationResult):
    """Warn if models array is empty (common bug)."""
    models = data.get("models", [])

    if not models:
        result.add_warning("models array is empty - catalog will show no models")


def validate_python_version_matches_dockerfile(
    data: dict, result: ValidationResult, dockerfile_path: Path
):
    """Check that installation.python_version matches Dockerfile."""
    installation = data.get("installation", {})
    yaml_version = installation.get("python_version")

    if not yaml_version:
        return  # Optional field

    if not dockerfile_path.exists():
        return  # No Dockerfile to compare

    dockerfile_content = dockerfile_path.read_text(encoding="utf-8")

    # Extract Python version from Dockerfile
    # Patterns: FROM python:3.12-slim, python3.12, python3.10
    patterns = [
        r"FROM python:(\d+\.\d+)",
        r"python(\d+\.\d+)",
        r"Python (\d+\.\d+)",
    ]

    dockerfile_version = None
    for pattern in patterns:
        match = re.search(pattern, dockerfile_content)
        if match:
            dockerfile_version = match.group(1)
            break

    if dockerfile_version and yaml_version != dockerfile_version:
        result.add_error(
            f"Python version mismatch: engine.yaml '{yaml_version}' != Dockerfile '{dockerfile_version}'",
            f"Change installation.python_version to \"{dockerfile_version}\""
        )


def validate_gpu_requirement_matches_dockerfile(
    data: dict, result: ValidationResult, dockerfile_path: Path
):
    """Check that requires_gpu matches Dockerfile base image."""
    variants = data.get("variants", [])

    if not variants:
        return

    # Check first variant (primary)
    requires_gpu = variants[0].get("requires_gpu", False)

    if not dockerfile_path.exists():
        return

    dockerfile_content = dockerfile_path.read_text(encoding="utf-8")

    # Check if Dockerfile uses nvidia/cuda base image
    uses_cuda = "nvidia/cuda" in dockerfile_content or "FROM cuda" in dockerfile_content

    if requires_gpu and not uses_cuda:
        result.add_warning(
            "requires_gpu: true but Dockerfile doesn't use nvidia/cuda base image"
        )
    elif not requires_gpu and uses_cuda:
        result.add_warning(
            "requires_gpu: false but Dockerfile uses nvidia/cuda base image"
        )


# =============================================================================
# Main Validation
# =============================================================================

def find_engine_yamls() -> list[Path]:
    """Find all engine.yaml files (excluding templates)."""
    engine_dirs = [
        REPO_ROOT / "tts",
        REPO_ROOT / "stt",
        REPO_ROOT / "text_processing",
        REPO_ROOT / "audio_analysis",
    ]

    yamls = []
    for engine_dir in engine_dirs:
        if not engine_dir.exists():
            continue

        for engine_yaml in engine_dir.glob("*/engine.yaml"):
            # Skip templates
            if "_template" in str(engine_yaml):
                continue
            yamls.append(engine_yaml)

    return sorted(yamls)


def validate_engine(engine_yaml: Path, schema: dict) -> ValidationResult:
    """Validate a single engine.yaml file."""
    result = ValidationResult(engine_yaml)

    # Load YAML
    try:
        with open(engine_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        result.add_error(f"YAML parse error: {e}")
        return result

    if not data:
        result.add_error("Empty YAML file")
        return result

    # Schema validation
    validate_schema(data, schema, result)

    # Cross-file validation
    validate_name_matches_directory(data, result)
    validate_engine_type_matches_parent(data, result)
    validate_default_model_exists(data, result)
    validate_models_not_empty(data, result)

    # Dockerfile checks
    dockerfile_path = engine_yaml.parent / "Dockerfile"
    validate_python_version_matches_dockerfile(data, result, dockerfile_path)
    validate_gpu_requirement_matches_dockerfile(data, result, dockerfile_path)

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate engine.yaml files")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )
    parser.add_argument(
        "--fix", action="store_true", help="Show suggested fixes"
    )
    args = parser.parse_args()

    # Load schema
    schema = load_schema()

    # Find all engine.yaml files
    engine_yamls = find_engine_yamls()

    if not engine_yamls:
        print("[WARN] No engine.yaml files found")
        sys.exit(0)

    print(f"Validating {len(engine_yamls)} engines...\n")

    # Validate each engine
    results: list[ValidationResult] = []
    for engine_yaml in engine_yamls:
        result = validate_engine(engine_yaml, schema)
        results.append(result)

    # Print results
    passed = 0
    failed = 0
    warnings = 0

    for result in results:
        rel_path = result.engine_path.relative_to(REPO_ROOT)

        if result.passed:
            status = "[OK]"
            passed += 1
        else:
            status = "[FAIL]"
            failed += 1

        if result.warnings:
            warnings += len(result.warnings)

        print(f"{status} {rel_path}")

        if args.verbose or not result.passed:
            for error in result.errors:
                print(f"      ERROR: {error}")

            if args.fix and result.fixes:
                for fix in result.fixes:
                    print(f"      FIX: {fix}")

        if args.verbose and result.warnings:
            for warning in result.warnings:
                print(f"      WARN: {warning}")

    # Summary
    print()
    print(f"Results: {passed} passed, {failed} failed, {warnings} warnings")

    if failed > 0:
        print("\nRun with --fix to see suggested fixes")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
