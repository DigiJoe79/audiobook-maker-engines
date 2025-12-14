#!/usr/bin/env python3
"""
Generate catalog.json from engine.yaml files.

Scans all {type}/{name}/engine.yaml files and builds catalog.json.
Header info comes from catalog-config.yaml in the repo root.
"""

import json
import yaml
import sys
from datetime import datetime, timezone
from pathlib import Path

# Engine types to scan
ENGINE_TYPES = ["tts", "stt", "text", "audio"]


def load_catalog_config(repo_root: Path) -> dict:
    """Load catalog header configuration."""
    config_path = repo_root / "catalog-config.yaml"
    if not config_path.exists():
        print(f"Error: {config_path} not found")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_engines(repo_root: Path) -> list[tuple[str, Path]]:
    """Find all engine.yaml files and return (type, path) tuples."""
    engines = []
    for engine_type in ENGINE_TYPES:
        type_dir = repo_root / engine_type
        if not type_dir.exists():
            continue

        for engine_dir in type_dir.iterdir():
            if not engine_dir.is_dir():
                continue

            yaml_path = engine_dir / "engine.yaml"
            if yaml_path.exists():
                engines.append((engine_type, yaml_path))

    return engines


def transform_engine(engine_type: str, yaml_path: Path) -> dict:
    """Transform engine.yaml to catalog entry format."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Extract model names
    models = config.get("models", [])
    model_names = [m.get("name", "default") for m in models]

    # Extract default parameters from parameter_schema
    default_params = {}
    param_schema = config.get("config", {}).get("parameter_schema", {})
    for param_name, param_def in param_schema.items():
        if "default" in param_def:
            default_params[param_name] = param_def["default"]

    # Build catalog entry
    entry = {
        "name": config.get("name"),
        "type": engine_type,
        "displayName": config.get("display_name", config.get("name")),
        "description": config.get("description", ""),
        "upstream": config.get("upstream", {}),
        "variants": config.get("variants", []),
        "supportedLanguages": config.get("supported_languages", []),
        "models": model_names,
        "defaultModel": config.get("default_model", model_names[0] if model_names else "default"),
        "defaultParameters": default_params,
    }

    return entry


def generate_catalog(repo_root: Path) -> dict:
    """Generate complete catalog.json content."""
    config = load_catalog_config(repo_root)

    catalog = {
        "catalogVersion": config.get("catalog_version", "1.0"),
        "minAppVersion": config.get("min_app_version", "1.0.0"),
        "registry": config.get("registry", ""),
        "lastUpdated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "engines": [],
    }

    engines = find_engines(repo_root)
    print(f"Found {len(engines)} engine(s)")

    for engine_type, yaml_path in engines:
        try:
            entry = transform_engine(engine_type, yaml_path)
            catalog["engines"].append(entry)
            print(f"  + {entry['name']} ({engine_type})")
        except Exception as e:
            print(f"  ! Error processing {yaml_path}: {e}")
            sys.exit(1)

    return catalog


def main():
    # Determine repo root (script is in scripts/)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    # Allow override via command line
    if len(sys.argv) > 1:
        repo_root = Path(sys.argv[1])

    print(f"Scanning {repo_root}")
    catalog = generate_catalog(repo_root)

    # Write catalog.json
    output_path = repo_root / "catalog.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"Generated {output_path}")
    print(f"  Engines: {len(catalog['engines'])}")


if __name__ == "__main__":
    main()
