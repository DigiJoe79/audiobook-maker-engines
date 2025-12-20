#!/usr/bin/env python3
"""
Generate catalog.yaml from engine.yaml files.

Scans all {type}/{name}/engine.yaml files and builds catalog.yaml.
Header info comes from catalog-config.yaml in the repo root.
Uses snake_case throughout (same as engine.yaml).
"""

import yaml
import sys
from datetime import datetime, timezone
from pathlib import Path

# Engine type directories to scan (maps engine_type to directory name)
ENGINE_TYPE_DIRS = {
    "tts": "tts",
    "stt": "stt",
    "text": "text_processing",
    "audio": "audio_analysis",
}


def load_catalog_config(repo_root: Path) -> dict:
    """Load catalog header configuration."""
    config_path = repo_root / "catalog-config.yaml"
    if not config_path.exists():
        print(f"Error: {config_path} not found")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_engines(repo_root: Path) -> list[tuple[str, Path]]:
    """
    Find all engine.yaml files and return (type, path) tuples.

    Only includes engines that:
    - Have a Dockerfile (are dockerized)
    - Are not _template directories
    """
    engines = []
    for engine_type, dir_name in ENGINE_TYPE_DIRS.items():
        type_dir = repo_root / dir_name
        if not type_dir.exists():
            continue

        for engine_dir in type_dir.iterdir():
            if not engine_dir.is_dir():
                continue

            # Skip template directories
            if engine_dir.name.startswith("_"):
                continue

            yaml_path = engine_dir / "engine.yaml"
            dockerfile_path = engine_dir / "Dockerfile"

            # Only include engines with both engine.yaml AND Dockerfile
            if yaml_path.exists() and dockerfile_path.exists():
                engines.append((engine_type, yaml_path))
            elif yaml_path.exists():
                print(f"  - Skipping {engine_dir.name}: no Dockerfile")

    return engines


def transform_engine(engine_type: str, yaml_path: Path) -> dict:
    """Transform engine.yaml to catalog entry format (preserving snake_case)."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Build catalog entry - all keys in snake_case
    entry = {
        "name": config.get("name"),
        "engine_type": engine_type,
        "display_name": config.get("display_name", config.get("name")),
        "description": config.get("description", ""),
        "upstream": config.get("upstream", {}),
        "variants": config.get("variants", []),
        "supported_languages": config.get("supported_languages", []),
        "constraints": config.get("constraints", {}),
        "capabilities": config.get("capabilities", {}),
        "parameters": config.get("parameters", {}),
        "models": config.get("models", []),
        "default_model": config.get("default_model", ""),
    }

    return entry


def generate_catalog(repo_root: Path) -> dict:
    """Generate complete catalog.yaml content."""
    config = load_catalog_config(repo_root)

    catalog = {
        "catalog_version": config.get("catalog_version", "1.0"),
        "min_app_version": config.get("min_app_version", "1.0.0"),
        "registry": config.get("registry", ""),
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    if len(sys.argv) > 1:
        repo_root = Path(sys.argv[1])

    print(f"Scanning {repo_root}")
    catalog = generate_catalog(repo_root)

    # Write catalog.yaml (not JSON!)
    output_path = repo_root / "catalog.yaml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(catalog, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Generated {output_path}")
    print(f"  Engines: {len(catalog['engines'])}")


if __name__ == "__main__":
    main()
