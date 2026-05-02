"""
Merge personas from 4chan and PersonaHub into a single file.

Output format:
{"id": 0, "persona": "...", "source": "4chan|personahub"}
"""

import json
import argparse
from pathlib import Path


def load_personas(filepath: str, source: str) -> list:
    """Load personas from JSONL file and add source field."""
    personas = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            personas.append({
                "persona": data["persona"],
                "source": source
            })
    return personas


def main():
    # Load config
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f).get("merge_personas", {})
    else:
        config = {}

    parser = argparse.ArgumentParser(description="Merge personas from multiple sources")
    parser.add_argument("--input_4chan", default=config.get("input_4chan", "data/output/persona_selected_1285.jsonl"))
    parser.add_argument("--input_personahub", default=config.get("input_personahub", "data/output/personahub_selected_1000.jsonl"))
    parser.add_argument("--output", default=config.get("output", "data/output/personas_merged.jsonl"))
    args = parser.parse_args()

    # Resolve paths
    base_dir = Path(__file__).parent
    input_4chan = base_dir / args.input_4chan
    input_personahub = base_dir / args.input_personahub
    output_path = base_dir / args.output

    # Load personas
    print(f"Loading 4chan personas from {input_4chan}...")
    personas_4chan = load_personas(str(input_4chan), "4chan")
    print(f"  Loaded {len(personas_4chan)} personas")

    print(f"Loading PersonaHub personas from {input_personahub}...")
    personas_personahub = load_personas(str(input_personahub), "personahub")
    print(f"  Loaded {len(personas_personahub)} personas")

    # Merge and assign IDs
    all_personas = personas_4chan + personas_personahub
    for i, p in enumerate(all_personas):
        p["id"] = i

    # Reorder fields: id first
    all_personas = [{"id": p["id"], "persona": p["persona"], "source": p["source"]} for p in all_personas]

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for p in all_personas:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f"\nMerged {len(all_personas)} personas to {output_path}")
    print(f"  - 4chan: {len(personas_4chan)}")
    print(f"  - PersonaHub: {len(personas_personahub)}")


if __name__ == "__main__":
    main()
