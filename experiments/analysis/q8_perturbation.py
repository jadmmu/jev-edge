import argparse
import json
from pathlib import Path


CATEGORIES = {"control", "spelling", "grammar", "formatting", "paraphrase", "additional_detail"}


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_requests(path):
    result = {}
    for item in records(path):
        result[item["id"]] = item["request"]
    return result


def diff_paths(first, second, prefix=""):
    if isinstance(first, dict) and isinstance(second, dict):
        paths = []
        for key in sorted(set(first) | set(second)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in first or key not in second:
                paths.append(path)
            else:
                paths.extend(diff_paths(first[key], second[key], path))
        return paths
    if isinstance(first, list) and isinstance(second, list):
        paths = []
        for index in range(max(len(first), len(second))):
            path = f"{prefix}[{index}]"
            if index >= len(first) or index >= len(second):
                paths.append(path)
            else:
                paths.extend(diff_paths(first[index], second[index], path))
        return paths
    return [] if first == second else [prefix]


def validate_request(request):
    if not isinstance(request, dict) or not isinstance(request.get("state"), dict):
        raise ValueError("request state must be an object")
    if not isinstance(request["state"].get("ticket"), str):
        raise ValueError("request state.ticket must be text")
    questions = request.get("questions")
    if not isinstance(questions, dict) or set(questions) != {"team"}:
        raise ValueError("pilot request must contain the team question")
    team = questions["team"]
    if team.get("type") != "choice" or set(team.get("criteria", {})) != {"billing", "shipping", "technical"}:
        raise ValueError("pilot team question schema changed")
    if not isinstance(team.get("instructions"), str):
        raise ValueError("team instructions must be text")


def validate_pilot(pilot_path, source_path):
    source = source_requests(source_path)
    items = records(pilot_path)
    parent_ids = {"q001_billing_duplicate", "q002_password_lockout", "q003_missing_package", "q004_invoice_copy"}
    if len(items) != 24 or len({item["condition_id"] for item in items}) != 24:
        raise ValueError("pilot must contain 24 unique conditions")
    if {item["parent_question_id"] for item in items} != parent_ids:
        raise ValueError("pilot parent IDs do not match selected questions")
    categories = {parent: [item for item in items if item["parent_question_id"] == parent]
                  for parent in parent_ids}
    if any({item["variant_type"] for item in values} != CATEGORIES for values in categories.values()):
        raise ValueError("each parent must contain exactly six variant categories")
    review_rows = []
    for item in items:
        parent = item["parent_question_id"]
        if parent not in source:
            raise ValueError(f"unknown parent {parent}")
        request = item["request"]
        validate_request(request)
        source_request = source[parent]
        if item["variant_type"] == "control":
            if json.dumps(request, separators=(",", ":")) != json.dumps(source_request, separators=(",", ":")):
                raise ValueError(f"control is not byte-equivalent to source request for {parent}")
            if item["original_text"] != item["modified_text"]:
                raise ValueError(f"control text differs for {parent}")
            if item["semantic_equivalence"] != "identical":
                raise ValueError(f"control equivalence label is invalid for {parent}")
        else:
            if item["text_field_path"] != "state.ticket":
                raise ValueError(f"unsupported text field for {item['condition_id']}")
            if item["request"]["state"]["ticket"] != item["modified_text"]:
                raise ValueError(f"modified text mismatch for {item['condition_id']}")
            if item["original_text"] != source_request["state"]["ticket"]:
                raise ValueError(f"original text mismatch for {item['condition_id']}")
            paths = diff_paths(source_request, request)
            if paths != ["state.ticket"]:
                raise ValueError(f"unexpected request differences for {item['condition_id']}: {paths}")
        review_rows.append({"condition_id": item["condition_id"], "parent_question_id": parent,
                            "variant_type": item["variant_type"], "semantic_equivalence": item["semantic_equivalence"],
                            "original_text": item["original_text"], "modified_text": item["modified_text"]})
    return {"conditions": len(items), "parents": len(parent_ids), "categories_per_parent": 6, "review_rows": review_rows}


def write_review(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| Condition | Parent | Variant | Equivalence review | Original text | Modified text |",
             "|---|---|---|---|---|---|"]
    for row in rows:
        values = [row[key].replace("|", "\\|") for key in ("condition_id", "parent_question_id", "variant_type",
                                                               "semantic_equivalence", "original_text", "modified_text")]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    report = validate_pilot(args.pilot, args.source)
    write_review(report["review_rows"], args.review)
    print(json.dumps({"conditions": report["conditions"], "parents": report["parents"],
                      "categories_per_parent": report["categories_per_parent"]}, separators=(",", ":")))


if __name__ == "__main__":
    main()
