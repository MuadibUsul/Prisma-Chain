"""Collect GEMM E2E reports from challenger logs and classify PASS/FAIL.

The challenger prints exactly one JSON report object; this script extracts
the last {...} block from each log and applies the acceptance rules:

  honest E2E: outcome must be optimistic_unchallenged
  fraud E2E:  outcome must be challenger_wins

Usage: python3 collect_results.py --results-dir ./gemm-results
"""

import argparse
import json
import sys
from pathlib import Path


def extract_report(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    start = text.rfind("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="./gemm-results")
    args = ap.parse_args()
    results_dir = Path(args.results_dir)

    summary = {"honest": {"status": "NOT TESTED"}, "fraud": {"status": "NOT TESTED"}}
    honest_log = results_dir / "honest.log"
    fraud_log = results_dir / "fraud.log"

    if honest_log.exists():
        report = extract_report(honest_log)
        if report and report.get("outcome") == "optimistic_unchallenged":
            summary["honest"] = {"status": "PASS", "report": report}
        else:
            summary["honest"] = {"status": "FAIL", "report": report}

    if fraud_log.exists():
        report = extract_report(fraud_log)
        if report and report.get("outcome") == "challenger_wins":
            summary["fraud"] = {"status": "PASS", "report": report}
        else:
            summary["fraud"] = {"status": "FAIL", "report": report}

    out = results_dir / "e2e-summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    ok = summary["honest"]["status"] == "PASS" and summary["fraud"]["status"] == "PASS"
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
