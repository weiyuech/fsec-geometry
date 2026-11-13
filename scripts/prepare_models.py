"""Download the model snapshots used by named jobs and record the resolved Hub commits."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi, snapshot_download
from fsa.data import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", nargs="?", type=Path, default=ROOT / "jobs/main.json")
    parser.add_argument("--only", nargs="+", metavar="JOB")
    parser.add_argument("--output", type=Path, default=ROOT / "raw/model_revisions.json")
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text())["jobs"]
    if args.only:
        known = {j["name"] for j in jobs}
        if set(args.only) - known:
            parser.error("Unknown job(s): " + ", ".join(sorted(set(args.only) - known)))
        jobs = [j for j in jobs if j["name"] in args.only]
    models = sorted({j["model"] for j in jobs if "model" in j})
    records = json.loads(args.output.read_text()) if args.output.exists() else {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    for name in models:
        repo, revision = MODELS[name]
        # A previous capture stays pinned across retries and later invocations.
        old = records.get(name)
        info = api.model_info(repo, revision=old["sha"] if old else revision)
        sha = info.sha
        files = [s.rfilename for s in info.siblings if "/" not in s.rfilename]
        weights = [f for f in files if f.endswith(".safetensors")]
        if not weights:
            weights = [f for f in files if f.startswith("pytorch_model") and f.endswith(".bin")]
        if not weights:
            raise ValueError(f"No checkpoint weights listed for {repo} at {sha}")
        selected = weights + [f for f in files if f.endswith((".json", ".model", ".txt", ".tiktoken"))]
        snapshot_download(repo, revision=sha, allow_patterns=selected)
        records[name] = dict(repo=repo, revision=revision, sha=sha)
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(records, indent=2) + "\n")
        tmp.replace(args.output)
        print(name, sha, flush=True)


if __name__ == "__main__":
    main()
