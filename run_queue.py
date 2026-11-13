"""Run a list of jobs back to back on one GPU. Resumable; never stops on a failed job.

    python run_queue.py jobs/main.json [--only NAME ...] [--dry]

Jobs sharing a model spec run consecutively on one loaded model, so the GPU is
only idle while weights load from disk. Each job writes results/<name>/ and a
DONE marker; rerunning skips finished jobs and finished items inside a job.
"""
import argparse
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def log(*a):
    print(time.strftime("[%m-%d %H:%M:%S]"), *a, flush=True)


def model_key(job):
    return json.dumps({k: job.get(k) for k in ("model", "perturb_layer", "measure_layer", "shuffle_seed",
                                               "dtype", "batch", "mode")},
                      sort_keys=True)


def save_status(path, status):
    with open(path + ".tmp", "w") as handle:
        json.dump(status, handle, indent=1)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--results", default=os.path.join(ROOT, "raw", "results"))
    args = ap.parse_args()

    spec = json.load(open(args.jobs))
    if args.only and set(args.only) - {j["name"] for j in spec["jobs"]}:
        ap.error("Unknown job in --only")
    jobs = [j for j in spec["jobs"] if not args.only or j["name"] in args.only]
    defaults = spec.get("defaults", {})
    if args.dry:
        for j in jobs:
            done = os.path.exists(os.path.join(args.results, j["name"], "DONE"))
            print(("done   " if done else "pending"), j["name"], j["exp"], j.get("model", "-"))
        return

    import torch
    from fsa import experiments, toy
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log("GPU:", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")

    status_path = os.path.join(args.results, "queue_status.json")
    os.makedirs(args.results, exist_ok=True)
    status = json.load(open(status_path)) if os.path.isfile(status_path) else {}
    engine = tok = cur_key = None
    for j in jobs:
        j = {**defaults, **j}
        out_dir = os.path.join(args.results, j["name"])
        if os.path.exists(os.path.join(out_dir, "DONE")):
            log("skip (done):", j["name"])
            if not str(status.get(j["name"], "")).startswith("done"):
                status[j["name"]] = "done"
            continue
        t0 = time.time()
        try:
            if j["exp"] == "toy":
                if engine is not None:                       # free the LLM before the toy fills the GPU
                    engine.close(); del engine, tok
                    engine = tok = cur_key = None
                    if args.device == "cuda":
                        torch.cuda.empty_cache()
                toy.run_toy(out_dir, device=args.device, log=log,
                            **{k: v for k, v in j.items() if k not in ("name", "exp", "model")})
            else:
                key = model_key(j)
                if key != cur_key:
                    if engine is not None:
                        engine.close()
                        engine = tok = cur_key = None
                        if args.device == "cuda":
                            torch.cuda.empty_cache()
                    log("loading", j["model"], "shuffle_seed=", j.get("shuffle_seed"))
                    engine, tok = experiments.make_engine(
                        j["model"], perturb_layer=j.get("perturb_layer", 2), measure_layer=j.get("measure_layer"),
                        batch=j.get("batch", 4096), mode=j.get("mode", "auto"), shuffle_seed=j.get("shuffle_seed"),
                        device=args.device, dtype=j.get("dtype", "bfloat16"), log=log)
                    cur_key = key
                kw = {k: v for k, v in j.items() if k not in ("name", "exp", "model", "perturb_layer",
                                                              "measure_layer", "batch", "mode", "shuffle_seed", "dtype")}
                experiments.EXPERIMENTS[j["exp"]](engine, tok, out_dir, log=log, **kw)
            open(os.path.join(out_dir, "DONE"), "w").write(time.strftime("%F %T"))
            status[j["name"]] = f"done in {(time.time() - t0) / 60:.1f} min"
            log("finished", j["name"], status[j["name"]])
        except Exception:
            status[j["name"]] = "FAILED"
            log("FAILED", j["name"], "\n" + traceback.format_exc())
            if engine is not None:
                engine._inj = engine._meas = None
            if args.device == "cuda":
                torch.cuda.empty_cache()
        save_status(status_path, status)
    save_status(status_path, status)
    log("queue finished:", json.dumps(status))
    return int(any(status.get(j["name"]) == "FAILED" for j in jobs))


if __name__ == "__main__":
    sys.exit(main())
