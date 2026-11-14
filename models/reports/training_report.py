#!/usr/bin/env python3
"""Summarize per-model, per-fold resource usage and training progress from
Slurm accounting (sacct) plus slurm/logs/.

train.sbatch/submit_all.sh submit one 5-fold array job per model, named after
the model (--job-name="$MODEL"), writing slurm/logs/<model>_fold<N>_<jobid>.out
/.err. Job history comes from sacct (finds every attempt on record - OOM/failed
first tries included, not just the latest successful one); resource numbers
(elapsed, GPU type, peak RAM) come from sacct too. Log files are matched up via
sacct's JobIDRaw (the real %j used in the log filename, distinct from the
"array_master_task" id sacct prints by default) - but log files get
overwritten/deleted across retries, so not every sacct record has one on disk
any more. Use --require-log to only show rows that still have a log to inspect.

Early-stopping / last-epoch detection is log-text parsing, not framework
introspection, and each model logs training progress differently:
  - gears/sclambda: no early-stopping mechanism at all (gears/gears.py's
    train() and sclambda/model.py's Model.train() both always run the full
    fixed epoch count).
  - scgpt: DOES implement its own early stopping (train_predict.py's own
    patience=10 loop, EARLY_STOP=10) - distinct from Lightning's, logs
    "early stop at epoch N" if it ever triggers (it didn't in any fold seen
    so far - all 5 ran the full fixed 15 epochs).
  - state/presage: use PyTorch Lightning's EarlyStopping callback, which
    prints a distinct "Signaling Trainer to stop" line when triggered - a
    reliable yes/no signal, and pairs with Lightning's per-epoch progress-bar
    lines for "last epoch reached".
  - tabicl: no epoch-based training loop appears in its logs at all (n/a) -
    it's a frozen pretrained foundation model, no per-fold training happens.

None of the 6 models print their *configured* ceiling anywhere in logs -
CONFIGURED_MAX below is read directly out of each model's own train_predict.py
(its --epochs/--max-steps argparse default), not derived from logs. state's
ceiling is in *steps* (--max-steps, default 10000), not epochs - its own logs
only expose Lightning's derived per-epoch progress-bar counter, not steps
directly, so its "reached" number is approximated from val-check bookkeeping
(the val_check_interval Trainer was built with, times how many validation
checks ran before EarlyStopping's patience - both of those numbers are
themselves read straight out of the log text, not guessed).

Usage:
    reports/training_report.py                       # last 2 days, implemented perturbation models
    reports/training_report.py --since 2026-07-17
    reports/training_report.py --models gears scgpt
    reports/training_report.py --require-log          # only rows with a surviving log file

Note: sacct's accounting history has a finite retention window (cluster-
configured, commonly 30-90 days) - this only works for runs still in it.
"""
import argparse
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from utils.count_params import count_params as count_model_params
from utils.count_params import find_checkpoint

DEFAULT_MODELS = ["gears", "presage", "scgpt", "sclambda", "tabicl"]
FIELDS = ["JobID", "JobIDRaw", "JobName", "State", "Elapsed", "NodeList", "AllocTRES", "MaxRSS", "ReqMem"]
MODELS_DIR = Path(__file__).resolve().parent.parent  # reports/ -> models/
LOGS_DIR = MODELS_DIR / "slurm" / "logs"

# Best-effort: how each model prints "an epoch finished" in slurm/logs/*.out|.err.
EPOCH_PATTERNS = {
    "gears": re.compile(r"^Epoch (\d+): Train Overall MSE", re.MULTILINE),
    "scgpt": re.compile(r"epoch (\d+) done in", re.IGNORECASE),
    "sclambda": re.compile(r"Epoch (\d+) complete!"),
    "presage": re.compile(r"Epoch (\d+):"),  # Lightning progress-bar line
    "state": re.compile(r"Epoch (\d+):"),  # Lightning progress-bar line
    "tabicl": None,  # no epoch-based training loop observed in tabicl logs
}
# Two different early-stopping signals: Lightning's EarlyStopping(verbose=True)
# message (state/presage) and scgpt's own patience loop's log line - see docstring.
EARLY_STOP_PATTERNS = [
    re.compile(r"did not improve in the last \d+ records.*Signaling Trainer to stop", re.IGNORECASE),
    re.compile(r"early stop at epoch \d+", re.IGNORECASE),
]
# state-only: approximating steps reached from Lightning's own bookkeeping text.
VAL_CHECK_INTERVAL_RE = re.compile(r"'val_check_interval':\s*(\d+)")
PATIENCE_RE = re.compile(r"did not improve in the last (\d+) records")
IMPROVED_RE = re.compile(r"Metric \S+ improved")

# Configured training ceiling per model, read directly out of each model's own
# train_predict.py --epochs/--max-steps argparse default (see module docstring) -
# not derivable from logs. state's is in steps, everyone else's in epochs.
CONFIGURED_MAX = {
    "gears": (20, "epochs"),
    "scgpt": (15, "epochs"),
    "sclambda": (200, "epochs"),
    "presage": (1000, "epochs"),
    "state": (10000, "steps"),
    "tabicl": (None, None),
}


def run_sacct(models, since):
    cmd = [
        "sacct",
        f"--name={','.join(models)}",
        f"-S{since}",
        "--units=G",
        "-P",
        f"--format={','.join(FIELDS)}",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    lines = out.splitlines()
    header, rows = lines[0].split("|"), [line.split("|") for line in lines[1:]]
    return [dict(zip(header, r)) for r in rows]


def gpu_type(alloc_tres):
    for part in alloc_tres.split(","):
        if part.startswith("gres/gpu:"):
            return part.split(":", 1)[1].split("=")[0]
    return "?"


def pct(maxrss, reqmem):
    try:
        return f"{float(maxrss.rstrip('G')) / float(reqmem.rstrip('Gn')) * 100:.0f}%"
    except (ValueError, ZeroDivisionError):
        return "-"


def elapsed_to_seconds(elapsed):
    h, m, s = map(int, elapsed.split(":"))
    return h * 3600 + m * 60 + s


def seconds_to_hm(seconds):
    h, rem = divmod(round(seconds), 3600)
    m = rem // 60
    return f"{h}h {m}m"


def params_to_millions(n):
    return f"{n / 1e6:.1f}M"


def find_logs(model, fold, jobid_raw):
    stem = LOGS_DIR / f"{model}_fold{fold}_{jobid_raw}"
    out, err = stem.with_suffix(".out"), stem.with_suffix(".err")
    return out if out.exists() else None, err if err.exists() else None


def analyze_progress(model, out_path, err_path):
    """Returns (last_epoch, early_stop, reached, is_approx) - reached is a raw int
    (in CONFIGURED_MAX[model]'s unit) suitable for averaging across folds; format it
    against CONFIGURED_MAX yourself (see progress_str)."""
    text = "".join(p.read_text(errors="ignore") for p in (out_path, err_path) if p is not None)
    if not text:
        return "-", "-", None, False
    pattern = EPOCH_PATTERNS.get(model)
    last_epoch = "-"
    if pattern:
        matches = pattern.findall(text)
        if matches:
            last_epoch = str(max(int(m) for m in matches))
    early_stop = "yes" if any(p.search(text) for p in EARLY_STOP_PATTERNS) else ("no" if last_epoch != "-" else "n/a")

    total, _unit = CONFIGURED_MAX.get(model, (None, None))
    if total is None:
        return last_epoch, early_stop, None, False
    if model == "state":
        # ceiling is in steps, not the epoch counter above - approximate reached-steps from
        # how many validation checks ran (improved + the patience-length losing streak that
        # triggered the stop) times the val_check_interval Trainer was built with.
        vci_m, pat_m = VAL_CHECK_INTERVAL_RE.search(text), PATIENCE_RE.search(text)
        if early_stop == "yes" and vci_m and pat_m:
            reached = (len(IMPROVED_RE.findall(text)) + int(pat_m.group(1))) * int(vci_m.group(1))
            return last_epoch, early_stop, reached, True
        return last_epoch, early_stop, (total if early_stop == "no" else None), False
    if last_epoch != "-":
        return last_epoch, early_stop, int(last_epoch), False
    return last_epoch, early_stop, None, False


def progress_str(model, reached, is_approx):
    total, unit = CONFIGURED_MAX.get(model, (None, None))
    if total is None or reached is None:
        return "n/a" if total is None else "-"
    prefix = "~" if is_approx else ""
    return f"{prefix}{reached:.0f}/{total} {unit}"


SUMMARY_COLUMNS = [
    ("model", "Model"), ("params", "Params"), ("gpu", "GPU"), ("folds", "Folds"),
    ("avg_elapsed", "Avg Elapsed"), ("avg_maxrss", "Avg MaxRSS"),
    ("avg_progress", "Avg Progress"), ("early_stop", "Early Stopped"),
]


def build_summary_rows(args, by_model):
    """One row per (model, gpu type) - completed runs only. Grouping order matters: rows for
    a given model are emitted consecutively, which write_html relies on for its rowspan merge."""
    rows = []
    for model in args.models:
        completed = [r for r in by_model.get(model, []) if r["state"] == "COMPLETED"]
        if not completed:
            continue
        ckpt = find_checkpoint(model, args.dataset)
        params_str = params_to_millions(count_model_params(ckpt)[0]) if ckpt else "-"
        by_gpu = defaultdict(list)
        for r in completed:
            by_gpu[r.get("gpu", "?")].append(r)
        for gpu in sorted(by_gpu):
            group = by_gpu[gpu]
            avg_elapsed_s = sum(elapsed_to_seconds(r["elapsed"]) for r in group) / len(group)
            rss_vals = [float(r["maxrss"].rstrip("G")) for r in group if r.get("maxrss", "-") not in ("-", "0")]
            avg_rss = f"{sum(rss_vals) / len(rss_vals):.1f}G" if rss_vals else "-"
            stops = {r["early_stop"] for r in group}
            stop_str = stops.pop() if len(stops) == 1 else "/".join(sorted(stops))
            reached_vals = [r["reached"] for r in group if r["reached"] is not None]
            avg_reached = sum(reached_vals) / len(reached_vals) if reached_vals else None
            avg_progress = progress_str(model, avg_reached, any(r["approx"] for r in group))
            folds_str = ",".join(sorted((r["fold"] for r in group), key=int))
            rows.append({
                "model": model, "params": params_str, "gpu": gpu, "folds": folds_str,
                "avg_elapsed": seconds_to_hm(avg_elapsed_s), "avg_maxrss": avg_rss,
                "avg_progress": avg_progress, "early_stop": stop_str,
            })
    return rows


def print_summary_table(rows):
    print("Per-model x GPU-type summary (completed runs only; MaxRSS = peak host RAM, not GPU memory):")
    if not rows:
        print("(no completed runs)")
        return
    widths = {key: max(len(label), max(len(str(r[key])) for r in rows)) + 2 for key, label in SUMMARY_COLUMNS}
    print("".join(f"{label:<{widths[key]}}" for key, label in SUMMARY_COLUMNS))
    print("-" * sum(widths.values()))
    for r in rows:
        print("".join(f"{r[key]!s:<{widths[key]}}" for key, _ in SUMMARY_COLUMNS))


def write_markdown(rows, path):
    lines = [
        "| " + " | ".join(label for _, label in SUMMARY_COLUMNS) + " |",
        "|" + "|".join("---" for _ in SUMMARY_COLUMNS) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r[key]) for key, _ in SUMMARY_COLUMNS) + " |")
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def write_html(rows, path):
    # Merge ("fuse") the model and params cells via rowspan across a model's consecutive
    # GPU-type rows - the actual reason to pick HTML over Markdown, which has no rowspan.
    span = defaultdict(int)
    for r in rows:
        span[r["model"]] += 1
    lines = [
        "<table>",
        "  <thead><tr>" + "".join(f"<th>{label}</th>" for _, label in SUMMARY_COLUMNS) + "</tr></thead>",
        "  <tbody>",
    ]
    seen = set()
    for r in rows:
        cells = []
        for key, _ in SUMMARY_COLUMNS:
            if key in ("model", "params"):
                if r["model"] in seen:
                    continue  # already rendered on this model's first row, via rowspan
                cells.append(f'<td rowspan="{span[r["model"]]}">{r[key]}</td>')
            else:
                cells.append(f"<td>{r[key]}</td>")
        lines.append("    <tr>" + "".join(cells) + "</tr>")
        seen.add(r["model"])
    lines += ["  </tbody>", "</table>"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default="now-2days", help="sacct -S value, e.g. 2026-07-17 or now-7days (default: now-2days)")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--require-log", action="store_true", help="only show rows whose slurm/logs/ .out or .err file still exists on disk")
    ap.add_argument("--dataset", default="replogle22k562", help="used only to locate each model's checkpoint for the params column")
    ap.add_argument("--out", default="figures/training_summary.html",
                     help="also write the per-model x GPU-type summary table here (relative to models/, same convention as "
                          "plot_training_curves.py's own --out) - .md for Markdown, .html for HTML (HTML merges a model's "
                          "multiple GPU-type rows via rowspan on the model/params cells; Markdown can't). "
                          "Pass '' to skip writing a file. Default: figures/training_summary.html")
    args = ap.parse_args()

    rows = run_sacct(args.models, args.since)

    # Merge each job's top-level record (State/Elapsed/NodeList/ReqMem/JobIDRaw)
    # with its .batch step (MaxRSS/AllocTRES gpu type); .extern carries neither.
    merged = {}
    for r in rows:
        base, _, step = r["JobID"].partition(".")
        rec = merged.setdefault(base, {})
        if step == "":
            rec.update(name=r["JobName"], state=r["State"], elapsed=r["Elapsed"],
                       node=r["NodeList"], reqmem=r["ReqMem"], jobid_raw=r["JobIDRaw"])
        elif step == "batch":
            rec.update(maxrss=r["MaxRSS"] or "-", gpu=gpu_type(r["AllocTRES"]))

    by_model = defaultdict(list)
    for jobid, rec in merged.items():
        if "name" not in rec or rec["name"] not in args.models:
            continue
        rec["fold"] = jobid.rsplit("_", 1)[-1] if "_" in jobid else "?"
        rec["jobid"] = jobid
        out_path, err_path = find_logs(rec["name"], rec["fold"], rec["jobid_raw"])
        rec["log"] = "yes" if (out_path or err_path) else "no"
        if args.require_log and rec["log"] == "no":
            continue
        rec["last_epoch"], rec["early_stop"], rec["reached"], rec["approx"] = analyze_progress(rec["name"], out_path, err_path)
        by_model[rec["name"]].append(rec)

    cols = (f"{'model':<10} {'fold':<5} {'state':<14} {'elapsed':<10} {'gpu':<6} {'maxrss':<8} {'%mem':<6} "
            f"{'log':<4} {'progress':<18} stopped")
    print(cols)
    print("-" * len(cols))
    for model in args.models:
        for rec in sorted(by_model.get(model, []), key=lambda r: (r["fold"], r["jobid"])):
            print(f"{model:<10} {rec['fold']:<5} {rec.get('state','-'):<14} {rec.get('elapsed','-'):<10} "
                  f"{rec.get('gpu','?'):<6} {rec.get('maxrss','-'):<8} "
                  f"{pct(rec.get('maxrss','-'), rec.get('reqmem','-')):<6} "
                  f"{rec['log']:<4} {progress_str(model, rec['reached'], rec['approx']):<18} {rec['early_stop']}")

    summary_rows = build_summary_rows(args, by_model)
    print()
    print_summary_table(summary_rows)

    if args.out:
        summary_out = MODELS_DIR / args.out
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        if summary_out.suffix in (".html", ".htm"):
            write_html(summary_rows, summary_out)
        elif summary_out.suffix in (".md", ".markdown"):
            write_markdown(summary_rows, summary_out)
        else:
            raise SystemExit(f"--out must end in .md or .html, got: {summary_out.suffix or '(no extension)'}")


if __name__ == "__main__":
    main()
