"""One-command runner for the lab GPU machine.

Run from the repository root with:  python runviffd.py
(The name has no underscores so it can be typed through Remote Desktop.)

Steps, each logged to outputs/runviffd_log.txt and skipped when already done:
  1. Write the VIFFD manifest and download the selected test videos (parallel ranged downloads).
  2. Evaluate SSIM, the 10-epoch proposed model, the 20-epoch proposed model and the 3D-CNN baseline on VIFFD.
  3. Regenerate ROC/PR curves, cross-action and robustness results for the 10-epoch model,
     without overwriting the existing 20-epoch outputs.
  4. Compute McNemar's test (10-epoch model vs baseline) from the saved ablation predictions.
  5. Pack every new result into ../fvresults.tar for transfer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "outputs" / "runviffd_log.txt"
PY = sys.executable
UA = {"User-Agent": "forensic-vision-dataset-tool"}
CKPT_10EP = "outputs/ablation/variant_e_10ep/best.pt"
CKPT_20EP = "outputs/ablation/variant_e/best.pt"
CKPT_BASE = "outputs/ablation/variant_a/best.pt"


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(args: list[str]) -> bool:
    log("RUN " + " ".join(args))
    result = subprocess.run([PY, *args], cwd=ROOT)
    log(f"exit code {result.returncode}")
    return result.returncode == 0


# ---------------------------------------------------------------- downloads
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_range(url: str, piece: Path, start: int, end: int) -> None:
    for _ in range(6):
        have = piece.stat().st_size if piece.exists() else 0
        if have == end - start + 1:
            return
        try:
            request = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start + have}-{end}"})
            with urllib.request.urlopen(request, timeout=120) as response, piece.open("ab") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
        except Exception as exc:  # retried below
            log(f"  retry range {start}: {exc}")
    if not piece.exists() or piece.stat().st_size != end - start + 1:
        raise RuntimeError(f"range {start}-{end} incomplete")


def download_one(row: dict, root: Path, parts: int = 6) -> str:
    target = root / row["local_path"]
    if target.exists() and sha256(target) == row["sha256"]:
        return "present"
    url = row["download_url"]
    if not url.startswith("https://data.mendeley.com/public-files/datasets/"):
        raise ValueError("unexpected download host")
    size = int(row["size_bytes"])
    step = -(-size // parts)
    spans = [(start, min(start + step, size) - 1) for start in range(0, size, step)]
    pieces = [target.with_name(f"{target.name}.r{start}") for start, _ in spans]
    target.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(len(spans)) as pool:
        list(pool.map(lambda item: fetch_range(url, item[0], *item[1]), zip(pieces, spans)))
    joined = target.with_name(f"{target.name}.joined")
    digest = hashlib.sha256()
    with joined.open("wb") as out:
        for piece in pieces:
            data = piece.read_bytes()
            digest.update(data)
            out.write(data)
    for piece in pieces:
        piece.unlink()
    if digest.hexdigest() != row["sha256"]:
        joined.unlink()
        raise RuntimeError("SHA-256 mismatch")
    joined.replace(target)
    return "downloaded"


def download_viffd() -> bool:
    if not run(["scripts/download_viffd.py", "--manifest-only"]):
        return False
    root = ROOT / "data" / "viffd"
    with (root / "viffd_test_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    failures = 0

    def task(row: dict) -> None:
        nonlocal failures
        try:
            status = download_one(row, root)
        except Exception as exc:
            failures += 1
            status = f"FAILED ({exc})"
        log(f"  {status}: {row['filename']}")

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(task, rows))
    log(f"VIFFD download finished with {failures} failures")
    return failures == 0


# ---------------------------------------------------------------- evaluations
def evaluate_viffd() -> None:
    jobs = [
        ("outputs/reports_viffd", ["--method", "ssim"]),
        ("outputs/reports_viffd_10ep", ["--method", "model", "--checkpoint", CKPT_10EP, "--arch", "variant_e"]),
        ("outputs/reports_viffd_20ep", ["--method", "model", "--checkpoint", CKPT_20EP, "--arch", "variant_e"]),
        ("outputs/reports_viffd_baseline", ["--method", "model", "--checkpoint", CKPT_BASE, "--arch", "variant_a"]),
    ]
    for out_dir, extra in jobs:
        method = extra[1]
        if (ROOT / out_dir / f"viffd_{method}_metrics.json").exists():
            log(f"skip (done): {out_dir}")
            continue
        run(["scripts/evaluate_viffd.py", *extra, "--output-dir", out_dir])


def regenerate_10ep_figures() -> None:
    """generate_roc_auc_evaluation.py writes to fixed folders, so back them up and restore them."""
    target = ROOT / "outputs" / "reports_10ep_analysis"
    if (target / "comprehensive_evaluation_metrics.json").exists():
        log("skip (done): 10-epoch figures")
        return
    watched = [ROOT / "outputs" / "reports_enhanced", ROOT / "docs" / "figures"]
    backup = ROOT / "outputs" / "runviffd_backup"
    if backup.exists():
        shutil.rmtree(backup)
    for folder in watched:
        if folder.exists():
            shutil.copytree(folder, backup / folder.name)
    try:
        run(["scripts/generate_roc_auc_evaluation.py", "--checkpoint", CKPT_10EP])
        target.mkdir(parents=True, exist_ok=True)
        for name in [
            "comprehensive_evaluation_metrics.json",
            "roc_auc_curve.png",
            "precision_recall_curve.png",
            "action_domain_generalization.png",
            "robustness_perturbation_curves.png",
        ]:
            for folder in watched:
                if (folder / name).exists():
                    shutil.copy2(folder / name, target / name)
    finally:
        for folder in watched:
            saved = backup / folder.name
            if saved.exists():
                shutil.rmtree(folder, ignore_errors=True)
                shutil.copytree(saved, folder)
        shutil.rmtree(backup, ignore_errors=True)
        log("restored original reports_enhanced and docs/figures")


def mcnemar_from_predictions() -> None:
    from scipy.stats import binomtest, chi2

    def load(path: str) -> dict[str, bool]:
        with (ROOT / path).open(newline="", encoding="utf-8") as handle:
            return {r["sample_id"]: r["target_index"] == r["pred_index"] for r in csv.DictReader(handle)}

    base = load("outputs/ablation/variant_a/test_predictions.csv")
    prop = load("outputs/ablation/variant_e_10ep/test_predictions.csv")
    if set(base) != set(prop):
        log("McNemar: sample ids differ between prediction files; skipped")
        return
    both = sum(base[k] and prop[k] for k in base)
    only_base = sum(base[k] and not prop[k] for k in base)
    only_prop = sum(prop[k] and not base[k] for k in base)
    neither = sum(not base[k] and not prop[k] for k in base)
    discordant = only_base + only_prop
    statistic = (abs(only_base - only_prop) - 1) ** 2 / discordant if discordant else 0.0
    result = {
        "comparison": "variant_a (3D-CNN baseline) vs variant_e_10ep (proposed, 10 epochs)",
        "n": len(base),
        "contingency_table": [[both, only_base], [only_prop, neither]],
        "baseline_correct": both + only_base,
        "proposed_correct": both + only_prop,
        "chi2_continuity_corrected": statistic,
        "p_value_chi2": float(chi2.sf(statistic, 1)) if discordant else 1.0,
        "p_value_exact": float(binomtest(only_prop, discordant, 0.5).pvalue) if discordant else 1.0,
    }
    out = ROOT / "outputs" / "reports_10ep_analysis" / "mcnemar_10ep_vs_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"McNemar: {json.dumps(result)}")


def pack_results() -> None:
    archive = ROOT.parent / "fvresults.tar"
    with tarfile.open(archive, "w") as tar:
        for name in [
            "outputs/reports_viffd",
            "outputs/reports_viffd_10ep",
            "outputs/reports_viffd_20ep",
            "outputs/reports_viffd_baseline",
            "outputs/reports_10ep_analysis",
            "outputs/runviffd_log.txt",
        ]:
            if (ROOT / name).exists():
                tar.add(ROOT / name, arcname=name)
    log(f"Packed results: {archive} ({archive.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    log(f"Python {sys.version.split()[0]}")
    try:
        import torch

        log(f"CUDA available: {torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    except Exception as exc:
        log(f"torch import failed: {exc}")
    for step in (mcnemar_from_predictions, regenerate_10ep_figures, download_viffd, evaluate_viffd):
        try:
            step()
        except Exception as exc:
            log(f"STEP FAILED {step.__name__}: {exc!r}")
    pack_results()
    log("ALL DONE")


if __name__ == "__main__":
    main()
