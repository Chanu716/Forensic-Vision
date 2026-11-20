"""Tests for VIFFD selection, video-level decisions and the evaluation flow."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from forensic_vision.datasets.viffd import (
    VIFFD_DOWNLOAD_PREFIX,
    VIFFD_TEST_FOLDER_ID,
    classify_filename,
    find_source_tokens,
    parse_deletion_range,
    select_test_videos,
    write_viffd_manifests,
)
from forensic_vision.evaluation.video_level import decide_video_label, summarize_video_predictions
from forensic_vision.models.three_d_cnn import build_model
from forensic_vision.preprocessing.video_io import write_video_frames
from scripts.evaluate_viffd import build_report, evaluate_videos, load_manifest, make_clip_predictor


def metadata_entry(filename: str, marker: str, folder_id: str = VIFFD_TEST_FOLDER_ID) -> dict:
    return {
        "filename": filename,
        "folder_id": folder_id,
        "size": 1000,
        "content_details": {
            "sha256_hash": marker * 64,
            "download_url": f"{VIFFD_DOWNLOAD_PREFIX}r3ss3v53sj/files/{marker}/file_downloaded",
        },
    }


class TestViffdFilenames(unittest.TestCase):
    def test_classification(self) -> None:
        self.assertEqual(classify_filename("insert 1-125 a to b.avi"), "frame_insertion")
        self.assertEqual(classify_filename("delete 25 - 75 huabin walk2.avi"), "frame_deletion")
        self.assertEqual(classify_filename("copy 1-25 to 26-50 x.mp4"), "frame_duplication")
        self.assertEqual(classify_filename("finger1.mp4"), "authentic")

    def test_deletion_range_is_zero_based(self) -> None:
        self.assertEqual(parse_deletion_range("delete 25 -75 x.mp4"), (24, 74))
        self.assertEqual(parse_deletion_range("delete 100-125haiyong walk2_1.mp4"), (99, 124))
        self.assertIsNone(parse_deletion_range("delete 75-10 x.mp4"))
        self.assertIsNone(parse_deletion_range("finger1.mp4"))

    def test_source_tokens_handle_real_name_variants(self) -> None:
        self.assertEqual(
            find_source_tokens("insert 100-end haiyong ball2_1to nemo ball 4.avi"),
            ["haiyongball2", "nemoball4"],
        )
        self.assertEqual(
            find_source_tokens("insert 1-125 nemo ball4_1 to wemen ball 7.avi"),
            ["nemoball4", "womenball7"],
        )
        self.assertEqual(
            find_source_tokens("insert 1- 475 huabin walk2_1 to dalman run3.avi"),
            ["huabinwalk2", "salmanrun3"],
        )
        self.assertEqual(find_source_tokens("salman walk8.avi"), ["salmanwalk7"])
        self.assertEqual(find_source_tokens("cannon ixus boy walk1 (10).avi"), ["canon"])


class TestViffdSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = {
            "files": [
                metadata_entry("salman walk7.avi", "a"),
                metadata_entry("salman walk8.avi", "a"),
                metadata_entry("salman run3.avi", "b"),
                metadata_entry("insert 1-50 salman run3 to salman walk 7.avi", "c"),
                metadata_entry("delete 10-20 finger1.mp4", "d"),
                metadata_entry("delete 1 - 100 to black finger1.avi", "e"),
                metadata_entry("copy 1-20 to 30-50 finger1.mp4", "f"),
                metadata_entry("training only.avi", "1", folder_id="another-folder"),
            ]
        }

    def test_selection_removes_duplicates_and_out_of_scope_files(self) -> None:
        videos, excluded = select_test_videos(self.metadata)
        self.assertEqual(
            sorted(video.filename for video in videos),
            sorted(
                [
                    "salman walk7.avi",
                    "salman run3.avi",
                    "insert 1-50 salman run3 to salman walk 7.avi",
                    "delete 10-20 finger1.mp4",
                    "delete 1 - 100 to black finger1.avi",
                ]
            ),
        )
        reasons = {item.filename: item.reason for item in excluded}
        self.assertEqual(reasons["salman walk8.avi"], "exact_duplicate_of:salman walk7.avi")
        self.assertEqual(reasons["copy 1-20 to 30-50 finger1.mp4"], "class_out_of_scope")

    def test_related_videos_share_a_cluster(self) -> None:
        videos = {video.filename: video for video in select_test_videos(self.metadata)[0]}
        insertion = videos["insert 1-50 salman run3 to salman walk 7.avi"]
        self.assertEqual(insertion.cluster_id, videos["salman walk7.avi"].cluster_id)
        self.assertEqual(insertion.cluster_id, videos["salman run3.avi"].cluster_id)
        self.assertNotEqual(insertion.cluster_id, videos["delete 10-20 finger1.mp4"].cluster_id)

    def test_only_clean_deletions_are_localization_eligible(self) -> None:
        videos = {video.filename: video for video in select_test_videos(self.metadata)[0]}
        self.assertTrue(videos["delete 10-20 finger1.mp4"].localization_eligible)
        self.assertEqual(videos["delete 10-20 finger1.mp4"].gt_start, 9)
        self.assertFalse(videos["delete 1 - 100 to black finger1.avi"].localization_eligible)
        self.assertFalse(videos["insert 1-50 salman run3 to salman walk 7.avi"].localization_eligible)

    def test_local_paths_ignore_remote_filenames(self) -> None:
        self.metadata["files"].append(metadata_entry("../../outside.avi", "9"))
        videos = select_test_videos(self.metadata)[0]
        outside = next(video for video in videos if video.filename == "../../outside.avi")
        self.assertEqual(outside.local_path, f"videos/{'9' * 16}.avi")

    def test_unexpected_download_host_is_rejected(self) -> None:
        self.metadata["files"][0]["content_details"]["download_url"] = "https://example.com/a.avi"
        with self.assertRaises(ValueError):
            select_test_videos(self.metadata)


class TestVideoDecisions(unittest.TestCase):
    def test_decision_rule_branches(self) -> None:
        insertion = np.array([[0.2, 0.7, 0.1], [0.9, 0.05, 0.05]])
        deletion = np.array([[0.1, 0.1, 0.8], [0.9, 0.05, 0.05]])
        weak_deletion = np.array([[0.85, 0.03, 0.12], [0.9, 0.05, 0.05]])
        quiet = np.array([[0.95, 0.03, 0.02], [0.9, 0.05, 0.05]])
        self.assertEqual(decide_video_label(insertion, "authentic"), "frame_insertion")
        self.assertEqual(decide_video_label(deletion, "frame_deletion"), "frame_deletion")
        self.assertEqual(decide_video_label(weak_deletion, "frame_deletion"), "frame_deletion")
        self.assertEqual(decide_video_label(weak_deletion, "authentic"), "authentic")
        self.assertEqual(decide_video_label(quiet, "authentic"), "authentic")

    def test_summary_counts_and_clustered_interval(self) -> None:
        names = ["authentic", "frame_insertion", "frame_deletion"]
        y_true = names * 4
        y_pred = names * 3 + ["authentic", "frame_insertion", "authentic"]
        clusters = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        summary = summarize_video_predictions(y_true, y_pred, names, clusters, n_bootstrap=50)
        self.assertEqual(summary["correct"], 11)
        self.assertEqual(summary["num_source_clusters"], 4)
        self.assertEqual(summary["per_class"]["frame_deletion"]["support"], 4)
        self.assertEqual(summary["confusion_matrix"][2], [1, 0, 3])
        self.assertEqual(len(summary["bootstrap_ci_95_by_source_cluster"]["accuracy"]), 2)


class TestViffdEvaluationFlow(unittest.TestCase):
    def _write_dataset(self, root: Path) -> Path:
        rng = np.random.default_rng(0)
        names = {"a": "finger1.avi", "b": "insert 1-5 finger1 to baby 3.avi", "c": "delete 6-9 baby 3.avi"}
        metadata = {"files": [metadata_entry(name, marker) for marker, name in names.items()]}
        videos, excluded = select_test_videos(metadata)
        manifest_path, _ = write_viffd_manifests(videos, excluded, root)
        for video in videos:
            frames = rng.integers(0, 255, size=(14, 32, 32, 3), dtype=np.uint8)
            write_video_frames(frames, root / video.local_path, 10.0)
        return manifest_path

    def test_manifest_round_trip_and_model_flow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._write_dataset(Path(directory))
            videos = load_manifest(manifest_path)
            self.assertEqual(len(videos), 3)
            model = build_model(arch="3dcnn", in_channels=3, num_classes=3).eval()
            rows = evaluate_videos(
                videos=videos,
                method="model",
                image_size=(32, 32),
                clip_length=9,
                clip_stride=4,
                localization_cfg={"use_multiscale": False},
                predictor=make_clip_predictor(model, torch.device("cpu"), batch_size=2),
            )
            self.assertTrue(all(row["status"] == "evaluated" for row in rows))
            self.assertTrue(all(row["num_clips"] == 3 for row in rows))
            report = build_report(rows, "model", seed=0)
            self.assertEqual(report["num_evaluated_videos"], 3)
            self.assertIn("clip_mean", report["decisions"])
            self.assertEqual(report["pipeline_by_container"]["avi"]["num_videos"], 3)

    def test_short_videos_are_reported_not_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._write_dataset(Path(directory))
            rows = evaluate_videos(
                videos=load_manifest(manifest_path),
                method="model",
                image_size=(32, 32),
                clip_length=49,
                clip_stride=24,
                localization_cfg={"use_multiscale": False},
                predictor=lambda clips: np.zeros((len(clips), 3)),
            )
            self.assertEqual({row["status"] for row in rows}, {"too_short"})

    def test_manifest_csv_contains_no_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._write_dataset(Path(directory))
            with manifest_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(row["local_path"].startswith("videos/") for row in rows))


if __name__ == "__main__":
    unittest.main()
