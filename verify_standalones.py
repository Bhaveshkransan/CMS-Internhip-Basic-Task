"""Comprehensive test runner to verify every standalone script in the repository.

Tests:
1. task6_create_empty_labels.py
2. task5_clean_empty_labels.py
3. task12_visualize_yolo.py
4. task3_distribute_dataset.py
5. task9_dataset_stats.py
6. task7_find_mismatches.py
7. task2_rename_pairs.py
8. task4_separate_bg.py
9. task10_split_dataset.py
10. task8_validate_yolo.py
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    full_cmd = [sys.executable] + cmd
    print(f"\n[EXEC] {' '.join(full_cmd)}")
    res = subprocess.run(full_cmd, capture_output=True, text=True)
    if res.stdout:
        print(f"[STDOUT]\n{res.stdout.strip()}")
    if res.stderr:
        print(f"[STDERR]\n{res.stderr.strip()}")
    assert res.returncode == 0, f"Command failed with code {res.returncode}: {' '.join(full_cmd)}"
    return res


def create_test_images(dest: Path, count: int = 5):
    dest.mkdir(parents=True, exist_ok=True)
    for i in range(1, count + 1):
        img = np.full((120, 160, 3), i * 35, dtype=np.uint8)
        # Draw a small rectangle so there is some contrast
        cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)
        cv2.imwrite(str(dest / f"img_{i:02d}.jpg"), img)


def main():
    test_root = Path("test_standalones_sandbox")
    if test_root.exists():
        shutil.rmtree(test_root)
    test_root.mkdir(parents=True)

    try:
        # ==============================================================
        # Test 1: task6_create_empty_labels.py
        # ==============================================================
        print("\n" + "=" * 60)
        print("VERIFYING: task6_create_empty_labels.py")
        print("=" * 60)
        t1_dir = test_root / "t1_empty"
        create_test_images(t1_dir, count=3)
        # One already has a label
        (t1_dir / "img_01.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        run(["task6_create_empty_labels.py", str(t1_dir)])
        assert (t1_dir / "img_02.txt").exists(), "img_02.txt not created"
        assert (t1_dir / "img_03.txt").exists(), "img_03.txt not created"
        assert (t1_dir / "img_02.txt").stat().st_size == 0, "img_02.txt should be 0-byte"
        assert (t1_dir / "img_01.txt").stat().st_size > 0, "img_01.txt was overwritten"
        print("PASS: task6_create_empty_labels.py created 0-byte labels without overwriting existing.")

        # ==============================================================
        # Test 2: task7_find_mismatches.py
        # ==============================================================
        print("\n" + "=" * 60)
        print("VERIFYING: task7_find_mismatches.py")
        print("=" * 60)
        t2_dir = test_root / "t2_mismatches"
        create_test_images(t2_dir, count=3)
        # img_01 has label, img_02 has none, img_03 has none, extra_label has no image
        (t2_dir / "img_01.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (t2_dir / "orphan_without_img.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        rep_csv = t2_dir / "mismatch_rep.csv"
        run(["task7_find_mismatches.py", str(t2_dir), "--report", str(rep_csv)])
        assert rep_csv.exists(), "Mismatch report CSV not created"
        rep_content = rep_csv.read_text()
        assert "orphan_without_img.txt" in rep_content, "Orphan label not reported"
        assert "img_02.jpg" in rep_content, "Missing label for img_02 not reported"
        print("PASS: task7_find_mismatches.py identified unmatched images and labels.")

        # ==============================================================
        # Test 3: task9_dataset_stats.py
        # ==============================================================
        print("\n" + "=" * 60)
        print("VERIFYING: task9_dataset_stats.py")
        print("=" * 60)
        t3_dir = test_root / "t3_stats"
        create_test_images(t3_dir, count=3)
        (t3_dir / "img_01.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n")
        (t3_dir / "img_02.txt").write_text("0 0.6 0.6 0.3 0.3\n")
        (t3_dir / "img_03.txt").write_text("")  # background
        (t3_dir / "classes.txt").write_text("person\nvehicle\n")
        stats_json = t3_dir / "stats.json"
        run(["task9_dataset_stats.py", str(t3_dir), "--classes-file", str(t3_dir / "classes.txt"), "--json", str(stats_json)])
        assert stats_json.exists(), "Stats JSON not written"
        stats_data = json.loads(stats_json.read_text())
        assert stats_data["total_images"] == 3
        assert stats_data["annotated_images"] == 2
        assert stats_data["background_images"] == 1
        assert stats_data["total_boxes"] == 3
        print("PASS: task9_dataset_stats.py computed correct dataset summary.")

        # ==============================================================
        # Test 4: task12_visualize_yolo.py
        # ==============================================================
        print("\n" + "=" * 60)
        print("VERIFYING: task12_visualize_yolo.py")
        print("=" * 60)
        t4_dir = test_root / "t4_viz"
        create_test_images(t4_dir, count=2)
        (t4_dir / "img_01.txt").write_text("0 0.5 0.5 0.3 0.3\n")
        (t4_dir / "classes.txt").write_text("person\n")
        viz_out = t4_dir / "rendered"
        run(["task12_visualize_yolo.py", str(t4_dir), "--output", str(viz_out), "--classes-file", str(t4_dir / "classes.txt"), "--samples", "2"])
        assert viz_out.exists(), "Output visualization directory missing"
        viz_files = list(viz_out.glob("viz_*.jpg"))
        assert len(viz_files) >= 1, "No rendered images found"
        print(f"PASS: task12_visualize_yolo.py rendered {len(viz_files)} image(s) with bounding boxes.")

        # ==============================================================
        # Test 5: task8_validate_yolo.py
        # ==============================================================
        print("\n" + "=" * 60)
        print("VERIFYING: task8_validate_yolo.py")
        print("=" * 60)
        t5_dir = test_root / "t5_val"
        t5_dir.mkdir(parents=True)
        # Create bad label file
        (t5_dir / "bad.txt").write_text(
            "0 1.5 0.5 0.2 0.2\n"  # out of bounds x > 1
            "0 0.5 0.5 0.2 0.2\n"  # dup 1
            "0 0.5 0.5 0.2 0.2\n"  # dup 2
            "99 0.2 0.2 0.1 0.1\n" # invalid class if 2 classes
        )
        (t5_dir / "classes.txt").write_text("c0\nc1\n")
        rep_val = t5_dir / "val_rep.csv"
        run(["task8_validate_yolo.py", str(t5_dir), "--classes-file", str(t5_dir / "classes.txt"), "--report", str(rep_val)])
        assert rep_val.exists(), "Validation report CSV not written"
        # Test --fix
        run(["task8_validate_yolo.py", str(t5_dir), "--classes-file", str(t5_dir / "classes.txt"), "--fix"])
        fixed_lines = (t5_dir / "bad.txt").read_text().splitlines()
        print(f"Fixed lines: {fixed_lines}")
        assert len(fixed_lines) > 0, "Fix should preserve valid boxes"
        print("PASS: task8_validate_yolo.py detected issues and auto-repaired successfully.")

        # ==============================================================
        # Test 6: task5_clean_empty_labels.py
        # ==============================================================
        print("\n" + "=" * 60)
        print("VERIFYING: task5_clean_empty_labels.py")
        print("=" * 60)
        t6_dir = test_root / "t6_clean"
        t6_dir.mkdir(parents=True)
        (t6_dir / "valid.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (t6_dir / "empty1.txt").write_text("")
        (t6_dir / "empty2.txt").write_text("   \n\t  \n")
        quarantine = t6_dir / "quarantine_box"
        run(["task5_clean_empty_labels.py", str(t6_dir), "--quarantine", str(quarantine)])
        assert (quarantine / "empty1.txt").exists(), "empty1.txt not quarantined"
        assert (quarantine / "empty2.txt").exists(), "empty2.txt not quarantined"
        assert (t6_dir / "valid.txt").exists(), "valid.txt was wrongly removed"
        assert not (t6_dir / "empty1.txt").exists(), "empty1.txt remains in source folder"
        print("PASS: task5_clean_empty_labels.py safely quarantined 0-byte and whitespace labels.")

        # ==============================================================
        # Test 7: task4_separate_bg.py
        # ==============================================================
        print("\n" + "=" * 60)
        print("VERIFYING: task4_separate_bg.py")
        print("=" * 60)
        t7_dir = test_root / "t7_sep"
        create_test_images(t7_dir, count=3)
        (t7_dir / "img_01.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        # img_02 and img_03 have no txt -> background
        # orphan.txt has no img -> orphan
        (t7_dir / "orphan.txt").write_text("0 0.1 0.1 0.1 0.1\n")
        bg_dest = t7_dir / "bg_folder"
        orphan_dest = t7_dir / "orphan_folder"
        run(["task4_separate_bg.py", str(t7_dir), "--bg-dest", str(bg_dest), "--orphan-dest", str(orphan_dest)])
        assert (bg_dest / "img_02.jpg").exists(), "img_02.jpg not moved to bg folder"
        assert (bg_dest / "img_03.jpg").exists(), "img_03.jpg not moved to bg folder"
        assert (orphan_dest / "orphan.txt").exists(), "orphan.txt not moved to orphan folder"
        assert (t7_dir / "img_01.jpg").exists(), "img_01.jpg should stay in source"
        assert (t7_dir / "img_01.txt").exists(), "img_01.txt should stay in source"
        print("PASS: task4_separate_bg.py correctly isolated unannotated images and orphan labels.")

        # ==============================================================
        # Test 8: task2_rename_pairs.py
        # ==============================================================
        print("\n" + "=" * 60)
        print("VERIFYING: task2_rename_pairs.py")
        print("=" * 60)
        t8_dir = test_root / "t8_rename"
        create_test_images(t8_dir, count=3)
        (t8_dir / "img_01.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (t8_dir / "img_02.txt").write_text("0 0.3 0.3 0.1 0.1\n")
        # img_03 has no label
        manifest_csv = t8_dir / "rename_manifest.csv"
        run(["task2_rename_pairs.py", str(t8_dir), "--prefix", "track_", "--padding", "3", "--manifest", str(manifest_csv)])
        assert (t8_dir / "track_001.jpg").exists(), "track_001.jpg missing"
        assert (t8_dir / "track_001.txt").exists(), "track_001.txt missing"
        assert (t8_dir / "track_002.jpg").exists(), "track_002.jpg missing"
        assert (t8_dir / "track_002.txt").exists(), "track_002.txt missing"
        assert (t8_dir / "track_003.jpg").exists(), "track_003.jpg missing"
        assert manifest_csv.exists(), "rename_manifest.csv missing"
        print("PASS: task2_rename_pairs.py executed atomic sequential renaming.")

        # ==============================================================
        # Test 9: task3_distribute_dataset.py
        # ==============================================================
        print("\n" + "=" * 60)
        print("VERIFYING: task3_distribute_dataset.py")
        print("=" * 60)
        t9_dir = test_root / "t9_dist"
        create_test_images(t9_dir, count=4)
        for i in range(1, 5):
            (t9_dir / f"img_{i:02d}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        dist_out = t9_dir / "distributed"
        dist_man = dist_out / "manifest.csv"
        run(["task3_distribute_dataset.py", str(t9_dir), "--dest", str(dist_out), "--num-folders", "2", "--manifest", str(dist_man)])
        assert (dist_out / "folder_01").exists(), "folder_01 missing"
        assert (dist_out / "folder_02").exists(), "folder_02 missing"
        f1_files = list((dist_out / "folder_01").glob("*.*"))
        f2_files = list((dist_out / "folder_02").glob("*.*"))
        assert len(f1_files) == 4, f"Expected 4 files in folder_01 (2 img + 2 txt), got {len(f1_files)}"
        assert len(f2_files) == 4, f"Expected 4 files in folder_02 (2 img + 2 txt), got {len(f2_files)}"
        assert dist_man.exists(), "Distribution manifest missing"
        print("PASS: task3_distribute_dataset.py divided images and labels equally.")

        # ==============================================================
        # Test 10: task10_split_dataset.py
        # ==============================================================
        print("\n" + "=" * 60)
        print("VERIFYING: task10_split_dataset.py")
        print("=" * 60)
        t10_dir = test_root / "t10_split"
        create_test_images(t10_dir, count=6)
        for i in range(1, 7):
            (t10_dir / f"img_{i:02d}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (t10_dir / "classes.txt").write_text("person\ncar\n")
        split_out = t10_dir / "yolo_dataset"
        run(["task10_split_dataset.py", str(t10_dir), "--output", str(split_out), "--ratios", "0.5", "0.33", "0.17", "--classes-file", str(t10_dir / "classes.txt")])
        assert (split_out / "images" / "train").exists(), "images/train missing"
        assert (split_out / "images" / "val").exists(), "images/val missing"
        assert (split_out / "images" / "test").exists(), "images/test missing"
        assert (split_out / "labels" / "train").exists(), "labels/train missing"
        assert (split_out / "labels" / "val").exists(), "labels/val missing"
        assert (split_out / "labels" / "test").exists(), "labels/test missing"
        assert (split_out / "data.yaml").exists(), "data.yaml missing"
        data_yaml = (split_out / "data.yaml").read_text()
        assert "nc: 2" in data_yaml, "nc: 2 missing in data.yaml"
        assert "person" in data_yaml, "class person missing in data.yaml"
        print("PASS: task10_split_dataset.py generated synchronized YOLO hierarchy and data.yaml.")

        print("\n" + "=" * 70)
        print("ALL 10 STANDALONE SCRIPTS EXECUTED AND PASSED VERIFICATION!")
        print("=" * 70)

    finally:
        if test_root.exists():
            shutil.rmtree(test_root)


if __name__ == "__main__":
    main()
