"""
Unit tests for src/data_loader.py
"""

import unittest
import io
import pandas as pd
from pathlib import Path
from src.data_loader import (
    check_dataset_exists,
    load_dataset,
    get_dataset_download_instructions,
    DEFAULT_DATA_FILENAME
)


class TestDataLoader(unittest.TestCase):

    def test_dataset_missing_check(self):
        """Verify check_dataset_exists returns False for non-existent file."""
        non_existent = Path("data/non_existent_file_xyz_123.csv")
        self.assertFalse(check_dataset_exists(non_existent))

    def test_load_dataset_missing_raises_error(self):
        """Verify load_dataset raises FileNotFoundError when path is missing."""
        with self.assertRaises(FileNotFoundError):
            load_dataset("data/definitely_not_here.csv")

    def test_load_dataset_from_stream(self):
        """Verify safe loading and schema detection from in-memory CSV stream."""
        csv_content = """id, dur , proto , spkts , dpkts , label , attack_cat
1, 0.12, tcp, 10, 8, 0, Normal
2, 0.45, udp, 4, 2, 1, Exploits
3, 0.05, tcp, 20, 15, 1, DoS
4, 0.88, tcp, 2, 1, 0, Normal
"""
        stream = io.StringIO(csv_content)
        df, meta = load_dataset(stream)

        self.assertEqual(len(df), 4)
        self.assertTrue(meta["has_label"])
        self.assertTrue(meta["has_attack_cat"])
        # Columns should be stripped of whitespace
        self.assertIn("dur", df.columns)
        self.assertIn("proto", df.columns)
        self.assertEqual(meta["label_col"], "label")

    def test_load_dataset_without_target(self):
        """Verify metadata handles CSV lacking the 'label' column gracefully."""
        csv_content = """dur, proto, spkts
0.12, tcp, 10
0.45, udp, 4
"""
        stream = io.StringIO(csv_content)
        df, meta = load_dataset(stream)

        self.assertFalse(meta["has_label"])
        self.assertIsNone(meta["label_col"])
        self.assertEqual(len(df), 2)

    def test_download_instructions(self):
        """Verify setup instructions dictionary contains valid paths and links."""
        instructions = get_dataset_download_instructions()
        self.assertEqual(instructions["expected_filename"], DEFAULT_DATA_FILENAME)
        self.assertIn("data", instructions["target_relative_path"])
        self.assertTrue(len(instructions["sources"]) >= 2)


if __name__ == "__main__":
    unittest.main()
