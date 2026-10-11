"""Diphoton plots use stored masses, validity, and explicit pair counting."""

from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.plot_diphoton_mass import diphoton_masses, main


COLUMNS = ["pt_1", "pt_2", "type_id_1", "type_id_2", "pair_mass", "pair_mass_valid"]


def sample():
    return {"pair_obj": torch.tensor([
        [[100, 60, 3, 3, 120, 1], [100, 20, 3, 3, 70, 1], [60, 20, 3, 3, 40, 1]],
        [[50, 30, 3, 3, 90, 1], [50, 30, 0, 3, 80, 1], [50, 30, 3, 3, 999, 1]],
    ], dtype=torch.float32), "pair_obj_columns": COLUMNS,
        "pair_slot_mask": torch.tensor([[True, True, True], [True, True, False]]),
        "label": torch.tensor([2, 1])}


class DiphotonMassTests(unittest.TestCase):
    def test_all_pairs_exclude_padding_and_nonphotons_and_use_saved_masses(self):
        mass, rows = diphoton_masses(sample())
        np.testing.assert_array_equal(mass, [120, 70, 40, 90])
        np.testing.assert_array_equal(rows, [0, 0, 0, 1])

    def test_leading_pair_is_one_entry_per_event(self):
        mass, rows = diphoton_masses(sample(), pairs="leading")
        np.testing.assert_array_equal(mass, [120, 90])
        np.testing.assert_array_equal(rows, [0, 1])

    def test_invalid_leading_pair_does_not_fall_back_to_another_pair(self):
        data = sample()
        data["pair_obj"][0, 0, -1] = 0
        mass, rows = diphoton_masses(data, pairs="leading")
        np.testing.assert_array_equal(mass, [90])
        np.testing.assert_array_equal(rows, [1])
        np.testing.assert_array_equal(diphoton_masses(data)[0], [70, 40, 90])

    def test_invalid_mass_or_presence_never_enters_histogram(self):
        for column, value in (("pair_mass", float("nan")), ("pair_mass", float("inf")),
                              ("pair_mass", -1), ("pair_mass_valid", 0),
                              ("pt_1", 0), ("pt_2", float("nan"))):
            with self.subTest(column=column, value=value):
                data = sample()
                data["pair_obj"][0, 0, COLUMNS.index(column)] = value
                self.assertNotIn(120, diphoton_masses(data)[0])

    def test_valid_zero_mass_is_not_treated_as_padding(self):
        data = sample()
        data["pair_obj"][0, 0, COLUMNS.index("pair_mass")] = 0
        self.assertIn(0, diphoton_masses(data)[0])

    def test_pt_cuts_are_strict_and_independent_of_member_order(self):
        mass, _ = diphoton_masses(sample(), leading_pt_min=50, subleading_pt_min=20, sum_pt_min=100)
        np.testing.assert_array_equal(mass, [120])
        data = sample()
        data["pair_obj"][0, 0, :2] = torch.tensor([60, 100])
        np.testing.assert_array_equal(diphoton_masses(data, pairs="leading", subleading_pt_min=30)[0], [120])

    def test_schema_resolves_photon_codes_instead_of_assuming_three(self):
        data = sample()
        data["pair_obj"][..., 2:4][data["pair_obj"][..., 2:4] == 3] = 7
        data["pair_obj_feature_schema"] = [
            {"name": name, "categories": {"0": "jets", "7": "photons"}}
            for name in ("type_id_1", "type_id_2")]
        np.testing.assert_array_equal(diphoton_masses(data)[0], [120, 70, 40, 90])

    def test_missing_columns_masks_and_invalid_options_are_rejected(self):
        for key in ("pair_obj", "pair_obj_columns", "pair_slot_mask"):
            data = sample()
            del data[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                diphoton_masses(data)
        for kwargs in ({"pairs": "best"}, {"sum_pt_min": -1}, {"leading_pt_min": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                diphoton_masses(sample(), **kwargs)

    def test_cli_writes_png_and_reports_range_overflow_and_labels(self):
        with tempfile.TemporaryDirectory(prefix="diphoton_plot_") as temporary:
            root = Path(temporary)
            source, output = root / "mixture.pt", root / "plots" / "mass.png"
            torch.save(sample(), source)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                main([str(source), "--output", str(output), "--bins", "20",
                      "--mass-range", "50", "100", "--log-y"])
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertIn("Selected 4 pairs from 2 / 2 events", stdout.getvalue())
            self.assertIn("Stored label 2: 3 entries", stdout.getvalue())
            self.assertIn("Total: 4 entries; below range=1; above range=1", stdout.getvalue())

    def test_cli_supports_unlabelled_data_and_default_output(self):
        data = sample()
        del data["label"]
        with tempfile.TemporaryDirectory(prefix="diphoton_plot_") as temporary:
            source = Path(temporary) / "mixture.pt"
            torch.save(data, source)
            with redirect_stdout(io.StringIO()):
                main([str(source), "--pairs", "leading"])
            self.assertTrue(source.with_suffix(".diphoton_mass.png").exists())

    def test_cli_rejects_empty_selection_and_invalid_plot_options(self):
        with tempfile.TemporaryDirectory(prefix="diphoton_plot_") as temporary:
            source = Path(temporary) / "mixture.pt"
            torch.save(sample(), source)
            for options in (["--sum-pt-min", "1000"], ["--bins", "0"],
                            ["--mass-range", "150", "100"], ["--output", str(source)]):
                with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main([str(source), *options])


if __name__ == "__main__":
    unittest.main()
