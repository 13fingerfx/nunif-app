import unittest
import numpy as np
import trimesh

from actor_scan.core import measure


def sphere_head(radius=50.0):
    return trimesh.creation.icosphere(subdivisions=4, radius=radius)


class TestMeasure(unittest.TestCase):
    def setUp(self):
        self.mesh = sphere_head()
        self.r = 50.0

    def test_caliper(self):
        self.assertAlmostEqual(
            measure.caliper((self.r, 0, 0), (-self.r, 0, 0)), 100.0)

    def test_loop_is_great_circle(self):
        length = measure.loop_length(
            self.mesh, (self.r, 0, 0), (0, self.r, 0), (-self.r, 0, 0))
        self.assertAlmostEqual(length, 2 * np.pi * self.r, delta=0.5)

    def test_arc_chooses_side_by_over_hint(self):
        a, b = (self.r, 0, 0), (-self.r, 0, 0)
        top = measure.arc_length(self.mesh, a, b, (0, 0, self.r))
        self.assertAlmostEqual(top, np.pi * self.r, delta=0.5)
        bottom = measure.arc_length(self.mesh, a, b, (0, 0, -self.r))
        self.assertAlmostEqual(bottom, np.pi * self.r, delta=0.5)

    def test_chart_computation_and_missing(self):
        landmarks = {
            "pronasale": (self.r, 0, 0),
            "back_of_head": (-self.r, 0, 0),
            "l_ear_top": (0, self.r, 0),
            "r_ear_top": (0, -self.r, 0),
            "crown": (0, 0, self.r),
        }
        chart = measure.compute_chart(self.mesh, landmarks)
        self.assertAlmostEqual(chart["nose_to_back_of_head"], 100.0)
        self.assertAlmostEqual(chart["ear_to_ear_over_crown"],
                               np.pi * self.r, delta=0.5)
        self.assertIn("head_at_brow_circ", chart["_missing"])
        self.assertNotIn("head_at_brow_circ", chart)

    def test_compare_charts(self):
        target = {"nose_to_back_of_head": 195.0, "neck_width": 120.0}
        measured = {"nose_to_back_of_head": 197.5}
        diff = measure.compare_charts(measured, target)
        self.assertAlmostEqual(diff["nose_to_back_of_head"]["delta"], 2.5)
        self.assertNotIn("neck_width", diff)

    def test_chart_ids_are_unique_and_complete(self):
        ids = [m.id for m in measure.MEASUREMENTS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 14)
        for m in measure.MEASUREMENTS:
            want = 2 if m.kind == "caliper" else 3
            self.assertEqual(len(m.landmarks), want, m.id)

    def test_collinear_plane_rejected(self):
        with self.assertRaises(ValueError):
            measure.loop_length(self.mesh, (1, 0, 0), (2, 0, 0), (3, 0, 0))


if __name__ == "__main__":
    unittest.main()
