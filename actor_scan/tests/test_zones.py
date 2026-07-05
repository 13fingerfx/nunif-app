import unittest

from actor_scan.core import zones


class TestZones(unittest.TestCase):
    def test_canonical_ids_resolve_to_themselves(self):
        for zid in zones.ZONES:
            self.assertEqual(zones.resolve(zid), zid)

    def test_loose_names(self):
        cases = {
            "under left eye": "l_under_eye",
            "Under L Eye": "l_under_eye",
            "L cheek": "l_cheek",
            "cheek l": "l_cheek",
            "Right Cheek": "r_cheek",
            "left-brow": "l_brow",
            "left eyebrow": "l_brow",
            "under chin": "under_chin",
            "moustache": "upper_lip",
            "between brows": "glabella",
            "crow's feet left": "l_crows_feet",
            "top of head": "crown",
            "neck": "throat",
        }
        for loose, want in cases.items():
            self.assertEqual(zones.resolve(loose), want, loose)

    def test_unknown_raises_with_suggestion(self):
        with self.assertRaises(KeyError) as ctx:
            zones.resolve("left chek")
        self.assertIn("mean", str(ctx.exception))

    def test_mirror(self):
        self.assertEqual(zones.mirror("l_cheek"), "r_cheek")
        self.assertEqual(zones.mirror("r_under_eye"), "l_under_eye")
        self.assertEqual(zones.mirror("chin"), "chin")

    def test_mirrors_are_symmetric_and_valid(self):
        for z in zones.ZONES.values():
            if z.mirror:
                self.assertIn(z.mirror, zones.ZONES)
                self.assertEqual(zones.ZONES[z.mirror].mirror, z.id)

    def test_hair_zones(self):
        scalp = zones.hair_zones("scalp")
        beard = zones.hair_zones("beard")
        self.assertIn("scalp", scalp)
        self.assertIn("crown", scalp)
        self.assertIn("chin", beard)
        self.assertIn("upper_lip", beard)
        self.assertNotIn("l_cheek", scalp + beard)
        both = zones.hair_zones()
        self.assertEqual(set(both), set(scalp) | set(beard))

    def test_groups(self):
        self.assertIn("l_under_eye", zones.in_group("eye"))
        self.assertIn("throat", zones.in_group("neck"))


if __name__ == "__main__":
    unittest.main()
