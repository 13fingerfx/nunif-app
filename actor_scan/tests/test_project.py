import os
import stat
import tempfile
import unittest

from actor_scan.core import project


class TestProject(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "proj")
        self.proj = project.Project.create(self.root)
        self.scan = os.path.join(self.tmp.name, "scan.ply")
        with open(self.scan, "w") as f:
            f.write("fake scan data")

    def tearDown(self):
        self.tmp.cleanup()

    def test_source_is_copied_hashed_readonly(self):
        dst = self.proj.add_source(self.scan)
        self.assertTrue(os.path.isfile(dst))
        # Write bits stripped (os.access lies under root, check the mode).
        mode = stat.S_IMODE(os.stat(dst).st_mode)
        self.assertEqual(mode & (stat.S_IWUSR | stat.S_IWGRP |
                                 stat.S_IWOTH), 0)
        self.assertIn("scan.ply", self.proj.journal["sources"])
        # Original stays where it was, still writable by its owner.
        self.assertTrue(stat.S_IMODE(os.stat(self.scan).st_mode) &
                        stat.S_IWUSR)

    def test_identical_readd_is_noop_but_conflict_refused(self):
        self.proj.add_source(self.scan)
        self.proj.add_source(self.scan)  # same content: fine
        with open(self.scan, "w") as f:
            f.write("DIFFERENT data")
        with self.assertRaises(ValueError):
            self.proj.add_source(self.scan)

    def test_outputs_are_versioned_never_reused(self):
        p1 = self.proj.new_output("bald.ply")
        self.assertTrue(p1.endswith("bald_v001.ply"))
        open(p1, "w").close()
        p2 = self.proj.new_output("bald.ply")
        self.assertTrue(p2.endswith("bald_v002.ply"))

    def test_journal_survives_reopen(self):
        self.proj.log("fill", params={"profile": "beard"},
                      inputs=["scan.ply"], outputs=["bald_v001.ply"])
        reopened = project.Project(self.root)
        self.assertEqual(reopened.journal["steps"][0]["step"], "fill")
        self.assertEqual(
            reopened.journal["steps"][0]["params"]["profile"], "beard")

    def test_guard_overwrite(self):
        with self.assertRaises(ValueError):
            project.guard_overwrite(self.scan, self.scan)
        project.guard_overwrite(self.scan + ".out", self.scan)  # fine


if __name__ == "__main__":
    unittest.main()
