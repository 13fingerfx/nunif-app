import unittest
import numpy as np
import trimesh

from actor_scan.core import commit


def open_sphere(radius=50.0, cut_z=35.0):
    """Sphere with its top cap faces removed (one boundary loop)."""
    m = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    centers = m.vertices[m.faces].mean(axis=1)
    m.update_faces(centers[:, 2] < cut_z)
    m.remove_unreferenced_vertices()
    return m


class TestBoundaryLoops(unittest.TestCase):
    def test_open_sphere_has_one_loop(self):
        m = open_sphere()
        loops, skipped = commit.boundary_loops(m)
        self.assertEqual(len(loops), 1)
        self.assertEqual(skipped, 0)
        # Loop vertices sit at the cut.
        ring = np.asarray(m.vertices)[loops[0]]
        self.assertTrue((ring[:, 2] > 25.0).all())

    def test_watertight_mesh_has_none(self):
        m = trimesh.creation.icosphere(subdivisions=2)
        loops, skipped = commit.boundary_loops(m)
        self.assertEqual(loops, [])
        self.assertEqual(skipped, 0)


class TestCommit(unittest.TestCase):
    def test_open_sphere_becomes_watertight_solid(self):
        m = open_sphere()
        self.assertFalse(m.is_watertight)
        solid, report = commit.commit(m)
        self.assertTrue(report["watertight"])
        self.assertTrue(report["winding_consistent"])
        self.assertEqual(report["loops_capped"], 1)
        # Volume close to the full sphere (flat cap trims a little).
        full = 4.0 / 3.0 * np.pi * 50.0 ** 3
        self.assertGreater(report["volume"], 0.93 * full)
        self.assertLess(report["volume"], 1.01 * full)

    def test_two_holes(self):
        m = open_sphere()
        centers = m.vertices[m.faces].mean(axis=1)
        m.update_faces(centers[:, 2] > -45.0)  # second hole at bottom
        m.remove_unreferenced_vertices()
        solid, report = commit.commit(m)
        self.assertTrue(report["watertight"])
        self.assertEqual(report["loops_capped"], 2)

    def test_floating_debris_dropped(self):
        m = open_sphere()
        junk = trimesh.creation.icosphere(subdivisions=1, radius=2.0)
        junk.apply_translation([0, 0, 80.0])
        both = trimesh.util.concatenate([m, junk])
        solid, report = commit.commit(both, keep="largest")
        self.assertEqual(report["components_dropped"], 1)
        self.assertTrue(report["watertight"])
        self.assertLess(solid.bounds[1][2], 60.0)  # junk gone

    def test_keep_all_preserves_large_components(self):
        a = trimesh.creation.icosphere(subdivisions=3, radius=50.0)
        b = trimesh.creation.icosphere(subdivisions=3, radius=30.0)
        b.apply_translation([200.0, 0, 0])
        both = trimesh.util.concatenate([a, b])
        solid, report = commit.commit(both, keep="all")
        self.assertEqual(report["components_dropped"], 0)
        self.assertTrue(report["watertight"])

    def test_nonmanifold_bowtie_repaired(self):
        """An edge shared by 4 faces (real scans have these) must be
        cut out and re-capped; found on scan 049, where it was the
        only thing between the raw scan and watertightness."""
        m = trimesh.creation.icosphere(subdivisions=3, radius=50.0)
        edge = m.faces[0][:2]
        apexes = [[70.0, 0, 0], [70.0, 5.0, 0]]
        extra_v = np.vstack([m.vertices, apexes])
        extra_f = np.vstack([m.faces,
                             [[edge[0], edge[1], len(m.vertices)],
                              [edge[0], edge[1], len(m.vertices) + 1]]])
        bowtie = trimesh.Trimesh(vertices=extra_v, faces=extra_f,
                                 process=False)
        self.assertFalse(bowtie.is_watertight)
        cleaned, removed = commit.drop_nonmanifold_faces(bowtie)
        self.assertGreaterEqual(removed, 3)
        solid, report = commit.commit(bowtie)
        self.assertTrue(report["watertight"])

    def test_stranded_deletion_output_commits_clean(self):
        """End-to-end with fill: junk deleted upstream, then committed."""
        from actor_scan.core import fill
        from actor_scan.tests.test_defects import head_with_cap
        mesh, _, cap = head_with_cap(spiky=True)
        junk = trimesh.creation.icosphere(subdivisions=1, radius=3.0)
        junk.apply_translation([0, 0, 70.0])
        both = trimesh.util.concatenate([mesh, junk])
        sel = np.concatenate([cap, np.ones(len(junk.vertices), bool)])
        repaired, _ = fill.fill_regions(both, sel, delete_stranded=True)
        solid, report = commit.commit(repaired)
        self.assertTrue(report["watertight"])
        self.assertTrue(report["winding_consistent"])


if __name__ == "__main__":
    unittest.main()
