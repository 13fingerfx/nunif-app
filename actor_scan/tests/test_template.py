import unittest
import numpy as np
import trimesh

from actor_scan.core import template
from actor_scan.tests.test_defects import head_with_cap


def random_similarity(seed=4):
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, np.pi)
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    rotation = trimesh.transformations.rotation_matrix(angle, axis)[:3, :3]
    scale = rng.uniform(0.01, 100.0)
    translation = rng.normal(size=3) * 10
    m = np.eye(4)
    m[:3, :3] = scale * rotation
    m[:3, 3] = translation
    return m


class TestSimilarity(unittest.TestCase):
    def test_umeyama_roundtrip(self):
        rng = np.random.default_rng(1)
        src = rng.normal(size=(8, 3)) * 5
        truth = random_similarity()
        dst = (np.hstack([src, np.ones((8, 1))]) @ truth.T)[:, :3]
        recovered = template.similarity_transform(src, dst)
        np.testing.assert_allclose(recovered, truth, atol=1e-9)

    def test_align_template_reports_rms(self):
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        marks_t = {"a": (1, 0, 0), "b": (0, 1, 0), "c": (0, 0, 1),
                   "d": (-1, 0, 0)}
        # Scan-space: same shape scaled x50 and shifted.
        marks_s = {k: (np.asarray(v) * 50.0 + [5, 6, 7]).tolist()
                   for k, v in marks_t.items()}
        aligned, matrix, rms = template.align_template(
            sphere, marks_t, marks_s)
        self.assertLess(rms, 1e-6)
        radii = np.linalg.norm(aligned.vertices - [5, 6, 7], axis=1)
        np.testing.assert_allclose(radii, 50.0, atol=1e-6)

    def test_refine_alignment_converges(self):
        """A mis-tethered sphere template snaps onto the trusted scan
        surface (scale + pose recovered by trimmed similarity-ICP)."""
        scan = trimesh.creation.icosphere(subdivisions=4, radius=50.0)
        tmpl = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        # Bad initial tether: 10% scale error, 2-unit offset.
        tmpl.apply_scale(55.0)
        tmpl.apply_translation([2.0, -1.5, 1.0])
        trusted = scan.vertices[scan.vertices[:, 2] < 20.0]  # "face"
        aligned, matrix, rms = template.refine_alignment(
            tmpl, trusted, iterations=15, allow_scale=True)
        radii = np.linalg.norm(aligned.vertices, axis=1)
        self.assertLess(abs(radii.mean() - 50.0), 0.5)
        self.assertLess(np.abs(aligned.vertices.mean(axis=0)).max(), 0.5)

    def test_refine_needs_points(self):
        tmpl = trimesh.creation.icosphere(subdivisions=1)
        with self.assertRaises(ValueError):
            template.refine_alignment(tmpl, np.zeros((5, 3)))

    def test_too_few_landmarks(self):
        sphere = trimesh.creation.icosphere(subdivisions=1)
        with self.assertRaises(ValueError):
            template.align_template(sphere, {"a": (0, 0, 0)},
                                    {"a": (1, 1, 1)})


class TestTemplateFill(unittest.TestCase):
    """Scan = sphere r=50 with a spiky cap; template = ellipsoid
    stretched to z*1.3. A template fill of the cap must follow the
    ellipsoid (taller skull), which plain biharmonic cannot invent."""

    @classmethod
    def setUpClass(cls):
        cls.mesh, _, cls.cap = head_with_cap(spiky=True)
        ellipsoid = trimesh.creation.icosphere(subdivisions=4, radius=50.0)
        ellipsoid.apply_scale([1.0, 1.0, 1.3])
        cls.template = ellipsoid
        cls.filled, cls.eff = template.template_fill(
            cls.mesh, cls.cap, ellipsoid, adherence=3.0, feather_rings=3)

    def _ellipsoid_distance(self, verts):
        scaled = verts / np.array([50.0, 50.0, 65.0])
        return np.abs(np.linalg.norm(scaled, axis=1) - 1.0)

    def test_interior_follows_template(self):
        from actor_scan.core.fill import _boundary_rings
        from actor_scan.core.defects import vertex_adjacency
        rings = _boundary_rings(self.eff, vertex_adjacency(self.mesh))
        deep = self.eff & (rings >= 6)
        self.assertTrue(deep.any())
        d_template = self._ellipsoid_distance(self.filled.vertices[deep])
        # Deep interior sits close to the ellipsoid (relative units)...
        self.assertLess(np.median(d_template), 0.03)
        # ...which the original spiky sphere surface did not.
        d_before = self._ellipsoid_distance(self.mesh.vertices[deep])
        self.assertGreater(np.median(d_before), 3 * np.median(d_template))

    def test_rim_tethered(self):
        from actor_scan.core.defects import vertex_adjacency
        adj = vertex_adjacency(self.mesh)
        rim = self.eff & (np.asarray(
            adj @ (~self.eff).astype(float)).reshape(-1) > 0)
        shift = np.linalg.norm(self.filled.vertices[rim] -
                               self.mesh.vertices[rim], axis=1)
        # Rim vertices stay near the scan (their spikes may relax, so
        # the bound is spike-scale, not template-scale: the ellipsoid
        # is ~15 units away at the pole).
        self.assertLess(np.median(shift), 3.0)

    def test_outside_selection_untouched(self):
        np.testing.assert_array_equal(self.filled.vertices[~self.eff],
                                      self.mesh.vertices[~self.eff])

    def test_zero_adherence_matches_biharmonic(self):
        from actor_scan.core import fill
        a, _ = template.template_fill(self.mesh, self.cap, self.template,
                                      adherence=0.0, feather_rings=0)
        b, _ = fill.fill_regions(self.mesh, self.cap)
        np.testing.assert_allclose(a.vertices, b.vertices, atol=1e-6)

    def test_locked_respected(self):
        locked = self.mesh.vertices[:, 0] > 20.0
        filled, eff = template.template_fill(
            self.mesh, self.cap, self.template, locked=locked)
        np.testing.assert_array_equal(filled.vertices[locked],
                                      self.mesh.vertices[locked])
        self.assertEqual((eff & locked).sum(), 0)

    def test_negative_adherence_rejected(self):
        with self.assertRaises(ValueError):
            template.template_fill(self.mesh, self.cap, self.template,
                                   adherence=-1.0)


if __name__ == "__main__":
    unittest.main()
