""" Photo <-> scan registration from user-clicked correspondences.

v1 is deliberately manual: the user marks matching points (a 3D point on
the scan, its pixel in the photo) and we solve a PnP pose. Deterministic
and debuggable; COLMAP/feature automation can replace the input source
later without changing anything downstream.

Camera convention here is OpenCV's: x right, y down, z forward from the
camera into the scene (distinct from the image-space light convention in
lightcal/photometric, which never meet a mesh).
"""
import json
import numpy as np
import cv2


class CameraPose:
    def __init__(self, rvec, tvec, camera_matrix, image_size):
        self.rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
        self.tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.image_size = tuple(int(v) for v in image_size)  # (w, h)

    @property
    def rotation(self):
        return cv2.Rodrigues(self.rvec)[0]

    @property
    def camera_center(self):
        return -self.rotation.T @ self.tvec

    def project(self, points3d):
        pts = np.asarray(points3d, dtype=np.float64).reshape(-1, 1, 3)
        proj, _ = cv2.projectPoints(pts, self.rvec, self.tvec,
                                    self.camera_matrix, None)
        return proj.reshape(-1, 2)

    def to_dict(self):
        return {
            "rvec": self.rvec.tolist(),
            "tvec": self.tvec.tolist(),
            "camera_matrix": self.camera_matrix.tolist(),
            "image_size": list(self.image_size),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["rvec"], d["tvec"], d["camera_matrix"], d["image_size"])

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


def camera_matrix(focal_px, image_size):
    w, h = image_size
    return np.array([[focal_px, 0.0, w / 2.0],
                     [0.0, focal_px, h / 2.0],
                     [0.0, 0.0, 1.0]])


def reprojection_error(pose, points3d, points2d):
    proj = pose.project(points3d)
    return float(np.linalg.norm(proj - np.asarray(points2d, dtype=np.float64),
                                axis=1).mean())


def solve_pose(points3d, points2d, image_size, focal_px=None):
    """PnP pose from >= 6 correspondences (>= 4 with known focal).

    focal_px=None sweeps a plausible focal range (0.4x..4x the image
    diagonal-ish span) and keeps the solution with the smallest
    reprojection error -- for photos with no usable EXIF.
    Returns (CameraPose, mean reprojection error in px).
    """
    pts3 = np.asarray(points3d, dtype=np.float64)
    pts2 = np.asarray(points2d, dtype=np.float64)
    if len(pts3) != len(pts2):
        raise ValueError("points3d and points2d must pair up")
    minimum = 4 if focal_px is not None else 6
    if len(pts3) < minimum:
        raise ValueError(f"need at least {minimum} correspondences")

    if focal_px is not None:
        candidates = [float(focal_px)]
    else:
        base = float(max(image_size))
        candidates = [base * m for m in
                      (0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.8, 4.0)]

    best = None
    for focal in candidates:
        k = camera_matrix(focal, image_size)
        ok, rvec, tvec = cv2.solvePnP(pts3.reshape(-1, 1, 3),
                                      pts2.reshape(-1, 1, 2), k, None,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        pose = CameraPose(rvec, tvec, k, image_size)
        err = reprojection_error(pose, pts3, pts2)
        if best is None or err < best[1]:
            best = (pose, err)
    if best is None:
        raise RuntimeError("solvePnP failed for every focal candidate")
    return best
