"""Register a scanned page to a template with ORB features + RANSAC homography.

Feature extraction runs on a downscaled copy (``max_dim``) for speed; keypoints
are scaled back so the homography maps full-resolution template pixels to
full-resolution scan pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import structure
from .templates import Template


@dataclass
class AlignSettings:
    max_dim: int = 1200  # longest side used for feature extraction
    n_features: int = 5000
    ratio: float = 0.75  # Lowe ratio test
    ransac_thresh: float = 6.0  # px, scan full-res
    min_inliers: int = 40
    min_inlier_ratio: float = 0.15
    min_area_frac: float = 0.15  # projected form must cover this much of the scan
    max_side_ratio: float = 2.0  # opposite sides of projected form (perspective limit)
    min_score: float = 0.35  # NCC between template and back-warped scan


@dataclass
class Alignment:
    template: Template
    H: np.ndarray | None  # template -> scan
    inliers: int
    matches: int
    score: float
    ok: bool
    reason: str = ""
    src: np.ndarray | None = None  # inlier points, template space
    dst: np.ndarray | None = None  # inlier points, scan space

    def local(self, center, radius: float = 350.0, min_points: int = 15, max_shift: float = 40.0) -> np.ndarray:
        """Homography fitted only to inliers near ``center`` (template coords).

        Photographed forms are rarely flat (KIEMS photos of booklet pages curl),
        so one global homography can be off by 10-20 px in places. A local fit
        tightens zones; it falls back to the global H when there's too little
        support or it disagrees wildly with the global fit."""
        if self.src is None:
            return self.H
        near = np.linalg.norm(self.src - np.float32(center), axis=1) < radius
        if near.sum() < min_points:
            return self.H
        Hl, _ = cv2.findHomography(self.src[near], self.dst[near], cv2.RANSAC, 4.0)
        if Hl is None:
            return self.H
        c = np.float32([[center]])
        if np.linalg.norm(cv2.perspectiveTransform(c, Hl) - cv2.perspectiveTransform(c, self.H)) > max_shift:
            return self.H
        return Hl


def _features(gray: np.ndarray, s: AlignSettings, page_long_side: float | None = None,
              n_features: int | None = None, max_dim: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """ORB features on a copy scaled so the *page* is ``max_dim`` long. For a
    template cut from a page (``page_long_side``), that is the page it came
    from, so its text is at the same size as in a scanned page."""
    scale = min(1.0, (max_dim or s.max_dim) / (page_long_side or max(gray.shape[:2])))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    orb = cv2.ORB_create(nfeatures=n_features or s.n_features, fastThreshold=10)
    kps, desc = orb.detectAndCompute(small, None)
    pts = np.float32([k.pt for k in kps]).reshape(-1, 2) / scale
    return pts, desc


def prepare(template: Template, s: AlignSettings) -> Template:
    template.keypoints, template.descriptors = _features(template.image, s, template.page_long_side,
                                                         template.n_features, template.feature_dim)
    template.mask = template.build_mask()
    template.regions = {n: structure.prepare_region(template.image, n, r) for n, r in template.structure.items()}
    return template


def _geometry_ok(H: np.ndarray, template: Template, scan_shape, s: AlignSettings) -> str:
    w, h = template.size
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    if not cv2.isContourConvex(q.astype(np.float32)):
        return "projected form is not convex"
    area = cv2.contourArea(q)
    min_frac = s.min_area_frac if template.min_area_frac is None else template.min_area_frac
    if area < min_frac * scan_shape[0] * scan_shape[1]:
        return f"projected form too small ({area / (scan_shape[0] * scan_shape[1]):.2f} of scan)"
    sides = np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)
    for a, b in ((sides[0], sides[2]), (sides[1], sides[3])):
        if max(a, b) / max(min(a, b), 1e-6) > s.max_side_ratio:
            return "implausible perspective"
    return ""


def _score(H: np.ndarray, template: Template, scan_gray: np.ndarray, size: int = 500) -> float:
    """Normalized cross-correlation of the template vs. the scan warped into template
    space, ignoring PII zones (they hold handwriting the blank template lacks)."""
    w, h = template.size
    f = size / max(w, h)
    sw, sh = max(1, round(w * f)), max(1, round(h * f))
    S = np.diag([f, f, 1.0])
    warped = cv2.warpPerspective(scan_gray, S @ np.linalg.inv(H), (sw, sh), flags=cv2.INTER_AREA)
    ref = cv2.resize(template.image, (sw, sh), interpolation=cv2.INTER_AREA)
    keep = cv2.resize(template.mask, (sw, sh), interpolation=cv2.INTER_NEAREST) == 0
    a = cv2.GaussianBlur(ref, (5, 5), 0)[keep].astype(np.float32)
    b = cv2.GaussianBlur(warped, (5, 5), 0)[keep].astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def align(scan_gray: np.ndarray, template: Template, s: AlignSettings,
          scan_feats: tuple[np.ndarray, np.ndarray | None] | None = None) -> Alignment:
    pts, desc = scan_feats if scan_feats is not None else \
        _features(scan_gray, s, n_features=template.n_features, max_dim=template.feature_dim)
    fail = lambda reason, inl=0, m=0: Alignment(template, None, inl, m, 0.0, False, reason)  # noqa: E731
    if desc is None or template.descriptors is None or len(desc) < 2:
        return fail("no features")

    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(template.descriptors, desc, k=2)
    good = [m for m, *rest in knn if rest and m.distance < s.ratio * rest[0].distance]
    min_inliers = template.min_inliers or s.min_inliers
    min_score = s.min_score if template.min_score is None else template.min_score
    if len(good) < min_inliers:
        return fail(f"too few matches ({len(good)})", m=len(good))

    src = template.keypoints[[m.queryIdx for m in good]]
    dst = pts[[m.trainIdx for m in good]]
    H, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, s.ransac_thresh, maxIters=5000, confidence=0.999)
    if H is None:
        return fail("homography failed", m=len(good))
    n_in = int(inl.sum())
    if n_in < min_inliers or n_in / len(good) < s.min_inlier_ratio:
        return fail(f"weak match ({n_in}/{len(good)} inliers)", n_in, len(good))
    if reason := _geometry_ok(H, template, scan_gray.shape, s):
        return fail(reason, n_in, len(good))
    score = _score(H, template, scan_gray)
    keep = inl.ravel().astype(bool)
    ok = score >= min_score
    return Alignment(template, H, n_in, len(good), score, ok,
                     "" if ok else f"low alignment score ({score:.2f})", src[keep], dst[keep])


def best_alignment(scan_gray: np.ndarray, templates: list[Template], s: AlignSettings,
                   form: str | None = None) -> Alignment:
    """Try every template (only those for ``form`` if it is known, e.g. from the
    QR code); return the passing alignment with the most inliers, or the best
    failure (for the review report) if none pass."""
    if form:
        # The form type is known (QR / KIEMS file name): only its templates may
        # match. A 34B page must never be redacted with a 34A layout.
        templates = [t for t in templates if not t.form or t.form == form]
        if not templates:
            return Alignment(None, None, 0, 0, 0.0, False, f"no template for form {form}")
    feats: dict = {}  # the scan's features, once per feature setting

    def scan_feats(t: Template):
        key = (t.feature_dim, t.n_features)
        if key not in feats:
            feats[key] = _features(scan_gray, s, n_features=t.n_features, max_dim=t.feature_dim)
        return feats[key]

    results = [align(scan_gray, t, s, scan_feats(t)) for t in templates]
    passing = [r for r in results if r.ok]
    pool = passing or results
    # Priority first: a 34B page showing the declaration block must be
    # redacted with the block template even if the results header matches better.
    return max(pool, key=lambda r: (r.template.priority if r.ok else 0, r.inliers, r.score))
