# actor_scan (planning stage)

Goal: a standalone tool that takes an actor's 3D scan (mesh + UVs, treated as
the dimensional ground truth) plus a set of HD reference photos of the same
actor, and:

1. Builds a much higher-resolution color texture for the scan from the photos.
2. Extrapolates fine surface detail (pores, fine wrinkles, lip texture, etc.)
   from the photos and bakes it into the mesh as real topology/displacement,
   not just a texture trick.
3. Saves detail as reusable, region-tagged overlays (L cheek, R cheek, throat,
   under-chin, under-L-eye, under-R-eye, ...) that can later be applied to
   other scans or from-scratch characters when no photos are available.

This directory currently contains only the stage-1 research pass. No code has
been written yet.

See [`docs/research.md`](docs/research.md) for the full write-up: what exists
already (open source and commercial), what's missing and would need custom
work, how this maps onto tooling already in this repo (`waifu2x`, `iw3`,
`dino`), and a platform recommendation (Mac vs PC).
