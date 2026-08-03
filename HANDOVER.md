# Handover: nunif-app

Stub — this is currently a clean fork with no local customization
documented. Fill out properly once real local work happens here.

## What this is

A personal fork of [nagadomi/nunif](https://github.com/nagadomi/nunif),
the upstream author's own "personal playground" collection of
PyTorch-based image/video tools: **waifu2x** (anime-style image
super-resolution, also GAN-based photo models), **iw3** (2D→3D/VR
conversion, including live desktop capture, with a self-hosted WebXR
viewer), **stlizer** (video stabilizer), **cliqa** (low-vision image
quality scoring for dataset filtering).

**The README is unchanged from upstream** — no local customization is
documented in this fork yet.

## Open questions (unresolved, ask the operator)

- Which of the four sub-tools (waifu2x/iw3/stlizer/cliqa) is actually in
  active use, and for what purpose? Given the operator's FX/production
  context, waifu2x upscaling or iw3 2D→3D/VR conversion are plausible
  guesses, not confirmed.
- Any intended local modifications, or is this meant to track upstream
  closely?

## Where the durable project record lives

`13fingerfx/second-brain-git` → `Projects/nunif-app.md`.
