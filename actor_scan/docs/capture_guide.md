# First real capture — checklist (no rig)

Goal of the first session: one Miraco-class scan + one photometric set of
a single face region (forehead or cheek is easiest), enough to run the
whole pipeline on real data. Twenty minutes, no special gear.

## Gear

- Your scanner (scan as you normally would).
- Any camera with manual/lockable settings — a phone is fine *if* you can
  disable the computational stuff (see failure modes below).
- A tripod, or the camera wedged rock-solid against something.
- One movable light: bare LED bulb, work light, or a phone torch.
  Smaller/barer is better — a big softbox smears light directions.
- A shiny sphere for calibration: chrome ball bearing, chrome drawer
  knob, silver Christmas bauble. 2–5 cm is ideal. It only needs a clean
  specular highlight.

## Procedure

1. **Scan first**, as usual. Export mesh (OBJ/PLY, any of them).
2. **Set the stage**: subject seated, head against a wall or headrest —
   the subject must not move between shots. Tape the ball somewhere near
   the face (cheek height, a few cm away), visible to the camera.
3. **Kill other light**: curtains closed, room lights off. The moving
   lamp should be the only meaningful light source.
4. **Lock the camera**: manual focus, manual exposure, fixed white
   balance, lowest practical ISO. Frame the target region + ball.
   Do not touch the camera again — use a timer or remote.
5. **Shoot 4 frames**, moving only the lamp between frames, about 1–1.5 m
   from the face, roughly 30–45° off the camera axis:
   - left of camera
   - right of camera
   - above camera
   - below camera (or a 4th distinct position)
   Hold the lamp still during each exposure. Same distance each time if
   you can — brightness differences between shots are partly forgiven by
   the solver, but consistency helps.
6. **One extra frame** with soft/ambient light (room lights on) for a
   clean color/albedo reference. Optional but cheap.
7. **HD texture photos** (for the color path later): as many well-lit,
   sharp close-ups as you like — these don't need any of the above
   discipline.

## Feed it to the pipeline

```bash
# circle = ball center x, y and radius in pixels (any image viewer)
python -m actor_scan light-calibrate shot_L.jpg shot_R.jpg shot_T.jpg shot_B.jpg \
    --circle CX CY R -o lights.json
python -m actor_scan normals shot_L.jpg shot_R.jpg shot_T.jpg shot_B.jpg \
    --lights lights.json --robust -o ps.npz
python -m actor_scan detail ps.npz --sigma 40 -o height.npy
# click 6+ matching points (mesh xyz <-> photo pixel) into corres.json
python -m actor_scan register scan.obj corres.json --image-size W H -o pose.json
python -m actor_scan bake scan.obj --view pose.json height.npy \
    --scale 0.15 --max-edge 0.4 -o enhanced.stl
```

The ball appears in every shot, so each frame carries its own light
calibration — the subject and camera stay fixed, only the lamp moves.

## Failure modes that will silently ruin the solve

- **Phone computational photography**: HDR, Night mode, Deep Fusion,
  "scene enhancement", beauty filters — all of them re-light or denoise
  differently per shot and break the shading physics. Use a manual/"pro"
  camera mode or a raw-capture app.
- **Auto-anything drifting between shots** (exposure, focus, WB).
- **Subject movement** between frames — even a few mm matters at pore
  scale. Headrest, short session, shoot fast.
- **A second light source** you forgot about: window leak, monitor glow,
  the scanner's own projector still running.
- **The lamp too close** (< ~4x the size of the captured region): light
  direction then varies across the face and the single-direction
  assumption degrades.
- **Glossy skin highlights**: strong specular breaks the Lambertian
  assumption. Matte the skin if you can (powder), or just avoid oily
  areas for the first test; --robust also helps.
