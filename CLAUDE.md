## second-brain sync

This project has a note at `13fingerfx/second-brain-git` → `/Projects/nunif-app.md`.

At the start of substantive work in this repo (planning, architecture
decisions, adding a new dependency), pull the latest `second-brain-git`
(add_repo if not already in session scope, then `git pull`) and read that
note, specifically the `## Suggestions from stars` section. Surface any
unchecked suggestions to the user before diving in — don't act on them
unprompted.

When the user accepts or rejects a suggestion, check the box and append a
short verdict (e.g. `- [x] [[Stars/foo-bar]] — integrated, see src/x.py`
or `- [x] [[Stars/foo-bar]] — rejected, wrong license`).

When something durable changes about this project (stack, architecture,
goals, status), update the note's frontmatter and relevant section, then
commit and push to second-brain-git. Keep the `stack`/`topics` frontmatter
field accurate — it's what the star-sync Routine matches new stars against.

Don't sync trivial/in-flight changes — only durable facts worth another
session (or the star-matcher) knowing about.

## Workshop context

For machine specs, network addresses, and what's running where,
consult `13fingerfx/Workshop_data` (`MACHINES.md` + `SERVICES.md`) before
assuming or asking — pull the latest copy first. If you learn something
new and durable about the workshop's tech that isn't recorded there yet,
add it (never secrets, never passwords/keys/tokens) — see that repo's
`AGENTS.md` for the exact mechanism.

