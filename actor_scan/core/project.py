""" Non-destructive project structure.

Hard rule of the whole tool: the original scan and photos are never
modified or overwritten. A project directory enforces that mechanically:

    project/
      sources/    original files, copied in, hashed, made read-only
      outputs/    every derived artifact, auto-versioned (_v001, ...)
      journal.json  append-only log of every step and its parameters

Every operation reads sources (or earlier outputs) and writes a NEW
output file; re-running a step yields the next version rather than
replacing the last. The journal is what makes a result reproducible
and auditable ("which mask, which fullness, which exemplar produced
this bald head?").
"""
import hashlib
import json
import os
import shutil
import stat
import time

JOURNAL = "journal.json"


def _sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class Project:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.sources_dir = os.path.join(self.root, "sources")
        self.outputs_dir = os.path.join(self.root, "outputs")
        self.journal_path = os.path.join(self.root, JOURNAL)
        if not os.path.isfile(self.journal_path):
            raise FileNotFoundError(
                f"{root} is not a project (missing {JOURNAL}); "
                "use Project.create()")
        with open(self.journal_path) as f:
            self.journal = json.load(f)

    @classmethod
    def create(cls, root):
        root = os.path.abspath(root)
        os.makedirs(os.path.join(root, "sources"), exist_ok=True)
        os.makedirs(os.path.join(root, "outputs"), exist_ok=True)
        journal_path = os.path.join(root, JOURNAL)
        if not os.path.isfile(journal_path):
            with open(journal_path, "w") as f:
                json.dump({"version": 1, "created": time.time(),
                           "sources": {}, "steps": []}, f, indent=2)
        return cls(root)

    def _save(self):
        with open(self.journal_path, "w") as f:
            json.dump(self.journal, f, indent=2)

    def add_source(self, path):
        """Copy a file into sources/, hash it, and make it read-only.
        Re-adding the identical file is a no-op; a different file with
        the same name is refused (originals are immutable)."""
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        digest = _sha256(path)
        name = os.path.basename(path)
        dst = os.path.join(self.sources_dir, name)
        recorded = self.journal["sources"].get(name)
        if recorded is not None:
            if recorded["sha256"] != digest:
                raise ValueError(
                    f"a different source named {name!r} already exists; "
                    "originals are never replaced")
            return dst
        shutil.copy2(path, dst)
        os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        self.journal["sources"][name] = {"sha256": digest,
                                         "added": time.time()}
        self._save()
        return dst

    def new_output(self, name):
        """Reserve a versioned output path: name.ext -> name_v001.ext,
        never reusing an existing file's path."""
        base, ext = os.path.splitext(name)
        version = 1
        while True:
            candidate = os.path.join(self.outputs_dir,
                                     f"{base}_v{version:03d}{ext}")
            if not os.path.exists(candidate):
                return candidate
            version += 1

    def log(self, step, params=None, inputs=None, outputs=None):
        self.journal["steps"].append({
            "time": time.time(),
            "step": step,
            "params": params or {},
            "inputs": inputs or [],
            "outputs": outputs or [],
        })
        self._save()


def guard_overwrite(output, *inputs):
    """Refuse an output path that would clobber any input file."""
    out = os.path.abspath(output)
    for path in inputs:
        if path and os.path.abspath(path) == out:
            raise ValueError(
                f"refusing to overwrite input {path!r}; originals are "
                "never erased -- pick a new output path")
