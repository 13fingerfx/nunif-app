""" Canonical face/head zone registry.

One shared vocabulary for overlay region tags, the fill/repair stage,
and (later) marketplace packs. Users and other tools can be sloppy --
"under left eye", "Left Cheek", "cheek l" -- and resolve() maps them all
to one canonical id, so a pack tagged r_cheek always fits a request for
"right cheek".

Zones are user-extensible in spirit (overlay files accept any tag; the
CLI only warns on unknown ones) -- this registry is the *shared* core
set, not a cage. hair marks zones that commonly carry hair in raw scans
("scalp" / "beard"): the bald/repair stage uses it to bias detection
and to name its outputs.
"""
from dataclasses import dataclass
import difflib


@dataclass(frozen=True)
class Zone:
    id: str
    label: str
    group: str          # face | eye | mouth | jaw | neck | ear | scalp
    mirror: str = None  # id of the opposite-side zone, if any
    hair: str = None    # None | "scalp" | "beard"


_ZONES = [
    Zone("forehead", "Forehead", "face"),
    Zone("glabella", "Glabella (between brows)", "face"),
    Zone("l_temple", "Left temple", "face", "r_temple"),
    Zone("r_temple", "Right temple", "face", "l_temple"),
    Zone("l_brow", "Left brow", "eye", "r_brow"),
    Zone("r_brow", "Right brow", "eye", "l_brow"),
    Zone("l_upper_eyelid", "Left upper eyelid", "eye", "r_upper_eyelid"),
    Zone("r_upper_eyelid", "Right upper eyelid", "eye", "l_upper_eyelid"),
    Zone("l_under_eye", "Under left eye", "eye", "r_under_eye"),
    Zone("r_under_eye", "Under right eye", "eye", "l_under_eye"),
    Zone("l_crows_feet", "Left crow's feet", "eye", "r_crows_feet"),
    Zone("r_crows_feet", "Right crow's feet", "eye", "l_crows_feet"),
    Zone("nose_bridge", "Nose bridge", "face"),
    Zone("nose", "Nose", "face"),
    Zone("nose_tip", "Nose tip", "face"),
    Zone("l_cheek", "Left cheek", "face", "r_cheek"),
    Zone("r_cheek", "Right cheek", "face", "l_cheek"),
    Zone("l_nasolabial", "Left nasolabial fold", "mouth", "r_nasolabial"),
    Zone("r_nasolabial", "Right nasolabial fold", "mouth", "l_nasolabial"),
    Zone("philtrum", "Philtrum", "mouth"),
    Zone("upper_lip", "Upper lip", "mouth", hair="beard"),
    Zone("lower_lip", "Lower lip", "mouth"),
    Zone("l_lip_corner", "Left lip corner", "mouth", "r_lip_corner"),
    Zone("r_lip_corner", "Right lip corner", "mouth", "l_lip_corner"),
    Zone("chin", "Chin", "jaw", hair="beard"),
    Zone("l_jaw", "Left jawline", "jaw", "r_jaw", hair="beard"),
    Zone("r_jaw", "Right jawline", "jaw", "l_jaw", hair="beard"),
    Zone("under_chin", "Under chin (submental)", "neck", hair="beard"),
    Zone("throat", "Throat", "neck"),
    Zone("nape", "Nape of neck", "neck", hair="scalp"),
    Zone("l_ear", "Left ear", "ear", "r_ear"),
    Zone("r_ear", "Right ear", "ear", "l_ear"),
    Zone("scalp", "Scalp", "scalp", hair="scalp"),
    Zone("hairline", "Hairline", "scalp", hair="scalp"),
    Zone("crown", "Crown", "scalp", hair="scalp"),
    Zone("l_sideburn", "Left sideburn", "scalp", "r_sideburn", hair="scalp"),
    Zone("r_sideburn", "Right sideburn", "scalp", "l_sideburn", hair="scalp"),
]

ZONES = {z.id: z for z in _ZONES}

_EXTRA_ALIASES = {
    "mustache": "upper_lip",
    "moustache": "upper_lip",
    "goatee": "chin",
    "double_chin": "under_chin",
    "neck": "throat",
    "between_brows": "glabella",
    "frown_lines": "glabella",
    "top_of_head": "crown",
}

_WORD_MAP = {"left": "l", "right": "r", "below": "under", "beneath": "under",
             "eyebrow": "brow", "cheeks": "cheek", "lips": "lip"}


def _tokens(name):
    s = str(name).strip().lower()
    for ch in "-/,.":
        s = s.replace(ch, " ")
    s = s.replace("'", "")
    out = []
    for tok in s.replace("_", " ").split():
        out.append(_WORD_MAP.get(tok, tok))
    return out


def _signature(name):
    return frozenset(_tokens(name))


_SIGNATURES = {}
for _z in _ZONES:
    _sig = _signature(_z.id)
    if _sig in _SIGNATURES:
        raise RuntimeError(f"ambiguous zone signature: {_z.id}")
    _SIGNATURES[_sig] = _z.id
for _alias, _target in _EXTRA_ALIASES.items():
    _SIGNATURES.setdefault(_signature(_alias), _target)


def resolve(name):
    """Canonical zone id for a loosely written name.

    Raises KeyError (with suggestions) if nothing matches; unknown
    custom tags are allowed at the file-format level, this is for
    the shared vocabulary only.
    """
    sig = _signature(name)
    if sig in _SIGNATURES:
        return _SIGNATURES[sig]
    joined = "_".join(_tokens(name))
    close = difflib.get_close_matches(joined, list(ZONES), n=3, cutoff=0.6)
    hint = f" (did you mean: {', '.join(close)}?)" if close else ""
    raise KeyError(f"unknown zone {name!r}{hint}")


def mirror(zone_id):
    """Opposite-side zone id, or the same id for unpaired zones."""
    z = ZONES[zone_id]
    return z.mirror or z.id


def in_group(group):
    return [z.id for z in _ZONES if z.group == group]


def hair_zones(kind=None):
    """Zones that commonly carry hair; kind None|'scalp'|'beard'."""
    return [z.id for z in _ZONES
            if z.hair is not None and (kind is None or z.hair == kind)]
