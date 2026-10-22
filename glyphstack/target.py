"""The target profile: which typeface faces a run measures, where on the card they are printed, and every setting tuned to that type

A profile is a TOML file. `rules` ships in `glyphstack/profiles` and is the default. `use` selects another by its name
there or by a path, and every stage reads the profile in use when it runs, so processes started for extraction select
the same profile before they work
"""
import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

BUILTIN = Path(__file__).parent / "profiles"
DEFAULT = "rules"
SECTIONS = ("paper", "segment", "templates", "geometry", "stack", "fit", "site")


@dataclass(frozen=True)
class Face:
    """One face of the typeface. Glyph keys are the prefix plus the character, cache files start with stem and site names end with suffix"""

    name: str
    full_name: str = ""
    prefix: str = ""
    suffix: str = ""
    stem: str = ""


@dataclass(frozen=True)
class Region:
    """A part of the card to read, as x, y, w, h fractions of the upright card crop, which is the card plus about 20 px"""

    name: str
    box: tuple

    def contains(self, cx, cy, shape):
        x, y, w, h = self.box
        return x <= cx / shape[1] <= x + w and y <= cy / shape[0] <= y + h


@dataclass(frozen=True)
class Profile:
    name: str
    path: str
    sha256: str
    faces: tuple
    regions: tuple
    settings: dict

    def __getitem__(self, section):
        return self.settings[section]

    @property
    def primary(self):
        """The face whose glyph keys are the bare character, the one the print scale and the shared fit are taken from"""
        return self.faces[0]

    def face(self, name):
        for f in self.faces:
            if f.name == name:
                return f
        raise KeyError(f"profile {self.name} has no face {name!r}")


def resolve(spec):
    """The file a profile name or path refers to"""
    p = Path(spec)
    if p.suffix == ".toml" or p.exists():
        return p
    return BUILTIN / f"{spec}.toml"


def load(spec=DEFAULT):
    """Reads and checks a profile"""
    path = resolve(spec)
    raw = path.read_bytes()
    d = tomllib.loads(raw.decode())
    faces = tuple(Face(**f) for f in d["faces"])
    if not faces or faces[0].prefix:
        raise ValueError(f"{path}: the first face is the primary one and takes no key prefix")
    for attr in ("name", "prefix", "suffix", "stem"):
        vals = [getattr(f, attr) for f in faces]
        if len(set(vals)) != len(vals):
            raise ValueError(f"{path}: every face needs its own {attr}")
    regions = tuple(Region(r["name"], tuple(r["box"])) for r in d["regions"])
    missing = [s for s in SECTIONS if s not in d]
    if missing:
        raise ValueError(f"{path}: missing sections {', '.join(missing)}")
    if d["paper"]["mode"] not in ("scan", "local"):
        raise ValueError(f"{path}: paper mode is scan or local")
    return Profile(d["name"], str(path), hashlib.sha256(raw).hexdigest(), faces, regions, {s: d[s] for s in SECTIONS})


_current = None


def use(spec):
    """Selects the profile every stage reads from now on"""
    global _current
    _current = load(spec)
    return _current


def get():
    """The profile in use, the default one until another is selected"""
    global _current
    if _current is None:
        _current = load(DEFAULT)
    return _current


def add_args(ap, profile_help="target profile, a name in glyphstack/profiles or a path to a TOML file, default rules"):
    """Adds --profile and the font options: --font for the primary face, --face-font NAME=PATH for others"""
    ap.add_argument("--profile", help=profile_help)
    ap.add_argument("--font", required=True, help="TrueType font of the profile's primary face, used as the measuring stick and never copied to the output")
    ap.add_argument("--face-font", action="append", default=[], metavar="NAME=PATH", help="font of another face of the profile, repeat for each face")
    ap.add_argument("--italic-font", help="the same as --face-font italic=PATH")


def font_paths(a):
    """{face name: font path} from the parsed options, checked against the profile in use"""
    p = get()
    out = {p.primary.name: a.font}
    pairs = list(a.face_font) + ([f"italic={a.italic_font}"] if a.italic_font else [])
    for pair in pairs:
        name, _, path = pair.partition("=")
        if not path:
            raise SystemExit(f"--face-font takes NAME=PATH, got {pair!r}")
        p.face(name)
        out[name] = path
    return out
