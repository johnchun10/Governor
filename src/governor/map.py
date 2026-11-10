"""The project map: what the codebase is made of and where each feature lives.

A store under
``.governor/map/`` holds *area* cards (a folder or module: its purpose, what it
contains, what it depends on) and *feature* cards (a name, an abstract, the
paths it lives in). The agent reads the map at the start of every run
(``MapContext`` puts a compact rendering in the system prompt, part of the cached
prefix) and extends it as it learns (the ``map_feature`` tool). ``MapSink``
records the paths a run actually read or wrote onto the feature cards that run
mapped, so a card shows the paths predicted for it against the ones confirmed by
use. ``/map`` surveys the repo (an explore run with an output schema),
refreshes the map, and inspects it. Cards link to chats and runs by id only.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from agent_core import (
    AgentSpec,
    ContextProvider,
    Event,
    Message,
    RunView,
    Sink,
    Tool,
    ToolContext,
    ToolError,
    tool,
)

from .commands import Command

if TYPE_CHECKING:
    from .session import Session

T = TypeVar("T")

BUDGET_TOKENS = 3000
"""What the rendered map may cost in every system prompt."""
CHARS_PER_TOKEN = 4
"""The estimate used for the budget; no tokenizer is loaded for it."""
READ_TOOLS = frozenset({"read_file"})
WRITE_TOOLS = frozenset({"write_file", "edit_file"})
"""The tool calls the sink counts as touching a path. grep, list_files and bash are not parsed."""
RESERVED_NAMES = frozenset({"survey", "refresh"})
"""``/map``'s verbs; a feature cannot take them as its name."""


# --- cards -------------------------------------------------------------------------------


@dataclass
class Area:
    """A folder (path ends with ``/``) or module, keyed by its repo-relative path."""

    path: str
    purpose: str
    contains: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Area:
        return cls(
            data["path"], data.get("purpose", ""), list(data.get("contains", [])), list(data.get("depends_on", []))
        )


@dataclass
class FeatureCard:
    """A feature: what it does and where it lives. ``paths`` are predicted (declared by the agent or
    the survey); ``read`` and ``wrote`` are confirmed by runs through ``MapSink``."""

    name: str
    abstract: str
    paths: list[str] = field(default_factory=list)
    read: list[str] = field(default_factory=list)
    wrote: list[str] = field(default_factory=list)
    runs: list[str] = field(default_factory=list)
    chats: list[str] = field(default_factory=list)
    updated: float = field(default_factory=time.time)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureCard:
        return cls(
            data["name"],
            data.get("abstract", ""),
            list(data.get("paths", [])),
            list(data.get("read", [])),
            list(data.get("wrote", [])),
            list(data.get("runs", [])),
            list(data.get("chats", [])),
            float(data.get("updated", 0.0)),
        )


@dataclass
class Survey:
    """When, from which commit and by which model the areas were last surveyed."""

    at: float | None = None
    commit: str | None = None
    model: str | None = None


@dataclass
class ProjectMap:
    areas: list[Area] = field(default_factory=list)
    features: list[FeatureCard] = field(default_factory=list)
    surveyed: Survey = field(default_factory=Survey)

    @property
    def empty(self) -> bool:
        return not self.areas and not self.features

    def feature(self, name: str) -> FeatureCard | None:
        wanted = name.strip().casefold()
        return next((f for f in self.features if f.name.casefold() == wanted), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "surveyed": asdict(self.surveyed),
            "areas": [asdict(a) for a in self.areas],
            "features": [asdict(f) for f in self.features],
        }

    @classmethod
    def from_dict(cls, data: Any) -> ProjectMap:
        if not isinstance(data, dict):
            raise TypeError(f"the map must be a JSON object, not {type(data).__name__}")
        surveyed = data.get("surveyed") or {}
        areas, features = data.get("areas", []), data.get("features", [])
        if not isinstance(surveyed, dict) or not isinstance(areas, list) or not isinstance(features, list):
            raise TypeError("surveyed must be an object; areas and features must be lists")
        return cls(
            [Area.from_dict(a) for a in areas],
            [FeatureCard.from_dict(f) for f in features],
            Survey(surveyed.get("at"), surveyed.get("commit"), surveyed.get("model")),
        )


# --- the store ---------------------------------------------------------------------------


_STORES: dict[Path, MapStore] = {}
_STORES_LOCK = threading.Lock()


class MapStore:
    """``<project>/.governor/map/map.json``: one file, written atomically, read on demand.

    ``MapStore.of(project_dir)`` hands the tool, the sink, the command and the context provider the
    same instance, so one lock orders every read-modify-write of a project's map within the process.
    """

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir.resolve()
        self.path = self.project_dir / ".governor" / "map" / "map.json"
        self.read_error: str | None = None
        """Why the file could not be read, set and cleared by ``load``; ``/map`` shows it."""
        self.write_error: str | None = None
        """Why the last write failed (the sink's, which must not raise); cleared by the next successful write."""
        self._lock = threading.Lock()

    @property
    def errors(self) -> list[str]:
        return [e for e in (self.read_error, self.write_error) if e]

    @classmethod
    def of(cls, project_dir: Path) -> MapStore:
        key = project_dir.resolve()
        with _STORES_LOCK:
            store = _STORES.get(key)
            if store is None:
                store = _STORES[key] = cls(key)
            return store

    def load(self) -> ProjectMap:
        """The stored map; empty when there is none, or when the file is unreadable (``read_error`` says so)."""
        try:
            project_map = ProjectMap.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            self.read_error = None
            return ProjectMap()
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            self.read_error = f"{self.path} is unreadable ({type(exc).__name__}: {exc}); /map survey rewrites it"
            return ProjectMap()
        self.read_error = None
        return project_map

    def save(self, project_map: ProjectMap) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(project_map.to_dict(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)

    def update(self, change: Callable[[ProjectMap], T], *, rewrite: bool = False) -> T:
        """Load, apply ``change`` and save under the store's lock. ``OSError`` from the write propagates, and
        an unreadable file is never written over unless ``rewrite`` (a survey) says so."""
        with self._lock:
            project_map = self.load()
            if self.read_error and not rewrite:
                raise OSError(self.read_error)
            result = change(project_map)
            try:
                self.save(project_map)
            except OSError as exc:
                self.write_error = f"couldn't write {self.path}: {exc}"
                raise
            self.write_error = self.read_error = None
            return result

    def relative(self, path: str) -> str | None:
        """``path`` as the map writes it: repo-relative, ``/`` separators, a folder ending with ``/``.
        None for the project root itself or anything outside the project."""
        raw = Path(path.strip()).expanduser()
        full = raw if raw.is_absolute() else self.project_dir / raw
        try:
            rel = full.resolve().relative_to(self.project_dir).as_posix()
        except (ValueError, OSError):
            return None
        if rel == ".":
            return None
        if full.is_dir() or (not full.exists() and path.rstrip().endswith("/")):
            rel += "/"
        return rel

    def map_feature(self, name: str, paths: list[str], abstract: str) -> FeatureCard:
        """Create or update the card named ``name``: abstract and predicted paths change, confirmations stay."""
        return self.update(lambda m: _upsert(m, name, paths, abstract))

    def confirm(self, name: str, read: Iterable[str], wrote: Iterable[str], run_id: str, chat_id: str | None) -> None:
        """Record what a run that mapped ``name`` actually read and wrote. A name with no card is ignored."""

        def apply(m: ProjectMap) -> None:
            card = m.feature(name)
            if card is None:
                return
            card.read = sorted({*card.read, *read})
            card.wrote = sorted({*card.wrote, *wrote})
            if run_id not in card.runs:
                card.runs.append(run_id)
            if chat_id and chat_id not in card.chats:
                card.chats.append(chat_id)
            card.updated = time.time()

        self.update(apply)

    def fill(self, data: dict[str, Any], *, commit: str | None, model: str | None) -> ProjectMap:
        """Take a survey's output: the areas are replaced, the features upserted (confirmations kept)."""

        def apply(m: ProjectMap) -> ProjectMap:
            areas: list[Area] = []
            for item in data.get("areas", []):
                rel = self.relative(str(item.get("path", "")))
                if rel is None or any(a.path == rel for a in areas):
                    continue
                contains = self._relative_all(item.get("contains", []))
                depends = [p for p in self._relative_all(item.get("depends_on", [])) if p != rel]
                areas.append(Area(rel, _one_line(str(item.get("purpose", ""))), contains, depends))
            m.areas = areas
            for item in data.get("features", []):
                paths = self._relative_all(item.get("paths", []))
                name = str(item.get("name", "")).strip()
                if name and name.casefold() not in RESERVED_NAMES and paths:
                    _upsert(m, name, paths, str(item.get("abstract", "")))
            m.surveyed = Survey(time.time(), commit, model)
            return m

        return self.update(apply, rewrite=True)

    def _relative_all(self, paths: Any) -> list[str]:
        if not isinstance(paths, list):
            return []
        out: list[str] = []
        for p in paths:
            rel = self.relative(str(p))
            if rel is not None and rel not in out:
                out.append(rel)
        return out


def _upsert(m: ProjectMap, name: str, paths: list[str], abstract: str) -> FeatureCard:
    card = m.feature(name)
    if card is None:
        card = FeatureCard(name.strip(), "")
        m.features.append(card)
    card.abstract = abstract.strip()
    card.paths = list(paths)
    card.updated = time.time()
    return card


def _one_line(text: str, limit: int | None = None) -> str:
    line = " ".join(text.strip().split())
    return line if limit is None or len(line) <= limit else line[: limit - 1] + "…"


# --- rendering: what the agent reads ------------------------------------------------------

NOTE = "When a task teaches you what a feature is or where it lives, record it with map_feature."
"""Added for runs that have the tool; the tool's own description says the rest."""


def estimate(text: str) -> int:
    """Tokens in ``text``, estimated at ``CHARS_PER_TOKEN`` characters each."""
    return -(-len(text) // CHARS_PER_TOKEN)


def tree(areas: list[Area]) -> list[tuple[int, Area]]:
    """Areas in reading order with their depth: roots (contained by no other area) first, then their
    ``contains`` children. Each area appears once; a cycle or an orphan starts at depth 0."""
    by_path = {a.path: a for a in areas}
    contained = {c for a in areas for c in a.contains if c in by_path and c != a.path}
    order: list[tuple[int, Area]] = []
    seen: set[str] = set()

    def walk(area: Area, depth: int) -> None:
        if area.path in seen:
            return
        seen.add(area.path)
        order.append((depth, area))
        for child in area.contains:
            if (c := by_path.get(child)) is not None:
                walk(c, depth + 1)

    for area in areas:
        if area.path not in contained:
            walk(area, 0)
    for area in areas:
        walk(area, 0)
    return order


def render(project_map: ProjectMap, budget: int = BUDGET_TOKENS, *, note: bool = False) -> str:
    """The map as the agent reads it, kept within about ``budget`` tokens; ``note`` adds the line
    asking the model to call ``map_feature``.

    Over budget, in order: purposes and abstracts are clipped, the deepest areas are dropped,
    then the features' path lists, then features, then root areas; a last line says what was left out.
    """
    m = project_map
    rows = tree(m.areas)
    features = list(m.features)
    purpose_limit, abstract_limit, with_paths = 400, 1000, True
    omitted_areas = omitted_features = 0
    surveyed = time.strftime("%Y-%m-%d", time.localtime(m.surveyed.at)) if m.surveyed.at else "never"

    def area_line(depth: int, a: Area) -> str:
        deps = f" -> {', '.join(a.depends_on)}" if a.depends_on else ""
        return "  " * (depth + 1) + f"{a.path}: {_one_line(a.purpose, purpose_limit)}{deps}"

    def feature_line(f: FeatureCard) -> str:
        paths = f" [{', '.join(f.paths)}]" if with_paths and f.paths else ""
        return f"  {f.name}: {_one_line(f.abstract, abstract_limit)}{paths}"

    def build() -> str:
        lines = [f'<project_map areas={len(m.areas)} features={len(m.features)} surveyed="{surveyed}">']
        if rows:
            lines.append("Areas (path: purpose -> depends on):")
            lines += [area_line(d, a) for d, a in rows]
        if features:
            lines.append("Features (name: abstract [paths]):")
            lines += [feature_line(f) for f in features]
        if omitted_areas or omitted_features:
            lines.append(f"({omitted_areas} areas and {omitted_features} features omitted; /map shows them all)")
        lines.append("</project_map>")
        if note:
            lines.append(NOTE)
        return "\n".join(lines)

    text = build()
    while estimate(text) > budget:
        over = (estimate(text) - budget) * CHARS_PER_TOKEN
        if purpose_limit > 100:
            purpose_limit, abstract_limit = 100, 240
        elif any(d > 0 for d, _ in rows):
            while over > 0 and any(d > 0 for d, _ in rows):
                i = max(range(len(rows)), key=lambda k: (rows[k][0], k))
                over -= len(area_line(*rows[i])) + 1
                del rows[i]
                omitted_areas += 1
        elif with_paths:
            with_paths = False
        elif features:
            while over > 0 and features:
                over -= len(feature_line(features.pop())) + 1
                omitted_features += 1
        elif rows:
            while over > 0 and rows:
                over -= len(area_line(*rows.pop())) + 1
                omitted_areas += 1
        else:
            break
        text = build()
    return text


def can_map(spec: AgentSpec) -> bool:
    """Whether ``spec``'s tool patterns reach ``map_feature`` (good enough for the built-in profiles)."""
    return "-map_feature" not in spec.tools and ("map_feature" in spec.tools or "*" in spec.tools)


def opens_with(history: tuple[Message, ...], prefix: tuple[Message, ...]) -> bool:
    """Whether ``history`` begins with ``prefix``: same roles and same first content blocks. Only the
    first block is compared because the loop appends a context block to the newest user message."""
    if len(history) < len(prefix):
        return False
    return all(h.role == p.role and h.content[:1] == p.content[:1] for h, p in zip(history, prefix, strict=False))


class MapContext:
    """The project map in the system prompt, the cached prefix; nothing when the project has no map.

    The map belongs to the run's project (``run.cwd``, as ``AgentsMdContext`` reads it). A conversation
    keeps the text it started with: the map is rendered when a depth-0 run begins a conversation and
    reused for its later turns and its Q&A, so the cached prefix stays byte-identical while the store
    changes underneath (a ``map_feature`` call mid-turn, a survey). The conversation is known by its
    first exchange (the first user message, then the first reply once there is one); a new conversation,
    a resumed chat that began differently, and every subagent read the store as it is.
    """

    def __init__(self) -> None:
        self._pinned: tuple[tuple[Message, ...], str | None] | None = None
        """The opening of the conversation last rendered for (one or two messages), and its text."""

    def system(self, run: RunView) -> str | None:
        history = run.messages[:-1]
        if run.depth == 0 and history and self._pinned is not None and opens_with(history, self._pinned[0]):
            prefix, text = self._pinned
            if len(prefix) < 2 <= len(history):
                self._pinned = (history[:2], text)
            return text
        m = MapStore.of(run.cwd).load()
        text = None if m.empty else render(m, note=can_map(run.spec))
        if run.depth == 0 and run.messages:
            self._pinned = (run.messages[:2] if len(history) >= 2 else run.messages[:1], text)
        return text

    def before_turn(self, run: RunView) -> str | None:
        return None


# --- the tool ------------------------------------------------------------------------------


def feature_tool(store: MapStore) -> Tool:
    # "allow", although it writes: only the map store under .governor/ changes, never the user's files,
    # and a headless session must be able to call it. /map <feature> is where the user reads what it wrote.
    @tool(permission="allow")
    def map_feature(ctx: ToolContext, name: str, paths: list[str], abstract: str) -> str:
        """Record a feature in the project map: what it is and where it lives. Call it when a
        survey or a task teaches you what a feature does or which files it spans, and again
        when you learn the card is wrong. Calling it for a name that exists updates that card;
        the paths runs actually touched are kept.

        Args:
            name: the feature as a person would name it ("saved chats", "/qa mode"), used as its id.
            paths: repo-relative files and folders it lives in (folders end with /).
            abstract: two to four sentences: what it does, how it works, what it depends on.
        """
        if not name.strip():
            raise ToolError("name is empty")
        if name.strip().casefold() in RESERVED_NAMES:
            raise ToolError(f"{name!r} is reserved; name the feature as a person would")
        clean: list[str] = []
        for p in paths:
            rel = store.relative(p)
            if rel is None:
                raise ToolError(f"{p!r} is outside the project; give repo-relative paths")
            if rel not in clean:
                clean.append(rel)
        if not clean:
            raise ToolError("paths is empty; give at least one file or folder")
        try:
            card = store.map_feature(name, clean, abstract)
        except OSError as exc:
            raise ToolError(f"couldn't save the map: {exc}") from exc
        on_disk = sum(1 for p in clean if (store.project_dir / p).exists())
        return f"mapped {card.name}: {len(clean)} paths, {on_disk} on disk"

    return map_feature


# --- the sink: confirmed paths -------------------------------------------------------------


@dataclass
class _Touched:
    read: set[str] = field(default_factory=set)
    wrote: set[str] = field(default_factory=set)
    features: list[str] = field(default_factory=list)


class MapSink:
    """Records onto each feature card a run mapped (``map_feature``) the paths the run read or wrote.

    A run tree is one unit: a subagent's reads count for the depth-0 run that delegated to it. A
    tool call counts only when its result was ``ok``, so refused, invalid and failed calls never
    confirm a path. The card is updated when the depth-0 run ends, however it ended; nothing is
    created here. The sink never raises: ``EventBus.emit`` has no isolation, so a failed write is
    kept on ``store.write_error`` for ``/map`` instead of ending the run.
    """

    def __init__(self, store: MapStore, chat_id: Callable[[], str | None] = lambda: None) -> None:
        self.store = store
        self.chat_id = chat_id
        self._root: dict[str, str] = {}
        """run_id -> the depth-0 run_id of its tree."""
        self._pending: dict[tuple[str, str], tuple[str, str]] = {}
        """(run_id, tool call id) -> (kind, path or feature name), until the result arrives."""
        self._touched: dict[str, _Touched] = {}
        """depth-0 run_id -> what its tree has touched and mapped so far."""

    def handle(self, event: Event) -> None:
        data = event.data
        if event.type == "run_start":
            root = self._root.get(event.parent_run_id or "", event.run_id)
            self._root[event.run_id] = root
            self._touched.setdefault(root, _Touched())
        elif event.type == "tool_call":
            name = data.get("name")
            kind = "read" if name in READ_TOOLS else "wrote" if name in WRITE_TOOLS else None
            key = "path" if kind else "name" if name == "map_feature" else None
            value = (data.get("input") or {}).get(key) if key else None
            if isinstance(value, str):
                self._pending[(event.run_id, data["id"])] = (kind or "feature", value)
        elif event.type == "tool_result":
            item = self._pending.pop((event.run_id, data["id"]), None)
            touched = self._touched.get(self._root.get(event.run_id, ""))
            if item is None or touched is None or data.get("outcome") != "ok":
                return
            kind, value = item
            if kind == "feature":
                if value not in touched.features:
                    touched.features.append(value)
            elif (rel := self.store.relative(value)) is not None:
                (touched.read if kind == "read" else touched.wrote).add(rel)
        elif event.type == "run_end" and event.depth == 0:
            touched = self._touched.pop(event.run_id, None)
            tree_runs = {r for r, root in self._root.items() if root == event.run_id}
            self._root = {r: root for r, root in self._root.items() if root != event.run_id}
            self._pending = {k: v for k, v in self._pending.items() if k[0] not in tree_runs}
            if touched is not None and touched.features:
                chat_id = self.chat_id()
                for name in touched.features:
                    try:
                        self.store.confirm(name, touched.read, touched.wrote, event.run_id, chat_id)
                    except Exception as exc:  # the last line of defence: EventBus.emit isolates nothing
                        self.store.write_error = f"couldn't record the paths of {name!r}: {exc}"


# --- the survey ----------------------------------------------------------------------------

SURVEY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "areas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "repo-relative; a folder ends with /"},
                    "purpose": {"type": "string", "description": "one line"},
                    "contains": {"type": "array", "items": {"type": "string"}},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["path", "purpose"],
                "additionalProperties": False,
            },
        },
        "features": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "abstract": {"type": "string", "description": "two to four sentences"},
                    "paths": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "abstract", "paths"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["areas"],
    "additionalProperties": False,
}

SURVEY_SYSTEM = """You survey a project for its map. You read, list and search files; you cannot modify anything.

Areas are folders and modules: give every top-level folder and file that matters and the main modules in \
the source folders, skipping caches, build output, virtualenvs and .governor/. For each: its repo-relative \
path (a folder ends with /), a one-line purpose, the paths it contains (direct children that are areas \
themselves) and the paths it depends on (imports, reads, calls).

Features are what the project does, from a user's or maintainer's point of view, each spanning the paths \
it lives in: a two-to-four-sentence abstract of what it does and how, and those paths.

List or grep first, then read only what you need (READMEs, package inits, module docstrings). Keep it \
compact: an agent reads the whole map at the start of every run within about 3000 tokens, so prefer \
15-40 areas, one line each, and 5-15 features. Your final reply is only the JSON object the schema asks for."""

SURVEY_PROMPT = "Survey this project and reply with its map."


def survey_spec(profiles: dict[str, AgentSpec]) -> AgentSpec:
    """The explore profile (read-only tools, the subagent model) turned into the survey: its own prompt,
    the map as its output schema, room for a long read."""
    base = profiles.get("explore") or profiles["general"]
    read_only = ["read_file", "list_files", "grep"]
    return replace(
        base,
        name="map-survey",
        description="Surveys the project for its map.",
        system_prompt=SURVEY_SYSTEM,
        tools=read_only if "*" in base.tools else [t for t in base.tools if t != "task"],
        output_schema=SURVEY_SCHEMA,
        max_turns=max(base.max_turns, 60),
        subagent=False,
    )


def refresh_prompt(current: ProjectMap, changes: str | None) -> str:
    """The survey prompt seeded with the whole current map (not the budgeted rendering, which clips and
    omits) and the change list since it was surveyed."""
    seed = {
        "areas": [asdict(a) for a in current.areas],
        "features": [{"name": f.name, "abstract": f.abstract, "paths": f.paths} for f in current.features],
    }
    changed = (
        f"<changes>\n{changes}\n</changes>"
        if changes
        else "(no change list available: not a git checkout, or no survey commit)"
    )
    return (
        f"{SURVEY_PROMPT} The current map is below; update it for what changed, keep what is still right, "
        f"and reply with the whole map.\n\n<current_map>\n{json.dumps(seed, ensure_ascii=False)}\n</current_map>"
        f"\n\n{changed}"
    )


def _git(project_dir: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(project_dir), *args], capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_head(project_dir: Path) -> str | None:
    return _git(project_dir, "rev-parse", "HEAD")


def changes_since(project_dir: Path, commit: str | None) -> str | None:
    """``git diff --stat`` from ``commit`` to the working tree plus untracked files, or None without git,
    a commit, or when the commit is gone (a rebase)."""
    if not commit:
        return None
    diff = _git(project_dir, "diff", "--stat", commit)
    if diff is None:
        return None
    untracked = _git(project_dir, "ls-files", "--others", "--exclude-standard") or ""
    extra = [p for p in untracked.splitlines() if not p.startswith(".governor/")]
    parts = [diff or "(no committed changes)"] + ([f"untracked: {', '.join(extra)}"] if extra else [])
    return "\n".join(parts)


# --- the command ---------------------------------------------------------------------------


def status_line(m: ProjectMap) -> str:
    when = (
        f"surveyed {time.strftime('%b %d %H:%M', time.localtime(m.surveyed.at))}"
        + (f" at {m.surveyed.commit[:7]}" if m.surveyed.commit else "")
        if m.surveyed.at
        else "not surveyed"
    )
    return f"map: {len(m.areas)} areas · {len(m.features)} features · {when} · ~{estimate(render(m))} tokens in every system prompt"


def _inspect(session: Session, store: MapStore) -> None:
    m = store.load()
    for error in store.errors:
        session.say(error, "error")
    if m.empty:
        if not store.read_error:
            session.say("no project map yet; /map survey builds one (an explore run that reads the repo)", "dim")
        return
    session.say(status_line(m), "dim")
    session.say(render(m))


def _show(session: Session, store: MapStore, name: str) -> None:
    m = store.load()
    card = m.feature(name)
    if card is None:
        names = ", ".join(f.name for f in m.features) or "none yet"
        session.say(f"no feature named {name!r} in the map; features: {names}", "error")
        return
    confirmed = {**dict.fromkeys(card.read, "read"), **dict.fromkeys(card.wrote, "wrote")}
    covered: set[str] = set()
    lines = [f"{card.name}: {card.abstract}"]

    def disk(p: str) -> str:
        return "" if (store.project_dir / p).exists() else " · not on disk"

    for p in sorted(card.paths):
        if p.endswith("/"):  # a predicted folder is confirmed by the files touched under it
            under = sorted(c for c in confirmed if c.startswith(p))
            covered.update(under)
            state = f"predicted, confirmed ({len(under)} files under it)" if under else "predicted, not touched yet"
            lines.append(f"  {p}  {state}{disk(p)}")
            lines += [f"    {c}  {confirmed[c]}{disk(c)}" for c in under]
        else:
            covered.add(p)
            state = f"predicted, confirmed ({confirmed[p]})" if p in confirmed else "predicted, not touched yet"
            lines.append(f"  {p}  {state}{disk(p)}")
    lines += [f"  {c}  touched ({confirmed[c]}), not predicted{disk(c)}" for c in sorted(set(confirmed) - covered)]
    session.say("\n".join(lines))
    session.say(
        f"{len(card.runs)} runs · {len(card.chats)} chats · updated {time.strftime('%b %d %H:%M', time.localtime(card.updated))}",
        "dim",
    )


def _survey(session: Session, store: MapStore, refresh: bool) -> None:
    spec = survey_spec(session.runtime.profiles)
    prompt = SURVEY_PROMPT
    if refresh:
        current = store.load()
        if current.empty:
            session.say("nothing to refresh yet; surveying instead", "dim")
        else:
            prompt = refresh_prompt(current, changes_since(store.project_dir, current.surveyed.commit))
    session.say(f"{'refreshing' if refresh else 'surveying'} the project…", "dim")
    # The survey reads the repo, not the map: its own run gets every context provider but this feature's.
    without_map = [p for p in session.runtime.context_providers if not isinstance(p, MapContext)]
    turn = session.run(spec, prompt, runtime=replace(session.runtime, context_providers=without_map), quiet=True)
    if turn.error is not None:
        session.say(turn.error, "error")
        return
    if turn.stopped:
        session.say("stopped; the map is unchanged", "dim")
        return
    if turn.result is None or turn.model is None:
        return
    if not isinstance(turn.result.structured, dict):
        session.say(f"the survey returned no map (stopped: {turn.result.stop}); the map is unchanged", "error")
        return
    try:
        m = store.fill(turn.result.structured, commit=git_head(store.project_dir), model=turn.model.alias)
    except OSError as exc:
        session.say(f"couldn't save the map: {exc}", "error")
        return
    session.say(f"map: {len(m.areas)} areas · {len(m.features)} features · {turn.status()}", "dim")
    if session.history:
        session.say("this conversation keeps the map it started with; /clear starts one with the new map", "dim")


def map_command(session: Session, arg: str) -> None:
    """``/map`` shows the map; ``/map survey`` builds it, ``/map refresh`` updates it, ``/map <feature>`` shows a card."""
    store = MapStore.of(session.project_dir)
    if arg in RESERVED_NAMES:
        _survey(session, store, refresh=arg == "refresh")
    elif arg:
        _show(session, store, arg)
    else:
        _inspect(session, store)


def _completions(session: Session) -> list[str]:
    return ["survey", "refresh", *(f.name for f in MapStore.of(session.project_dir).load().features)]


COMMAND = Command("map", "the project map: show it, survey or refresh it, or show a feature", map_command, _completions)


# --- the feature's registrations ------------------------------------------------------------


def tools(project_dir: Path) -> list[Tool]:
    return [feature_tool(MapStore.of(project_dir))]


def context_providers() -> list[ContextProvider]:
    return [MapContext()]


def sinks(session: Session) -> list[Sink]:
    store = MapStore.of(session.project_dir)
    return [MapSink(store, lambda: session.chat.id if session.chat is not None else None)]
