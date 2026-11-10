"""The project map, driven headlessly: a survey fills the store, the map reaches the system prompt and a
conversation keeps it, the sink confirms the paths a run touched, map_feature updates a card, /map inspects."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from agent_core import AgentSpec, Message, TextBlock, ToolContext, ToolError
from agent_core.testing import ScriptedProvider, text, tool_call, tool_calls
from governor.chats import Chat
from governor.config import Settings
from governor.map import (
    BUDGET_TOKENS,
    NOTE,
    SURVEY_PROMPT,
    SURVEY_SCHEMA,
    Area,
    FeatureCard,
    MapStore,
    ProjectMap,
    can_map,
    changes_since,
    estimate,
    feature_tool,
    git_head,
    render,
    survey_spec,
    tree,
)
from governor.runtime import build_session
from governor.session import Session

pytestmark = pytest.mark.usefixtures("isolated")

SURVEY = {
    "areas": [
        {"path": "src/", "purpose": "Source packages", "contains": ["src/app/"]},
        {"path": "src/app", "purpose": "The app", "contains": ["src/app/core.py"], "depends_on": []},
        {"path": "./src/app/core.py", "purpose": "The loop", "depends_on": ["src/app/"]},
        {"path": "README.md", "purpose": "Overview"},
    ],
    "features": [{"name": "the loop", "abstract": "Runs things.", "paths": ["src/app/core.py", "README.md"]}],
}
READ_ONLY = ["read_file", "list_files", "grep"]


class Says:
    """The one Interaction method the map's command needs; records what was said, by kind."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def say(self, text: str, kind: str = "plain") -> None:
        self.lines.append((kind, text))

    def said(self, kind: str | None = None) -> str:
        return "\n".join(t for k, t in self.lines if kind is None or k == kind)


def project(root: Path) -> Path:
    (root / "src" / "app").mkdir(parents=True)
    (root / "src" / "app" / "core.py").write_text("x = 1\n")
    (root / "README.md").write_text("# demo\n")
    return root


def make_session(project_dir: Path, provider: ScriptedProvider, io: Says | None = None, **kwargs) -> Session:
    Settings(provider="openai").save()
    return build_session(project_dir, io=io, provider_for=lambda info: provider, **kwargs)  # type: ignore[arg-type]


def tool_results(turn) -> list:
    return [b for m in turn.result.messages for b in m.content if b.type == "tool_result"]


def no_git(monkeypatch: pytest.MonkeyPatch, isolated: Path) -> None:
    """The temp project must not count as part of an enclosing checkout, wherever pytest puts it."""
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(isolated.parent))


def git(root: Path, *args: str) -> str:
    env = {
        **os.environ,
        **{f"GIT_{w}_{k}": v for w in ("AUTHOR", "COMMITTER") for k, v in (("NAME", "t"), ("EMAIL", "t@x"))},
    }
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True, env=env).stdout


# --- the survey ----------------------------------------------------------------------------


def test_survey_fills_the_store(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    no_git(monkeypatch, isolated)
    project(isolated)
    provider = ScriptedProvider([tool_call("list_files", {"pattern": "**/*"}), text(json.dumps(SURVEY))])
    io = Says()
    session = make_session(isolated, provider, io)

    session.submit("/map survey")
    request = provider.requests[0]
    assert (
        request.output_schema == SURVEY_SCHEMA and request.model == "gpt-6-luna"
    )  # explore's read-only tools, its model
    assert [t.name for t in request.tools] == READ_ONLY
    assert "You survey a project" in request.system and "<project_map" not in request.system
    assert request.messages[-1].text == SURVEY_PROMPT
    m = MapStore.of(isolated).load()
    assert [a.path for a in m.areas] == ["src/", "src/app/", "src/app/core.py", "README.md"]  # paths normalised
    assert m.areas[0].contains == ["src/app/"] and m.areas[2].depends_on == ["src/app/"]
    [card] = m.features
    assert card.name == "the loop" and card.paths == ["src/app/core.py", "README.md"] and card.read == []
    assert m.surveyed.at is not None and m.surveyed.model == "openai/luna" and m.surveyed.commit is None
    assert session.history is None  # the survey is a side run, not part of the conversation
    assert "map: 4 areas · 1 features · [luna]" in io.said("dim")
    assert "keeps the map it started with" not in io.said()  # no conversation to tell

    spec = survey_spec(session.runtime.profiles)
    assert spec.max_turns == 60 and spec.tools == READ_ONLY and not spec.subagent
    assert survey_spec({"general": AgentSpec("general", tools=["*"])}).tools == READ_ONLY  # never the write tools


def test_a_survey_that_returns_no_map_leaves_the_store_alone(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = ScriptedProvider([text("not json"), text("still not json")])
    io = Says()
    session = make_session(isolated, provider, io)
    session.submit("/map survey")
    assert "the survey returned no map (stopped: invalid_output)" in io.said("error")
    assert MapStore.of(isolated).load().empty and not MapStore.of(isolated).path.exists()


def test_refresh_on_an_empty_store_surveys_and_a_survey_tells_a_conversation(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    project(isolated)
    provider = ScriptedProvider([text(json.dumps(SURVEY)), text("hello"), text(json.dumps(SURVEY))])
    io = Says()
    session = make_session(isolated, provider, io)
    session.submit("/map refresh")
    assert "nothing to refresh yet; surveying instead" in io.said("dim")
    assert provider.requests[0].messages[-1].text == SURVEY_PROMPT
    session.ask("hi")
    session.submit("/map survey")
    assert "this conversation keeps the map it started with; /clear starts one with the new map" in io.said("dim")


def test_refresh_keeps_confirmations_and_seeds_the_prompt(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    no_git(monkeypatch, isolated)
    project(isolated)
    store = MapStore.of(isolated)
    store.fill(SURVEY, commit=None, model="m")
    store.confirm("the loop", ["README.md"], ["notes.md"], "r1", "c1")
    updated = {
        "areas": [{"path": "src/", "purpose": "Source"}],
        "features": [{"name": "The Loop", "abstract": "Still runs things.", "paths": ["src/"]}],
    }
    provider = ScriptedProvider([text(json.dumps(updated))])
    io = Says()
    session = make_session(isolated, provider, io)

    session.submit("/map refresh")
    request = provider.requests[0]
    prompt = request.messages[-1].text
    assert "update it for what changed" in prompt and "<current_map>" in prompt
    assert '"path": "src/app/core.py", "purpose": "The loop"' in prompt  # the whole store, not the budgeted rendering
    assert '"name": "the loop", "abstract": "Runs things."' in prompt and '"read"' not in prompt
    assert "no change list available" in prompt  # the temp project is not a git checkout
    assert "<project_map" not in request.system  # the survey reads the repo, not the map
    m = store.load()
    assert [a.path for a in m.areas] == ["src/"]
    [card] = m.features
    assert card.name == "the loop" and card.abstract == "Still runs things." and card.paths == ["src/"]
    assert card.read == ["README.md"] and card.wrote == ["notes.md"] and card.runs == ["r1"] and card.chats == ["c1"]
    assert "refreshing the project…" in io.said("dim") and "map: 1 areas · 1 features" in io.said("dim")


def test_refresh_lists_the_changes_since_the_survey(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    project(isolated)
    git(isolated, "init", "-q")
    git(isolated, "add", ".")
    git(isolated, "commit", "-q", "-m", "base")
    head = git_head(isolated)
    assert head is not None and len(head) == 40
    store = MapStore.of(isolated)
    store.fill(SURVEY, commit=head, model="m")
    (isolated / "README.md").write_text("# demo\nmore\n")  # edited, not committed
    (isolated / "new.py").write_text("y = 2\n")  # untracked, like .governor/map/map.json itself
    provider = ScriptedProvider([text(json.dumps(SURVEY))])
    session = make_session(isolated, provider, Says())

    session.submit("/map refresh")
    prompt = provider.requests[0].messages[-1].text
    assert "<changes>" in prompt and "README.md |" in prompt
    assert re.search(r"untracked: .*new\.py", prompt) and ".governor" not in prompt
    assert store.load().surveyed.commit == head  # the refresh re-read the same commit: nothing new committed
    assert changes_since(isolated, "0" * 40) is None  # a commit that is gone (a rebase) means no change list


def test_fill_cleans_what_the_model_sent(isolated: Path):
    project(isolated)
    dirty = {
        "areas": [
            {"path": "src/", "purpose": "A"},
            {"path": "src", "purpose": "the same folder again"},
            {
                "path": "README.md",
                "purpose": "R\nmulti  line",
                "depends_on": ["README.md", "/elsewhere"],
                "contains": "oops",
            },
            {"path": "/elsewhere", "purpose": "outside"},
        ],
        "features": [
            {"name": "Refresh", "abstract": "a reserved name", "paths": ["README.md"]},
            {"name": "outside", "abstract": "no path in the project", "paths": ["/elsewhere"]},
            {"name": "ok", "abstract": "kept", "paths": ["src/", "src/"]},
        ],
    }
    m = MapStore.of(isolated).fill(dirty, commit=None, model=None)
    assert [a.path for a in m.areas] == ["src/", "README.md"]
    assert m.areas[1].purpose == "R multi line" and m.areas[1].depends_on == [] and m.areas[1].contains == []
    assert [(f.name, f.paths) for f in m.features] == [("ok", ["src/"])]


# --- the map in the system prompt ----------------------------------------------------------


def test_the_map_reaches_the_system_prompt_and_a_conversation_keeps_it(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    project(isolated)
    provider = ScriptedProvider([text(r) for r in ("none", "one", "**why**", "two", "resumed", "fresh")])
    session = make_session(isolated, provider)

    session.ask("no map yet")
    assert "<project_map" not in provider.requests[0].system  # no map: nothing is injected
    session.submit("/clear")
    store = MapStore.of(isolated)
    store.fill(SURVEY, commit="abc1234def", model="m")

    session.ask("first")
    system = provider.requests[1].system
    head = system.index("<project_map areas=4 features=1 surveyed=")
    assert system.index("<environment>") < head  # after the builtin providers, part of the cached prefix
    assert (
        "\n  src/: Source packages\n    src/app/: The app\n      src/app/core.py: The loop -> src/app/\n  README.md: Overview\n"
        in system
    )
    assert "\n  the loop: Runs things. [src/app/core.py, README.md]\n</project_map>\n" in system
    assert system.rstrip().endswith(NOTE)  # the general profile has map_feature
    assert provider.requests[1].messages[-1].text == "first"  # before_turn adds nothing

    store.map_feature("saved chats", ["README.md"], "Learned mid-conversation.")
    session.submit("/qa why?")  # Q&A repeats the conversation's prompt byte for byte
    session.submit("/done")
    session.ask("second")
    assert provider.requests[2].system == system and provider.requests[3].system == system  # the pinned text

    other = Chat.new("first", "openai", "openai/astra", "general")  # another chat with the same opener
    other.messages = [Message.user("first"), Message("assistant", (TextBlock("a different reply"),))]
    session.resume(other)
    session.ask("continue")
    assert (
        "saved chats: Learned mid-conversation." in provider.requests[4].system
    )  # it began differently: current store
    session.submit("/clear")
    session.ask("first")  # a new conversation with the old opener
    assert "\n  saved chats: Learned mid-conversation. [README.md]\n" in provider.requests[5].system


def test_subagents_read_the_current_map_and_leave_the_pin_alone(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    project(isolated)
    store = MapStore.of(isolated)
    store.fill(SURVEY, commit=None, model="m")
    provider = ScriptedProvider(
        [
            text("one"),
            tool_call("task", {"agent": "explore", "prompt": "look"}),
            tool_call("read_file", {"path": "README.md"}),  # the subagent's turns
            text("report"),
            text("two"),
            text("three"),
        ]
    )
    session = make_session(isolated, provider)
    session.ask("first")
    pinned = provider.requests[0].system
    store.map_feature("saved chats", ["README.md"], "Learned.")
    session.ask("second")
    session.ask("third")
    assert provider.requests[1].system == pinned and provider.requests[4].system == pinned  # the parent's turns
    assert provider.requests[5].system == pinned
    explore = provider.requests[2].system
    assert "<project_map" in explore and "saved chats: Learned." in explore and NOTE not in explore


def test_the_rendered_map_stays_within_budget():
    areas = [Area("pkg/", "root package", [f"pkg/m{i}.py" for i in range(400)])]
    areas += [Area(f"pkg/m{i}.py", f"module {i}: " + "does things " * 20) for i in range(400)]
    features = [FeatureCard(f"feature {i}", "An abstract sentence. " * 30, [f"pkg/m{i}.py"]) for i in range(40)]
    big = render(ProjectMap(areas, features))
    assert estimate(big) <= BUDGET_TOKENS
    assert big.startswith("<project_map areas=401 features=40") and "\n  pkg/: root package\n" in big
    assert " areas and 0 features omitted; /map shows them all)\n</project_map>" in big
    assert "  feature 0: An abstract sentence." in big and len(big.splitlines()) > 20

    small = render(ProjectMap(areas[:3], features[:1]))
    assert "omitted" not in small and ("does things " * 20).strip() in small  # within budget, nothing is clipped
    assert NOTE not in small and render(ProjectMap(areas[:3]), note=True).endswith(NOTE)
    assert can_map(AgentSpec("g", tools=["*"])) and can_map(AgentSpec("g", tools=["read_file", "map_feature"]))
    assert not can_map(AgentSpec("e", tools=["read_file"])) and not can_map(AgentSpec("e", tools=["*", "-map_feature"]))


def test_pruning_drops_path_lists_then_features_then_root_areas():
    features = [
        FeatureCard(f"feature {i}", "An abstract sentence. " * 15, [f"pkg/m{i}.py", f"pkg/n{i}.py"]) for i in range(300)
    ]
    roots = [Area(f"root{i}/", "a root") for i in range(10)]
    rendered = render(ProjectMap(roots, features))
    assert estimate(rendered) <= BUDGET_TOKENS and rendered.startswith("<project_map areas=10 features=300")
    assert "[pkg/" not in rendered  # path lists go before any feature
    kept = re.findall(r"^  feature (\d+): ", rendered, re.M)
    assert kept[0] == "0" and len(kept) < 300 and all(int(k) == i for i, k in enumerate(kept))  # the last ones go first
    assert f"(0 areas and {300 - len(kept)} features omitted; /map shows them all)" in rendered
    assert all(f"  root{i}/: a root" in rendered for i in range(10))

    wide = [Area(f"root{i}/", "purpose " * 30) for i in range(500)]
    rendered = render(ProjectMap(wide))
    assert estimate(rendered) <= BUDGET_TOKENS and "  root0/: " in rendered and "root499/" not in rendered
    assert re.search(r"\(\d+ areas and 0 features omitted", rendered)


def test_tree_renders_cycles_and_orphans_once():
    looped = [Area("a/", "A", ["b/"]), Area("b/", "B", ["a/", "zzz/"]), Area("c/", "C", ["a/"])]
    assert [(d, a.path) for d, a in tree(looped)] == [(0, "c/"), (1, "a/"), (2, "b/")]
    pure = [Area("a/", "A", ["b/"]), Area("b/", "B", ["a/"])]
    assert [(d, a.path) for d, a in tree(pure)] == [(0, "a/"), (1, "b/")]
    rendered = render(ProjectMap(pure))
    assert rendered.count("a/: A") == 1 and rendered.count("b/: B") == 1 and "zzz" not in rendered


# --- the sink ------------------------------------------------------------------------------


def test_the_sink_confirms_touched_paths(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GOVERNOR_ALLOW", "write_file")  # edit_file still needs approval, which no one can give
    project(isolated)
    reads = [("read_file", {"path": "README.md"}), ("read_file", {"path": "missing.md"})]
    writes = [
        ("write_file", {"path": "notes.md", "content": "n"}),
        ("edit_file", {"path": "README.md", "old_string": "demo", "new_string": "Demo"}),
    ]
    core = str(isolated / "src" / "app" / "core.py")
    provider = ScriptedProvider(
        [
            tool_call("task", {"agent": "explore", "prompt": "look at core.py"}),
            tool_call("read_file", {"path": "src/app/core.py"}),  # the subagent's turns
            text("core.py sets x"),
            tool_calls([*reads, *writes]),
            tool_call(
                "map_feature", {"name": "the loop", "paths": ["src/app/core.py", "/elsewhere/x"], "abstract": "x"}
            ),
            tool_call("map_feature", {"name": "the loop", "paths": [core, "docs/"], "abstract": "Runs."}),
            tool_call("map_feature", {"name": "docs", "paths": ["README.md"], "abstract": "Reading."}),
            text("done"),
            tool_call("map_feature", {"name": "The Loop", "paths": ["README.md"], "abstract": "Updated."}),
            text("ok"),
        ]
    )
    session = make_session(isolated, provider)

    turn = session.ask("map the loop")
    assert turn.ok and turn.result is not None
    results = tool_results(turn)
    assert any(r.is_error and "'/elsewhere/x' is outside the project" in r.content for r in results)
    assert any(r.content == "mapped the loop: 2 paths, 1 on disk" for r in results)
    m = MapStore.of(isolated).load()
    card = m.feature("the loop")
    assert card is not None and card.abstract == "Runs." and card.paths == ["src/app/core.py", "docs/"]
    assert card.read == ["README.md", "src/app/core.py"]  # the subagent's read rolls up; missing.md failed
    assert card.wrote == ["notes.md"]  # edit_file was refused
    assert card.runs == [turn.result.run_id] and card.chats == []  # the chat exists only once the turn is saved
    docs = m.feature("docs")  # every feature the run mapped gets the whole tree's paths
    assert docs is not None and docs.read == card.read and docs.wrote == card.wrote and docs.runs == card.runs

    second = session.ask("again")
    assert second.ok and second.result is not None and session.chat is not None
    card = MapStore.of(isolated).load().feature("the loop")
    assert card is not None and card.name == "the loop" and card.abstract == "Updated." and card.paths == ["README.md"]
    assert card.read == ["README.md", "src/app/core.py"] and card.wrote == ["notes.md"]  # confirmations stay
    assert card.runs == [turn.result.run_id, second.result.run_id] and card.chats == [session.chat.id]


# --- the tool and the store ----------------------------------------------------------------


def test_map_feature_updates_a_card(isolated: Path, ctx: ToolContext):
    project(isolated)
    store = MapStore.of(isolated)
    tool = feature_tool(store)
    assert tool.name == "map_feature" and tool.permission == "allow"
    assert tool.input_schema["required"] == ["name", "paths", "abstract"]

    assert tool.fn(ctx, name="Saved chats", paths=["src/app/core.py", "src/app/core.py"], abstract="First take.") == (
        "mapped Saved chats: 1 paths, 1 on disk"
    )
    store.confirm("saved chats", ["README.md"], [], "run1", "chat1")
    store.confirm("unknown", ["README.md"], [], "run1", "chat1")  # no card: nothing is created
    tool.fn(ctx, name="saved chats", paths=[str(isolated / "src" / "app"), "./README.md"], abstract=" Second take. ")
    [card] = store.load().features
    assert card.name == "Saved chats" and card.abstract == "Second take." and card.paths == ["src/app/", "README.md"]
    assert card.read == ["README.md"] and card.runs == ["run1"] and card.chats == ["chat1"]

    with pytest.raises(ToolError, match="outside the project"):
        tool.fn(ctx, name="x", paths=["../up.py"], abstract="a")
    with pytest.raises(ToolError, match="paths is empty"):
        tool.fn(ctx, name="x", paths=[], abstract="a")
    with pytest.raises(ToolError, match="name is empty"):
        tool.fn(ctx, name=" ", paths=["README.md"], abstract="a")
    with pytest.raises(ToolError, match="reserved"):
        tool.fn(ctx, name="Survey", paths=["README.md"], abstract="a")
    assert store.relative(".") is None and store.relative(str(isolated)) is None
    assert (
        store.relative("README.md/") == "README.md" and store.relative("new/") == "new/"
    )  # a slash never makes a file a folder
    assert MapStore.of(isolated) is store  # one store per project


def test_map_command_inspects(isolated: Path):
    project(isolated)
    io = Says()
    session = make_session(isolated, ScriptedProvider(), io)
    session.submit("/map")
    assert "no project map yet; /map survey builds one" in io.said("dim")

    store = MapStore.of(isolated)
    store.fill(SURVEY, commit="abc1234def", model="m")
    store.confirm("the loop", ["src/app/core.py", "src/app/__init__.py"], [], "r1", None)
    session.submit("/map")
    assert "map: 4 areas · 1 features · surveyed " in io.said("dim") and " at abc1234 · ~" in io.said("dim")
    shown = io.said("plain")
    assert "<project_map areas=4" in shown and NOTE not in shown  # the note is for the model

    session.submit("/map the loop")
    shown = io.said("plain")
    assert "the loop: Runs things." in shown
    assert "\n  README.md  predicted, not touched yet\n" in shown
    assert shown.endswith("\n  src/app/__init__.py  touched (read), not predicted · not on disk")  # unpredicted last
    assert "\n  src/app/core.py  predicted, confirmed (read)\n" in shown
    assert "1 runs · 0 chats · updated " in io.said("dim")

    store.map_feature(
        "the app", ["src/app/", "docs/"], "The app."
    )  # a predicted folder is confirmed by the files under it
    store.confirm("the app", ["src/app/core.py", "README.md"], ["src/app/new.py", "README.md"], "r2", None)
    io.lines.clear()
    session.submit("/map the app")
    assert io.said("plain") == (
        "the app: The app.\n"
        "  docs/  predicted, not touched yet · not on disk\n"
        "  src/app/  predicted, confirmed (2 files under it)\n"
        "    src/app/core.py  read\n"
        "    src/app/new.py  wrote · not on disk\n"
        "  README.md  touched (wrote), not predicted"  # written after being read: wrote wins
    )

    session.submit("/map nothing")
    assert "no feature named 'nothing' in the map; features: the loop, the app" in io.said("error")
    assert session.commands["map"].args is not None
    assert session.commands["map"].args(session) == ["survey", "refresh", "the loop", "the app"]


@pytest.mark.parametrize(
    "content",
    ["{not json", "[]", '"text"', '{"surveyed": "x"}', '{"features": [{"abstract": "x"}]}', '{"areas": "nope"}'],
)
def test_an_unreadable_map_never_stops_a_run(isolated: Path, monkeypatch: pytest.MonkeyPatch, content: str):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    project(isolated)
    store = MapStore.of(isolated)
    store.path.parent.mkdir(parents=True)
    store.path.write_text(content)
    provider = ScriptedProvider(
        [tool_call("map_feature", {"name": "x", "paths": ["README.md"], "abstract": "a"}), text("fine")]
    )
    io = Says()
    session = make_session(isolated, provider, io)
    turn = session.ask("hello")
    assert turn.ok  # neither the context provider nor the sink raised
    [result] = tool_results(turn)
    assert result.is_error and "couldn't save the map" in result.content and "is unreadable" in result.content
    assert store.path.read_text() == content  # an unreadable file is never written over by a tool
    session.submit("/map")
    assert "map.json is unreadable (" in io.said("error") and "/map survey rewrites it" in io.said("error")
    assert "no project map yet" not in io.said()
    store.fill(SURVEY, commit=None, model=None)  # the survey may rewrite it
    assert store.read_error is None and len(store.load().areas) == 4


def test_a_failed_write_is_reported_not_raised(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    project(isolated)
    store = MapStore.of(isolated)

    def failing(*args, **kwargs):
        raise OSError("disk full")

    provider = ScriptedProvider(
        [tool_call("map_feature", {"name": "x", "paths": ["README.md"], "abstract": "a"}), text("ok")]
    )
    io = Says()
    session = make_session(isolated, provider, io)
    with monkeypatch.context() as patched:
        patched.setattr(store, "confirm", failing)  # the sink's write at run_end fails
        turn = session.ask("map x")
    assert turn.ok and not tool_results(turn)[0].is_error  # the tool's own write succeeded; the run was never disturbed
    session.submit("/map")
    assert "couldn't record the paths of 'x': disk full" in io.said("error")
    assert "<project_map areas=0 features=1" in io.said()

    provider = ScriptedProvider(
        [tool_call("map_feature", {"name": "y", "paths": ["README.md"], "abstract": "a"}), text("ok")]
    )
    io = Says()
    session = make_session(isolated, provider, io)
    with monkeypatch.context() as patched:
        patched.setattr(store, "save", failing)  # now the tool's write fails: the model is told, nothing crashes
        turn = session.ask("map y")
    assert (
        turn.ok
        and tool_results(turn)[0].is_error
        and "couldn't save the map: disk full" in tool_results(turn)[0].content
    )
    session.submit("/map")
    assert "couldn't write" in io.said("error")

    store.map_feature("z", ["README.md"], "a")  # the next successful write clears the error
    io.lines.clear()
    session.submit("/map")
    assert io.said("error") == "" and "features=2" in io.said()
