# User guide

This guide shows you how to set up Governor and use it day to day. You don't
need to read the code. For how it's built, see `README.md`.

## What is this?

Governor is an AI agent you run from your terminal. It takes over the whole terminal
window, like other modern coding agents. You give it a task in plain
English, such as *"find where we parse dates and add a test for it"*. It works out the
steps by itself: it reads files, searches, edits code and runs commands until the task
is done. Then it tells you what it did.

It can use models from three places, and you choose which:

- **Amazon Bedrock**: billed to your AWS account. Claude, OpenAI and other models.
- **Anthropic API**: Claude, billed to an Anthropic account.
- **OpenAI API**: GPT models, billed to an OpenAI account.

A few terms used below:

- **Provider**: where the models run (one of the three above).
- **Agent**: an AI worker with a role, a set of tools and a model. Three come built in.
- **Tool**: an action the agent can take, such as reading a file or running a command.
- **Subagent**: a helper agent that the main agent starts for part of a task. Several can work at the same time.
- **Model**: the AI behind an agent. Bigger models are smarter and cost more.

## 1. Install (one time)

You need Python 3.11 or newer.

```bash
git clone https://github.com/johnchun10/Governor.git
cd Governor
python3.12 -m venv .venv
source .venv/bin/activate        # run this in every new terminal
pip install -e .
```

## 2. Start it

Go to the project you want the agent to work on and run `govern`:

```bash
cd ~/some/project
govern
```

The first time, it asks you to **pick a provider** (use the arrow keys and Enter), then
offers to **sign you in**:

- **Anthropic or OpenAI**: paste an API key. Governor shows where to create one.
- **Amazon Bedrock**: choose AWS console sign-in (`aws login`), an SSO profile
  (`aws sso login`), or a Bedrock API key, then the region. The AWS options open
  the usual AWS sign-in in your browser; they need the
  [AWS CLI](https://aws.amazon.com/cli/).

After signing in, it sends one tiny test message (a fraction of a cent) to check
that it works. Your choices are saved in `~/.config/governor/`, and keys go in
`~/.config/governor/credentials`, which only you can read. Next time, `govern`
starts straight away.

## 3. Give it a task

Type what you want and press Enter:

```
› explain what this project does and how it is organised
```

While it works, you'll see each step it takes (`⏺ read_file(...)`, `⏺ grep(...)`),
and its answer appears word by word as it is written. At the end it prints a line with
the tokens used and an estimated cost. The conversation carries on: your next message
can refer to the last answer. `/clear` starts a fresh conversation, and `/chats` brings an
earlier one back (section 4).

- **Ctrl-J** adds a new line, for longer tasks.
- **↑ / ↓** bring back earlier inputs.
- **Shift-Tab** cycles the effort (default → low → medium → high → max): how hard the
  model thinks on the next call. **Ctrl-1…4** set low…max and **Ctrl-0** the default, in
  terminals that report Ctrl+digit (those with the kitty keyboard protocol: Ghostty, kitty,
  WezTerm, foot…; Apple's Terminal doesn't, so use Shift-Tab or `/effort` there).
- **Esc** or **Ctrl-C** stops a running task at once: the model call in progress is
  cancelled, no tool runs, and the conversation is left as it was before that message.
- **Ctrl-D** (on an empty input) or `/exit` quits.
- The window scrolls with the mouse wheel. To select text for copying, hold **Shift**
  (**Option** in macOS Terminal/iTerm) while dragging.

**The agent asks before it does anything risky.** Before writing a file or running a
command, it stops and asks:

```
Allow write_file(path='fib.py', content='...')?
> 1. Yes
  2. Yes, and don't ask again for write_file this session
  3. No
```

If you say no, the agent is told and tries something else. Reading and searching
files never needs approval.

## 4. Commands

Type `/` and a list of commands appears above the input: ↑↓ to pick, Tab to complete,
Enter to run. Pickers (providers, models, permissions) open as a box in the middle of the
screen: ↑↓ and Enter, or Esc to cancel.

| Command | What it does |
| --- | --- |
| `/provider` | Switch provider (Bedrock, Anthropic, OpenAI…) |
| `/model` | Pick the models from the current provider's list: ←/→ switch between the main model, the cheaper one subagents use, and the one that answers `/qa`. `/model sol` sets the main model straight away |
| `/effort` | How hard the model thinks: low, medium, high, max, or the model's default. Applies to every call, subagents included |
| `/login` | Sign in to a provider, or see which ones you're signed in to |
| `/logout` | Forget a provider's saved key |
| `/agent` | Switch agent (see below) |
| `/doctor` | Check sign-in and send each current model a tiny test message |
| `/qa` | Ask questions about the conversation so far (see below). `/qa why did you pick X?` asks straight away. Inside Q&A, `/model` picks the model that answers and `/done` leaves |
| `/chats` | List this project's saved chats and resume one (see below) |
| `/map` | The project map: what the codebase is made of and where each feature lives, read by the agent at the start of every run (see below). `/map survey` builds it, `/map refresh` updates it, `/map <feature>` shows one feature |
| `/budget` | Cap what this session spends; a run stops when the cap is reached (see Costs). `/budget 2` sets $2, `/budget off` clears it |
| `/clear` | Start a new conversation |
| `/cost` | Tokens and cost so far this session |
| `/yolo` | Stop asking for approval. Use only in a folder you don't mind changing |
| `/verbose` | Also show tool output and all model text |
| `/help` | List commands |
| `/exit` | Quit Governor (or press Ctrl-D on an empty input) |

The bar at the bottom always shows the current provider, models, agent, cost and, when set, the effort.

**Ask about the conversation with `/qa`.** After a discussion or a task, `/qa` switches
the input to questions about it: *why did you change that file?*, *what did we decide
about retries?* Answers appear in a box under the conversation, and nothing in the box
becomes part of the conversation. When you leave, with Esc on an empty input or with
`/done`, the box disappears.

- Each question starts fresh, from the conversation as it was when you typed `/qa`.
- Start a line with `+` to follow up on the last answer instead. The input box's title
  says which one you're about to send.
- The agent can't use tools in Q&A; it answers from what is already in the conversation.

Questions go to the model the conversation used. It already has the conversation in
its prompt cache, so a question costs a fraction of sending the whole conversation
again (the `cache r` figure on the cost line). Inside Q&A, `/model` can pick a cheaper model
(`/model same` goes back), but that model starts without the cache and can't see the first
model's private reasoning.

**Resume a chat with `/chats`.** Every completed turn is saved in the project, under
`.governor/chats/`. `/chats` lists them, newest first, with their first prompt, turn count
and model; pick one and the conversation comes back exactly as it was, transcript included,
so your next message continues it and `/qa` works on it. A chat resumed on the same provider
and model reuses its prompt cache. If you have switched providers since, the chat still
resumes, on the current main model, but without the old provider's cache or private reasoning.

**Give the agent a map of the project with `/map`.** `/map survey` sends a read-only
explore run (on the subagent model) through the repo and saves what it finds in
`.governor/map/map.json`: *areas* (each folder or module, its purpose, what it contains
and depends on) and *features* (a name, a short abstract, the files it lives in). From
then on the map goes into the system prompt of every run (after the AGENTS.md text),
within about 3k tokens, so the agent starts a task knowing where things are instead of
rediscovering them. The agent adds to
the map as it works (a `map_feature` tool line in the log), and each feature card keeps
two lists: the paths predicted for it and the paths runs actually read or wrote.
`/map` shows the map as the agent sees it; `/map <feature>` shows one card with the two
lists side by side; `/map refresh` re-surveys with the current map and the git changes
since the last survey as its starting point. A conversation keeps the map it started
with, so its prompt cache holds; `/clear` or a new conversation picks up what changed.
While it runs you see its subagent and tool-call summary lines, then the status line; the
map itself is not printed (`/map` shows it). Delete the file to start over.

**Each provider remembers its own models.** If you pick `sol` on OpenAI, switch to
Anthropic and pick `sonnet`, then switch back, OpenAI is still on `sol`.

| Provider | Models (main · subagent by default) |
| --- | --- |
| `bedrock` | `opus` · `haiku`, also `fable`, `sonnet`, `astra`, `sol`, `luna`, `gpt-oss`, `gpt-oss-small`, `nova-pro` |
| `mantle` | `opus` · `sonnet` (Claude on Bedrock through Anthropic's API format) |
| `anthropic` | `opus` · `haiku`, also `fable`, `sonnet` |
| `openai` | `astra` · `luna`, also `sol`, `terra` |

`/model` → *Other model id…* accepts any id the provider knows. On Bedrock it lists
the ids your AWS account can actually use, with Tab completion.

## 5. The built-in agents

| Agent | Best for | Model | Can change files? |
| --- | --- | --- | --- |
| `general` (default) | Most tasks: reading, writing and running code | main | Yes, with approval |
| `planner` | Big tasks. It splits the work and sends it to subagents, often in parallel | main | No, it delegates |
| `explore` | Quick read-only research ("where is X handled?") | subagent (cheap) | No |

With `/agent planner`, a task such as *"review each module in ./src and list the three
riskiest pieces of code"* starts several `explore` subagents at once and combines
their reports. Subagents appear indented with a `↳` marker.

## 6. Costs

- Every answer ends with a line like `12034 in / 851 out tokens · $0.0652 · 41.3s`.
  `/cost` shows the session total, and the bottom bar keeps a running count.
- **Anthropic and OpenAI** bill at their list prices, which Governor uses, so
  these costs are close. OpenAI charges more for prompts over 272K input tokens,
  which the cost line doesn't account for.
- **Bedrock** costs are **estimates**: AWS bills at its own rates. AWS doesn't
  publish prices for its `astra`, `sol` and `luna`, so those show $0 until a
  maintainer adds prices to `.governor/models.toml`.
- `cache r/w` on the cost line is the part of the prompt read from, or written to, the
  provider's prompt cache. Reads cost a small fraction of normal input. The cache
  lasts about 5 minutes after its last use, so the first message (or `/qa` question)
  after a longer pause costs more than the ones right after it.
- Rough guide: `haiku`/`luna` are very cheap, `terra` is cheap, `sonnet`/`sol` are
  mid-range, and `opus`, `astra` and `fable` cost the most. For experiments, set `/model` to a cheap model first.

### A spending cap: `/budget`

`/budget` opens a picker of caps (or type one: `/budget 2.50`); `/budget off` removes it. The
cap is remembered across sessions and shown in the bottom bar while it is set. Governor adds
up the cost of every model call as it happens, subagents and `/qa` included, and when the
session's total reaches the cap it stops the running task the way Esc does: no more model
calls, pending tool calls are refused, and the conversation is left as it was before that
message (the line says `stopped by budget`). The cap is checked after each call, so the
session can go over it by the calls already under way: one in a plain task, one per
subagent when several run at once. If the call that crossed it was the answer itself, the
answer is kept. Once reached, every new task stops before its first call until you raise
or clear the cap. Bedrock costs are estimates (above), so the
cap is too.

## 7. Run history (traces) and saved chats

Every session is saved to `.governor/runs/<date>-tui.jsonl` inside the project folder.
It records every step: what the model said, which tools it used, the results and
the cost. Use these files to look back at what happened, compare two approaches, or
attach to a bug report. The path is printed when you quit.

Conversations themselves are saved separately, one file per chat in `.governor/chats/`,
which is what `/chats` reads. Delete a file there to forget a chat. The project map
(`/map`) is `.governor/map/map.json`.

## 8. Customise without coding

All of these go in a `.governor/` folder inside your project.

**Your own agent.** Create `.governor/agents/reviewer.md`:

```markdown
---
name: reviewer
description: Reviews code changes for bugs and unclear naming.
model: main          # main, subagent, or a model name such as sonnet
tools: [read_file, list_files, grep, bash]
---
You are a careful code reviewer. Look at the recent changes (use `git diff`), point out
bugs and confusing code, and suggest concrete fixes. Do not edit files.
```

Then choose it with `/agent reviewer`. If your file has the same name as a built-in
agent, yours replaces it.

**Project instructions.** Put an `AGENTS.md` in your project describing conventions
(for example "use pytest" or "never touch the migrations folder"). The agent reads it
automatically at the start of every task.

**Pre-approve tools.** If you trust the agent with certain tools in a project, put
this in the project's `.env` file:

```bash
GOVERNOR_ALLOW=write_file,edit_file     # still asks for bash
```

## 9. Scripts and automation

Governor runs only as the terminal UI; `govern` takes no arguments and there is no
scripted mode. `GOVERNOR_PROVIDER`, `GOVERNOR_MODEL` and `GOVERNOR_EFFORT` in the
environment override the saved choices for a session, and `GOVERNOR_ALLOW` pre-approves
tools (section 8). Programs that need a run without the UI use the Python API (see
`README.md`, "Library use").

## Troubleshooting

| You see | What to do |
| --- | --- |
| `not signed in`, `no … credentials found`, `401`, `session has expired` | `/login` and pick the provider |
| `on-demand throughput isn't supported` | The Bedrock model id needs a region prefix (`us.` or `global.`). `/model` → *Other model id…* lists valid ids |
| `AccessDenied … model access` / `use case` | In the AWS console, go to Bedrock → Model access, enable the model, and submit the Anthropic use-case form (once per account) |
| `… is not available for this account` | The AWS account has no agreement for that model (`/doctor` shows `agreement=NOT_AVAILABLE`). Ask AWS Support to enable it, and pick another model with `/model` meanwhile |
| `AccessDenied` (no other detail) | Your AWS role needs Bedrock permissions. Ask whoever manages the account |
| `not in this region's inference profiles` | `/doctor` lists similar ids. Pick one with `/model` → *Other model id…* |
| `ThrottlingException` / `429` | Too many requests. Governor retries automatically; if it keeps happening, the account's quota needs raising |
| It stopped with `stopped: max_turns` | The task needed more steps. Say "continue", or split the task |
| `stopped: invalid_tool_calls` | The model kept calling a tool with bad arguments. Try a stronger model, or rephrase the task |

## FAQ

**Is my code sent anywhere?** Only to the provider you chose, as part of the model
requests: Amazon Bedrock in your AWS account, Anthropic, or OpenAI. Traces stay on
your machine in `.governor/runs/`.

**Can it break things?** It can only change files or run commands when you approve
it (unless you use `/yolo` or `GOVERNOR_ALLOW`). Use git so any change is easy to undo.

**Why did it make a mistake?** Models sometimes guess when they're missing
information. Give it more context in the task, add project notes to `AGENTS.md`, or try
a stronger model with `/model`.

**How do I add a real new tool or feature?** That needs code. See `README.md` →
"Extending it", and `AGENTS.md` for where each kind of change goes.
