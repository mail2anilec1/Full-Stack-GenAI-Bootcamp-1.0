# LangChain Tutorial Script: Groq + VS Code Debugging Guide

This folder contains a runnable Python companion to the repository's
[`langchain_latest_version.ipynb`](../../langchain_latest_version.ipynb)
notebook:

- [`langchain_latest_version_groq.py`](./langchain_latest_version_groq.py) —
  small, named demos that you can run independently and step through in a
  debugger.

The script uses Groq for chat completions instead of OpenAI. It is an
educational companion, not a full document-ingestion RAG application: tools
named `search_knowledge_base` and `search_web` return placeholder text.

## 1. Requirements

- Python 3.10 or newer.
- A Groq API key for demos that call a chat model.
- VS Code's **Python** and **Python Debugger** extensions to use the debugger.

The `document`, `messages`, and `prompts` demos run locally and do not need a
Groq API key. The other demos make calls to Groq. The `deep-agent` demo also
requires the optional `deepagents` package.

## 2. Install dependencies

Open a terminal at the repository root:

```bash
cd /workspaces/Full-Stack-GenAI-Bootcamp-1.0
```

Create and activate a virtual environment if you do not already have one:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the packages used by this script:

```bash
python -m pip install -U langchain langchain-core langchain-groq python-dotenv pydantic
```

For the optional Deep Agents example:

```bash
python -m pip install -U deepagents
```

This repository also has a `.venv_genai` environment. If you use it, select
that interpreter in VS Code and run the script with:

```bash
./.venv_genai/bin/python RAG_END2END/Langchain/langchain_latest_version_groq.py --demo prompts
```

## 3. Configure Groq

Create or update `.env` in the repository root (the same directory as
`requirements.txt`) and add your key:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Do not commit `.env` or paste your key into Python source code. The script
loads this repository-root `.env` file itself, so it works even if the
terminal's current directory is not the script's folder.

The default chat model is `openai/gpt-oss-20b`. Despite the `openai/` prefix in
that model ID, the script sends chat requests through Groq's LangChain
integration. If you need another model available to your Groq account, set
`GROQ_MODEL` in `.env`:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your-groq-model-id
```

Groq is only the chat-completion provider here; it does not supply embeddings
through `ChatGroq`. The notebook's OpenAI embedding model is not used by this
script. For a RAG project, choose a separate embeddings model/provider (for
example local Hugging Face embeddings).

## 4. Run a demo from the terminal

From the repository root, print the available choices:

```bash
python RAG_END2END/Langchain/langchain_latest_version_groq.py --help
```

Run one example by name:

```bash
python RAG_END2END/Langchain/langchain_latest_version_groq.py --demo prompts
```

The default demo is `chat`, so this also works:

```bash
python RAG_END2END/Langchain/langchain_latest_version_groq.py
```

### Demo reference

| `--demo` value | What it demonstrates | Calls Groq? |
|---|---|---:|
| `document` | LangChain `Document` content, ID, and metadata | No |
| `messages` | System/human/AI messages and trimming conversation history | No |
| `prompts` | Text/chat prompts, history placeholders, partial prompts | No |
| `chat` | `invoke`, `batch`, and `stream` on a chat model | Yes |
| `parsers` | String, JSON, Pydantic, and native structured output | Yes |
| `tools` | A local calculator, model tool calls, and an agent | Yes |
| `agents` | An agent response matching a Pydantic schema | Yes |
| `middleware` | Hooks before and after a model call | Yes |
| `deep-agent` | Optional Deep Agents example with a placeholder search tool | Yes |
| `all` | Runs all demos except `deep-agent` | Yes |

Start with `document`, `messages`, or `prompts` to learn and debug without
network requests. `--demo all` makes several Groq requests and may take longer
or encounter account rate limits; run individual demos while learning.

## 5. Set up VS Code debugging

### Select the Python interpreter

1. Open the repository root in VS Code.
2. Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`).
3. Choose **Python: Select Interpreter**.
4. Select the environment where you installed the dependencies, such as
   `.venv` or `.venv_genai`.

The selected interpreter must have `langchain`, `langchain-groq`,
`python-dotenv`, and `pydantic` installed.

### Add a launch configuration

1. Open **Run and Debug** (`Ctrl+Shift+D` or `Cmd+Shift+D`).
2. Choose **create a launch.json file** and select **Python Debugger**.
3. Add the following configuration to the `configurations` array in
   `.vscode/launch.json`:

```json
{
    "name": "LangChain: local prompts demo",
    "type": "debugpy",
    "request": "launch",
    "program": "${workspaceFolder}/RAG_END2END/Langchain/langchain_latest_version_groq.py",
    "args": ["--demo", "prompts"],
    "console": "integratedTerminal",
    "justMyCode": true
}
```

To debug a Groq-backed example, change the arguments, for example:

```json
"args": ["--demo", "chat"]
```

Make sure `.env` has a valid `GROQ_API_KEY` before starting a Groq-backed
debug session. The Python code loads `.env` from the repository root; the
launch configuration does not need to contain the API key.

### Place breakpoints and start

1. Open `langchain_latest_version_groq.py`.
2. Click in the gutter next to a line number to set a breakpoint (a red dot).
   For example, set one inside `demo_prompts()` or on the `model.invoke(...)`
   call inside `demo_chat()`.
3. In the Run and Debug view, choose **LangChain: local prompts demo** and
   press **F5**. Use **F10** to step over, **F11** to step into, and
   **Shift+F11** to step out.
4. Inspect **Variables**, **Watch**, and **Call Stack** in the Run and Debug
   view. Add expressions such as `chat_prompt`, `response.content`, or
   `result["messages"]` to **Watch** when those variables are in scope.
5. Press **F5** again to continue to the next breakpoint. Use **Shift+F5** to
   stop the session.

Use `justMyCode: true` to focus on the tutorial code. If you specifically need
to step into installed LangChain package code, change it to `false`; stepping
into third-party libraries can be noisy.

### Debugging suggested learning paths

**Prompt construction, no API key**

1. Set `--demo` to `prompts`.
2. Break in `demo_prompts()` on the `PromptTemplate.from_template(...)` line.
3. Step over each `.invoke(...)` call.
4. Inspect `.to_messages()` to see how input variables become system and human
   messages, and how `MessagesPlaceholder` inserts the history.

**Single Groq request**

1. Set `--demo` to `chat`.
2. Break on `model = build_chat_model()` and inspect the selected model.
3. Step to `model.invoke(...)` and continue (`F5`) to wait for the API response.
4. Inspect `response.content`. The batch and stream sections then demonstrate
   the same model with different call patterns.

**Structured output**

1. Set `--demo` to `parsers`.
2. Break on `json_chain.invoke(...)` or `pydantic_chain.invoke(...)`.
3. Step over the request and inspect the parsed dictionary or Pydantic model.
4. If the provider/model rejects native structured output, inspect the
   traceback and try a Groq model that supports the requested feature.

**Tool calling and agents**

1. Set `--demo` to `tools` or `agents`.
2. Break on `model.bind_tools(...)` or `agent.invoke(...)`.
3. Inspect `tool_request.tool_calls` to see the model's proposed tool call.
4. In the agent example, step through the returned message list and inspect
   the final assistant message or `structured_response`.

## 6. How to use the script as a learning/debugging file

The script is deliberately organized into small functions:

- `build_chat_model()` handles Groq model creation and checks for a missing
  key.
- Each `demo_*()` function corresponds to a concept from the notebook.
- `DEMO_FUNCTIONS` maps the CLI demo name to its Python function.
- `main()` parses `--demo`, loads `.env`, and calls only the selected function.

This separation means you can put a breakpoint inside one concept without
running all the other examples. A local-only function can also be invoked
directly from the VS Code Python terminal after importing the module:

```python
from RAG_END2END.Langchain.langchain_latest_version_groq import demo_prompts

demo_prompts()
```

The directories are not configured as Python packages in every environment,
so running with `--demo prompts` or using the VS Code launch configuration is
the most reliable approach.

## 7. Troubleshooting

### `GROQ_API_KEY is not set`

- Confirm that `.env` is at the repository root, not inside `Langchain/`.
- Check that the variable is spelled `GROQ_API_KEY`.
- Restart the debug session after editing `.env`.
- To test setup without a key, run `--demo document`, `--demo messages`, or
  `--demo prompts`.

### `ModuleNotFoundError`

- Check the interpreter selected in VS Code.
- Install the package into that same interpreter, for example:

  ```bash
  python -m pip install -U langchain langchain-core langchain-groq python-dotenv pydantic
  ```

- The `deep-agent` demo additionally requires `deepagents`.

### Groq authentication, model, or rate-limit error

- Verify that the API key is valid and active in your Groq account.
- Verify that `GROQ_MODEL` names a model currently available to your account.
- Check the exception traceback and Groq account limits. Do not paste the API
  key into the traceback, a screenshot, or source control.
- Run just one API-backed demo instead of `--demo all`.

### Structured output or tool calling fails

Structured output and tool calling depend on the selected model's capabilities
and the current Groq/LangChain integration. The `agents` demo uses
LangChain's `ToolStrategy` so its structured response does not request Groq
JSON mode at the same time as the knowledge-base tool. The local
`multiply.invoke(...)` call in the `tools` demo does not require Groq; the
model-bound tool call and agent portion do.

## 8. Relationship to the notebook

The notebook contains architecture notes, code fragments, and examples that
depend on live provider APIs. This script turns the runnable core concepts
into selectable demonstrations, with docstrings and normal Python functions
that are easier to inspect in a debugger. It uses Groq for chat instead of
OpenAI. Notebook examples requiring a real image URL or undefined Deep Agents
configuration objects are intentionally not copied as runnable demos.
