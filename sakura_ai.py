#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║                  🌸 SAKURA AI 🌸                            ║
║  A powerful unrestricted AI assistant with advanced memory   ║
║  Model: Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-    ║
║         Distilled                                            ║
║  Purpose: Unrestricted intelligent chatting, coding, tasks   ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import datetime
import hashlib
import pickle
import re
import subprocess
import platform
import textwrap
import readline  # for better terminal input experience
from pathlib import Path
from typing import Optional

# ─── Dependency Bootstrap ────────────────────────────────────────────────────

# ─── Dependency Bootstrap ────────────────────────────────────────────────────

def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args, "-q"])

def _pkg_ok(import_name):
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False

def ensure_deps():
    """
    Install / upgrade everything, then restart the process once via os.execv so
    all new package versions are loaded cleanly with no stale cached imports.
    SAKURA_DEPS_READY=1 prevents an infinite restart loop.
    """
    if os.environ.get("SAKURA_DEPS_READY") == "1":
        return  # already bootstrapped

    # Basic packages
    for pkg, imp in [
        ("torch",         "torch"),
        ("accelerate",    "accelerate"),
        ("sentencepiece", "sentencepiece"),
        ("colorama",      "colorama"),
        ("rich",          "rich"),
        ("tiktoken",      "tiktoken"),
    ]:
        if not _pkg_ok(imp):
            print(f"[sakura] Installing {pkg}...")
            _pip(pkg)

    # huggingface_hub must be current — old versions break new transformers source
    print("[sakura] Ensuring huggingface_hub is latest...")
    _pip("huggingface_hub", "--upgrade")

    # Check whether qwen3_5 is already registered
    qwen_ok = False
    try:
        # Temporarily import to probe — may still be the old version at this point
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        qwen_ok = "qwen3_5" in CONFIG_MAPPING
    except Exception:
        pass

    if not qwen_ok:
        print("[sakura] Installing transformers from source (Qwen3.5 support)...")
        _pip("git+https://github.com/huggingface/transformers.git", "--upgrade")

    # Restart the process so Python loads fresh versions of everything
    print("[sakura] Restarting with updated packages...\n")
    os.environ["SAKURA_DEPS_READY"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

ensure_deps()

# ─── Imports ─────────────────────────────────────────────────────────────────

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from colorama import Fore, Back, Style, init as colorama_init
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.table import Table
from rich import box

colorama_init(autoreset=True)
console = Console()

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_ID        = "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"
MEMORY_FILE     = Path.home() / ".sakura_memory.pkl"
SESSION_LOG_DIR = Path.home() / ".sakura_sessions"
MAX_MEMORY      = 2000          # max stored memory items
MAX_CTX_TURNS   = 30           # turns kept in active context window
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE     = torch.float16 if DEVICE == "cuda" else torch.float32

# ─── Memory System ───────────────────────────────────────────────────────────

class SakuraMemory:
    """
    Advanced hierarchical memory system:
    - Short-term: current session turns
    - Long-term: persisted cross-session facts, preferences, notes
    - Episodic: past conversation summaries stored by session
    - Semantic:  user profile, extracted entities, topics
    """

    def __init__(self):
        self.short_term: list[dict] = []        # [{role, content, ts}]
        self.long_term:  dict       = {}        # {key: value}
        self.episodic:   list[dict] = []        # [{session_id, summary, ts}]
        self.semantic:   dict       = {         # structured user knowledge
            "user_profile": {},
            "facts":        [],
            "preferences":  [],
            "topics":       [],
        }
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        SESSION_LOG_DIR.mkdir(exist_ok=True)
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if MEMORY_FILE.exists():
            try:
                with open(MEMORY_FILE, "rb") as f:
                    data = pickle.load(f)
                self.long_term = data.get("long_term", {})
                self.episodic  = data.get("episodic",  [])
                self.semantic  = data.get("semantic",  self.semantic)
                console.print(f"[dim]🌸 Memory loaded: {len(self.episodic)} sessions, "
                              f"{len(self.semantic['facts'])} facts[/dim]")
            except Exception as e:
                console.print(f"[yellow]⚠ Could not load memory: {e}[/yellow]")

    def save(self):
        try:
            with open(MEMORY_FILE, "wb") as f:
                pickle.dump({
                    "long_term": self.long_term,
                    "episodic":  self.episodic,
                    "semantic":  self.semantic,
                }, f)
        except Exception as e:
            console.print(f"[yellow]⚠ Could not save memory: {e}[/yellow]")

    def save_session_log(self):
        log_file = SESSION_LOG_DIR / f"session_{self.session_id}.json"
        try:
            with open(log_file, "w") as f:
                json.dump(self.short_term, f, indent=2, default=str)
        except Exception:
            pass

    # ── Short-term ───────────────────────────────────────────────────────────

    def add_turn(self, role: str, content: str):
        self.short_term.append({
            "role": role, "content": content,
            "ts": datetime.datetime.now().isoformat()
        })

    def get_context(self, n: int = MAX_CTX_TURNS) -> list[dict]:
        """Return last N turns for context window."""
        turns = self.short_term[-n:]
        return [{"role": t["role"], "content": t["content"]} for t in turns]

    # ── Long-term / Semantic ──────────────────────────────────────────────────

    def remember(self, key: str, value):
        self.long_term[key] = value
        self.save()

    def recall(self, key: str):
        return self.long_term.get(key)

    def add_fact(self, fact: str):
        if fact not in self.semantic["facts"]:
            self.semantic["facts"].append(fact)
            if len(self.semantic["facts"]) > MAX_MEMORY:
                self.semantic["facts"] = self.semantic["facts"][-MAX_MEMORY:]
            self.save()

    def get_memory_context(self) -> str:
        """Compose a memory context block for the system prompt."""
        parts = []
        if self.semantic["facts"]:
            recent_facts = self.semantic["facts"][-20:]
            parts.append("Known facts about the user:\n" + "\n".join(f"- {f}" for f in recent_facts))
        if self.semantic["preferences"]:
            parts.append("User preferences:\n" + "\n".join(f"- {p}" for p in self.semantic["preferences"][-10:]))
        if self.episodic:
            recent = self.episodic[-3:]
            summaries = "\n".join(f"- [{e['ts'][:10]}] {e['summary']}" for e in recent)
            parts.append(f"Recent past sessions:\n{summaries}")
        if self.long_term:
            lt_str = "\n".join(f"- {k}: {v}" for k, v in list(self.long_term.items())[-15:])
            parts.append(f"Stored memories:\n{lt_str}")
        return "\n\n".join(parts) if parts else ""

    def archive_session(self, summary: str):
        self.episodic.append({
            "session_id": self.session_id,
            "summary":    summary,
            "turns":      len(self.short_term),
            "ts":         datetime.datetime.now().isoformat(),
        })
        if len(self.episodic) > MAX_MEMORY:
            self.episodic = self.episodic[-MAX_MEMORY:]
        self.save()
        self.save_session_log()

    def show_stats(self):
        table = Table(title="🌸 Sakura Memory Stats", box=box.ROUNDED, style="pink1")
        table.add_column("Category", style="bold magenta")
        table.add_column("Count", style="cyan")
        table.add_row("Short-term turns",     str(len(self.short_term)))
        table.add_row("Long-term memories",   str(len(self.long_term)))
        table.add_row("Episodic sessions",    str(len(self.episodic)))
        table.add_row("Known facts",          str(len(self.semantic["facts"])))
        table.add_row("Preferences",          str(len(self.semantic["preferences"])))
        console.print(table)


# ─── Model Loader ────────────────────────────────────────────────────────────

class SakuraModel:
    def __init__(self):
        self.tokenizer = None
        self.model     = None
        self.loaded    = False

    def load(self):
        console.print(Panel(
            f"[bold pink1]Loading model:[/bold pink1] [cyan]{MODEL_ID}[/cyan]\n"
            f"[bold]Device:[/bold] [yellow]{DEVICE.upper()}[/yellow]  |  "
            f"[bold]Dtype:[/bold] [yellow]{TORCH_DTYPE}[/yellow]\n"
            f"[dim]This may take a few minutes on first run...[/dim]",
            title="🌸 Sakura AI - Model Init",
            border_style="magenta"
        ))

        # Common kwargs
        load_kwargs = dict(
            torch_dtype=TORCH_DTYPE,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Safety / speed optimisations
        if DEVICE == "cuda":
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = "cpu"

        with console.status("[bold magenta]Loading tokenizer...[/bold magenta]"):
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                padding_side="left",
            )

        with console.status("[bold magenta]Loading model weights (please wait)...[/bold magenta]"):
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, **load_kwargs
            )
            if DEVICE == "cpu":
                self.model.eval()

        self.loaded = True
        console.print("[bold green]✓ Model loaded successfully![/bold green]\n")

    def generate(
        self,
        messages:    list[dict],
        max_new_tokens: int  = 2048,
        temperature: float   = 0.7,
        top_p:       float   = 0.95,
        do_sample:   bool    = True,
        stream:      bool    = True,
    ) -> str:
        # Build chat-template prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(DEVICE if DEVICE == "cuda" else "cpu")

        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

        if stream:
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            gen_kwargs["streamer"] = streamer

            thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()

            output_tokens = []
            console.print("\n[bold magenta]Sakura 🌸[/bold magenta] ", end="")
            for tok in streamer:
                print(tok, end="", flush=True)
                output_tokens.append(tok)
            print()  # newline after streaming done
            thread.join()
            return "".join(output_tokens)
        else:
            with torch.no_grad():
                out = self.model.generate(**gen_kwargs)
            new_ids = out[0][input_ids.shape[-1]:]
            return self.tokenizer.decode(new_ids, skip_special_tokens=True)


# ─── Command Handlers ────────────────────────────────────────────────────────

class CommandHandler:
    """Handle /slash commands."""

    def __init__(self, memory: SakuraMemory, model: SakuraModel):
        self.memory = memory
        self.model  = model
        self.COMMANDS = {
            "/help":       self.cmd_help,
            "/memory":     self.cmd_memory,
            "/remember":   self.cmd_remember,
            "/recall":     self.cmd_recall,
            "/fact":       self.cmd_fact,
            "/clear":      self.cmd_clear,
            "/history":    self.cmd_history,
            "/sessions":   self.cmd_sessions,
            "/run":        self.cmd_run,
            "/temp":       self.cmd_temp,
            "/tokens":     self.cmd_tokens,
            "/save":       self.cmd_save,
            "/sysinfo":    self.cmd_sysinfo,
            "/exit":       self.cmd_exit,
            "/quit":       self.cmd_exit,
        }
        self.temperature = 0.7
        self.max_tokens  = 2048

    def handle(self, line: str) -> Optional[bool]:
        """Returns True if handled, None if not a command."""
        parts = line.strip().split(None, 1)
        cmd   = parts[0].lower()
        arg   = parts[1] if len(parts) > 1 else ""
        if cmd in self.COMMANDS:
            self.COMMANDS[cmd](arg)
            return True
        return None

    # ── individual commands ───────────────────────────────────────────────────

    def cmd_help(self, _):
        table = Table(title="🌸 Sakura AI Commands", box=box.ROUNDED, style="magenta")
        table.add_column("Command",     style="bold cyan",   no_wrap=True)
        table.add_column("Description", style="white")
        cmds = [
            ("/help",              "Show this help"),
            ("/memory",            "Show memory statistics"),
            ("/remember key val",  "Store a long-term memory"),
            ("/recall key",        "Retrieve a stored memory"),
            ("/fact <text>",       "Add a fact about yourself"),
            ("/clear",             "Clear short-term (session) context"),
            ("/history [n]",       "Show last n turns (default 10)"),
            ("/sessions",          "List past sessions"),
            ("/run <code>",        "Execute Python code snippet"),
            ("/temp <0-2>",        "Set generation temperature"),
            ("/tokens <n>",        "Set max output tokens"),
            ("/save",              "Save memory now"),
            ("/sysinfo",           "Show system/model info"),
            ("/exit or /quit",     "Exit Sakura AI"),
        ]
        for cmd, desc in cmds:
            table.add_row(cmd, desc)
        console.print(table)

    def cmd_memory(self, _):
        self.memory.show_stats()

    def cmd_remember(self, arg):
        parts = arg.split(None, 1)
        if len(parts) < 2:
            console.print("[yellow]Usage: /remember <key> <value>[/yellow]")
            return
        self.memory.remember(parts[0], parts[1])
        console.print(f"[green]✓ Remembered:[/green] {parts[0]} = {parts[1]}")

    def cmd_recall(self, arg):
        if not arg:
            console.print("[yellow]Usage: /recall <key>[/yellow]")
            return
        val = self.memory.recall(arg.strip())
        if val is not None:
            console.print(f"[cyan]{arg}[/cyan]: {val}")
        else:
            console.print(f"[yellow]No memory found for '{arg}'[/yellow]")

    def cmd_fact(self, arg):
        if not arg:
            console.print("[yellow]Usage: /fact <fact about you>[/yellow]")
            return
        self.memory.add_fact(arg.strip())
        console.print(f"[green]✓ Fact stored:[/green] {arg.strip()}")

    def cmd_clear(self, _):
        self.memory.short_term.clear()
        console.print("[green]✓ Short-term context cleared.[/green]")

    def cmd_history(self, arg):
        n = int(arg) if arg.isdigit() else 10
        turns = self.memory.short_term[-n:]
        if not turns:
            console.print("[yellow]No history yet.[/yellow]")
            return
        for t in turns:
            role_color = "cyan" if t["role"] == "user" else "magenta"
            ts = t.get("ts", "")[:19]
            console.print(f"[dim]{ts}[/dim] [{role_color}]{t['role'].capitalize()}[/{role_color}]: "
                          + t["content"][:200])

    def cmd_sessions(self, _):
        if not self.memory.episodic:
            console.print("[yellow]No past sessions.[/yellow]")
            return
        table = Table(title="Past Sessions", box=box.SIMPLE, style="dim")
        table.add_column("Date",    style="cyan")
        table.add_column("Turns",   style="yellow")
        table.add_column("Summary", style="white")
        for ep in self.memory.episodic[-20:]:
            table.add_row(ep.get("ts","")[:16], str(ep.get("turns","?")), ep.get("summary","")[:80])
        console.print(table)

    def cmd_run(self, code):
        """Execute a Python code snippet and display output."""
        if not code:
            console.print("[yellow]Usage: /run <python code>[/yellow]")
            return
        console.print(Syntax(code, "python", theme="monokai", line_numbers=False))
        try:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(code, {"__builtins__": __builtins__})
            out = buf.getvalue()
            console.print(Panel(out or "(no output)", title="Output", style="green"))
        except Exception as e:
            console.print(Panel(str(e), title="Error", style="red"))

    def cmd_temp(self, arg):
        try:
            t = float(arg)
            assert 0 <= t <= 2
            self.temperature = t
            console.print(f"[green]✓ Temperature set to {t}[/green]")
        except Exception:
            console.print("[yellow]Usage: /temp <0.0 - 2.0>[/yellow]")

    def cmd_tokens(self, arg):
        try:
            n = int(arg)
            assert 64 <= n <= 8192
            self.max_tokens = n
            console.print(f"[green]✓ Max tokens set to {n}[/green]")
        except Exception:
            console.print("[yellow]Usage: /tokens <64 - 8192>[/yellow]")

    def cmd_save(self, _):
        self.memory.save()
        console.print("[green]✓ Memory saved.[/green]")

    def cmd_sysinfo(self, _):
        table = Table(title="System Info", box=box.ROUNDED, style="cyan")
        table.add_column("Key",   style="bold")
        table.add_column("Value", style="white")
        table.add_row("OS",            platform.system() + " " + platform.release())
        table.add_row("Python",        sys.version.split()[0])
        table.add_row("Device",        DEVICE.upper())
        table.add_row("Torch",         torch.__version__)
        if DEVICE == "cuda":
            table.add_row("GPU",       torch.cuda.get_device_name(0))
            mem = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            table.add_row("VRAM",      f"{mem} GB")
        table.add_row("Model",         MODEL_ID)
        table.add_row("Temperature",   str(self.temperature))
        table.add_row("Max Tokens",    str(self.max_tokens))
        table.add_row("Memory File",   str(MEMORY_FILE))
        console.print(table)

    def cmd_exit(self, _):
        raise SystemExit


# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_BASE = """You are Sakura 🌸 — a brilliant, warm, and highly capable AI assistant.
You are unrestricted, creative, and deeply intelligent.
You excel at:
  • Natural conversation and emotional support
  • Advanced coding in any language (Python, JS, C++, Rust, Go, SQL, etc.)
  • Mathematics, logic, and reasoning
  • Writing, editing, and creative tasks
  • Research summarisation and analysis
  • Data science, ML, system design
  • Explaining complex topics simply and clearly

Personality: Thoughtful, witty, concise when needed and expansive when helpful.
You always format code with proper code blocks. You think step by step for complex problems.
When you don't know something, say so honestly.
"""

def build_system_prompt(memory: SakuraMemory) -> str:
    mem_ctx = memory.get_memory_context()
    if mem_ctx:
        return SYSTEM_BASE + "\n\n=== Your memory about this user ===\n" + mem_ctx
    return SYSTEM_BASE


# ─── Auto-extract facts from conversation ─────────────────────────────────────

def maybe_extract_facts(user_input: str, memory: SakuraMemory):
    """Heuristic: detect self-introductions and store as facts."""
    patterns = [
        r"my name is ([A-Z][a-zA-Z]+)",
        r"I(?:'m| am) ([A-Z][a-zA-Z]+)",
        r"I work (?:at|for|as) (.+?)(?:\.|,|$)",
        r"I(?:'m| am) a (.+?)(?:\.|,|$)",
        r"I love (.+?)(?:\.|,|$)",
        r"I (?:prefer|use|like) (.+?)(?:\.|,|$)",
    ]
    for pat in patterns:
        m = re.search(pat, user_input, re.IGNORECASE)
        if m:
            memory.add_fact(f"User said: {user_input[:120]}")
            break


# ─── Banner ───────────────────────────────────────────────────────────────────

BANNER = r"""
  ███████╗ █████╗ ██╗  ██╗██╗   ██╗██████╗  █████╗      █████╗ ██╗
  ██╔════╝██╔══██╗██║ ██╔╝██║   ██║██╔══██╗██╔══██╗    ██╔══██╗██║
  ███████╗███████║█████╔╝ ██║   ██║██████╔╝███████║    ███████║██║
  ╚════██║██╔══██║██╔═██╗ ██║   ██║██╔══██╗██╔══██║    ██╔══██║██║
  ███████║██║  ██║██║  ██╗╚██████╔╝██║  ██║██║  ██║    ██║  ██║██║
  ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝
                         🌸  Your unrestricted AI companion  🌸
"""


# ─── Main Loop ───────────────────────────────────────────────────────────────

def main():
    # Print banner
    console.print(f"[bold magenta]{BANNER}[/bold magenta]")
    console.print(Panel(
        "[bold]Name:[/bold]    [pink1]Sakura AI 🌸[/pink1]\n"
        "[bold]Model:[/bold]   [cyan]Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled[/cyan]\n"
        "[bold]Purpose:[/bold] [white]Unrestricted intelligent chatting — coding, analysis,\n"
        "         reasoning, creativity, emotional support & more.[/white]\n"
        "[bold]Device:[/bold]  [yellow]" + DEVICE.upper() + "[/yellow]\n"
        "[dim]Type /help for commands. Type /exit to quit.[/dim]",
        title="🌸 Welcome",
        border_style="magenta",
        padding=(1, 2),
    ))

    # Init memory and model
    memory   = SakuraMemory()
    sakura   = SakuraModel()
    handler  = CommandHandler(memory, sakura)

    sakura.load()

    console.print("[dim]Ready. Start chatting below!\n[/dim]")

    try:
        while True:
            # ── Input ────────────────────────────────────────────────────────
            try:
                user_input = input(f"{Fore.CYAN}You ▶ {Style.RESET_ALL}").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            # ── Commands ─────────────────────────────────────────────────────
            try:
                if handler.handle(user_input):
                    continue
            except SystemExit:
                break

            # ── Memory hooks ─────────────────────────────────────────────────
            maybe_extract_facts(user_input, memory)
            memory.add_turn("user", user_input)

            # ── Build message list ────────────────────────────────────────────
            system_prompt = build_system_prompt(memory)
            messages = [{"role": "system", "content": system_prompt}] + memory.get_context()

            # ── Generate ─────────────────────────────────────────────────────
            start = time.time()
            try:
                response = sakura.generate(
                    messages,
                    max_new_tokens=handler.max_tokens,
                    temperature=handler.temperature,
                )
            except Exception as e:
                console.print(f"[red]Generation error: {e}[/red]")
                continue

            elapsed = time.time() - start
            console.print(f"\n[dim]⏱ {elapsed:.1f}s[/dim]\n")

            # ── Store response ────────────────────────────────────────────────
            memory.add_turn("assistant", response)

    except KeyboardInterrupt:
        pass

    # ── Teardown ─────────────────────────────────────────────────────────────
    console.print("\n[bold magenta]🌸 Goodbye! Saving memory...[/bold magenta]")

    # Auto-summarise session if long enough
    if len(memory.short_term) >= 4:
        topics = []
        for t in memory.short_term[:6]:
            if t["role"] == "user":
                topics.append(t["content"][:60])
        summary = "Topics: " + " | ".join(topics[:3])
        memory.archive_session(summary)

    memory.save()
    console.print("[green]✓ Memory saved. See you next time![/green]")


if __name__ == "__main__":
    main()
