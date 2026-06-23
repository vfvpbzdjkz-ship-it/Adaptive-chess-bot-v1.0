#!/usr/bin/env python3
"""
OUROBOROS GUI — Windows desktop monitor and configurator for the OUROBOROS chess bot.

Connects to a running OUROBOROS instance (local or Railway) and shows:
  • Live chess board with move highlights
  • Win / Draw / Loss record
  • Play mode (Lichess / Self-play) with countdowns
  • Training dashboard: loss sparklines, buffer fill, data composition, ELO chart
  • Settings panel for Lichess token, Hugging Face key, and bot URL

Requirements:
    pip install requests
    Python 3.10+ with tkinter (bundled with the standard Windows Python installer)

Run:
    python ouroboros_gui.py
    -- or double-click launch_gui.bat --
"""

import json
import os
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ── Colour palette (matches the web viewer) ──────────────────────────────────
BG      = "#1a1a2e"
BG2     = "#141428"
BG3     = "#1e1e3a"
GOLD    = "#c89b3c"
GREEN   = "#6fcf6f"
RED     = "#cf6f6f"
BLUE    = "#5a8fd6"
PURPLE  = "#9b6fcf"
GREY    = "#666666"
GREY2   = "#888888"
FG      = "#e0e0e0"
FG2     = "#bbbbbb"

LIGHT_SQ = "#f0d9b5"
DARK_SQ  = "#b58863"
HL_FROM  = "#f6f669"
HL_TO    = "#cdd26a"

PIECE_UNICODE = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟',
}
PIECE_FG = {p: ('#ffffff' if p.isupper() else '#1a1a1a') for p in PIECE_UNICODE}

DEFAULT_FEN   = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
SETTINGS_FILE = Path("gui_settings.json")
POLL_INTERVAL = 3.0   # seconds


# ── Settings persistence ──────────────────────────────────────────────────────

def _load_settings() -> dict:
    defaults = {"bot_url": "http://localhost:8080", "lichess_token": "", "hf_token": ""}
    try:
        if SETTINGS_FILE.exists():
            defaults.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    except Exception:
        pass
    return defaults


def _save_settings(s: dict) -> None:
    SETTINGS_FILE.write_text(json.dumps(s, indent=2), encoding="utf-8")


# ── FEN utilities ─────────────────────────────────────────────────────────────

def _parse_fen(fen: str) -> list:
    """Return 8×8 grid of piece chars ('' for empty) from FEN string."""
    pos = (fen or DEFAULT_FEN).split()[0]
    grid = []
    for rank in pos.split('/'):
        row = []
        for ch in rank:
            if ch.isdigit():
                row.extend([''] * int(ch))
            else:
                row.append(ch)
        grid.append(row)
    return grid


def _diff_fen(fen1: str, fen2: str):
    """Return (from_squares, to_squares) as sets of 'row,col' strings."""
    if not fen1 or not fen2:
        return set(), set()
    g1, g2 = _parse_fen(fen1), _parse_fen(fen2)
    frm, to = set(), set()
    for r in range(8):
        for c in range(8):
            p1 = g1[r][c] if r < len(g1) and c < len(g1[r]) else ''
            p2 = g2[r][c] if r < len(g2) and c < len(g2[r]) else ''
            if p1 and not p2:
                frm.add(f"{r},{c}")
            elif p1 != p2 and p2:
                to.add(f"{r},{c}")
    return frm, to


# ── Chess board widget ────────────────────────────────────────────────────────

class ChessBoard(tk.Canvas):
    SQ = 58   # square size in pixels; board is SQ*8 × SQ*8

    def __init__(self, parent, **kw):
        size = self.SQ * 8
        super().__init__(parent, width=size, height=size,
                         bg=DARK_SQ, highlightthickness=2,
                         highlightbackground=GOLD, **kw)
        self._fen      = DEFAULT_FEN
        self._flipped  = False
        self._from_sqs: set = set()
        self._to_sqs:   set = set()
        # Try several fonts; first hit wins
        self._piece_font = None
        self.after(50, self.draw)

    def _font(self):
        if self._piece_font is None:
            size = int(self.SQ * 0.65)
            for name in ("Segoe UI Symbol", "Symbola", "DejaVu Sans", "Arial Unicode MS"):
                self._piece_font = (name, size)
                break
        return self._piece_font

    def set_position(self, fen: str, flipped: bool = False,
                     from_sqs=None, to_sqs=None):
        self._fen      = fen or DEFAULT_FEN
        self._flipped  = flipped
        self._from_sqs = set(from_sqs or [])
        self._to_sqs   = set(to_sqs or [])
        self.draw()

    def draw(self):
        self.delete("all")
        grid = _parse_fen(self._fen)
        sq   = self.SQ
        font = self._font()
        for vrow in range(8):
            for vcol in range(8):
                fr = 7 - vrow if self._flipped else vrow
                fc = 7 - vcol if self._flipped else vcol
                rank   = 8 - fr
                light  = (rank + fc) % 2 == 0
                sq_key = f"{fr},{fc}"
                if sq_key in self._from_sqs:
                    bg = HL_FROM
                elif sq_key in self._to_sqs:
                    bg = HL_TO
                else:
                    bg = LIGHT_SQ if light else DARK_SQ
                x0, y0 = vcol * sq, vrow * sq
                self.create_rectangle(x0, y0, x0 + sq, y0 + sq,
                                      fill=bg, outline="")
                piece = grid[fr][fc] if fr < len(grid) and fc < len(grid[fr]) else ''
                if piece:
                    glyph = PIECE_UNICODE.get(piece, piece)
                    fg    = PIECE_FG.get(piece, FG)
                    self.create_text(x0 + sq // 2, y0 + sq // 2,
                                     text=glyph, fill=fg,
                                     font=font, anchor="center")


# ── Sparkline widget ──────────────────────────────────────────────────────────

class Sparkline(tk.Canvas):
    def __init__(self, parent, color=GOLD, width=160, height=36, **kw):
        super().__init__(parent, width=width, height=height,
                         bg=BG3, highlightthickness=0, **kw)
        self._color = color
        self._vals: list = []
        self.bind("<Configure>", lambda _e: self._draw())

    def set_values(self, vals: list):
        self._vals = list(vals)
        self._draw()

    def _draw(self):
        self.delete("all")
        vals = self._vals
        if len(vals) < 2:
            return
        w = self.winfo_width()  or int(self["width"])
        h = self.winfo_height() or int(self["height"])
        mn, mx = min(vals), max(vals)
        rng = mx - mn or 1.0
        pts = []
        for i, v in enumerate(vals):
            x = i / (len(vals) - 1) * w
            y = (1.0 - (v - mn) / rng) * (h - 6) + 3
            pts.extend([x, y])
        self.create_line(*pts, fill=self._color, width=1.8,
                         smooth=True, joinstyle="round", capstyle="round")
        lx, ly = pts[-2], pts[-1]
        self.create_oval(lx - 3, ly - 3, lx + 3, ly + 3,
                         fill=self._color, outline="")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lbl(parent, text="", fg=FG2, bg=BG2, font=None, anchor="w", **kw):
    return tk.Label(parent, text=text, fg=fg, bg=bg,
                    font=font or ("Segoe UI", 9), anchor=anchor, **kw)


def _section_label(parent, text, bg=BG2):
    tk.Label(parent, text=text, fg=GOLD, bg=bg,
             font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(10, 4))


def _hr(parent, bg_frame=BG2):
    tk.Frame(parent, bg=BG3, height=1).pack(fill="x", pady=8)


def _fmt(n) -> str:
    if n is None:
        return "—"
    if isinstance(n, float):
        return f"{n:.4f}"
    return f"{n:,}"


def _countdown_str(remaining: int) -> str:
    m, s = divmod(max(0, remaining), 60)
    return f"{m}:{s:02d}"


# ── Stat box (used in training tab) ──────────────────────────────────────────

class StatBox(tk.Frame):
    def __init__(self, parent, label: str, spark_color=GREY2, **kw):
        super().__init__(parent, bg=BG3, padx=12, pady=10, **kw)
        tk.Label(self, text=label, fg=GREY, bg=BG3,
                 font=("Segoe UI", 7)).pack(anchor="w")
        self.val_lbl = tk.Label(self, text="—", fg=FG, bg=BG3,
                                font=("Segoe UI", 14, "bold"))
        self.val_lbl.pack(anchor="w")
        self.sparkline = Sparkline(self, color=spark_color, width=140, height=30)
        self.sparkline.pack(anchor="w", pady=(4, 0))

    def set(self, text: str, color=FG):
        self.val_lbl.config(text=text, fg=color)

    def set_spark(self, vals: list):
        if vals:
            self.sparkline.set_values(vals)


# ── Main application ──────────────────────────────────────────────────────────

class OuroborosGUI:
    def __init__(self, root: tk.Tk):
        self.root     = root
        self.settings = _load_settings()
        self._state:  dict = {}
        self._prev_fen: str | None = None
        self._stop    = threading.Event()

        root.title("OUROBOROS — Live Monitor")
        root.configure(bg=BG)
        root.minsize(900, 680)
        root.resizable(True, True)

        self._apply_ttk_style()
        self._build_ui()
        self._start_polling()
        self._tick()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── ttk style ─────────────────────────────────────────────────────────────

    def _apply_ttk_style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TNotebook",     background=BG,  borderwidth=0)
        s.configure("TNotebook.Tab", background=BG3, foreground=FG2,
                     padding=[14, 6], font=("Segoe UI", 9))
        s.map("TNotebook.Tab",
              background=[("selected", BG2)],
              foreground=[("selected", GOLD)])

    # ── UI skeleton ────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Title bar
        hdr = tk.Frame(self.root, bg=BG, pady=8)
        hdr.pack(fill="x", padx=20)
        tk.Label(hdr, text="♜  OUROBOROS", fg=GOLD, bg=BG,
                 font=("Segoe UI", 18, "bold")).pack(side="left")
        tk.Label(hdr, text="self-learning chess bot", fg=GREY2, bg=BG,
                 font=("Segoe UI", 9)).pack(side="left", padx=(10, 0), pady=(6, 0))
        self._conn_lbl = tk.Label(hdr, text="● Not connected",
                                   fg=RED, bg=BG, font=("Segoe UI", 8))
        self._conn_lbl.pack(side="right")

        # Notebook
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        live_tab     = tk.Frame(nb, bg=BG2)
        train_tab    = tk.Frame(nb, bg=BG2)
        settings_tab = tk.Frame(nb, bg=BG2)

        nb.add(live_tab,     text="  Live Game  ")
        nb.add(train_tab,    text="  Training  ")
        nb.add(settings_tab, text="  Settings  ")

        self._build_live_tab(live_tab)
        self._build_training_tab(train_tab)
        self._build_settings_tab(settings_tab)

    # ── Live Game tab ─────────────────────────────────────────────────────────

    def _build_live_tab(self, parent):
        # Board column (left)
        left = tk.Frame(parent, bg=BG2)
        left.pack(side="left", fill="y", padx=(16, 10), pady=16)

        self._badge_frame = tk.Frame(left, bg=BG2)
        self._badge_frame.pack(fill="x", pady=(0, 5))
        self._badge_live    = tk.Label(self._badge_frame, text="● LIVE",
                                       fg=GREEN, bg="#1b4a1b",
                                       font=("Segoe UI", 8, "bold"), padx=8, pady=2)
        self._badge_ended   = tk.Label(self._badge_frame, text="ENDED",
                                       fg=GREY2, bg=BG3,
                                       font=("Segoe UI", 8, "bold"), padx=8, pady=2)
        self._badge_revenge = tk.Label(self._badge_frame, text="REVENGE",
                                       fg=RED, bg="#4a1b1b",
                                       font=("Segoe UI", 8, "bold"), padx=8, pady=2)

        self._matchup_lbl = tk.Label(left, text="Waiting for a game…",
                                      fg=GREY2, bg=BG2,
                                      font=("Segoe UI", 10))
        self._matchup_lbl.pack(pady=(0, 8))

        self._board = ChessBoard(left)
        self._board.pack()

        self._game_link = tk.Label(left, text="", fg=GOLD, bg=BG2,
                                    font=("Segoe UI", 9, "underline"),
                                    cursor="hand2")
        self._game_link.pack(pady=(6, 0))
        self._game_link.bind("<Button-1>", self._open_game)
        self._game_url = ""

        # Info column (right)
        right = tk.Frame(parent, bg=BG2)
        right.pack(side="left", fill="both", expand=True,
                   padx=(0, 16), pady=16)

        def _stat_row(label):
            fr = tk.Frame(right, bg=BG2)
            fr.pack(fill="x", pady=2)
            tk.Label(fr, text=label, fg=GREY, bg=BG2,
                     font=("Segoe UI", 8), width=22, anchor="w").pack(side="left")
            v = tk.Label(fr, text="—", fg=FG, bg=BG2,
                         font=("Segoe UI", 10, "bold"), anchor="w")
            v.pack(side="left")
            return v

        _section_label(right, "PLAY STATUS")
        self._mode_val             = _stat_row("Mode")
        self._mode_switch_val      = _stat_row("Mode switch in")
        self._challenge_val        = _stat_row("Next challenge in")

        _hr(right)
        _section_label(right, "RECORD")
        self._wins_val   = _stat_row("Wins")
        self._draws_val  = _stat_row("Draws")
        self._losses_val = _stat_row("Losses")
        self._score_val  = _stat_row("Score")

        _hr(right)
        _section_label(right, "CONTROLS")
        self._force_btn = tk.Button(
            right, text="▶  Play One Game",
            fg=GOLD, bg=BG3,
            activeforeground=BG, activebackground=GOLD,
            relief="flat", bd=0,
            highlightthickness=1, highlightbackground=GOLD,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2", padx=12, pady=6,
            command=self._force_game,
        )
        self._force_btn.pack(anchor="w", pady=(0, 6))

        _hr(right)
        _section_label(right, "BACKEND")
        self._native_val = _stat_row("Encoding")

    # ── Training tab ──────────────────────────────────────────────────────────

    def _build_training_tab(self, parent):
        # Scrollable inner frame
        canvas  = tk.Canvas(parent, bg=BG2, highlightthickness=0)
        scrolly = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner   = tk.Frame(canvas, bg=BG2)
        canvas.configure(yscrollcommand=scrolly.set)
        scrolly.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_resize(e):
            canvas.itemconfig(win_id, width=e.width)
        def _on_inner_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.bind("<Configure>",     _on_resize)
        inner.bind( "<Configure>",     _on_inner_configure)

        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        pad = {"padx": 16}

        # Stat boxes grid
        grid_frame = tk.Frame(inner, bg=BG2)
        grid_frame.pack(fill="x", **pad, pady=(16, 8))
        for c in range(3):
            grid_frame.columnconfigure(c, weight=1)

        self._box_sp    = StatBox(grid_frame, "SELF-PLAY GAMES",  BLUE)
        self._box_steps = StatBox(grid_frame, "TRAINING STEPS",   GREY2)
        self._box_spm   = StatBox(grid_frame, "STEPS / MIN",      GREY2)
        self._box_tloss = StatBox(grid_frame, "TOTAL LOSS",       GREY2)
        self._box_ploss = StatBox(grid_frame, "POLICY LOSS",      BLUE)
        self._box_vloss = StatBox(grid_frame, "VALUE LOSS",       PURPLE)

        for i, box in enumerate([self._box_sp, self._box_steps, self._box_spm,
                                  self._box_tloss, self._box_ploss, self._box_vloss]):
            box.grid(row=i // 3, column=i % 3, padx=5, pady=5, sticky="nsew")

        tk.Frame(inner, bg=BG3, height=1).pack(fill="x", **pad, pady=8)

        # Buffer
        buf_frame = tk.Frame(inner, bg=BG2)
        buf_frame.pack(fill="x", **pad, pady=(0, 8))

        _section_label(buf_frame, "REPLAY BUFFER", bg=BG2)
        self._buf_canvas = tk.Canvas(buf_frame, bg=BG3, height=10,
                                      highlightthickness=0)
        self._buf_canvas.pack(fill="x")
        self._buf_canvas.bind("<Configure>", self._redraw_buf)
        self._buf_fill_pct = 0.0

        self._buf_info_lbl = tk.Label(buf_frame, text="", fg=GREY2, bg=BG2,
                                       font=("Segoe UI", 8))
        self._buf_info_lbl.pack(anchor="w", pady=(3, 0))

        # Data composition
        _section_label(buf_frame, "DATA COMPOSITION", bg=BG2)
        self._comp_canvas = tk.Canvas(buf_frame, bg=BG3, height=12,
                                       highlightthickness=0)
        self._comp_canvas.pack(fill="x")
        self._comp_canvas.bind("<Configure>", self._redraw_comp)
        self._comp_data: dict = {}

        self._comp_lbl = tk.Label(buf_frame, text="", fg=GREY2, bg=BG2,
                                   font=("Segoe UI", 8))
        self._comp_lbl.pack(anchor="w", pady=(4, 0))

        # Legend dots
        leg_frame = tk.Frame(buf_frame, bg=BG2)
        leg_frame.pack(anchor="w", pady=(4, 0))
        for color, label in [(BLUE, "Self-play"), (GOLD, "Lichess"), (PURPLE, "Imitation")]:
            dot = tk.Label(leg_frame, text="■", fg=color, bg=BG2,
                           font=("Segoe UI", 9))
            dot.pack(side="left")
            tk.Label(leg_frame, text=label, fg=GREY2, bg=BG2,
                     font=("Segoe UI", 8)).pack(side="left", padx=(0, 12))

        tk.Frame(inner, bg=BG3, height=1).pack(fill="x", **pad, pady=8)

        # ELO section
        elo_frame = tk.Frame(inner, bg=BG2)
        elo_frame.pack(fill="x", **pad, pady=(0, 16))

        elo_left = tk.Frame(elo_frame, bg=BG2)
        elo_left.pack(side="left", padx=(0, 24))
        tk.Label(elo_left, text="INTERNAL ELO", fg=GOLD, bg=BG2,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self._elo_num   = tk.Label(elo_left, text="1500", fg=GOLD, bg=BG2,
                                    font=("Segoe UI", 28, "bold"))
        self._elo_num.pack(anchor="w")
        self._elo_delta = tk.Label(elo_left, text="", fg=GREY2, bg=BG2,
                                    font=("Segoe UI", 9))
        self._elo_delta.pack(anchor="w")

        elo_right = tk.Frame(elo_frame, bg=BG2)
        elo_right.pack(side="left", fill="both", expand=True)
        tk.Label(elo_right, text="ELO HISTORY", fg=GOLD, bg=BG2,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self._elo_chart = Sparkline(elo_right, color=GOLD, width=380, height=80)
        self._elo_chart.pack(fill="x")

    # ── Settings tab ─────────────────────────────────────────────────────────

    def _build_settings_tab(self, parent):
        outer = tk.Frame(parent, bg=BG2)
        outer.pack(fill="both", expand=True, padx=30, pady=20)

        def _entry_group(label, sublabel, key, show=""):
            tk.Label(outer, text=label, fg=GOLD, bg=BG2,
                     font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(14, 1))
            tk.Label(outer, text=sublabel, fg=GREY2, bg=BG2,
                     font=("Segoe UI", 8)).pack(anchor="w", pady=(0, 4))
            e = tk.Entry(outer, bg=BG3, fg=FG, insertbackground=FG,
                         font=("Segoe UI", 10), relief="flat",
                         highlightthickness=1,
                         highlightbackground=BG3, highlightcolor=GOLD,
                         show=show)
            e.pack(fill="x", ipady=6)
            e.insert(0, self.settings.get(key, ""))
            return e

        tk.Label(outer, text="Connection & API Tokens",
                 fg=FG, bg=BG2, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(outer,
                 text="Connect the GUI to your running bot, and store your API credentials.",
                 fg=GREY2, bg=BG2, font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 0))

        tk.Frame(outer, bg=BG3, height=1).pack(fill="x", pady=(10, 0))

        self._url_e = _entry_group(
            "Bot URL",
            "Railway public URL  (e.g. https://mybot.up.railway.app)  or  http://localhost:8080",
            "bot_url",
        )

        self._lichess_e = _entry_group(
            "Lichess API Token",
            "Required scopes:  bot:play   challenge:read   challenge:write",
            "lichess_token", show="•",
        )

        self._hf_e = _entry_group(
            "Hugging Face Token",
            "Write-access token from huggingface.co/settings/tokens  (for model sync)",
            "hf_token", show="•",
        )

        # Show / hide toggles
        toggle_row = tk.Frame(outer, bg=BG2)
        toggle_row.pack(anchor="w", pady=(6, 0))

        def _toggle_visibility(entry, btn):
            if entry["show"] == "•":
                entry.config(show="")
                btn.config(text="Hide")
            else:
                entry.config(show="•")
                btn.config(text="Show")

        for label_text, entry_widget in [("Show Lichess token", self._lichess_e),
                                          ("Show HF token", self._hf_e)]:
            b = tk.Button(toggle_row, text=label_text,
                          fg=GREY2, bg=BG3, relief="flat",
                          font=("Segoe UI", 8), padx=8, pady=3,
                          cursor="hand2")
            b.config(command=lambda e=entry_widget, b=b: _toggle_visibility(e, b))
            b.pack(side="left", padx=(0, 6))

        tk.Frame(outer, bg=BG3, height=1).pack(fill="x", pady=16)

        save_btn = tk.Button(
            outer, text="  Save Settings  ",
            fg=BG, bg=GOLD, activeforeground=BG, activebackground="#a07830",
            font=("Segoe UI", 10, "bold"), relief="flat",
            padx=16, pady=8, cursor="hand2",
            command=self._save_settings_action,
        )
        save_btn.pack(anchor="w")

        tk.Frame(outer, bg=BG3, height=1).pack(fill="x", pady=16)

        note_text = (
            "ℹ  Railway deployment — set these as environment variables in your project:\n\n"
            "     LICHESS_TOKEN = <your lichess token>\n"
            "     HF_TOKEN = <your hugging face token>\n\n"
            "     Railway → your project → Variables → + New Variable\n\n"
            "ℹ  The GUI polls the Bot URL every 3 seconds.  After changing the URL,\n"
            "     click Save Settings — the new address will be used on the next poll."
        )
        tk.Label(outer, text=note_text, fg=GREY2, bg=BG2,
                 font=("Segoe UI", 8), justify="left",
                 anchor="w").pack(anchor="w")

    # ── Polling ───────────────────────────────────────────────────────────────

    def _start_polling(self):
        t = threading.Thread(target=self._poll_loop, daemon=True)
        t.start()

    def _poll_loop(self):
        while not self._stop.is_set():
            if not HAS_REQUESTS:
                self.root.after(0, self._set_conn, False,
                                "requests not installed — run: pip install requests")
                self._stop.wait(5)
                continue
            url = self.settings.get("bot_url", "http://localhost:8080").rstrip("/")
            try:
                resp = requests.get(f"{url}/api/state", timeout=5)
                resp.raise_for_status()
                data = resp.json()
                self.root.after(0, self._apply_state, data)
                self.root.after(0, self._set_conn, True, f"Connected · {url}")
            except Exception as e:
                short = str(e)[:80]
                self.root.after(0, self._set_conn, False, f"Not connected · {short}")
            self._stop.wait(POLL_INTERVAL)

    def _set_conn(self, ok: bool, msg: str):
        self._conn_lbl.config(
            text=f"● {msg}",
            fg=GREEN if ok else RED,
        )

    # ── State application ─────────────────────────────────────────────────────

    def _apply_state(self, d: dict):
        self._state = d

        # ── Live tab ──
        game_id  = d.get("game_id")
        last_id  = d.get("last_game_id")
        opp      = d.get("opponent_username")
        our_col  = d.get("our_color")
        fen      = d.get("current_fen", DEFAULT_FEN)
        revenge  = d.get("is_revenge", False)
        bot_name = d.get("bot_username", "OUROBOROS")

        # Badges
        for w in self._badge_frame.winfo_children():
            w.pack_forget()
        if game_id:
            self._badge_live.pack(side="left", padx=(0, 4))
            if revenge:
                self._badge_revenge.pack(side="left")
        elif last_id:
            self._badge_ended.pack(side="left")

        # Matchup
        if game_id and opp and our_col:
            white = bot_name if our_col == "white" else opp
            black = bot_name if our_col == "black" else opp
            self._matchup_lbl.config(text=f"♙ {white}  vs  ♟ {black}", fg=FG2)
        else:
            self._matchup_lbl.config(text="Waiting for a game…", fg=GREY2)

        # Board
        flipped = (our_col == "black")
        frm_sqs, to_sqs = _diff_fen(self._prev_fen, fen)
        self._board.set_position(fen, flipped=flipped,
                                 from_sqs=frm_sqs, to_sqs=to_sqs)
        self._prev_fen = fen

        # Lichess link
        view_id = game_id or last_id
        if view_id:
            self._game_url = f"https://lichess.org/{view_id}"
            self._game_link.config(text="Open on Lichess  ↗")
        else:
            self._game_url = ""
            self._game_link.config(text="")

        # Mode
        mode = d.get("play_mode", "lichess")
        if mode == "lichess":
            self._mode_val.config(text="● Lichess", fg=GREEN)
        else:
            self._mode_val.config(text="△ Self-play", fg=GOLD)

        # Force-game button state
        if game_id:
            self._force_btn.config(text="■  In Game", state="disabled")
        else:
            self._force_btn.config(text="▶  Play One Game", state="normal",
                                   fg=GOLD)

        # Record
        w_n  = d.get("total_wins",   0)
        dr_n = d.get("total_draws",  0)
        l_n  = d.get("total_losses", 0)
        played = w_n + dr_n + l_n
        self._wins_val.config(  text=str(w_n),  fg=GREEN)
        self._draws_val.config( text=str(dr_n), fg=GREY2)
        self._losses_val.config(text=str(l_n),  fg=RED)
        if played:
            pct = round((w_n + 0.5 * dr_n) / played * 100)
            self._score_val.config(text=f"{pct}%", fg=FG)
        else:
            self._score_val.config(text="—", fg=GREY)

        # Native backend
        hn = d.get("has_native")
        if hn is True:
            self._native_val.config(text="Rust (native)", fg=GREEN)
        elif hn is False:
            self._native_val.config(text="Pure Python", fg=GOLD)
        else:
            self._native_val.config(text="Unknown", fg=GREY)

        # ── Training tab ──
        sp    = d.get("selfplay_games",  0)
        steps = d.get("train_steps",     0)
        spm   = d.get("steps_per_min",   0)
        loss  = d.get("last_loss")
        ploss = d.get("policy_loss")
        vloss = d.get("value_loss")
        lh    = d.get("loss_history",   [])
        ph    = d.get("ploss_history",  [])
        vh    = d.get("vloss_history",  [])
        bf    = d.get("buffer_fill",     0)
        bc    = d.get("buffer_cap", 1_000_000)
        elo   = d.get("ladder_elo",   1500.0)
        eloh  = d.get("elo_history",    [])
        src   = d.get("buffer_sources", {})

        self._box_sp.set(   f"{sp:,}")
        self._box_steps.set(f"{steps:,}")
        self._box_spm.set(  f"{spm:.1f}" if spm else "—")
        self._box_tloss.set(_fmt(loss))
        self._box_ploss.set(_fmt(ploss))
        self._box_vloss.set(_fmt(vloss))

        if lh: self._box_tloss.set_spark(lh)
        if ph: self._box_ploss.set_spark(ph)
        if vh: self._box_vloss.set_spark(vh)

        # Buffer
        self._buf_fill_pct = min(1.0, bf / bc) if bc > 0 else 0.0
        self._redraw_buf()
        self._buf_info_lbl.config(
            text=f"{self._buf_fill_pct*100:.0f}%  —  {bf:,} / {bc:,} positions")

        # Composition
        self._comp_data = src
        self._redraw_comp()
        sp_n = src.get("selfplay",   0)
        li_n = src.get("live",       0)
        im_n = src.get("imitation",  0)
        self._comp_lbl.config(
            text=f"Self-play: {sp_n:,}   Lichess: {li_n:,}   Imitation: {im_n:,}")

        # ELO
        self._elo_num.config(text=str(round(elo)))
        if len(eloh) >= 2:
            delta = elo - eloh[-2]
            sign  = "+" if delta > 0 else ""
            color = GREEN if delta > 0.5 else (RED if delta < -0.5 else GREY2)
            self._elo_delta.config(
                text=f"{sign}{delta:.0f} from last match", fg=color)
        else:
            self._elo_delta.config(text="Awaiting first ladder match", fg=GREY)
        if len(eloh) >= 2:
            self._elo_chart.set_values(eloh)

    # ── Canvas redraws ────────────────────────────────────────────────────────

    def _redraw_buf(self, _event=None):
        c = self._buf_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 2 or h < 2:
            return
        c.create_rectangle(0, 0, w, h, fill=BG3, outline="")
        fw = max(0, int(w * self._buf_fill_pct))
        if fw:
            c.create_rectangle(0, 0, fw, h, fill=GOLD, outline="")

    def _redraw_comp(self, _event=None):
        c = self._comp_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 2 or h < 2:
            return
        src   = self._comp_data
        total = sum(src.values()) if src else 0
        c.create_rectangle(0, 0, w, h, fill=BG3, outline="")
        if total:
            sp_w = int(w * src.get("selfplay",  0) / total)
            li_w = int(w * src.get("live",      0) / total)
            im_w = w - sp_w - li_w
            x = 0
            for width, color in [(sp_w, BLUE), (li_w, GOLD), (im_w, PURPLE)]:
                if width > 0:
                    c.create_rectangle(x, 0, x + width, h, fill=color, outline="")
                    x += width

    # ── Countdown tick (runs in main thread every second) ─────────────────────

    def _tick(self):
        now = time.time()
        d   = self._state

        sw = d.get("mode_switch_at", 0) or 0
        if sw > now:
            self._mode_switch_val.config(
                text=_countdown_str(int(sw - now)), fg=FG2)
        else:
            self._mode_switch_val.config(text="—", fg=GREY)

        nc   = d.get("next_challenge_at", 0) or 0
        mode = d.get("play_mode", "lichess")
        gid  = d.get("game_id")
        if mode == "lichess" and not gid and nc > 0:
            rem = max(0, int(nc - now))
            self._challenge_val.config(
                text="Challenge sent!" if rem == 0 else _countdown_str(rem),
                fg=GREEN if rem == 0 else GOLD,
            )
        else:
            self._challenge_val.config(text="—", fg=GREY)

        self.root.after(1000, self._tick)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _force_game(self):
        if not HAS_REQUESTS:
            messagebox.showerror("Missing dependency",
                                 "pip install requests\nthen restart the GUI.")
            return
        url = self.settings.get("bot_url", "http://localhost:8080").rstrip("/")
        self._force_btn.config(text="Sending…", state="disabled")

        def _do():
            try:
                resp = requests.post(f"{url}/api/force-game", timeout=5)
                ok   = resp.json().get("ok", False)
            except Exception:
                ok = False
            color = GREEN if ok else RED
            label = "✓  Challenge sent!" if ok else "✗  Failed"
            self.root.after(0, lambda: self._force_btn.config(
                text=label, fg=color, state="disabled"))
            time.sleep(2.5)
            self.root.after(0, lambda: self._force_btn.config(
                text="▶  Play One Game", fg=GOLD, state="normal"))

        threading.Thread(target=_do, daemon=True).start()

    def _open_game(self, _event=None):
        if self._game_url:
            import webbrowser
            webbrowser.open(self._game_url)

    def _save_settings_action(self):
        self.settings["bot_url"]       = self._url_e.get().strip()
        self.settings["lichess_token"] = self._lichess_e.get().strip()
        self.settings["hf_token"]      = self._hf_e.get().strip()
        _save_gui_settings(self.settings)
        messagebox.showinfo("Saved",
                            "Settings saved to gui_settings.json\n\n"
                            "The new bot URL will be used on the next poll.")

    def _on_close(self):
        self._stop.set()
        self.root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.title("OUROBOROS")

    # Attempt to set a chess-piece window icon (silent if it fails)
    try:
        img = tk.PhotoImage(width=1, height=1)
        root.iconphoto(True, img)
    except Exception:
        pass

    if not HAS_REQUESTS:
        import tkinter.messagebox as mb
        mb.showwarning(
            "Missing dependency",
            "The 'requests' library is not installed.\n\n"
            "Open a terminal and run:\n    pip install requests\n\n"
            "The GUI will open but cannot connect until requests is available.",
        )

    OuroborosGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
