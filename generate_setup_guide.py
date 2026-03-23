#!/usr/bin/env python3
"""Generate setup guide PDF using reportlab."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted, HRFlowable,
    ListFlowable, ListItem, Table, TableStyle
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

OUTPUT = "setup_guide.pdf"

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=letter,
    leftMargin=0.85 * inch,
    rightMargin=0.85 * inch,
    topMargin=0.85 * inch,
    bottomMargin=0.85 * inch,
)

styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle(
    "Title", parent=styles["Title"], fontSize=20, spaceAfter=4, textColor=colors.HexColor("#1a1a2e")
)
subtitle_style = ParagraphStyle(
    "Subtitle", parent=styles["Normal"], fontSize=11, spaceAfter=14,
    textColor=colors.HexColor("#555555"), alignment=TA_CENTER
)
h1_style = ParagraphStyle(
    "H1", parent=styles["Heading1"], fontSize=13, spaceBefore=14, spaceAfter=4,
    textColor=colors.HexColor("#1a1a2e"), borderPad=2,
)
h2_style = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontSize=11, spaceBefore=10, spaceAfter=3,
    textColor=colors.HexColor("#2c3e6b"),
)
body_style = ParagraphStyle(
    "Body", parent=styles["Normal"], fontSize=9.5, leading=14, spaceAfter=4,
)
note_style = ParagraphStyle(
    "Note", parent=styles["Normal"], fontSize=9, leading=13,
    textColor=colors.HexColor("#7a0000"), leftIndent=12,
    borderPad=4, backColor=colors.HexColor("#fff3f3"),
)
code_style = ParagraphStyle(
    "Code", parent=styles["Code"], fontSize=8.2, leading=12,
    backColor=colors.HexColor("#f4f4f4"), leftIndent=10, rightIndent=10,
    spaceAfter=6, spaceBefore=3,
)
bullet_style = ParagraphStyle(
    "Bullet", parent=styles["Normal"], fontSize=9.5, leading=14,
    leftIndent=16, spaceAfter=2,
)
label_style = ParagraphStyle(
    "Label", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#444444"),
)

def code(text):
    return Preformatted(text, code_style)

def h1(text):
    return Paragraph(text, h1_style)

def h2(text):
    return Paragraph(text, h2_style)

def body(text):
    return Paragraph(text, body_style)

def note(text):
    return Paragraph(f"<b>Note:</b> {text}", note_style)

def bullet(text):
    return Paragraph(f"• {text}", bullet_style)

def sp(n=6):
    return Spacer(1, n)

def hr():
    return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=4, spaceBefore=4)

# ── mount table ──────────────────────────────────────────────────────────────
mount_data = [
    [Paragraph("<b>Host path</b>", label_style), Paragraph("<b>Purpose</b>", label_style)],
    [Paragraph("<font name='Courier' size='8'>~/.claude/</font>", label_style),
     Paragraph("Claude Code config, session data, and memory", label_style)],
    [Paragraph("<font name='Courier' size='8'>~/.claude.json</font>", label_style),
     Paragraph("Claude Code auth token / account info", label_style)],
    [Paragraph("<font name='Courier' size='8'>~/.config/gh/</font>", label_style),
     Paragraph("GitHub CLI credentials", label_style)],
]
mount_table = Table(mount_data, colWidths=[2.5 * inch, 4.2 * inch])
mount_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eaf6")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
]))

# ── build story ──────────────────────────────────────────────────────────────
story = []

story.append(Paragraph("MTH9899 Project — Clone &amp; Run Setup Guide", title_style))
story.append(Paragraph("How to reproduce the DevContainer environment on a new machine", subtitle_style))
story.append(hr())
story.append(sp(4))

# ── Why Docker / overview box ─────────────────────────────────────────────────
overview_style = ParagraphStyle(
    "Overview", parent=styles["Normal"], fontSize=9.5, leading=14,
    backColor=colors.HexColor("#eef2ff"), leftIndent=12, rightIndent=12,
    spaceBefore=4, spaceAfter=4, borderPad=8,
    textColor=colors.HexColor("#1a1a2e"),
)
overview_bullet = ParagraphStyle(
    "OverviewBullet", parent=styles["Normal"], fontSize=9.5, leading=14,
    backColor=colors.HexColor("#eef2ff"), leftIndent=28, rightIndent=12,
    spaceAfter=3, textColor=colors.HexColor("#1a1a2e"),
)
story.append(Paragraph("<b>Why a Docker DevContainer?</b>", overview_style))
story.append(sp(2))
story.append(Paragraph(
    "This project runs Claude Code inside a Docker DevContainer rather than directly on the host. "
    "The container acts as a controlled, reproducible sandbox with several concrete benefits:",
    overview_style,
))
story.append(sp(3))
story.append(Paragraph(
    "• <b>Network firewall (tinyproxy)</b> — outbound traffic is restricted to a whitelist: "
    "Anthropic APIs, GitHub, PyPI, and npm. Claude Code cannot exfiltrate data or reach "
    "arbitrary internet endpoints, which limits the blast radius of any runaway or misbehaving agent.",
    overview_bullet,
))
story.append(Paragraph(
    "• <b>Reproducible environment</b> — every collaborator gets the same OS, Python version, "
    "and package set. No 'works on my machine' issues.",
    overview_bullet,
))
story.append(Paragraph(
    "• <b>Isolated filesystem</b> — Claude Code can only read/write files inside the container. "
    "Host files (outside the mounted paths) are not accessible.",
    overview_bullet,
))
story.append(Paragraph(
    "• <b>Swap pre-allocated</b> — the image reserves 4 GB of swap so large dataset loads "
    "don't OOM-kill the kernel.",
    overview_bullet,
))
story.append(Paragraph(
    "• <b>Credential sharing via bind-mounts</b> — Claude Code and GitHub CLI credentials "
    "are passed in from the host via targeted mounts rather than copied into the image, "
    "so they stay in sync and are never baked into a Docker layer.",
    overview_bullet,
))
story.append(sp(8))
story.append(hr())
story.append(sp(4))

# ── Section 1: Prerequisites ─────────────────────────────────────────────────
story.append(h1("1. Prerequisites (install once on your host machine)"))
story.append(body("The DevContainer expects three things to already exist on your host before you open the repo:"))
story.append(sp(4))

story.append(h2("1a. Docker Desktop (or Docker Engine)"))
story.append(bullet("macOS / Windows: install <b>Docker Desktop</b> from docker.com"))
story.append(bullet("Linux: install <b>Docker Engine</b> + the Compose plugin via your package manager"))
story.append(sp(2))

story.append(h2("1b. VS Code + Dev Containers extension"))
story.append(bullet("Install <b>Visual Studio Code</b>"))
story.append(bullet("Install the <b>Dev Containers</b> extension (ms-vscode-remote.remote-containers)"))
story.append(sp(2))

story.append(h2("1c. Claude Code — logged in"))
story.append(body("The container bind-mounts <font name='Courier' size='9'>~/.claude/</font> and "
                  "<font name='Courier' size='9'>~/.claude.json</font> from the host. "
                  "These files are created when you log in to Claude Code on the host."))
story.append(sp(3))
story.append(code("npm install -g @anthropic-ai/claude-code\nclaude          # follow the browser-based login flow"))
story.append(note("Both ~/.claude/ and ~/.claude.json must exist before opening the DevContainer. "
                  "The container will fail to start if either path is missing."))
story.append(sp(2))

story.append(h2("1d. GitHub CLI — logged in"))
story.append(body("The container bind-mounts <font name='Courier' size='9'>~/.config/gh/</font>. "
                  "Authenticate on the host first:"))
story.append(sp(3))
story.append(code("# macOS\nbrew install gh\n\n# Linux\n# see https://cli.github.com for your distro\n\ngh auth login   # choose HTTPS + browser or token"))
story.append(sp(8))

# ── Section 2: Clone & Open ───────────────────────────────────────────────────
story.append(hr())
story.append(h1("2. Clone the Repository"))
story.append(code("git clone https://github.com/<org>/MTH9899-Project.git\ncd MTH9899-Project"))
story.append(body("Replace <font name='Courier' size='9'>&lt;org&gt;</font> with the actual GitHub username or org."))
story.append(sp(8))

# ── Section 3: Open in DevContainer ──────────────────────────────────────────
story.append(hr())
story.append(h1("3. Open in DevContainer"))
story.append(body("VS Code detects the <font name='Courier' size='9'>.devcontainer/</font> folder automatically."))
story.append(sp(4))
story.append(bullet("<b>Command Palette</b> → <i>Dev Containers: Reopen in Container</i>"))
story.append(bullet("<b>Or</b>: click the green <b>&gt;&lt;</b> button (bottom-left) → <i>Reopen in Container</i>"))
story.append(sp(6))
story.append(body("What happens automatically on first build:"))
story.append(sp(3))

build_steps = [
    ("Docker image built", "Ubuntu base + Node 20 + Claude Code (npm) + Python 3 + all packages"),
    ("4 GB swap file", "Pre-allocated in the image to prevent OOM during large data loads"),
    ("tinyproxy firewall", "postStartCommand runs init-proxy.sh — restricts outbound traffic to "
                           "Anthropic, GitHub, PyPI, and npm only"),
    ("pip install", "postCreateCommand runs: pip install -r requirements.txt "
                    "(adds xgboost + matplotlib on top of the image packages)"),
    ("Mounts activated", "~/.claude, ~/.claude.json, and ~/.config/gh are bind-mounted read-write "
                         "into the container so Claude Code and gh work immediately"),
]

for name, desc in build_steps:
    story.append(Paragraph(
        f"• <b>{name}</b> — {desc}", bullet_style
    ))
story.append(sp(8))

# ── Section 4: Mount reference ────────────────────────────────────────────────
story.append(hr())
story.append(h1("4. What the DevContainer Mounts from the Host"))
story.append(sp(4))
story.append(mount_table)
story.append(sp(6))
story.append(note("These are read-write mounts. Changes made inside the container (e.g., new Claude "
                  "memory files) persist on the host, and vice-versa."))
story.append(sp(8))

# ── Section 5: Run the project ────────────────────────────────────────────────
story.append(hr())
story.append(h1("5. Running the Project"))

story.append(h2("Install remaining dependencies (if not already done by postCreateCommand)"))
story.append(code("pip install -r requirements.txt --break-system-packages"))

story.append(h2("Run tests"))
story.append(code("pytest"))

story.append(h2("Interactive notebook (full pipeline)"))
story.append(code("jupyter lab     # then open run.ipynb"))

story.append(h2("CLI — generate features from OOS data (Mode 1)"))
story.append(code("python3 main.py -m 1 -i oos_data -o /tmp/features -s 20150101 -e 20151231"))

story.append(h2("CLI — generate predictions (Mode 2)"))
story.append(code("python3 main.py -m 2 -i /tmp/features -o /tmp/preds -p saved_model -s 20150101 -e 20151231"))
story.append(sp(8))

# ── Section 6: Data ───────────────────────────────────────────────────────────
story.append(hr())
story.append(h1("6. Data (gitignored — obtain separately)"))
story.append(body("All three raw data directories are listed in <font name='Courier' size='9'>.gitignore</font> "
                  "and must be sourced from the course data share:"))
story.append(sp(4))
story.append(bullet("<font name='Courier' size='9'>DailyData/</font> — one CSV per trading day, "
                    "OHLCV + adjustment factors (2010–2014)"))
story.append(bullet("<font name='Courier' size='9'>data_intraday/</font> — one CSV per trading day, "
                    "15-minute residual return snapshots (2010–2014)"))
story.append(bullet("<font name='Courier' size='9'>oos_data/</font> — held-out 2015 test data "
                    "(daily_data/ + intraday_data/ subdirs); now gitignored — obtain from the course share"))
story.append(sp(6))
story.append(body("Place all three directories at the repo root. "
                  "The saved model in <font name='Courier' size='9'>saved_model/</font> is committed "
                  "and ready to use — Mode 2 predictions require only "
                  "<font name='Courier' size='9'>oos_data/</font>, not the full training set."))
story.append(sp(4))
story.append(note(
    "oos_data/ was previously committed to the repo and has since been added to .gitignore. "
    "Adding a path to .gitignore does NOT remove already-tracked files from git history or the "
    "working tree. To fully untrack it, run: "
    "git rm -r --cached oos_data/  then commit. "
    "Until that step is taken, git will continue to track any previously committed files in that directory."
))
story.append(sp(8))

# ── Section 7: Troubleshooting ────────────────────────────────────────────────
story.append(hr())
story.append(h1("7. Common Issues"))

issues = [
    ("Container fails to start",
     "Check that ~/.claude/, ~/.claude.json, and ~/.config/gh/ all exist on the host. "
     "Run claude and gh auth login on the host, then rebuild the container."),
    ("OOM / kernel kill during data load",
     "The image pre-allocates 4 GB of swap. The postStartCommand enables it via swapon. "
     "If it was skipped, run: sudo swapon /swapfile inside the container."),
    ("python command not found",
     "Use python3 — the python alias is not installed in this container."),
    ("xgboost import error",
     "Run: pip install xgboost --break-system-packages"),
    ("Network blocked inside container",
     "tinyproxy allows only Anthropic, GitHub, PyPI, and npm. "
     "Other outbound requests will be blocked by design."),
]

for problem, solution in issues:
    story.append(Paragraph(f"<b>{problem}</b>", ParagraphStyle(
        "IssueTitle", parent=body_style, spaceBefore=6, spaceAfter=1,
        textColor=colors.HexColor("#2c3e6b"),
    )))
    story.append(Paragraph(solution, ParagraphStyle(
        "IssueSol", parent=body_style, leftIndent=14, spaceAfter=2,
    )))

story.append(sp(10))
story.append(hr())
story.append(Paragraph(
    "MTH9899 Project — Setup Guide — generated 2026-03-23",
    ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8,
                   textColor=colors.HexColor("#aaaaaa"), alignment=TA_CENTER)
))

doc.build(story)
print(f"Written: {OUTPUT}")
