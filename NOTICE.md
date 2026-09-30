# NOTICE

IntelliFlow — Enterprise Multi-Agent AI Assistant
=================================================

This project is a derivative work. It began as a fork of an MIT-licensed
open-source project and has been substantially reworked.

Upstream project
----------------

    LangGraph Multi-Agent Chatbot
    https://github.com/arbaz-builds/langgraph-multi-agent
    Copyright (c) 2026 Arbaz (Pokemon455)  —  https://github.com/arbaz-builds
    Licensed under the MIT License

Portions of this repository — including the LangGraph graph design, the
`add_messages` state reducer pattern, the bounded tool-calling loop, the
FastAPI `/health` and `/chat` entry points, the LangGraph PostgreSQL
checkpointing integration, and various defensive coding patterns documented in
`docs/PROJECT_AUDIT.md` §12 — originate from that project and remain
copyright (c) 2026 Arbaz (Pokemon455).

IntelliFlow modifications and new subsystems were written for this portfolio
project and are described in `docs/CHANGELOG.md` and
`docs/PROJECT_ORIGIN.md` §4.

The upstream MIT License text is retained verbatim at the repository root in
the `LICENSE` file. Those terms continue to apply to the upstream portions of
this codebase.

Full attribution, the reference-vs-modification breakdown, and the
third-party component list: see `docs/PROJECT_ORIGIN.md`.

Attribution request
-------------------

If you present this project publicly, please credit both the upstream author
and this derivative work:

    IntelliFlow — Enterprise Multi-Agent AI Assistant.
    Built on the MIT-licensed langgraph-multi-agent by Mohammad Arbaz,
    substantially reworked for portfolio purposes.
