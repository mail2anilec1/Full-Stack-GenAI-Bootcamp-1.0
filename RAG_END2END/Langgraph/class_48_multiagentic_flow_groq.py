#!/usr/bin/env python
# coding: utf-8

# # LangGraph Multi-Agent Patterns
#
# Each pattern lives in its own script. Run only the script you want to study;
# each one starts its own model calls and does not execute the other examples.
#
# | # | Pattern | Main control style | Typical model calls in this test |
# |---:|---|---|---:|
# | 1 | Router | Select one specialist | 2 |
# | 2 | Subagents / Agent-as-Tool | Main agent delegates through tools | Variable |
# | 3 | Supervisor | Supervisor repeatedly selects workers | Variable, bounded |
# | 4 | Handoff | Triage transfers control to one specialist | 2 |
# | 5 | Swarm | Agents hand control to one another | Variable |
# | 6 | Hierarchical Supervisor | Top supervisor selects a team subgraph | 3 |
# | 7 | Parallel Collaboration with Send | Dynamic fan-out and aggregation | N + 2 |
# | 8 | Coordinator-Guided Peer Network | Coordinator repeatedly selects a peer | Variable, bounded |
#
# Run a pattern from the repository root, for example:
#
#     .venv_genai/bin/python RAG_END2END/Langgraph/class_48_01_router_multiagent_groq.py
#
# Available standalone scripts:
# 1. Router Multi-Agent: class_48_01_router_multiagent_groq.py
# 2. Subagents / Agent-as-Tool: class_48_02_subagents_agent_as_tool_groq.py
# 3. Supervisor Multi-Agent: class_48_03_supervisor_multiagent_groq.py
# 4. Handoff Multi-Agent: class_48_04_handoff_multiagent_groq.py
# 5. Swarm Multi-Agent: class_48_05_swarm_multiagent_groq.py
# 6. Hierarchical Supervisor: class_48_06_hierarchical_supervisor_groq.py
# 7. Parallel Multi-Agent Collaboration with `Send`: class_48_07_parallel_collaboration_groq.py
# 8. Coordinator-Guided Peer Network: class_48_08_coordinator_guided_peer_network_groq.py
#
# Set GROQ_MODEL in .env to choose another supported model.
# The default is openai/gpt-oss-20b.