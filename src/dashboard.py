"""
Unified Swarm Dashboard
=======================
Visualizes the health, memory, and activity of the Nexus Swarm.

Run with: streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import time
import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

try:
    from backend.core.swarm import SwarmContext, SwarmEvent
    from backend.core.plugin_system import AgentRegistry
except ImportError:
    st.error("Backend modules not found. Run from valid environment.")
    st.stop()

st.set_page_config(page_title="Nexus Swarm Monitor", layout="wide", page_icon="🐝")

st.title("🐝 Nexus Swarm Intelligence Dashboard")

# --- Sidebar Controls ---
st.sidebar.header("Swarm Controls")
refresh_rate = st.sidebar.slider("Refresh Rate (s)", 1, 60, 5)
show_memory = st.sidebar.checkbox("Show Vector Memory", True)

# --- Initialize ---
if 'swarm' not in st.session_state:
    st.session_state.swarm = SwarmContext()
    st.session_state.swarm.init_vector_memory()
    
if 'registry' not in st.session_state:
    st.session_state.registry = AgentRegistry()
    st.session_state.registry.discover_agents()

swarm = st.session_state.swarm
registry = st.session_state.registry

# --- Metrics Row ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Active Agents", len(registry.agents))
with col2:
    pending_tasks = len(swarm.get_pending_tasks())
    st.metric("Pending Tasks", pending_tasks)
with col3:
    insights_count = 0
    if hasattr(swarm, 'memory_collection') and swarm.memory_collection:
        insights_count = swarm.memory_collection.count()
    st.metric("Stored Insights", insights_count)
with col4:
    last_event = "None"
    if swarm._message_history:
        last = swarm._message_history[-1]
        last_event = f"{last.type.name} ({int(time.time() - last.timestamp)}s ago)"
    st.metric("Last Event", last_event)

# --- Main Layout ---

tab1, tab2, tab3 = st.tabs(["🔴 Live Activity", "🧠 Cluster Memory", "🤖 Agent Health"])

with tab1:
    st.subheader("Recent Swarm Events")
    if swarm._message_history:
        events = []
        for msg in reversed(swarm._message_history[-20:]):
            events.append({
                "Time": time.strftime('%H:%M:%S', time.localtime(msg.timestamp)),
                "Type": msg.type.name,
                "Source": msg.source_agent,
                "Content": str(msg.content)[:100] + "..."
            })
        st.dataframe(pd.DataFrame(events), use_container_width=True)
    else:
        st.info("No events recorded yet.")

with tab2:
    st.subheader("Vector Memory Contents")
    query = st.text_input("Search Insights", placeholder="e.g., sales trends")
    if query:
        results = swarm.query_insights(query, n_results=5)
        for res in results:
            with st.expander(f"Insight ({res.get('metadata', {}).get('agent', 'Unknown')})"):
                st.write(res.get('content'))
                st.caption(f"Metadata: {res.get('metadata')}")
    else:
        # Show all (limited) if no query
        if hasattr(swarm, 'memory_collection') and swarm.memory_collection:
            count = swarm.memory_collection.count()
            if count > 0:
                peek = swarm.memory_collection.peek(limit=10)
                # Reformat peek results for display
                # peek returns {'ids': [], 'embeddings': [], 'metadatas': [], 'documents': []}
                if peek and peek['documents']:
                    for i, doc in enumerate(peek['documents']):
                         meta = peek['metadatas'][i] if peek['metadatas'] else {}
                         st.info(f"**{meta.get('agent', 'System')}**: {doc}")
            else:
                 st.write("Memory is empty.")

with tab3:
    st.subheader("Agent Registry")
    agents = []
    for name, agent in registry.agents.items():
        meta = agent.metadata
        agents.append({
            "Name": meta.name,
            "Version": meta.version,
            "Capabilities": [c.name for c in meta.capabilities],
            "Priority": meta.priority
        })
    st.dataframe(pd.DataFrame(agents), use_container_width=True)

# Auto-refresh logic
time.sleep(refresh_rate)
st.rerun()
