
import streamlit as st
import pandas as pd
from utils import summarize_fcar, get_departments,add_sidebar_logo,logo,get_dept_profile,inject_custom_css
from pathlib import Path

inject_custom_css()

st.set_page_config(page_title="Assessment Hub ", page_icon=str(logo), layout="wide")
# st.image("/Users/judy.matar/Desktop/assessment_hub/data/lau-logo copy.jpg", width=200)
add_sidebar_logo()
st.markdown("<h1 style='color:#046d5a;'>Assessment Hub</h1>", unsafe_allow_html=True)
# st.title("Assessment Hub")
st.caption("Local dashboard for Assessment, FCAR collection, and live insights.")

st.markdown("<h1 style='color:#046d5a;'>LAU FCAR System</h1>", unsafe_allow_html=True)
st.caption("Restricted access – LAU course assessment and accreditation users only.")

# ---- Load users from secrets.toml ----
def load_users():
    auth = st.secrets.get("auth", {})
    
    users = auth.get("users", [])
    passwords = auth.get("passwords", [])
    roles = auth.get("roles", [])

    user_map = {}

    for i in range(len(users)):
        user_map[users[i]] = {
            "password": passwords[i],
            "role": roles[i] if i < len(roles) else "user"
        }

    return user_map


VALID_USERS = load_users()

def show_login():
    st.subheader("Login")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        user = VALID_USERS.get(username)
        if not user or user["password"] != password:
            st.error("Invalid username or password.")
        else:
            st.session_state["user"] = username
            st.session_state["role"] = user["role"]
            st.success(f"Welcome, {username}!")
            st.rerun()

def show_home():
    st.success(f"Logged in as **{st.session_state['user']}** "
               f"({st.session_state['role']})")
    # ----- picture and intro -----
    left, right = st.columns([2, 1], vertical_alignment="center")
    with left:
        st.markdown("<h2 style='color:#046d5a;'>What is this tool?</h2>", unsafe_allow_html=True)
        st.markdown(""" 
        A local app that helps departments of the SoAS- LAU collect FCAR reports, track progress, summarize findings, and view live dashboards all in one place.
        """ )
    
        st.markdown("<h2 style='color:#046d5a;'>What can you do?</h2>", unsafe_allow_html=True)
        st.page_link("pages/FCAR_Submit.py", label="Submit and manage FCARs",icon="📝")
        st.page_link("pages/FCARs_Stats.py", label="View FCARs Stats",icon="📝")
        # st.page_link("pages/Assessment_Plans.py", label="Submit and manage Assessment plans", icon="📝")
        st.page_link("pages/AI_Assistant.py", label="Use the AI Assistant for quick summaries and briefings", icon="🤖")
    with right:
        picSoAS = Path("data/logo_SoAS_LAU.png")
        st.image(str(picSoAS))
        
        if st.button("Logout"):
            for k in ["user", "role"]:
                st.session_state.pop(k, None)
            st.rerun()

if "user" not in st.session_state:
    show_login()
else:
    show_home()




# ----- ASSESSMENT PROCESS FLOW -----

import graphviz
st.markdown("<h2 style='color:#046d5a;'>Assessment Process</h2>", unsafe_allow_html=True)
# st.title("Assessment Process")

g = graphviz.Digraph("SimpleFlow", format="svg")
g.attr( splines="ortho", nodesep="0.1", ranksep="0.1",fontsize="2",shape="box")
#g.attr()
# Core steps (minimal)
g.node("plan", "Assessment Plan")
g.node("teach", "Teach and Measure\n(CLO evidence)")
g.node("fcar", "Draft FCAR\n(Findings & Actions)")
g.node("pss", "Program Review/Self study")
g.node("prd", "External Reviewers")
g.node("rr", "Reviewers Report")
g.node("change","Implement Changes")

# Main flow
g.edge("plan", "teach")
g.edge("teach", "fcar")
g.edge("fcar", "pss")
g.edge("pss", "prd")
g.edge("prd", "rr")
g.edge("rr", "change")
# Simple feedback loops
g.edge("change", "plan", label="repeat cycle", style="dotted")

st.graphviz_chart(g)
