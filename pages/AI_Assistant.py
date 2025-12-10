# pages/05_AI_Assistant.py
import os
import streamlit as st
from openai import OpenAI
from utils import get_fcar, ai_summarize_fcar,logo,add_sidebar_logo,inject_custom_css
inject_custom_css()
st.set_page_config(page_title="AI Assistant", page_icon=str(logo), layout="wide")
add_sidebar_logo()
st.markdown("<h1 style='color:#046d5a;'>Ask AI</h1>", unsafe_allow_html=True)
# st.title("Ask AI")
st.subheader("AI Chatbot Assistant to assist with LAU course assessment")

# --- API key (env var or Streamlit secrets) ---
API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.error(
        "No OpenAI API key found. Set OPENAI_API_KEY as an environment variable "
        "or add it to .streamlit/secrets.toml."
    )
    st.stop()

client = OpenAI(api_key=API_KEY)

# --- Initialize chat history in session state ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                 "You are the LAU Assessment Assistant embedded inside the LAU SoAS FCAR web app.\n\n"
                "Your knowledge and answers MUST stay within these topics:\n"
                "- How to use this FCAR system: Submit / Edit page, AI extraction from Word FCARs, "
                "attachments, saving to the database, and viewing results in the dashboard.\n"
                "- How LAU course assessment is done using this tool: grade distribution, PCs/SLOs, "
                "performance standards, course delivery (teaching strategies, content, resources, assessment "
                "measures), instructor reflection, and how these support program-level assessment and accreditation.\n\n"
                "If the user asks about anything outside LAU course assessment or this FCAR app "
                "(e.g., general coding help, random AI topics, personal questions, non-LAU topics), reply:\n"
                "\"I can only help with this FCAR system and LAU course assessment.\"\n\n"
                "Communication style:\n"
                "- Use clear, simple, professional language.\n"
                "- Prefer short paragraphs and numbered steps when explaining workflows.\n"
                "- When asked \"how assessment goes\" or similar, describe the process using this tool: "
                "1) Instructor fills FCAR, 2) PCs and grades recorded, 3) data stored in DB, "
                "4) dashboards summarize results for departments and accreditation.\n"
                "- Ask early in the conversation who the person is (e.g., instructor, course coordinator, "
                "department chair, assessment officer, admin) so you can tailor your explanations.\n"
           
            ),
        }
    ]
    # Optional: friendly greeting
    st.session_state.messages.append(
        {"role": "assistant", "content": "Hi! I’m the LAU FCAR Assistant. "
                "Are you an instructor, coordinator, chair, or admin?"}
    )

# --- Render existing history ---
for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.write(m["content"])

# --- User input box (handles Enter) ---
user_text = st.chat_input("Ask about FCAR, dashboards, or assessment workflow…")

if user_text:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    # Optional: ground the assistant with current FCAR summary as system context
    try:
        fcar_df = get_fcar()
        fcar_context = ai_summarize_fcar(fcar_df) if fcar_df is not None else ""
    except Exception:
        fcar_context = ""

    # Build prompt with optional context
    prompt_messages = list(st.session_state.messages)
    if fcar_context:
        prompt_messages.insert(
            1,
            {
                "role": "system",
                "content": f"Context from current FCAR data:\n{fcar_context}",
            },
        )

    # --- Call OpenAI Chat Completions ---
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",   # choose your model
            messages=prompt_messages,
            temperature=0.2,
        )
        reply = resp.choices[0].message.content.strip()
    except Exception as e:
        reply = f"Error talking to the model: {e}"

    # Show and store assistant reply
    with st.chat_message("assistant"):
        st.write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

# --- Sidebar tools ---
# with st.sidebar:
#     if st.button("🧹 Clear chat"):
#         st.session_state.messages = st.session_state.messages[:1]  # keep system prompt only
#         st.experimental_rerun()
# --- Sidebar tools ---
with st.sidebar:
    st.markdown("### AI Assistant Help")
    st.markdown(
        "- Ask how to fill the FCAR form\n"
        "- Ask how PCs/SLOs and % met are used\n"
        "- Ask how the dashboard summarizes FCAR data\n"
        "- Ask how this supports accreditation reports\n"
    )
    if st.button("🧹 Clear chat"):
        # Reset to system + first greeting
        base_system = st.session_state.messages[0]
        st.session_state.messages = [base_system]
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "Hi again! I can help with this FCAR system and "
                    "LAU course assessment. Are you an instructor, coordinator, chair, or admin?"
                ),
            }
        )
        