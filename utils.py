
import pandas as pd
import streamlit as st
import os 
from pathlib import Path
DATA_DIR = Path(__file__).parent / "data"
logo = Path(__file__).parent / "data" / "lau-logo.jpg"
# add a logo
def add_sidebar_logo():
    logowhite = Path(__file__).parent / "data" / "lau-white-nobc.png"
    with st.sidebar:
        st.image(str(logowhite), width=200)


def add_page_logo():
    logo = Path(__file__).parent / "data" / "lau-logo.jpg"
    left, _ = st.columns([1, 9])
    with left:
        st.image(str(logo), width=160)

def _path(name:str):
    return DATA_DIR / name

def load_csv(name:str) -> pd.DataFrame:
    p = _path(name)
    if p.exists():
        return pd.read_csv(p)
    else:
        return pd.DataFrame()

def save_csv(df:pd.DataFrame, name:str):
    df.to_csv(_path(name), index=False)

def get_departments():
    return load_csv("departments.csv")

def get_courses():
    return load_csv("courses.csv")

def get_fcar():
    return load_csv("fcar.csv")

def upsert_fcar(row:dict):
    df = get_fcar()
    if df.empty:
        df = pd.DataFrame([row])
    else:
        if "fcar_id" in row and (df["fcar_id"]==row["fcar_id"]).any():
            df.loc[df["fcar_id"]==row["fcar_id"], :] = row
        else:
            next_id = (df["fcar_id"].max() if "fcar_id" in df.columns and pd.notna(df["fcar_id"]).any() else 0) + 1
            row["fcar_id"] = next_id
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_csv(df, "fcar.csv")
    return df

def summarize_fcar():
    df = get_fcar()
    if df.empty:
        return {}, pd.DataFrame()
    by_dept = df.groupby("dept_id").size().reset_index(name="count")
    status = df["status"].value_counts().reset_index()
    status.columns = ["status","count"]
    return {"total": len(df), "by_dept": by_dept, "status": status}, df

def ai_summarize_fcar(df:pd.DataFrame) -> str:
    """Very simple heuristic summarizer to avoid external API calls."""
    if df.empty:
        return "No FCAR entries yet. Once submissions are received, a summary of findings and actions will show here."
    top_gaps = df["findings"].dropna().head(5).tolist()
    actions = df["actions"].dropna().head(5).tolist()
    lines = []
    lines.append(f"- {len(df)} FCAR submissions in the system.")
    if "status" in df.columns:
        s = df["status"].value_counts().to_dict()
        s_str = ", ".join([f"{k}: {v}" for k,v in s.items()])
        lines.append(f"- Status split → {s_str}.")
    if top_gaps:
        lines.append("- Common gaps mentioned: " + "; ".join(top_gaps))
    if actions:
        lines.append("- Sample actions planned: " + "; ".join(actions))
    return "\n".join(lines)
def get_dept_profile():
    return load_csv("dept_profile.csv")

def normalize_accred(val: str) -> str:
    s = str(val).strip().lower()
    if "in progress" in s:
        return "In progress"
    if "yes" in s and "no" in s:
        return "Partial"
    if s.startswith("yes"):
        return "Yes"
    if s.startswith("no"):
        return "No"
    return "Unknown"

def summarize_dept_profile(df):
    if df is None or df.empty or "Accredited?" not in df.columns:
        return {"Yes": 0, "In progress": 0, "Partial": 0, "No": 0, "Unknown": 0}
    mapped = df["Accredited?"].map(normalize_accred)
    return mapped.value_counts().to_dict()

# ==== Dashboard helpers: accreditation badges & timelines ====
def status_badge_text(status: str) -> str:
    """Return an emoji badge for accreditation status."""
    s = normalize_accred(status)
    mapping = {
        "Yes": "🟢 Yes",
        "In progress": "🟠 In progress",
        "Partial": "🟡 Partial",
        "No": "🔴 No",
        "Unknown": "⚪ Unknown",
    }
    return mapping.get(s, "⚪ Unknown")

def split_steps(text: str):
    """Split 'Next Steps' text into a clean list (prefers ';', falls back to '.')."""
    if not isinstance(text, str) or not text.strip():
        return []
    parts = [p.strip(" -•\u2022\t\n\r ") for p in (text.split(";") if ";" in text else text.split("."))]
    return [p for p in parts if p and len(p) > 1]

def timeline_html(steps, color="#046d5a", title=None):
    """Build a simple vertical timeline with CSS."""
    if not steps:
        return ""
    items = "".join(
        f"""
        <div class="timeline-item">
            <div class="timeline-dot" style="border-color:{color};"></div>
            <div class="timeline-content">{step}</div>
        </div>
        """ for step in steps
    )
    title_html = f'<div class="timeline-title">{title}</div>' if title else ""
    css = f"""
    <style>
      .timeline {{ position: relative; margin: 0.5rem 0 1.25rem 0; padding-left: 1.5rem; }}
      .timeline:before {{
          content: ""; position: absolute; left: 9px; top: 0; bottom: 0; width: 2px; background: #e5e7eb;
      }}
      .timeline-title {{ font-weight: 600; margin-bottom: 0.5rem; }}
      .timeline-item {{ position: relative; margin: 0 0 0.75rem 0; }}
      .timeline-dot {{
          position: absolute; left: -1px; top: 2px; width: 7px; height: 7px;
          border-radius: 999px; background: #fff; border: 4px solid {color};
          box-shadow: 0 0 0 1px #fff;
      }}
      .timeline-content {{ margin-left: 1rem; line-height: 1.35rem; }}
    </style>
    """
    return css + f'<div class="timeline">{title_html}{items}</div>'
def get_program_workflow():
    return load_csv("program_workflow.csv")

def summarize_program_workflow():
    import pandas as pd
    df = get_program_workflow()
    if df is None or df.empty:
        return pd.DataFrame()
    # Simple by-department snapshot
    return (
        df.groupby("dept_id")
          .agg(
              programs=("program", "nunique"),
              plans_complete=("assessment_plan_complete", "sum"),
              slo_clo_mapped=("slo_clo_mapped", "sum"),
          )
          .reset_index()
    )
def inject_custom_css():
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                background-color: #046d5a !important;
            }
            [data-testid="stSidebar"] * {
                color: white !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
# helper to turn pct into a colored circle
def pct_to_dot(p):
    try:
        p = float(p or 0)
    except Exception:
        p = 0
    if p > 70:
        return "🟢"
    elif p >= 60:
        return "🟡"
    else:
        return "🔴"
# Import for mongodb
def lazy_import_pymongo():
    try:
        import pymongo  # noqa: F401
        from pymongo import MongoClient, ASCENDING
        return pymongo, MongoClient, ASCENDING
    except Exception:
        st.error("Missing dependency `pymongo`. Install it: `pip install pymongo`")
        raise


def get_mongo_client():
    _, MongoClient, _ = lazy_import_pymongo()
    uri = os.getenv("MONGODB_URI") or st.secrets.get("MONGODB_URI", "")
    if not uri:
        st.error("MONGODB_URI not set (env or st.secrets).")
        return None
    try:
        return MongoClient(uri, serverSelectionTimeoutMS=5000)
    except Exception as e:
        st.error(f"Mongo connection failed: {e}")
        return None


def get_db(db_name: str = "fcar_db"):
    client = get_mongo_client()
    return client[db_name] if client else None


def ensure_indexes(db):
    # Unique FCAR id on headers; not unique on pcs since multiple rows per fcar
    _, _, ASCENDING = lazy_import_pymongo()
    db.fcar_headers.create_index([("fcar_id", ASCENDING)], unique=True)
    db.fcar_pcs.create_index([("fcar_id", ASCENDING)], unique=False)
    # Optional: text index for quick search
    db.fcar_headers.create_index(
        [("course_code_title", "text"), ("instructor_name", "text")],
        name="hdr_text_idx",
    )
