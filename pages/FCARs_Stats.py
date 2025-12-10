# import streamlit as st
# import pandas as pd
# from pathlib import Path
# from datetime import date
# from utils import get_departments, get_courses, upsert_fcar,logo,add_sidebar_logo,summarize_fcar,inject_custom_css
# inject_custom_css()
# st.set_page_config(page_title="FCAR ", page_icon=str(logo), layout="wide")
# # st.set_page_config(page_title="FCAR (Detailed)", page_icon="📝", layout="wide")
# st.markdown("<h1 style='color:#046d5a;'>FCAR Stats</h1>", unsafe_allow_html=True)
# # st.title("Faculty Course Assessment Report (FCAR) — Detailed")
# st.caption("Page for editing and submitting new FCAR entries")
# deps = get_departments()
# courses = get_courses()
# dept_ids = deps["dept_id"].tolist() if not deps.empty else []
# course_ids = courses["course_id"].tolist() if not courses.empty else []
# DATA_PATH = Path(__file__).parents[1] / "data" / "fcar_detail.csv"
# DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
# # # ===== FCAR summary =====
# summary, fcar_df = summarize_fcar()

# k1, k2, k3 = st.columns(3)
# k1.metric("Total FCAR", summary.get("total", 0))
# status_df = summary.get("status", pd.DataFrame(columns=["status", "count"]))
# k2.metric("Statuses", len(status_df))
# by_dept = summary.get("by_dept", pd.DataFrame())
# k3.metric("Active Departments", int(by_dept["dept_id"].nunique()) if not by_dept.empty else 0)

# st.subheader("FCAR Status Overview")
# if not status_df.empty:
#     st.bar_chart(status_df.set_index("status"))
# else:
#     st.info("No FCAR data yet.")

# st.subheader("All FCAR Table")
# st.dataframe(fcar_df,width='stretch')

# st.markdown("---")

# # Quick KPIs
# summary, df = summarize_fcar()
# col1, col2, col3 = st.columns(3)
# col1.metric("Total FCAR Submissions", summary.get("total", 0))
# if "status" in summary:
#     status_df = summary["status"]
# else:
#     status_df = pd.DataFrame(columns=["status","count"])
# col2.metric("Statuses Tracked", len(status_df))
# dept_count = summary.get("by_dept", pd.DataFrame())
# col3.metric("Departments Active", int(dept_count["dept_id"].nunique()) if not dept_count.empty else 0)
# st.subheader("Recent FCAR Entries")
# st.dataframe(df, width='stretch')
import pandas as pd
import streamlit as st

from pathlib import Path

# Reuse helpers from your submit page / utils
from utils import inject_custom_css, logo, pct_to_dot
from pages.FCAR_submit import get_db  # <-- change to your actual module name
# login guard
if "user" not in st.session_state:
    st.error("You must be logged in to access this page. Please go to the Home page and log in.")
    st.stop()

# Optional: show who is logged in in the sidebar
st.sidebar.markdown(f"👤 **User:** {st.session_state.get('user')} ({st.session_state.get('role')})")

# ---- Page setup ----
inject_custom_css()
# st.set_page_config(page_title="FCAR Dashboard", page_icon=str(logo), layout="wide")
st.markdown("<h1 style='color:#046d5a;'>FCAR Dashboard</h1>", unsafe_allow_html=True)
st.caption("Explore submitted FCARs by department, course, term, and SLO/PC performance.")

GRADE_LABELS = ["A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "F", "W", "I", "Total"]


# ---------- Data loaders ----------
@st.cache_data(show_spinner=False)
def load_fcar_headers():
    db = get_db("fcar_db")
    if not db:
        return pd.DataFrame()
    rows = list(db.fcar_headers.find({}))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # Make _id printable
    if "_id" in df.columns:
        df["_id"] = df["_id"].astype(str)

    # Coerce grade columns to numeric
    for col in [f"grade_{g}" for g in GRADE_LABELS]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


@st.cache_data(show_spinner=False)
def load_fcar_pcs():
    db = get_db("fcar_db")
    if not db:
        return pd.DataFrame()
    rows = list(db.fcar_pcs.find({}))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "_id" in df.columns:
        df["_id"] = df["_id"].astype(str)
    return df


headers_df = load_fcar_headers()
pcs_df = load_fcar_pcs()

if headers_df.empty:
    st.warning("No FCAR records found in MongoDB. Submit at least one FCAR first.")
    st.stop()

# ---------- Filters ----------
with st.sidebar:
    st.subheader("Filters")

    dept_options = ["(All)"] + sorted(
        [d for d in headers_df["dept_id"].dropna().unique().tolist() if d != ""]
    )
    dept_filter = st.selectbox("Department", dept_options)

    # Filtered course list depends on department
    filtered_hdr_for_courses = (
        headers_df if dept_filter == "(All)" else headers_df[headers_df["dept_id"] == dept_filter]
    )

    course_options = ["(All)"] + sorted(
        filtered_hdr_for_courses["course_code_title"].dropna().unique().tolist()
    )
    course_filter = st.selectbox("Course (Code & Title)", course_options)

    term_options = ["(All)"] + sorted(
        filtered_hdr_for_courses["term"].dropna().unique().tolist()
    )
    term_filter = st.selectbox("Term", term_options)

# Apply filters to headers
hdr_filtered = headers_df.copy()
if dept_filter != "(All)":
    hdr_filtered = hdr_filtered[hdr_filtered["dept_id"] == dept_filter]
if course_filter != "(All)":
    hdr_filtered = hdr_filtered[hdr_filtered["course_code_title"] == course_filter]
if term_filter != "(All)":
    hdr_filtered = hdr_filtered[hdr_filtered["term"] == term_filter]

if hdr_filtered.empty:
    st.info("No FCARs match your filter selection.")
    st.stop()

# Filter PCs using matching fcar_ids
fcar_ids = hdr_filtered["fcar_id"].unique().tolist()
pcs_filtered = pcs_df[pcs_df["fcar_id"].isin(fcar_ids)] if not pcs_df.empty else pcs_df

# ---------- Top-level metrics ----------
col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.metric("Number of FCARs", len(hdr_filtered))

with col_m2:
    distinct_courses = hdr_filtered["course_code_title"].nunique()
    st.metric("Distinct Courses", distinct_courses)

with col_m3:
    if not pcs_filtered.empty:
        avg_pct = pcs_filtered["pct"].mean()
        st.metric("% Met Standard (avg across PCs)", f"{avg_pct:.1f}%")
    else:
        st.metric("% Met Standard (avg across PCs)", "—")

st.markdown("---")

# ---------- Grade Distribution ----------
st.subheader("Grade Distribution")

grade_cols = [f"grade_{g}" for g in GRADE_LABELS if f"grade_{g}" in hdr_filtered.columns]
if grade_cols:
    grade_totals = hdr_filtered[grade_cols].sum()

    grade_chart_df = pd.DataFrame(
        {"Grade": [g.replace("grade_", "") for g in grade_cols], "Count": grade_totals.values}
    ).set_index("Grade")

    c1, c2 = st.columns([2, 1])

    with c1:
        st.bar_chart(grade_chart_df)

    with c2:
        st.dataframe(grade_chart_df, use_container_width=True)
else:
    st.info("No grade columns found to display distribution.")

st.markdown("---")

# ---------- PC / SLO Performance ----------
st.subheader("Student Learning Outcomes / PCs")

if pcs_filtered.empty:
    st.info("No PC/SLO rows found for the current filter.")
else:
    # Aggregate by PC code and/or title
    group_cols = ["code"]
    if "title" in pcs_filtered.columns:
        group_cols.append("title")

    pcs_agg = (
        pcs_filtered.groupby(group_cols, dropna=False)
        .agg(
            n_fcars=("fcar_id", "nunique"),
            n_rows=("fcar_id", "size"),
            avg_pct=("pct", "mean"),
            avg_students=("students", "mean"),
        )
        .reset_index()
    )

    # Sort worst to best performance
    pcs_agg = pcs_agg.sort_values("avg_pct")

    st.caption("Average % Met Standard per PC code (across filtered FCARs).")
    st.dataframe(pcs_agg, use_container_width=True)

    # Simple bar chart for avg % met
    if not pcs_agg.empty:
        chart_df = pcs_agg.set_index("code")[["avg_pct"]]
        st.bar_chart(chart_df)

st.markdown("---")

# ---------- Detailed FCAR Table ----------
st.subheader("Detailed FCAR Records")

cols_to_show = [
    "fcar_id",
    "submitted_on",
    "dept_id",
    "course_id",
    "course_code_title",
    "section",
    "semester_hours",
    "course_coordinator",
    "academic_year_semester",
    "instructor_name",
    "term",
]
cols_to_show = [c for c in cols_to_show if c in hdr_filtered.columns]

st.dataframe(hdr_filtered[cols_to_show], use_container_width=True)
