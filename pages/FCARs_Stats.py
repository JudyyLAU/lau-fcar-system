import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path

# Reuse helpers from your submit page / utils
from utils import inject_custom_css, logo, pct_to_dot,get_db,ensure_indexes

# login guard
if "user" not in st.session_state:
    st.error("You must be logged in to access this page. Please go to the Home page and log in.")
    st.stop()

# show who is logged in in the sidebar
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
    if db is None: 
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
    if db is None:
        return pd.DataFrame()
    rows = list(db.fcar_pcs.find({}))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "_id" in df.columns:
        df["_id"] = df["_id"].astype(str)
    return df




# Refresh button
if st.button("🔄 Refresh"):
    load_fcar_headers.clear()
    load_fcar_pcs.clear()
    st.rerun()
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

# # ---------- PC / SLO Performance ----------
# st.subheader("Student Learning Outcomes / PCs")

# if pcs_filtered.empty:
#     st.info("No PC/SLO rows found for the current filter.")
# else:
#     # Aggregate by PC code and/or title
#     group_cols = ["code"]
#     if "title" in pcs_filtered.columns:
#         group_cols.append("title")

#     pcs_agg = (
#         pcs_filtered.groupby(group_cols, dropna=False)
#         .agg(
#             n_fcars=("fcar_id", "nunique"),
#             n_rows=("fcar_id", "size"),
#             avg_pct=("pct", "mean"),
#             avg_students=("students", "mean"),
#         )
#         .reset_index()
#     )

#     # Sort worst to best performance
#     pcs_agg = pcs_agg.sort_values("avg_pct")

#     st.caption("Average % Met Standard per PC code (across filtered FCARs).")
#     st.dataframe(pcs_agg, use_container_width=True)

#     # Simple bar chart for avg % met
#     if not pcs_agg.empty:
#         chart_df = pcs_agg.set_index("code")[["avg_pct"]]
#         st.bar_chart(chart_df)

# st.markdown("---")
# ---------- PC / SLO Performance with different colors based on average----------
st.markdown("---")
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
            n_fcars=("fcar_id", "nunique"),        # how many distinct FCARs touched this PC
            avg_pct=("pct", "mean"),              # average % met
            avg_students=("students", "mean"),    # average number of students
        )
        .reset_index()
    )

    # Sort worst to best performance
    pcs_agg = pcs_agg.sort_values("avg_pct")

    st.caption("Average % Met Standard per PC code (across filtered FCARs).")
    st.dataframe(pcs_agg, use_container_width=True)

    # -------- PC vs Average % chart with highlighting < 60% --------
    if not pcs_agg.empty:
        # Use code as the x-axis; title in tooltip if available
        tooltip_cols = ["code", "avg_pct", "n_fcars", "avg_students"]
        if "title" in pcs_agg.columns:
            tooltip_cols.insert(1, "title")

        chart = (
            alt.Chart(pcs_agg)
            .mark_bar()
            .encode(
                x=alt.X("code:N", title="PC Code"),
                y=alt.Y("avg_pct:Q", title="Average % Met Standard"),
                color=alt.condition(
                    alt.datum.avg_pct < 60,
                    alt.value("#d62728"),   # highlight: below 60%
                    alt.value("#1ca023"),   # normal: 60% and above
                ),
                tooltip=tooltip_cols,
            )
            .properties(
                width="container",
                height=400,
                title="PC vs Average % Met Standard (highlighting < 60%)",
            )
        )

        st.altair_chart(chart, use_container_width=True)

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
