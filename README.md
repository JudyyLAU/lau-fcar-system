
# Assessment Hub (Local Streamlit App)

A local, private Streamlit app for:
- Collecting FCAR reports via structured forms
- Tracking progress per department/course/instructor
- Generating live dashboards and exportable summaries
- Hosting an "AI" assistant to auto-summarize trends 
- Managing assessment plans and course reviews

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```
Data is saved to `data/*.csv` for easy backup/versioning.

## Pages
- **Dashboard**: metrics, trends, data table
- **Submit FCAR**: form to add/edit FCAR entries
- **Assessment Plans**: list plans and link evidence
- **Program Review**: per-course snapshot & export
- **AI Assistant**: text area that summarizes FCAR trends from current data
- **Admin**: import/export CSVs, manage departments/courses/users

## Notes
- This starter keeps data in CSV files. Swap `utils.py` with a DB (SQLite/Postgres) later.

