LAU FCAR System – AI-Powered Assessment Assistant

A Streamlit-based web application developed to streamline the Faculty Course Assessment Report (FCAR) workflow at LAU.
The system allows faculty to upload FCARs, extract structured data using OpenAI, store them in MongoDB, and access a powerful AI Assistant that summarizes, analyzes, and supports accreditation-related tasks.

 Features
1. FCAR Upload & Auto-Extraction
Upload Word/PDF FCAR documents.
AI extracts:
  Course information
  CLOs/SLOs mappings
  Assessments and performance levels
  Recommendations

2. Dynamic Editing Interface
Auto-filled forms for quick review.
Editable tables with dynamic rows.
Percentage indicators and color-coded SLO achievement levels.

3. AI Chat Assistant
Integrated with OpenAI.
Supports:
Summaries of FCARs
Accreditation explanations
Data-driven insights

4. MongoDB Integration
Automatically stores FCAR data.
Can be extended for dashboards and KPIs.

5. Secure Login 
Can restrict access to specific users via st.secrets credentials.

Project Structure
├── pages/
│   ├── 01_Upload_FCAR.py
│   ├── 02_View_FCARs.py
│   ├── 03_Dashboard.py
│   ├── 05_AI_Assistant.py
├── utils.py
├── app.py
├── requirements.txt
├── README.md
└── data/

Installation
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/lau-fcar-system.git
cd lau-fcar-system

2. Create a virtual environment
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows

3. Install dependencies
pip install -r requirements.txt
 Configure Secrets
Create a .streamlit/secrets.toml file:
OPENAI_API_KEY = "your_openai_api_key"
[mongo]
uri = "your_mongodb_connection_string"
[auth]
username = "admin"
password = "1234"


In the app, access secrets with:
API_KEY = st.secrets["OPENAI_API_KEY"]
mongo_uri = st.secrets["mongo"]["uri"]

▶️ Run the App
streamlit run app.py


AI-Powered FCAR Workflow
Faculty upload FCAR
System extracts data using structured OpenAI prompts
Faculty review/edit
Data saved to MongoDB
AI assistant provides insights on demand

Future Work
Multi-department user roles
Advanced accreditation dashboards
SLA-based tracking
Full pipeline automation
Additional AI analytics (trends, weaknesses, recommendations)
