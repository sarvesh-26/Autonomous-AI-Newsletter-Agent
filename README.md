# AI News Digest

An autonomous newsletter generation system built using LangGraph, Gemini, and Streamlit.

The application researches the latest AI-related news, extracts key insights, summarizes top articles, generates a newsletter in Markdown and HTML formats, performs a self-review step, and provides downloadable output through a simple web interface.

## Features

- Automated AI news research
- Article extraction and summarization
- Weekly newsletter generation
- Markdown and HTML output
- Self-review and quality check
- Fully Autonomous mode
- Human-in-the-Loop mode
- Streamlit-based user interface

## Tech Stack

- Python
- LangGraph
- LangChain
- Google Gemini
- Streamlit
- BeautifulSoup
- Feedparser

## Project Structure

```text
newsletter-agent/
│
├── app.py
├── newsletter_agent.py
├── requirements.txt
├── .env.example
└── output/
```

## Setup

### Clone or Extract Project

Open the project folder in VS Code.

### Create Virtual Environment

```bash
py -m venv venv
```

### Activate Virtual Environment

PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

## Running the Application

### Start Streamlit UI

```bash
streamlit run app.py
```

or

```bash
py -m streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

### Run via Command Line

```bash
python newsletter_agent.py
```

## Workflow

1. Receive newsletter generation goal
2. Create execution plan
3. Research latest AI news
4. Extract article content
5. Summarize selected articles
6. Generate newsletter
7. Review generated content
8. Save output files

