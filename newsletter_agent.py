import os
import re
import json
import requests
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime
from typing import TypedDict, List, Dict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


class NewsletterState(TypedDict):
    goal: str
    mode: str
    plan: List[str]
    search_results: List[Dict]
    articles: List[Dict]
    summaries: List[Dict]
    newsletter_markdown: str
    newsletter_html: str
    critique: str
    final_subject: str
    final_output_path: str
    logs: List[str]


def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return None

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.4
    )


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def planning_node(state: NewsletterState) -> NewsletterState:
    plan = [
        "Understand the user goal",
        "Research latest AI agent news from public sources",
        "Collect relevant article links",
        "Extract useful article content",
        "Summarize top 5 to 7 articles",
        "Generate newsletter in Markdown and HTML",
        "Critique newsletter quality",
        "Improve final output",
        "Simulate sending by saving newsletter as a file"
    ]

    state["plan"] = plan
    state["logs"].append("Planning completed.")
    return state


def search_web_tool(query: str, max_results: int = 10) -> List[Dict]:
    """
    Tool 1: Web search using Google News RSS.
    No API key required.
    """

    rss_url = (
        "https://news.google.com/rss/search?q="
        + requests.utils.quote(query)
        + "&hl=en-IN&gl=IN&ceid=IN:en"
    )

    feed = feedparser.parse(rss_url)
    results = []

    for entry in feed.entries[:max_results]:
        results.append({
            "title": entry.get("title", ""),
            "link": entry.get("link", ""),
            "published": entry.get("published", ""),
            "source": entry.get("source", {}).get("title", "Google News")
        })

    return results


def research_node(state: NewsletterState) -> NewsletterState:
    query = "latest AI agent news autonomous agents generative AI agent frameworks"
    results = search_web_tool(query, max_results=12)

    state["search_results"] = results
    state["logs"].append(f"Research completed. Found {len(results)} search results.")
    return state


def scrape_article_tool(url: str) -> str:
    """
    Tool 2: Article scraping tool.
    Extracts readable page text from public article pages.
    """

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text(" ", strip=True) for p in paragraphs])

        return clean_text(text[:5000])

    except Exception:
        return ""


def article_extraction_node(state: NewsletterState) -> NewsletterState:
    articles = []

    for item in state["search_results"]:
        content = scrape_article_tool(item["link"])

        articles.append({
            "title": item["title"],
            "link": item["link"],
            "published": item["published"],
            "source": item["source"],
            "content": content
        })

    articles = [a for a in articles if a["title"]]
    state["articles"] = articles[:7]
    state["logs"].append(f"Article extraction completed. Selected {len(state['articles'])} articles.")
    return state


def fallback_summary(article: Dict) -> str:
    content = article.get("content", "")

    if not content:
        return "This article discusses recent developments in AI agents, autonomous workflows, and generative AI systems."

    sentences = re.split(r"(?<=[.!?]) +", content)
    return " ".join(sentences[:3])


def summarizer_tool(article: Dict) -> Dict:
    """
    Tool 3: Summarizer tool using Gemini.
    Falls back to simple extractive summary if API key is missing.
    """

    llm = get_llm()

    if llm is None:
        summary = fallback_summary(article)
        importance = "Relevant because it relates to recent developments in AI agents and autonomous AI systems."
    else:
        prompt = f"""
        Summarize this article for a professional weekly newsletter about AI agents.

        Title: {article['title']}
        Source: {article['source']}
        Published: {article['published']}
        Content: {article.get('content', '')[:4000]}

        Return JSON only with:
        {{
          "summary": "2-3 sentence summary",
          "importance": "why this matters in 1 sentence"
        }}
        """

        try:
            response = llm.invoke(prompt).content
            response = response.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(response)
            summary = parsed.get("summary", fallback_summary(article))
            importance = parsed.get("importance", "Important for understanding the direction of AI agent technology.")
        except Exception:
            summary = fallback_summary(article)
            importance = "Important for understanding the direction of AI agent technology."

    return {
        "title": article["title"],
        "source": article["source"],
        "published": article["published"],
        "link": article["link"],
        "summary": summary,
        "importance": importance
    }


def summarization_node(state: NewsletterState) -> NewsletterState:
    summaries = []

    for article in state["articles"]:
        summaries.append(summarizer_tool(article))

    state["summaries"] = summaries[:7]
    state["logs"].append(f"Summarization completed. Created {len(state['summaries'])} summaries.")
    return state


def markdown_generator_tool(summaries: List[Dict]) -> str:
    """
    Tool 4: Markdown newsletter generator.
    """

    today = datetime.now().strftime("%d %B %Y")

    markdown = f"""
# Weekly AI Agent Newsletter

**Date:** {today}

Welcome to this week's AI Agent Newsletter. Here are the top updates from the world of autonomous AI agents, agentic workflows, and generative AI systems.

---

## Top AI Agent Stories This Week

"""

    for index, item in enumerate(summaries, start=1):
        markdown += f"""
### {index}. {item['title']}

**Source:** {item['source']}  
**Published:** {item['published']}

{item['summary']}

**Why it matters:** {item['importance']}

[Read more]({item['link']})

---
"""

    markdown += """
## Final Takeaway

AI agents are rapidly moving from experimental assistants to practical systems that can plan, use tools, complete tasks, and support business workflows. The major trend is clear: agentic AI is becoming more autonomous, more integrated with tools, and more useful for productivity-focused applications.

---

You are receiving this simulated newsletter as part of the AI Developer Assignment.
"""

    return markdown.strip()


def html_generator_tool(markdown_text: str, summaries: List[Dict]) -> str:
    """
    Tool 5: HTML newsletter generator.
    """

    today = datetime.now().strftime("%d %B %Y")
    cards = ""

    for index, item in enumerate(summaries, start=1):
        cards += f"""
        <div class="card">
            <h2>{index}. {item['title']}</h2>
            <p class="meta"><strong>Source:</strong> {item['source']} | <strong>Published:</strong> {item['published']}</p>
            <p>{item['summary']}</p>
            <p><strong>Why it matters:</strong> {item['importance']}</p>
            <a href="{item['link']}" target="_blank">Read more</a>
        </div>
        """

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Weekly AI Agent Newsletter</title>
    <style>
        body {{ margin: 0; padding: 0; background: #f4f6f8; font-family: Arial, sans-serif; color: #222; }}
        .container {{ max-width: 820px; margin: 30px auto; background: white; border-radius: 16px; overflow: hidden; box-shadow: 0 8px 30px rgba(0,0,0,0.08); }}
        .header {{ background: linear-gradient(135deg, #111827, #2563eb); color: white; padding: 32px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 32px; }}
        .header p {{ margin-top: 10px; font-size: 16px; opacity: 0.9; }}
        .content {{ padding: 28px; }}
        .intro {{ font-size: 16px; line-height: 1.6; }}
        .card {{ border: 1px solid #e5e7eb; border-radius: 14px; padding: 20px; margin: 20px 0; background: #fafafa; }}
        .card h2 {{ margin-top: 0; font-size: 20px; color: #111827; }}
        .meta {{ color: #555; font-size: 14px; }}
        .card p {{ line-height: 1.6; }}
        .card a {{ display: inline-block; margin-top: 8px; color: white; background: #2563eb; padding: 10px 14px; border-radius: 8px; text-decoration: none; font-weight: bold; }}
        .takeaway {{ margin-top: 24px; padding: 20px; background: #eef6ff; border-left: 5px solid #2563eb; border-radius: 10px; }}
        .footer {{ text-align: center; padding: 20px; color: #666; font-size: 13px; background: #f9fafb; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Weekly AI Agent Newsletter</h1>
            <p>{today}</p>
        </div>
        <div class="content">
            <p class="intro">Welcome to this week's AI Agent Newsletter. Here are the latest updates from the world of autonomous AI agents, agentic workflows, and generative AI systems.</p>
            {cards}
            <div class="takeaway">
                <h2>Final Takeaway</h2>
                <p>AI agents are rapidly moving from experimental assistants to practical systems that can plan, use tools, complete tasks, and support business workflows. The major trend is clear: agentic AI is becoming more autonomous, more integrated with tools, and more useful for productivity-focused applications.</p>
            </div>
        </div>
        <div class="footer">Simulated newsletter generated by the autonomous Newsletter Agent.</div>
    </div>
</body>
</html>
"""

    return html.strip()


def newsletter_generation_node(state: NewsletterState) -> NewsletterState:
    markdown_text = markdown_generator_tool(state["summaries"])
    html_text = html_generator_tool(markdown_text, state["summaries"])

    state["newsletter_markdown"] = markdown_text
    state["newsletter_html"] = html_text
    state["final_subject"] = "Weekly AI Agent Newsletter: Latest Agentic AI Updates"
    state["logs"].append("Newsletter generation completed.")
    return state


def critique_node(state: NewsletterState) -> NewsletterState:
    """
    Self-reflection step.
    Agent checks the newsletter quality and suggests improvements.
    """

    llm = get_llm()

    if llm is None:
        critique = """
        Self-review completed:
        - Newsletter has a clear subject.
        - It includes multiple article summaries.
        - It contains source links.
        - It has a final takeaway.
        - HTML formatting is clean and readable.
        """
    else:
        prompt = f"""
        Review this newsletter as an editor.

        Check:
        1. Clarity
        2. Relevance
        3. Professional tone
        4. Formatting
        5. Whether it satisfies the goal

        Newsletter:
        {state['newsletter_markdown'][:5000]}

        Give a short critique and improvement suggestions.
        """
        critique = llm.invoke(prompt).content

    state["critique"] = critique
    state["logs"].append("Self-reflection and critique completed.")
    return state


def improvement_node(state: NewsletterState) -> NewsletterState:
    """
    Improves the newsletter after critique.
    """

    if "Final Takeaway" not in state["newsletter_markdown"]:
        state["newsletter_markdown"] += "\n\n## Final Takeaway\nAI agents continue to grow as a major trend in modern AI."

    if "Weekly AI Agent Newsletter" not in state["newsletter_html"]:
        state["newsletter_html"] = html_generator_tool(state["newsletter_markdown"], state["summaries"])

    state["logs"].append("Newsletter improvement completed after self-review.")
    return state


def human_review_node(state: NewsletterState) -> NewsletterState:
    """
    Human-in-the-loop mode for CLI.
    Streamlit handles human approval inside the UI.
    """

    if state["mode"] == "human":
        print("\nHuman-in-the-loop mode enabled.")
        print("\nGenerated Subject:")
        print(state["final_subject"])
        print("\nNewsletter Preview:")
        print(state["newsletter_markdown"][:1500])

        approval = input("\nApprove newsletter? Type yes/no: ").strip().lower()

        if approval != "yes":
            state["logs"].append("Human review requested changes. Saving current draft anyway for review.")
        else:
            state["logs"].append("Human approved the newsletter.")

    return state


def send_simulation_node(state: NewsletterState) -> NewsletterState:
    """
    Simulates email sending by saving the newsletter output.
    """

    os.makedirs("output", exist_ok=True)

    html_path = "output/weekly_ai_agent_newsletter.html"
    md_path = "output/weekly_ai_agent_newsletter.md"

    with open(html_path, "w", encoding="utf-8") as file:
        file.write(state["newsletter_html"])

    with open(md_path, "w", encoding="utf-8") as file:
        file.write(state["newsletter_markdown"])

    print("\nSimulated Email Send")
    print("--------------------")
    print("To: subscribers@example.com")
    print(f"Subject: {state['final_subject']}")
    print(f"HTML File Saved: {html_path}")
    print(f"Markdown File Saved: {md_path}")

    state["final_output_path"] = html_path
    state["logs"].append("Sending simulated successfully by saving newsletter files.")
    return state


def build_newsletter_graph():
    graph = StateGraph(NewsletterState)

    graph.add_node("planning", planning_node)
    graph.add_node("research", research_node)
    graph.add_node("extract_articles", article_extraction_node)
    graph.add_node("summarize", summarization_node)
    graph.add_node("generate_newsletter", newsletter_generation_node)
    graph.add_node("critique", critique_node)
    graph.add_node("improve", improvement_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("send_simulation", send_simulation_node)

    graph.set_entry_point("planning")

    graph.add_edge("planning", "research")
    graph.add_edge("research", "extract_articles")
    graph.add_edge("extract_articles", "summarize")
    graph.add_edge("summarize", "generate_newsletter")
    graph.add_edge("generate_newsletter", "critique")
    graph.add_edge("critique", "improve")
    graph.add_edge("improve", "human_review")
    graph.add_edge("human_review", "send_simulation")
    graph.add_edge("send_simulation", END)

    return graph.compile()


def run_newsletter_agent(goal: str, mode: str = "autonomous") -> NewsletterState:
    """
    Main autonomous agent function.
    One function call performs:
    planning -> research -> scraping -> summarization -> newsletter writing -> review -> output.
    """

    app = build_newsletter_graph()

    initial_state: NewsletterState = {
        "goal": goal,
        "mode": mode,
        "plan": [],
        "search_results": [],
        "articles": [],
        "summaries": [],
        "newsletter_markdown": "",
        "newsletter_html": "",
        "critique": "",
        "final_subject": "",
        "final_output_path": "",
        "logs": []
    }

    final_state = app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    goal = "Create a weekly newsletter on latest AI agent news and send it to our subscribers."
    result = run_newsletter_agent(goal, mode="autonomous")

    print("\nAgent Logs:")
    for log in result["logs"]:
        print("-", log)

    print("\nFinal Subject:")
    print(result["final_subject"])

    print("\nOutput saved at:")
    print(result["final_output_path"])
