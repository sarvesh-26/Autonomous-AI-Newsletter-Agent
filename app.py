import streamlit as st
from newsletter_agent import run_newsletter_agent


st.set_page_config(
    page_title="Newsletter Agent",
    page_icon="^-^",
    layout="wide"
)


st.title("Autonomous AI Newsletter Agent")

st.write(
    "This app creates a weekly newsletter on the latest AI agent news using an autonomous multi-step agent."
)

goal = st.text_area(
    "Enter your goal",
    value="Create a weekly newsletter on latest AI agent news and send it to our subscribers.",
    height=100
)

mode_option = st.radio(
    "Choose Agent Mode",
    ["Fully Autonomous", "Human-in-the-Loop"]
)

mode = "autonomous" if mode_option == "Fully Autonomous" else "human-ui"

run_button = st.button("Run Newsletter Agent")

if run_button:
    with st.spinner("Agent is working..."):
        result = run_newsletter_agent(goal, mode=mode)

    st.success("Newsletter generated successfully!")

    st.subheader("Agent Execution Logs")
    for log in result["logs"]:
        st.write("✅", log)

    st.subheader("Email Subject")
    st.code(result["final_subject"])

    st.subheader("Self-Reflection / Critique")
    st.write(result["critique"])

    st.subheader("Newsletter Preview")
    st.markdown(result["newsletter_markdown"])

    if mode_option == "Human-in-the-Loop":
        st.subheader("Human Review")
        edited_newsletter = st.text_area(
            "Edit newsletter before simulated sending",
            value=result["newsletter_markdown"],
            height=400
        )

        if st.button("Approve Final Newsletter"):
            st.success("Human approved newsletter. Sending simulated.")
            st.download_button(
                label="Download Edited Newsletter",
                data=edited_newsletter,
                file_name="weekly_ai_agent_newsletter.md",
                mime="text/markdown"
            )

    st.subheader("Download Output")

    st.download_button(
        label="Download HTML Newsletter",
        data=result["newsletter_html"],
        file_name="weekly_ai_agent_newsletter.html",
        mime="text/html"
    )

    st.download_button(
        label="Download Markdown Newsletter",
        data=result["newsletter_markdown"],
        file_name="weekly_ai_agent_newsletter.md",
        mime="text/markdown"
    )
