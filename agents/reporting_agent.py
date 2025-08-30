from crewai import Agent
from langchain_mistralai.chat_models import ChatMistralAI
import os

class ReportingAgent:
    @staticmethod
    def agent():
        llm = ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral/mistral-large-latest"
        )
        return Agent(
            role="HR Reporting Agent",
            goal="Generate recruitment reports and respond to HR queries.",
            backstory="An efficient summarizer and report generator for HR workflows.",
            llm=llm,
            allow_delegation=False
        )
