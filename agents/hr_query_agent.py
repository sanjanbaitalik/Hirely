from crewai import Agent
from langchain_mistralai.chat_models import ChatMistralAI
import os

class HRQueryAgent:
    @staticmethod
    def agent():
        llm = ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral/mistral-large-latest"
        )
        return Agent(
            role="HR Query Handler",
            goal="Interpret HR's job role queries to instruct other agents.",
            backstory=(
                "You are an intelligent HR assistant capable of interpreting HR's natural language queries about recruitment requirements. "
                "You clearly identify the requested job role and skills and instruct other agents accordingly."
            ),
            llm=llm,
            allow_delegation=True
        )