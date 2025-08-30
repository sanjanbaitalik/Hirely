from crewai import Agent
from langchain_mistralai.chat_models import ChatMistralAI
import os

class CommunicationAgent:
    @staticmethod
    def agent():
        llm = ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral/mistral-large-latest"
        )
        return Agent(
            role="Communication Expert",
            goal="Effectively contact candidates through email or WhatsApp.",
            backstory="Specialist in professional communication with candidates.",
            llm=llm,
            allow_delegation=False
        )
