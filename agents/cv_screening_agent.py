from crewai import Agent
from langchain_mistralai.chat_models import ChatMistralAI
import os

class CVScreeningAgent:
    @staticmethod
    def agent():
        llm = ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral/mistral-large-latest"
        )
        return Agent(
            role="CV Screener",
            goal="Screen and score CVs out of 10 according to job suitability.",
            backstory="Highly skilled in analyzing CVs swiftly and accurately.",
            llm=llm,
            allow_delegation=False
        )
