from crewai import Agent
from langchain_mistralai.chat_models import ChatMistralAI
import os

class InterviewSchedulerAgent:
    @staticmethod
    def agent():
        llm = ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral/mistral-large-latest"
        )
        return Agent(
            role="Interview Scheduler",
            goal="Coordinate interviews based on candidate availability.",
            backstory="A meticulous scheduler adept at managing calendars.",
            llm=llm,
            allow_delegation=False
        )
