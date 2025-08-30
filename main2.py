import os
import time
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from utils.db import get_chroma_client
from tasks.hr_tasks import scrape_and_store_profiles
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from typing import Any, Optional, Dict  # Add type annotations

# Load environment
load_dotenv()

# Setup ChromaDB
client = get_chroma_client()
collection = client.get_or_create_collection("candidate_profiles")

# Mistral setup
llm = ChatMistralAI(api_key=os.getenv('MISTRAL_API_KEY'))
embedding_fn = MistralAIEmbeddings(api_key=os.getenv('MISTRAL_API_KEY'))
vectorstore = Chroma(collection_name="candidate_profiles",
                     persist_directory="./data/chromadb_data",
                     embedding_function=embedding_fn)

def main():
    # Define tools using CrewAI's BaseTool with proper type annotations
    class ScrapeProfilesTool(BaseTool):
        name: str = "scrape_profiles"
        description: str = "Scrapes and stores candidate profiles in the database."
        
        def _run(self, *args: Any, **kwargs: Any) -> str:
            scrape_and_store_profiles(collection, embedding_fn)
            return "Profiles successfully scraped and stored in the database."

    class ScreenCandidatesTool(BaseTool):
        name: str = "screen_candidates"
        description: str = "Screens candidate CVs based on a query for Python skills."
        
        def _run(self, query: str = "Skills: Python", k: int = 5) -> str:
            docs = vectorstore.similarity_search(query, k=k)
            scored_candidates = []
            for i, doc in enumerate(docs):
                try:
                    # Add delay between API calls to avoid rate limiting
                    if i > 0:
                        print(f"Waiting 2 seconds before processing next candidate...")
                        time.sleep(2)
                        
                    scoring_prompt = f"Score candidate (1-10) based on: {query}\n{doc.page_content}"
                    score = llm.invoke(scoring_prompt)
                    scored_candidates.append({"profile": doc.metadata, "score": score.content})
                    print(f"Processed candidate {i+1}/{len(docs)}: {doc.metadata['name']}")
                    
                except Exception as e:
                    print(f"Error processing candidate {i+1}: {str(e)}")
                    if "rate limit" in str(e).lower():
                        print("Rate limit hit. Waiting 10 seconds...")
                        time.sleep(10)
                        try:
                            scoring_prompt = f"Score candidate (1-10) based on: {query}\n{doc.page_content}"
                            score = llm.invoke(scoring_prompt)
                            scored_candidates.append({"profile": doc.metadata, "score": score.content})
                            print(f"Successfully processed candidate {i+1} after waiting")
                        except Exception as e2:
                            print(f"Still failed after waiting: {str(e2)}")
                            scored_candidates.append({"profile": doc.metadata, "score": "N/A (API Error)"})
            
            result = "\n".join([f"{c['profile']['name']}: {c['score']}" for c in scored_candidates])
            return result

    class GenerateReportTool(BaseTool):
        name: str = "generate_report"
        description: str = "Generate an HR report based on candidate scores."
        
        def _run(self, candidate_scores: str) -> str:
            report_prompt = "Generate a comprehensive report on these candidates:\n" + candidate_scores
            report = llm.invoke(report_prompt)
            return report.content

    # Instantiate the tools
    scrape_profiles = ScrapeProfilesTool()
    screen_candidates = ScreenCandidatesTool()
    generate_report = GenerateReportTool()

    # Define agents with CrewAI
    profile_scraper = Agent(
        role="Profile Scraper",
        goal="Collect and store candidate profile information efficiently",
        backstory="You are an expert at gathering candidate data and organizing it for analysis.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[scrape_profiles]
    )
    
    cv_screener = Agent(
        role="CV Screener",
        goal="Evaluate candidates based on their skills and qualifications",
        backstory="You are a hiring manager with an eye for talent and technical skills.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[screen_candidates]
    )
    
    hr_reporter = Agent(
        role="HR Reporter",
        goal="Generate comprehensive reports on candidate evaluations",
        backstory="You summarize candidate information into clear, actionable reports for hiring teams.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[generate_report]
    )
    
    # Define tasks with CrewAI
    scraping_task = Task(
        description="Scrape and store candidate profiles",
        agent=profile_scraper,
        expected_output="Confirmation that profiles are stored successfully",
        context=["Use the scrape_profiles tool to gather and store candidate profiles."]
    )
    
    screening_task = Task(
        description="Screen candidate CVs for Python skills",
        agent=cv_screener,
        expected_output="List of candidates with scores",
        context=["Use the screen_candidates tool to evaluate candidates based on Python skills.",
                "Score each candidate on a scale of 1-10."]
    )
    
    reporting_task = Task(
        description="Generate a summary report of top candidates",
        agent=hr_reporter,
        expected_output="A comprehensive HR report summarizing top candidates",
        context=["Use the generate_report tool to create a detailed report based on the candidate scores from the CV screening."]
    )
    
    # Create the crew
    hr_crew = Crew(
        agents=[profile_scraper, cv_screener, hr_reporter],
        tasks=[scraping_task, screening_task, reporting_task],
        verbose=2,
        process=Process.sequential  # Execute tasks in sequence
    )
    
    # Execute the crew
    result = hr_crew.kickoff()
    print("\n\n=== FINAL RESULT ===")
    print(result)

if __name__ == "__main__":
    main()