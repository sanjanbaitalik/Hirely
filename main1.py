import os
import time
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain.vectorstores import Chroma
from utils.db import get_chroma_client
from tasks.hr_tasks import scrape_and_store_profiles

from agents.profile_scraper_agent import get_profile_scraper_agent
from agents.cv_screening_agent import get_cv_screening_agent
from agents.reporting_agent import get_reporting_agent
from agents.communication_agent import get_communication_agent  # ✅ New import
from agents.interview_scheduler_agent import get_interview_scheduler_agent  # ✅ New import

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

# Initialize Agents
profile_scraper_agent = get_profile_scraper_agent(llm)
cv_screening_agent = get_cv_screening_agent(llm)
communication_agent = get_communication_agent(llm)             # ✅ Initialize
interview_scheduler_agent = get_interview_scheduler_agent(llm) # ✅ Initialize
reporting_agent = get_reporting_agent(llm)

def main():
    # Step 1: Scrape Profiles
    scrape_and_store_profiles(collection, embedding_fn)

    # Step 2: CV Screening
    query = "Skills: Python"
    docs = vectorstore.similarity_search(query, k=5)

    scored_candidates = []
    for i, doc in enumerate(docs):
        try:
            if i > 0:
                print(f"Waiting 2 seconds before processing next candidate...")
                time.sleep(2)

            scoring_prompt = f"Score candidate (1-10) based on: {query}\n{doc.page_content}"
            score = llm.invoke(scoring_prompt)
            scored_candidates.append({"profile": doc.metadata, "score": score.content})
            print(f"Processed candidate {i+1}/{len(docs)}: {doc.metadata['name']}")

        except Exception as e:
            print(f"Error processing candidate {i+1}: {str(e)}")

    # Display screened candidates
    print("\n✅ Screened Candidates:")
    for candidate in scored_candidates:
        print(f"- {candidate['profile']['name']}: {candidate['score']}")

    # Step 3: Communication via Telegram or Email
    for candidate in scored_candidates:
        print(f"\n📩 Communicating with candidate: {candidate['profile']['name']}")
        # Example: Trigger agent (Pseudo-logic)
        communication_prompt = f"Send interview invitation to {candidate['profile']['name']} via preferred method."
        response = communication_agent.llm.invoke(communication_prompt)
        print(f"Communication status: {response.content}")
        time.sleep(1)  # Avoid rate limiting

    # Step 4: Schedule Interview via Outlook Calendar
    for candidate in scored_candidates:
        print(f"\n📅 Scheduling interview with: {candidate['profile']['name']}")
        scheduling_prompt = f"Schedule an interview with {candidate['profile']['name']} based on availability."
        response = interview_scheduler_agent.llm.invoke(scheduling_prompt)
        print(f"Scheduling status: {response.content}")
        time.sleep(1)  # Avoid rate limiting

    # Step 5: Generate Report
    print("\nWaiting 5 seconds before generating report...")
    time.sleep(5)

    if scored_candidates:
        try:
            report_prompt = "Generate a summary report of candidates:\n" + \
                        "\n".join([f"{c['profile']['name']} - Score: {c['score']}" for c in scored_candidates])
            report = reporting_agent.llm.invoke(report_prompt)
            print("\n📑 HR Report:\n", report.content)
        except Exception as e:
            print(f"\n❌ Error generating report: {str(e)}")

if __name__ == "__main__":
    main()
