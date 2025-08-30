from crewai import Agent
from langchain_mistralai.chat_models import ChatMistralAI
import os
import logging
from typing import List, Dict, Any, Optional
from ..utils.linkedin_scraper import LinkedInScraper
from ..utils.vector_store import ProfileVectorStore
from ..utils.rag_system import ProfileRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfileScraperAgent:
    def __init__(self):
        """Initialize the ProfileScraperAgent with required components"""
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.rapidapi_key = os.getenv("RAPIDAPI_KEY")
        
        # Initialize components if API keys are available
        if self.rapidapi_key:
            self.scraper = LinkedInScraper(api_key=self.rapidapi_key)
        else:
            logger.warning("RAPIDAPI_KEY not found. LinkedIn scraper will not work.")
            self.scraper = None
            
        # Initialize vector store
        self.vector_store = ProfileVectorStore(
            collection_name="linkedin_profiles",
            persist_directory="./data/chroma_db"
        )
        
        # Initialize RAG system if Mistral API key is available
        if self.api_key:
            self.rag = ProfileRAG(
                vector_store=self.vector_store,
                api_key=self.api_key
            )
        else:
            logger.warning("MISTRAL_API_KEY not found. RAG system will not work.")
            self.rag = None
            
    @staticmethod
    def agent(job_role=None):
        """Create a CrewAI agent for profile scraping"""
        llm = ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral/mistral-large-latest"
        )
        
        return Agent(
            role="Profile Scraper",
            goal=f"Find profiles matching the '{job_role}' job role",
            backstory="Expert in finding and collecting candidate profiles for technical positions.",
            llm=llm,
            allow_delegation=False
        )
        
    def collect_profiles(self, job_role: str, location: Optional[str] = None, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Collect real LinkedIn profiles for a job role
        
        Args:
            job_role: The job role to search for
            location: Optional location filter
            num_results: Number of profiles to collect
            
        Returns:
            List of collected profiles
        """
        if not self.scraper:
            logger.error("LinkedIn scraper not initialized. Cannot collect profiles.")
            return []
            
        # Find profile usernames
        usernames = self.scraper.find_profiles(job_role, location, num_results)
        
        # Collect profile details
        profiles = []
        for username in usernames:
            raw_profile = self.scraper.get_profile_details(username)
            if raw_profile:
                processed_profile = self.scraper.process_profile(raw_profile)
                profiles.append(processed_profile)
                
                # Add to vector store
                self.vector_store.add_profile(processed_profile)
                
        return profiles
        
    def analyze_candidates(self, job_role: str, job_description: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Analyze candidates using RAG
        
        Args:
            job_role: The job role to analyze
            job_description: Detailed job description
            n_results: Number of profiles to analyze
            
        Returns:
            Analysis results
        """
        if not self.rag:
            logger.error("RAG system not initialized. Cannot analyze candidates.")
            return {"error": "RAG system not initialized"}
            
        return self.rag.analyze_candidates(job_role, job_description, n_results)