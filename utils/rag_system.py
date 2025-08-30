from typing import List, Dict, Any, Optional
import logging
from .vector_store import ProfileVectorStore
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

logger = logging.getLogger(__name__)

class ProfileRAG:
    def __init__(self, vector_store: ProfileVectorStore, api_key: str, model: str = "mistral/mistral-large-latest"):
        """
        Initialize the RAG system for profile analysis
        
        Args:
            vector_store: Vector store containing profile data
            api_key: Mistral API key
            model: Mistral model to use
        """
        self.vector_store = vector_store
        self.llm = ChatMistralAI(api_key=api_key, model=model)
        
    def format_docs(self, docs: List[Dict[str, Any]]) -> str:
        """Format documents for context insertion"""
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            profile_doc = doc.get("document", "")
            metadata = doc.get("metadata", {})
            
            formatted = f"PROFILE {i}:\n"
            formatted += f"Name: {metadata.get('name', 'Unknown')}\n"
            formatted += f"Title: {metadata.get('title', 'Unknown')}\n"
            formatted += f"LinkedIn: {metadata.get('url', 'Unknown')}\n\n"
            formatted += f"{profile_doc}\n\n"
            formatted_docs.append(formatted)
            
        return "\n".join(formatted_docs)
        
    def analyze_candidates(self, job_role: str, job_description: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Analyze candidates for a job role using RAG
        
        Args:
            job_role: The job role to analyze
            job_description: Detailed job description
            n_results: Number of profiles to analyze
            
        Returns:
            Analysis results
        """
        # Construct search query
        search_query = f"{job_role} with skills matching: {job_description}"
        
        # Retrieve relevant profiles
        profiles = self.vector_store.search_profiles(search_query, n_results=n_results)
        if not profiles:
            return {"error": "No matching profiles found"}
            
        # Define RAG prompt
        template = """
        You are an expert HR talent analyst. You need to evaluate candidates for a job role using their LinkedIn profiles.

        JOB ROLE: {job_role}
        
        JOB DESCRIPTION:
        {job_description}
        
        CANDIDATE PROFILES:
        {formatted_docs}
        
        Analyze these candidates based on the job requirements and provide:
        
        1. Individual evaluation for each candidate (strengths, weaknesses, fit)
        2. Comparative ranking of candidates from best to worst fit
        3. Recommendation on whom to interview first
        
        Your analysis should be data-driven, focusing on relevant skills, experience, and qualifications.
        """
        
        # Create RAG chain
        prompt = ChatPromptTemplate.from_template(template)
        
        # Define the RAG chain
        rag_chain = (
            {"job_role": lambda x: x["job_role"],
             "job_description": lambda x: x["job_description"],
             "formatted_docs": lambda x: self.format_docs(x["docs"])}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Execute the chain
        try:
            result = rag_chain.invoke({
                "job_role": job_role,
                "job_description": job_description,
                "docs": profiles
            })
            
            return {
                "analysis": result,
                "profiles": profiles,
                "query": search_query
            }
        except Exception as e:
            logger.error(f"Error analyzing candidates: {e}")
            return {"error": f"Analysis failed: {str(e)}"}