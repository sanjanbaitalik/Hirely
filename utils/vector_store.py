import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ProfileVectorStore:
    def __init__(self, collection_name: str = "linkedin_profiles", persist_directory: Optional[str] = None):
        """
        Initialize the vector store for profile data
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database (None for in-memory)
        """
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        settings = Settings(persist_directory=persist_directory) if persist_directory else Settings()
        self.client = chromadb.Client(settings)
        
        # Initialize sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get or create collection
        if self.collection_name in [c.name for c in self.client.list_collections()]:
            self.collection = self.client.get_collection(name=self.collection_name)
        else:
            self.collection = self.client.create_collection(name=self.collection_name)
            
    def _create_profile_document(self, profile: Dict[str, Any]) -> str:
        """Create a text document from profile data"""
        doc = f"{profile.get('name', '')}\n{profile.get('title', '')}\n\n"
        
        # Add summary
        if profile.get('summary'):
            doc += f"Summary: {profile.get('summary')}\n\n"
            
        # Add experience
        doc += "Experience:\n"
        for exp in profile.get('experience', []):
            doc += f"- {exp.get('title')} at {exp.get('company')}, {exp.get('date_range')}\n"
            if exp.get('description'):
                doc += f"  {exp.get('description')}\n"
                
        # Add education
        doc += "\nEducation:\n"
        for edu in profile.get('education', []):
            doc += f"- {edu.get('degree')} in {edu.get('field')} from {edu.get('school')}, {edu.get('date_range')}\n"
            
        # Add skills
        doc += f"\nSkills: {', '.join(profile.get('skills', []))}"
        
        return doc
    
    def add_profile(self, profile: Dict[str, Any]) -> str:
        """
        Add a profile to the vector store
        
        Args:
            profile: Processed profile data dictionary
            
        Returns:
            ID of the added document
        """
        if not profile or not profile.get('username'):
            logger.warning("Cannot add invalid profile to vector store")
            return ""
            
        username = profile.get('username')
        document = self._create_profile_document(profile)
        
        # Generate embedding
        embedding = self.model.encode(document).tolist()
        
        # Create metadata
        metadata = {
            "username": username,
            "name": profile.get('name', ''),
            "title": profile.get('title', ''),
            "location": profile.get('location', ''),
            "url": profile.get('url', '')
        }
        
        # Create document ID
        doc_id = f"profile_{username}"
        
        # Add to collection
        try:
            self.collection.add(
                documents=[document],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[doc_id]
            )
            logger.info(f"Added profile {username} to vector store")
            return doc_id
        except Exception as e:
            logger.error(f"Error adding profile {username} to vector store: {e}")
            return ""
            
    def search_profiles(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for profiles matching a query
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of matching profile documents
        """
        query_embedding = self.model.encode(query).tolist()
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            if not results or 'documents' not in results or not results['documents']:
                logger.warning("No profiles found matching the query")
                return []
                
            matches = []
            for i, doc in enumerate(results['documents'][0]):
                matches.append({
                    "document": doc,
                    "metadata": results['metadatas'][0][i] if 'metadatas' in results else {},
                    "id": results['ids'][0][i] if 'ids' in results else f"result_{i}"
                })
            
            return matches
        except Exception as e:
            logger.error(f"Error searching profiles: {e}")
            return []