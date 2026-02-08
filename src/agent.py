"""
Agentic RAG System with Self-Reflection and Tool Use
"""
import logging
import json
from typing import List, Optional
from enum import Enum

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None

from src.retriever import RAGRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(Enum):
    """States in the agent reasoning loop"""
    INITIAL = "initial"
    RETRIEVING = "retrieving"
    ANALYZING = "analyzing"
    REFLECTING = "reflecting"
    GENERATING = "generating"
    COMPLETE = "complete"


class AgenticRAG:
    """
    Agentic RAG system with:
    - Tool calling (document retrieval)
    - Self-reflection (critic)
    - Answer generation
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        retriever: RAGRetriever,
        deployment_name: str = "gpt-4o-mini",
        api_version: str = "2025-01-01-preview",
        temperature: float = 0.3,
        max_iterations: int = 3
    ):
        """
        Initialize Agentic RAG with Azure OpenAI
        
        Args:
            api_key: Azure OpenAI API key
            endpoint: Azure OpenAI endpoint URL
            retriever: RAG retriever instance
            deployment_name: Chat deployment name
            api_version: API version
            temperature: Temperature for generation
            max_iterations: Maximum reflection iterations
        """
        if not AzureOpenAI:
            raise ImportError("openai is required. Install it with: pip install openai")
        
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        self.deployment_name = deployment_name
        self.retriever = retriever
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.conversation_history = []
        self.state = AgentState.INITIAL

    def _tool_retrieve_documents(self, query: str) -> str:
        """
        Tool: Retrieve relevant documents
        
        Args:
            query: Query to retrieve documents for
            
        Returns:
            Retrieved context as string
        """
        logger.info(f"Tool Call: Retrieving documents for query: {query}")
        self.state = AgentState.RETRIEVING
        context = self.retriever.retrieve_with_context(query)
        return context

    def _critic_evaluate_retrieved_docs(self, query: str, context: str) -> dict:
        """
        Critic: Self-evaluate retrieved documents
        
        Args:
            query: Original query
            context: Retrieved context
            
        Returns:
            Evaluation result with relevance score and feedback
        """
        logger.info("Critic: Evaluating retrieved documents...")
        self.state = AgentState.REFLECTING
        
        evaluation_prompt = f"""
        Evaluate the relevance of the following retrieved documents to the user query.
        
        User Query: {query}
        
        Retrieved Context:
        {context}
        
        Provide evaluation in JSON format with:
        - "relevance_score" (0-100): How relevant are the documents to the query?
        - "coverage" (0-100): How well do the documents cover the query topic?
        - "confidence" (0-100): How confident can we be in answering based on these docs?
        - "missing_aspects": List any aspects of the query not covered
        - "recommend_more_retrieval": Boolean, should we try different keywords?
        
        Return only valid JSON.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert document evaluator."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            # Parse JSON response
            response_text = response.choices[0].message.content.strip()
            evaluation = json.loads(response_text)
            return evaluation
        except Exception as e:
            logger.warning(f"Error in critic evaluation: {str(e)}")
            return {
                "relevance_score": 50,
                "coverage": 50,
                "confidence": 50,
                "missing_aspects": [],
                "recommend_more_retrieval": False
            }

    def _generate_answer(self, query: str, context: str) -> dict:
        """
        Generate final answer grounded in retrieved documents
        
        Args:
            query: User query
            context: Retrieved context with source information
            
        Returns:
            Dictionary with answer and sources
        """
        logger.info("Generating final answer...")
        self.state = AgentState.GENERATING
        
        # Extract source documents from context
        sources = []
        for line in context.split('\n'):
            if line.startswith('[Document'):
                # Extract filename from format: [Document 1: filename.txt]
                if ':' in line:
                    doc_info = line.split(':')[1].split(']')[0].strip()
                    sources.append(doc_info)
        
        answer_prompt = f"""
        You are a helpful QA assistant. Answer the following question based ONLY on the provided context.
        
        If the answer is not in the context, explicitly say so. Do not make up answers.
        At the end of your answer, provide citations like: [Source: filename.txt]
        
        User Question: {query}
        
        Context from Documents:
        {context}
        
        Provide a clear, well-structured answer. End with: "Sources: {', '.join(sources) if sources else 'No sources found'}"
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful QA assistant. Answer questions based only on provided context."},
                    {"role": "user", "content": answer_prompt}
                ],
                temperature=self.temperature
            )
            
            answer_text = response.choices[0].message.content
            return {
                "answer": answer_text,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": sources
            }

    def reason(self, query: str) -> dict:
        """
        Main reasoning loop with tool use and reflection
        
        Args:
            query: User query
            
        Returns:
            Result dictionary with answer and metadata
        """
        logger.info(f"Starting reasoning for query: {query}")
        self.state = AgentState.INITIAL
        
        result = {
            "query": query,
            "answer": None,
            "source_documents": [],
            "confidence": 0,
            "reasoning_steps": [],
            "iterations": 0
        }
        
        retrieved_context = None
        current_query = query
        
        for iteration in range(self.max_iterations):
            result["iterations"] += 1
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            # Step 1: Tool Call - Retrieve Documents
            retrieved_context = self._tool_retrieve_documents(current_query)
            
            result["reasoning_steps"].append({
                "step": "retrieve_documents",
                "iteration": iteration + 1,
                "docs_found": retrieved_context != "No relevant documents found in the knowledge base."
            })
            
            if retrieved_context == "No relevant documents found in the knowledge base.":
                logger.warning("No documents retrieved")
                break
            
            # Step 2: Self-Reflection / Critic
            evaluation = self._critic_evaluate_retrieved_docs(current_query, retrieved_context)
            
            result["reasoning_steps"].append({
                "step": "critic_evaluation",
                "iteration": iteration + 1,
                "evaluation": evaluation
            })
            
            result["confidence"] = evaluation.get("confidence", 50)
            
            # Check if we need more retrieval
            if (evaluation.get("recommend_more_retrieval", False) and 
                iteration < self.max_iterations - 1 and
                evaluation.get("relevance_score", 50) < 60):
                
                # Refine query for next iteration
                refinement_prompt = f"""
                The first retrieval attempt had low relevance. Suggest alternative keywords or 
                a rephrased query to better find relevant documents.
                
                Original Query: {query}
                Issues: {evaluation.get('missing_aspects', [])}
                
                Suggest only 1-2 alternative search terms, nothing else.
                """
                
                try:
                    response = self.client.chat.completions.create(
                        model=self.deployment_name,
                        messages=[
                            {"role": "system", "content": "You are a query refinement expert."},
                            {"role": "user", "content": refinement_prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=100
                    )
                    current_query = response.choices[0].message.content.strip()
                    logger.info(f"Refined query: {current_query}")
                    continue
                except:
                    pass
            
            break
        
        # Step 3: Generate Final Answer
        if retrieved_context and retrieved_context != "No relevant documents found in the knowledge base.":
            answer_result = self._generate_answer(query, retrieved_context)
            result["answer"] = answer_result["answer"]
            result["source_documents"] = answer_result.get("sources", [])
        else:
            result["answer"] = "I could not find relevant documents in the knowledge base to answer your question."
            result["source_documents"] = []
        
        self.state = AgentState.COMPLETE
        logger.info("Reasoning complete")
        
        return result

    def chat(self, user_input: str) -> str:
        """
        Chat interface for the agentic RAG
        
        Args:
            user_input: User question
            
        Returns:
            Agent response
        """
        result = self.reason(user_input)
        
        # Store in conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        self.conversation_history.append({
            "role": "assistant",
            "content": result["answer"],
            "metadata": {
                "confidence": result["confidence"],
                "iterations": result["iterations"]
            }
        })
        
        return result["answer"]

    def get_conversation_history(self) -> List[dict]:
        """Get conversation history"""
        return self.conversation_history

    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []
