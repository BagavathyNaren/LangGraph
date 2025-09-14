"""
Agentic RAG System Implementation
A retrieval agent that can decide when to retrieve context or respond directly.
"""

import getpass
import os
import warnings
warnings.filterwarnings("ignore")
from typing import Literal
from pydantic import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from langchain_core.messages import convert_to_messages
from dotenv import load_dotenv

load_dotenv()


def _set_env(key: str):
    """Set environment variable if not already set."""
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")


def setup_environment():
    """Setup API keys and environment."""
    _set_env("OPENAI_API_KEY")



def preprocess_documents():
    """Fetch and preprocess documents for RAG system."""
    print("Fetching documents...")

    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    docs = []

    # Ensure USER_AGENT is set
    headers = {"User-Agent": os.getenv("USER_AGENT", "agentic-rag-demo/1.0")}

    for url in urls:
        try:
            # First try with SSL verification
            loader = WebBaseLoader(url, requests_kwargs={"headers": headers})
            docs.extend(loader.load())
        except Exception as e:
            print(f"SSL/Request error for {url}: {e}")
            print("Retrying with SSL verify=False...")
            # Retry with SSL verification disabled (fallback)
            loader = WebBaseLoader(url, requests_kwargs={"headers": headers,"verify": False})
            docs.extend(loader.load())

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs)

    print(f"Created {len(doc_splits)} document chunks")
    return doc_splits

def create_retriever_tool_component(doc_splits):
    """Create a retriever tool from document splits."""
    print("Creating vector store and retriever...")
    
    try:
        # Create vector store with OpenAI embeddings
        vectorstore = InMemoryVectorStore.from_documents(
            documents=doc_splits, embedding=OpenAIEmbeddings()
        )
        retriever = vectorstore.as_retriever()
        
        # Create retriever tool
        retriever_tool = create_retriever_tool(
            retriever,
            "retrieve_blog_posts",
            "Search and return information about Lilian Weng blog posts.",
        )
        
        print("✓ Retriever tool created successfully")
        return retriever_tool
        
    except Exception as e:
        print(f"Error creating retriever tool: {e}")
        print("This might be due to missing OpenAI API key or network issues")
        raise


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class AgenticRAG:
    """Agentic RAG system that can decide when to retrieve or respond directly."""
    
    def __init__(self, retriever_tool):
        self.retriever_tool = retriever_tool
        self.response_model = init_chat_model("openai:gpt-4o-mini", temperature=0)
        self.grader_model = init_chat_model("openai:gpt-4o-mini", temperature=0)
        
        # Prompts
        self.GRADE_PROMPT = (
            "You are a grader assessing relevance of a retrieved document to a user question. \n "
            "Here is the retrieved document: \n\n {context} \n\n"
            "Here is the user question: {question} \n"
            "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
            "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
        )
        
        self.REWRITE_PROMPT = (
            "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
            "Here is the initial question:"
            "\n ------- \n"
            "{question}"
            "\n ------- \n"
            "Formulate an improved question:"
        )
        
        self.GENERATE_PROMPT = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n"
            "Question: {question} \n"
            "Context: {context}"
        )
    
    def generate_query_or_respond(self, state: MessagesState):
        """Call the model to generate a response or decide to retrieve."""
        response = (
            self.response_model
            .bind_tools([self.retriever_tool])
            .invoke(state["messages"])
        )
        return {"messages": [response]}
    
    def grade_documents(self, state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
        """Determine whether the retrieved documents are relevant to the question."""
        question = state["messages"][0].content
        context = state["messages"][-1].content
        
        prompt = self.GRADE_PROMPT.format(question=question, context=context)
        response = (
            self.grader_model
            .with_structured_output(GradeDocuments)
            .invoke([{"role": "user", "content": prompt}])
        )
        score = response.binary_score # type: ignore
        
        if score == "yes":
            print("✓ Documents are relevant - generating answer")
            return "generate_answer"
        else:
            print("✗ Documents not relevant - rewriting question")
            return "rewrite_question"
    
    def rewrite_question(self, state: MessagesState):
        """Rewrite the original user question for better retrieval."""
        messages = state["messages"]
        question = messages[0].content
        prompt = self.REWRITE_PROMPT.format(question=question)
        response = self.response_model.invoke([{"role": "user", "content": prompt}])
        print(f"Rewritten question: {response.content}")
        return {"messages": [{"role": "user", "content": response.content}]}
    
    def generate_answer(self, state: MessagesState):
        """Generate an answer based on the question and retrieved context."""
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = self.GENERATE_PROMPT.format(question=question, context=context)
        response = self.response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}
    
    def create_graph(self):
        """Create and compile the workflow graph."""
        workflow = StateGraph(MessagesState)
        
        # Define the nodes
        workflow.add_node("generate_query_or_respond", self.generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([self.retriever_tool]))
        workflow.add_node("rewrite_question", self.rewrite_question)
        workflow.add_node("generate_answer", self.generate_answer)
        
        # Define the edges
        workflow.add_edge(START, "generate_query_or_respond")
        
        # Decide whether to retrieve
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )
        
        # Grade documents and decide next step
        workflow.add_conditional_edges(
            "retrieve",
            self.grade_documents,
        )
        
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")
        
        # Compile the graph
        graph = workflow.compile()
        print("Workflow graph compiled successfully")
        return graph


def main():
    """Main function to run the Agentic RAG system."""
    print("Setting up Agentic RAG System...")
    
    # Setup environment
    setup_environment()
    
    # Preprocess documents
    doc_splits = preprocess_documents()
    
    # Create retriever tool
    retriever_tool = create_retriever_tool_component(doc_splits)
    
    # Test the retriever tool
    print("\nTesting retriever tool:")
    test_result = retriever_tool.invoke({"query": "types of reward hacking"})
    print(f"Retrieved: {test_result[:200]}...")
    
    # Create Agentic RAG system
    rag_system = AgenticRAG(retriever_tool)
    graph = rag_system.create_graph()
    
    # Test the system
    print("\n" + "="*60)
    print("Testing Agentic RAG System")
    print("="*60)
    
    # Test case 1: Question requiring retrieval
    test_questions = [
        "What does Lilian Weng say about types of reward hacking?",
        "Hello, how are you?",
        "What are some methods to prevent hallucination in LLMs?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nTest {i}: {question}")
        print("-" * 50)
        
        try:
            for chunk in graph.stream({
                "messages": [{"role": "user", "content": question}]
            }): # type: ignore
                for node, update in chunk.items():
                    print(f"Update from node: {node}")
                    if hasattr(update["messages"][-1], 'pretty_print'):
                        update["messages"][-1].pretty_print()
                    else:
                        print(update["messages"][-1])
                    print()
        except Exception as e:
            print(f"Error processing question: {e}")
    
    print("\nAgentic RAG system test completed!")
    
    # Interactive mode
    print("\n" + "="*60)
    print("Interactive Mode - Enter your questions (type 'quit' to exit)")
    print("="*60)
    
    while True:
        try:
            user_question = input("\nYour question: ").strip()
            if user_question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_question:
                continue
                
            print(f"\nProcessing: {user_question}")
            print("-" * 40)
            
            for chunk in graph.stream({
                "messages": [{"role": "user", "content": user_question}]
            }): # type: ignore
                for node, update in chunk.items():
                    if node == "generate_answer" or (node == "generate_query_or_respond" and not hasattr(update["messages"][-1], 'tool_calls')):
                        print(f"\nFinal Answer:")
                        if hasattr(update["messages"][-1], 'pretty_print'):
                            update["messages"][-1].pretty_print()
                        else:
                            print(update["messages"][-1].content)
                            
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()