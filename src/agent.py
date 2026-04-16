import math
import numpy as np
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langsmith import traceable

load_dotenv()


class DocMindAgent:
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
        self.llm = ChatOllama(model="llama3.2")
        self.tools     = self._build_tools()
        self.tool_map  = {t.name: t for t in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def _build_tools(self):
        rag = self.rag

        @tool
        def search_document(query: str) -> str:
            """Search the uploaded PDF document and retrieve relevant content.
            Use this tool when the user asks any specific question about
            the document content or wants to find information in the document.
            Input: a search query string."""
            query_emb          = rag.embedding_model.encode([query]).astype(np.float32)
            distances, indices = rag.index.search(query_emb, 3)
            chunks             = [rag.chunks[i] for i in indices[0]]
            return "\n\n---\n\n".join(chunks)

        @tool
        def summarise_document(section: str) -> str:
            """Generate a summary of the document or a specific section.
            Use this when the user asks for a summary or overview.
            Do NOT use search_document for summary requests.
            Input: 'full' for full summary, or a topic name."""
            if section == "full":
                indices = list(range(0, len(rag.chunks),
                               max(1, len(rag.chunks) // 10)))
                chunks  = [rag.chunks[i] for i in indices[:10]]
            else:
                query_emb       = rag.embedding_model.encode(
                    [section]
                ).astype(np.float32)
                _, idxs         = rag.index.search(query_emb, 5)
                chunks          = [rag.chunks[i] for i in idxs[0]]
            return "\n\n---\n\n".join(chunks)

        @tool
        def calculate(expression: str) -> str:
            """Evaluate a mathematical expression and return the exact result.
            Use this for any arithmetic, percentage, or numerical computation.
            Never compute math in your head — always use this tool.
            Input: a valid Python math expression like '1500 * 0.15'."""
            try:
                result = eval(expression, {"__builtins__": {}}, {
                    "abs": abs, "round": round, "min": min, "max": max,
                    "sum": sum, "pow": pow, "sqrt": math.sqrt,
                    "pi": math.pi, "e": math.e, "log": math.log,
                })
                return f"{expression} = {result}"
            except Exception as ex:
                return f"Error: {ex}"

        @tool
        def get_document_info(query: str) -> str:
            """Get metadata about the loaded document — pages, chunks, word count.
            Use this when the user asks about the document itself rather than
            its content, e.g. 'how long is this document'.
            Input: any string."""
            total_chars  = sum(len(c) for c in rag.chunks)
            approx_words = total_chars // 5
            return (
                f"Document: {rag.current_pdf}\n"
                f"Chunks: {len(rag.chunks)}\n"
                f"Approximate words: {approx_words:,}\n"
                f"Approximate pages: {len(rag.chunks) // 5}"
            )

        return [search_document, summarise_document,
                calculate, get_document_info]

    @traceable(name="docmind-agent-query")
    def query(self, user_input: str, max_iterations: int = 5) -> dict:
        """Run the agent loop — traced in LangSmith."""
        messages = [
            SystemMessage(content="""You are DocMind, an intelligent document 
assistant. You have access to tools to search documents, summarise content,
perform calculations, and get document metadata.

Always use the most appropriate tool for each question:
- Specific questions about content → search_document
- Summary or overview requests → summarise_document  
- Any math or calculations → calculate
- Questions about the document itself → get_document_info

Base your final answer on tool results, not your general knowledge."""),
            HumanMessage(content=user_input)
        ]

        tools_used = []

        for _ in range(max_iterations):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                return {
                    "question":   user_input,
                    "answer":     response.content,
                    "tools_used": tools_used,
                }

            for tool_call in response.tool_calls:
                name   = tool_call["name"]
                args   = tool_call["args"]
                fn     = self.tool_map.get(name)
                result = fn.invoke(args) if fn else f"Tool {name} not found"
                tools_used.append({"tool": name, "args": args,
                                   "result": str(result)[:200]})
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                ))

        return {
            "question":   user_input,
            "answer":     "Max iterations reached without final answer.",
            "tools_used": tools_used,
        }

    def is_loaded(self) -> bool:
        return self.rag.is_loaded()