
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# Suppress warnings
import warnings

warnings.filterwarnings('ignore')





class QAChainBuilder:
    """Builds and manages QA chains"""

    @staticmethod
    def create_prompt_template() -> PromptTemplate:
        """
        Step 10: Prompt Engineering - Design system prompt

        Returns:
            PromptTemplate instance
        """
        template = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.

IMPORTANT INSTRUCTIONS:
- Only use information from the provided context
- If the answer is not in the context, say "I don't have enough information to answer this question"
- Be concise but thorough
- Cite specific details from the context when possible
- Do not make up information

Context:
{context}

Question: {question}

Helpful Answer:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    @staticmethod
    def create_qa_chain(retriever, llm, return_source_documents: bool = True):
        """
        Step 9: Create the QA Chain - Connect retriever and LLM

        Chain types:
        - stuff: Put all docs into context (simple, works for small docs)
        - map_reduce: Summarize each doc, then combine
        - refine: Iteratively refine answer with each doc
        - map_rerank: Score each doc's answer, return best

        Args:
            retriever: Document retriever
            llm: Language model
            return_source_documents: Whether to return source documents

        Returns:
            RetrievalQA chain
        """
        print(f"\n  Creating QA chain...")
        print(f"   Chain type: stuff")
        print(f"   Return sources: {return_source_documents}")

        prompt = QAChainBuilder.create_prompt_template()

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # Simplest approach
            retriever=retriever,
            return_source_documents=return_source_documents,
            chain_type_kwargs={"prompt": prompt},
            verbose=False
        )

        print(f" QA chain created successfully")
        return qa_chain