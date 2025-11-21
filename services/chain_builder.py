import re
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
        Step 10: Prompt Engineering - Simplified prompt
        """
        template = """Context: {context}

    Question: {question}

    Answer:"""

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

    @staticmethod
    def post_process_answer(question: str, answer: str) -> str:
        """
        Post-process answers to make them more concise

        Args:
            question: The original question
            answer: The raw answer from the LLM

        Returns:
            Cleaned and concise answer
        """
        if not answer or not answer.strip():
            return answer

        answer = answer.strip()

        # CRITICAL: Remove prompt artifacts if they appear in the answer
        prompt_artifacts = [
            "Use the context below",
            "Context:",
            "Question:",
            "Instructions:",
            "Answer:",
            "Be direct and concise",
            "- Answer directly",
            "- Use ONLY information",
            "I don't have enough information",
        ]

        # If answer contains prompt text, try to extract just the actual answer
        for artifact in prompt_artifacts:
            if artifact in answer:
                # Split on the artifact and take the part after it
                parts = answer.split(artifact)
                if len(parts) > 1:
                    answer = parts[-1].strip()

        # Remove everything before "Answer:" if it exists
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()

        # Remove instruction-like text
        lines = answer.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            # Skip lines that look like instructions
            if line.startswith('-') or line.startswith('•'):
                continue
            if 'context' in line.lower() and 'use' in line.lower():
                continue
            if clean_lines or line:  # Start adding after we find content
                clean_lines.append(line)

        answer = ' '.join(clean_lines).strip()

        question_lower = question.lower()

        # For experience questions
        if any(keyword in question_lower for keyword in ['experience', 'how long', 'how many years']):
            # Extract year patterns like "4+ years" or "4 years"
            year_match = re.search(r'(\d+\+?\s*years?)', answer, re.IGNORECASE)
            if year_match:
                return year_match.group(1)

        # For name questions, extract just the name
        if any(keyword in question_lower for keyword in ['name', 'who is', 'candidate', 'person']):
            # Look for a proper name (capitalized words)
            name_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', answer)
            if name_match:
                return name_match.group(1)

            # If pattern doesn't match, try to extract first capitalized phrase
            name_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', answer)
            if name_match:
                return name_match.group(1)

        # For email questions
        if 'email' in question_lower or 'contact' in question_lower:
            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', answer)
            if email_match:
                return email_match.group(0)

        # For phone number questions
        if 'phone' in question_lower or 'number' in question_lower or 'contact' in question_lower:
            phone_match = re.search(r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}', answer)
            if phone_match:
                return phone_match.group(0)

        # Remove common verbose prefixes
        prefixes_to_remove = [
            "The candidate is ",
            "The candidate's name is ",
            "The name is ",
            "According to the context, ",
            "Based on the context, ",
            "Based on the information provided, ",
            "The answer is ",
            "It is ",
            "This is ",
            "That is ",
            "He has ",
            "She has ",
        ]

        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
                break

        # Take first sentence only
        answer = answer.split('.')[0].strip()

        return answer