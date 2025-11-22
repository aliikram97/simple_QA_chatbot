import re
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import warnings

warnings.filterwarnings('ignore')


class QAChainBuilder:
    """Builds and manages QA chains"""

    @staticmethod
    def create_prompt_template() -> PromptTemplate:
        """
        Step 10: Prompt Engineering - Improved prompt with strict instructions
        """
        template = """You are a helpful assistant analyzing a document. Use ONLY the information provided in the context below to answer the question. If the answer cannot be found in the context, say "I cannot answer this based on the provided document."

Context:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the context above
- Be direct and concise
- Do not use external knowledge
- Do not make assumptions or add information not in the context

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
        print(f"\n✓ Creating QA chain...")
        print(f"  Chain type: stuff")
        print(f"  Return sources: {return_source_documents}")

        prompt = QAChainBuilder.create_prompt_template()

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # Simplest approach
            retriever=retriever,
            return_source_documents=return_source_documents,
            chain_type_kwargs={"prompt": prompt},
            verbose=False
        )

        print(f"✓ QA chain created successfully")
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
        question_lower = question.lower()

        # Check if this is a question that expects a detailed answer
        detailed_question_keywords = [
            'summary', 'summarize', 'overview', 'describe', 'explain',
            'tell me about', 'what are', 'list', 'discuss', 'elaborate'
        ]
        is_detailed_question = any(kw in question_lower for kw in detailed_question_keywords)

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
            "- Answer based ONLY",
            "- Do not use external",
            "- Do not make assumptions",
        ]

        # Remove prompt artifacts from the beginning
        for artifact in prompt_artifacts:
            if answer.startswith(artifact):
                answer = answer[len(artifact):].strip()

        # If answer contains prompt text in the middle, try to extract just the actual answer
        for artifact in prompt_artifacts:
            if artifact in answer:
                parts = answer.split(artifact)
                if len(parts) > 1:
                    # Take the last part if it's substantial
                    potential_answer = parts[-1].strip()
                    if len(potential_answer) > 20:
                        answer = potential_answer

        # Remove everything before "Answer:" if it exists
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()

        # Remove instruction-like bullet points only (not content bullets)
        lines = answer.split('\n')
        clean_lines = []
        skip_mode = True  # Skip initial instruction-like lines

        for line in lines:
            line_stripped = line.strip()

            # Check if line looks like an instruction
            is_instruction = (
                    (line_stripped.startswith('-') or line_stripped.startswith('•')) and
                    any(word in line_stripped.lower() for word in ['context', 'based', 'only', 'use', 'do not'])
            )

            if is_instruction:
                continue

            # Once we find real content, stop skipping
            if line_stripped:
                skip_mode = False
                clean_lines.append(line_stripped)
            elif not skip_mode:
                clean_lines.append(line_stripped)

        answer = '\n'.join(clean_lines).strip()

        # For detailed questions (summaries, explanations), do minimal processing
        if is_detailed_question:
            # Only remove verbose prefixes
            prefixes_to_remove = [
                "According to the context, ",
                "Based on the context, ",
                "Based on the information provided, ",
                "The document states that ",
                "The document describes ",
                "The context shows that ",
            ]

            for prefix in prefixes_to_remove:
                if answer.lower().startswith(prefix.lower()):
                    answer = answer[len(prefix):].strip()
                    break

            return answer  # Return full answer for detailed questions

        # For factoid questions, extract specific information

        # For experience questions
        if any(keyword in question_lower for keyword in ['experience', 'how long', 'how many years']):
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
        if 'email' in question_lower:
            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', answer)
            if email_match:
                return email_match.group(0)

        # For phone number questions
        if 'phone' in question_lower or 'number' in question_lower:
            phone_match = re.search(r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}', answer)
            if phone_match:
                return phone_match.group(0)

        # Remove common verbose prefixes for short answers
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
            "They have ",
        ]

        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
                break

        # Only take first sentence for factoid questions
        if not is_detailed_question:
            answer = answer.split('.')[0].strip()

        return answer