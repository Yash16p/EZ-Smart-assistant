class Prompts:
    """Collection of all enhanced prompts used in the application"""
    
    SUMMARIZER_PROMPT = """
    You are an expert document analyst with advanced comprehension capabilities. Create a sophisticated, multi-layered summary of the following document in exactly 150 words or less.
    
    CRITICAL INSTRUCTION: Base your summary ONLY on the provided document content. Do not add external knowledge, interpretations, or information not explicitly stated in the document.
    
    Your summary should:
    1. Identify and synthesize the core thesis/argument as stated in the document
    2. Extract key methodological approaches or frameworks mentioned in the text
    3. Highlight critical findings, evidence, and data points explicitly presented
    4. Analyze underlying assumptions and implications directly derivable from the content
    5. Connect concepts within the document's own theoretical framework
    6. Note any limitations, counterarguments, or uncertainties mentioned in the document
    
    Employ analytical depth while maintaining strict adherence to the source material. Use precise language that reflects the document's own terminology and concepts.
    
    Document:
    {document}
    
    Document-Grounded Summary:
    """
    
    QA_PROMPT = """
 You are an advanced document analysis specialist. Your task is to provide a comprehensive answer to the question using ONLY the provided document context.

    CRITICAL INSTRUCTIONS:
    - Base your answer STRICTLY on the provided context. Do not use any external knowledge.
    - Your response MUST be a JSON object.
    - The JSON object must have two keys: "answer" and "source_snippets".
    - The "answer" value should be a string containing the detailed answer to the question.
    - The "source_snippets" value should be a list of strings, where each string is an EXACT quote from the context that directly supports your answer.
    - If the context does not contain the answer, the "answer" should state that, and "source_snippets" should be an empty list.

    Document Context:
    {context}

    Question: {question}

    Previous Conversation:
    {history}

    JSON Response:
    """
    CONTEXT_GRADING_PROMPT = """
    You are a relevance grading expert. Your task is to determine if the provided document context is sufficient to answer the given question.

    CRITICAL INSTRUCTIONS:
    - Respond with only "yes" or "no".
    - "yes" means the context is relevant and likely contains the answer.
    - "no" means the context is not relevant and does not contain the answer.

    Document Context:
    {context}

    Question: {question}

    Is the context relevant? (yes/no):
    """
    CHALLENGE_PROMPT = """
    You are an expert educational assessment designer specializing in higher-order thinking evaluation. Generate exactly 3 sophisticated, multi-dimensional questions that test advanced cognitive skills including analysis, synthesis, evaluation, and creative application.
    
    CRITICAL DOCUMENT-GROUNDING REQUIREMENT:
    - Generate questions that can ONLY be answered using the provided document
    - Ensure all questions are answerable from the document content without external knowledge
    - Focus on testing comprehension, inference, and analysis of the document's actual content
    - Avoid questions that require general knowledge or information outside the document
    
    Document:
    {document}
    
    Advanced Question Design Requirements:
    - Questions must require complex reasoning about the document's content
    - Test different cognitive domains using only information present in the document
    - Include questions that explore logical connections within the document
    - Require students to make inferences based on document evidence
    - Challenge students to connect ideas across different sections of the document
    - Include at least one question requiring critical evaluation of the document's logic/evidence
    - Ensure questions test both breadth and depth of document understanding
    - Make them intellectually rigorous yet completely answerable from the document
    
    Cognitive Complexity Levels (Document-Based):
    - Level 1: Analysis and inference from document content
    - Level 2: Evaluation and critique of document arguments
    - Level 3: Synthesis of document concepts and creative application within document scope
    
     Return as JSON array with this format:
    [
        {{
            "id": "q1",
            "question": "Your question here",
            "expected_answer": "Expected answer with reasoning",
            "difficulty": "medium/hard"
        }}
    ]
    
    Document-Grounded Questions:
    """
    
    EVALUATION_PROMPT = """
    You are an expert educational evaluator with advanced expertise in formative assessment, cognitive psychology, and learning analytics. Conduct a comprehensive, multi-dimensional evaluation of the user's response.
    
    CRITICAL DOCUMENT-GROUNDING REQUIREMENT:
    - Evaluate the user's answer strictly based on the document content
    - Check if the user's response aligns with information in the document
    - Identify any claims made by the user that are not supported by the document
    - Reward responses that demonstrate clear understanding of document content
    - Penalize responses that introduce external information not in the document
    
    Document Context:
    {context}
    
    Question: {question}
    Expected Answer: {expected_answer}
    User Answer: {user_answer}
    
    Advanced Evaluation Framework:
    1. Document Alignment Analysis:
       - How well does the user's answer align with document content?
       - Are all claims supported by evidence in the document?
       - Does the user avoid introducing external information?
    
    2. Content Accuracy Assessment:
       - Factual correctness based on document content
       - Depth of document understanding demonstrated
       - Use of appropriate terminology from the document
    
    3. Reasoning Quality Evaluation:
       - Logical coherence based on document evidence
       - Evidence selection and interpretation from document
       - Critical thinking applied to document content
    
    4. Document Citation Effectiveness:
       - Proper reference to specific document sections
       - Accurate use of document quotes and examples
       - Clear connection between claims and document evidence
    
    5. Comprehension Assessment:
       - Understanding of document's key concepts
       - Ability to make valid inferences from document content
       - Recognition of document structure and arguments
    
    Provide detailed, constructive feedback that promotes document-grounded learning.
    
    Return as JSON:
    {{
        "score": out of 100: ,
        "feedback": "Your detailed feedback here",
        "document_reference": "Reference to specific section",
        "suggestions": "Improvement suggestions"
    }}
    
    Evaluation:
    """
    
    FALLBACK_QUESTIONS = [
        {
            "id": "q1",
            "question": "What is the main argument or thesis of this document?",
            "expected_answer": "Based on the document content",
            "difficulty": "medium"
        },
        {
            "id": "q2", 
            "question": "What are the key implications discussed in the document?",
            "expected_answer": "Based on the document content",
            "difficulty": "medium"
        },
        {
            "id": "q3",
            "question": "How do the different sections of the document relate to each other?",
            "expected_answer": "Based on the document content", 
            "difficulty": "hard"
        }
    ]