"""
Prompt engineering utilities for the Notion RAG system.
Provides templates and utilities for various AI tasks.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of AI tasks."""
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    ANALYSIS = "analysis"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    TRANSLATION = "translation"
    CODE_GENERATION = "code_generation"


@dataclass
class PromptTemplate:
    """A prompt template with variables."""
    
    name: str
    template: str
    variables: List[str]
    description: str
    task_type: TaskType
    
    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable: {e}")
    
    def get_missing_variables(self, **kwargs) -> List[str]:
        """Get list of missing variables."""
        provided_vars = set(kwargs.keys())
        required_vars = set(self.variables)
        return list(required_vars - provided_vars)


class PromptLibrary:
    """Library of prompt templates for various tasks."""
    
    def __init__(self):
        """Initialize the prompt library."""
        self.templates: Dict[str, PromptTemplate] = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize default prompt templates."""
        
        # Summarization templates
        self.add_template(PromptTemplate(
            name="basic_summary",
            template="""Please provide a concise summary of the following text:

{text}

Summary:""",
            variables=["text"],
            description="Basic text summarization",
            task_type=TaskType.SUMMARIZATION
        ))
        
        self.add_template(PromptTemplate(
            name="detailed_summary",
            template="""Please provide a detailed summary of the following text, including key points, main ideas, and important details:

{text}

Detailed Summary:""",
            variables=["text"],
            description="Detailed text summarization with key points",
            task_type=TaskType.SUMMARIZATION
        ))
        
        self.add_template(PromptTemplate(
            name="bullet_point_summary",
            template="""Please provide a bullet-point summary of the following text, highlighting the main points and key takeaways:

{text}

Key Points:
•""",
            variables=["text"],
            description="Bullet-point summary format",
            task_type=TaskType.SUMMARIZATION
        ))
        
        # RAG-specific summarization template
        self.add_template(PromptTemplate(
            name="rag_summary",
            template="""You are an AI assistant that provides comprehensive summaries based on provided context documents.

Context:
{context}

Question: {question}

Please provide a detailed summary that addresses the question, including:
- Main points and key concepts
- Important details and insights
- Relevant examples or evidence
- Connections between different pieces of information

Guidelines:
- Use only information from the provided context
- Focus on content relevant to the question
- Be comprehensive but well-organized
- Cite specific sources when possible
- If the context doesn't contain enough information, acknowledge this

Summary:""",
            variables=["context", "question"],
            description="RAG detailed summary with context",
            task_type=TaskType.SUMMARIZATION
        ))
        
        # Question answering templates
        self.add_template(PromptTemplate(
            name="rag_qa",
            template="""You are a helpful AI assistant that answers questions based on the provided context documents.

Context:
{context}

Question: {question}

Guidelines:
1. Use only the information provided in the context documents to answer questions
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Cite specific documents or sources when possible
4. Be concise but thorough in your responses
5. If you're unsure about something, acknowledge the uncertainty

Answer:""",
            variables=["context", "question"],
            description="RAG question answering with context",
            task_type=TaskType.QUESTION_ANSWERING
        ))
        
        # Analysis templates
        self.add_template(PromptTemplate(
            name="content_analysis",
            template="""Please analyze the following content and provide insights on:

{text}

Analysis:
1. Main themes and topics
2. Key insights and findings
3. Potential applications or implications
4. Areas that need further exploration

Analysis:""",
            variables=["text"],
            description="Content analysis with structured insights",
            task_type=TaskType.ANALYSIS
        ))
        
        # RAG-specific analysis template
        self.add_template(PromptTemplate(
            name="rag_analysis",
            template="""You are an AI assistant that analyzes content based on provided context documents.

Context:
{context}

Question: {question}

Please provide a comprehensive analysis including:
1. Main themes and topics from the context
2. Key insights and findings relevant to the question
3. Potential applications or implications
4. Areas that need further exploration
5. Connections between different pieces of information

Guidelines:
- Use only information from the provided context
- Cite specific documents or sources when possible
- If the context doesn't contain enough information, acknowledge this
- Be thorough but well-structured in your analysis

Analysis:""",
            variables=["context", "question"],
            description="RAG content analysis with context",
            task_type=TaskType.ANALYSIS
        ))
        
        # Extraction templates
        self.add_template(PromptTemplate(
            name="key_points_extraction",
            template="""Please extract the key points and main ideas from the following text:

{text}

Key Points:
1.""",
            variables=["text"],
            description="Extract key points from text",
            task_type=TaskType.EXTRACTION
        ))
        
        # RAG-specific extraction template
        self.add_template(PromptTemplate(
            name="rag_extraction",
            template="""You are an AI assistant that extracts key information from provided context documents.

Context:
{context}

Question: {question}

Please extract the key points and main ideas that are relevant to the question:

Guidelines:
- Focus on information directly relevant to the question
- Extract concrete facts, concepts, and insights
- Organize information logically
- Cite specific sources when possible
- If the context doesn't contain relevant information, say so

Key Points:
1.""",
            variables=["context", "question"],
            description="RAG key points extraction with context",
            task_type=TaskType.EXTRACTION
        ))
        
        self.add_template(PromptTemplate(
            name="fact_extraction",
            template="""Please extract factual information from the following text. Focus on concrete facts, numbers, dates, names, and verifiable information:

{text}

Facts Extracted:
•""",
            variables=["text"],
            description="Extract factual information from text",
            task_type=TaskType.EXTRACTION
        ))
        
        # Classification templates
        self.add_template(PromptTemplate(
            name="topic_classification",
            template="""Please classify the following text into one or more relevant categories:

{text}

Categories to choose from:
{categories}

Classification:""",
            variables=["text", "categories"],
            description="Classify text into predefined categories",
            task_type=TaskType.CLASSIFICATION
        ))
        
        # Code generation templates
        self.add_template(PromptTemplate(
            name="code_explanation",
            template="""Please explain the following code in a clear and understandable way:

{code}

Explanation:""",
            variables=["code"],
            description="Explain code functionality",
            task_type=TaskType.CODE_GENERATION
        ))
        
        # Translation templates
        self.add_template(PromptTemplate(
            name="text_translation",
            template="""Please translate the following text to {target_language}:

{text}

Translation:""",
            variables=["text", "target_language"],
            description="Translate text to specified language",
            task_type=TaskType.TRANSLATION
        ))
    
    def add_template(self, template: PromptTemplate):
        """Add a new prompt template to the library."""
        self.templates[template.name] = template
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self.templates.get(name)
    
    def list_templates(self, task_type: Optional[TaskType] = None) -> List[PromptTemplate]:
        """List all templates, optionally filtered by task type."""
        if task_type:
            return [t for t in self.templates.values() if t.task_type == task_type]
        return list(self.templates.values())
    
    def format_template(self, name: str, **kwargs) -> str:
        """Format a template by name with provided variables."""
        template = self.get_template(name)
        if not template:
            raise ValueError(f"Template '{name}' not found")
        return template.format(**kwargs)


class Summarizer:
    """Utility class for text summarization."""
    
    def __init__(self, prompt_library: Optional[PromptLibrary] = None):
        """Initialize the summarizer."""
        self.prompt_library = prompt_library or PromptLibrary()
    
    def summarize(self, text: str, style: str = "basic", max_length: Optional[int] = None) -> str:
        """
        Summarize text using specified style.
        
        Args:
            text: Text to summarize
            style: Summary style ("basic", "detailed", "bullet")
            max_length: Optional maximum length for summary
            
        Returns:
            str: Generated summary
        """
        if style == "basic":
            template_name = "basic_summary"
        elif style == "detailed":
            template_name = "detailed_summary"
        elif style == "bullet":
            template_name = "bullet_point_summary"
        else:
            raise ValueError(f"Unknown summary style: {style}")
        
        prompt = self.prompt_library.format_template(template_name, text=text)
        
        # Note: This would typically be sent to an AI model
        # For now, we return the prompt template
        return prompt
    
    def extract_key_points(self, text: str) -> str:
        """Extract key points from text."""
        return self.prompt_library.format_template("key_points_extraction", text=text)
    
    def extract_facts(self, text: str) -> str:
        """Extract factual information from text."""
        return self.prompt_library.format_template("fact_extraction", text=text)


class ContentAnalyzer:
    """Utility class for content analysis."""
    
    def __init__(self, prompt_library: Optional[PromptLibrary] = None):
        """Initialize the content analyzer."""
        self.prompt_library = prompt_library or PromptLibrary()
    
    def analyze_content(self, text: str) -> str:
        """Analyze content and provide insights."""
        return self.prompt_library.format_template("content_analysis", text=text)
    
    def classify_topic(self, text: str, categories: List[str]) -> str:
        """Classify text into predefined categories."""
        categories_text = "\n".join([f"- {cat}" for cat in categories])
        return self.prompt_library.format_template("topic_classification", text=text, categories=categories_text)


class PromptBuilder:
    """Utility for building custom prompts."""
    
    @staticmethod
    def build_system_prompt(role: str, instructions: List[str], constraints: Optional[List[str]] = None) -> str:
        """
        Build a system prompt.
        
        Args:
            role: The role the AI should take
            instructions: List of instructions
            constraints: Optional list of constraints
            
        Returns:
            str: Formatted system prompt
        """
        prompt_parts = [f"You are {role}."]
        
        if instructions:
            prompt_parts.append("\nInstructions:")
            for i, instruction in enumerate(instructions, 1):
                prompt_parts.append(f"{i}. {instruction}")
        
        if constraints:
            prompt_parts.append("\nConstraints:")
            for i, constraint in enumerate(constraints, 1):
                prompt_parts.append(f"{i}. {constraint}")
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def build_rag_prompt(
        context: str,
        question: str,
        system_instructions: Optional[List[str]] = None,
        output_format: Optional[str] = None
    ) -> str:
        """
        Build a RAG prompt.
        
        Args:
            context: Context documents
            question: User question
            system_instructions: Optional system instructions
            output_format: Optional output format specification
            
        Returns:
            str: Formatted RAG prompt
        """
        prompt_parts = []
        
        # System instructions
        if system_instructions:
            prompt_parts.append("System Instructions:")
            for instruction in system_instructions:
                prompt_parts.append(f"- {instruction}")
            prompt_parts.append("")
        
        # Context
        prompt_parts.append("Context:")
        prompt_parts.append(context)
        prompt_parts.append("")
        
        # Question
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("")
        
        # Output format
        if output_format:
            prompt_parts.append(f"Please respond in the following format:")
            prompt_parts.append(output_format)
            prompt_parts.append("")
        
        prompt_parts.append("Answer:")
        
        return "\n".join(prompt_parts)


# Global prompt library instance
_prompt_library: Optional[PromptLibrary] = None


def get_prompt_library() -> PromptLibrary:
    """Get the global prompt library instance."""
    global _prompt_library
    if _prompt_library is None:
        _prompt_library = PromptLibrary()
    return _prompt_library


def get_summarizer() -> Summarizer:
    """Get a summarizer instance."""
    return Summarizer(get_prompt_library())


def get_content_analyzer() -> ContentAnalyzer:
    """Get a content analyzer instance."""
    return ContentAnalyzer(get_prompt_library()) 