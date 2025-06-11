import os
import json
from typing import Dict, List, Optional
from pathlib import Path
import PyPDF2
import numpy as np
from openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import fitz  # PyMuPDF
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

from dotenv import load_dotenv
load_dotenv("/Users/rahultaduri/Interview_Kickstart/Capstone_ResumeCoach/resume_analyzer_agent/backend/.env")  # Load environment variables from .env file

# Define the embedding model
embeddings = OpenAIEmbeddings(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",  # Using the same model as the LLM
    openai_api_key=os.environ.get("NEBIUS_API_KEY"),
    openai_api_base="https://api.studio.nebius.com/v1/"
)

class ResumeAnalyzerAgent:
    def __init__(self):
        # Initialize Nebius API client
        self.client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=os.environ.get("NEBIUS_API_KEY")
        )
        
        # Initialize LangChain LLM with Nebius API
        self.llm = ChatOpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=os.environ.get("NEBIUS_API_KEY"),
            model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
            temperature=0.6,
            max_tokens=2048,
            top_p=0.9,
            extra_body={
                "top_k": 50
            }
        )
        
        # Initialize paths
        self.resume_dir = Path("/Users/rahultaduri/Interview_Kickstart/Capstone_ResumeCoach/resume_analyzer_agent/resume_db/pdf")
        self.jd_dir = Path("/Users/rahultaduri/Interview_Kickstart/Capstone_ResumeCoach/resume_analyzer_agent/JD_db/txt")
        self.output_dir = Path("/Users/rahultaduri/Interview_Kickstart/Capstone_ResumeCoach/resume_analyzer_agent/output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize tools and agents
        self._setup_tools()
        self._setup_agents()
        
        # Initialize PDF styles
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=30
        ))
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12
        ))

    def _setup_tools(self):
        """Setup tools for the agents"""
        self.tools = [
            Tool(
                name="read_pdf",
                func=self.read_pdf,
                description="Read and extract text from a PDF file"
            ),
            Tool(
                name="read_txt",
                func=self.read_txt,
                description="Read text from a text file"
            ),
            Tool(
                name="calculate_similarity",
                func=self.calculate_similarity_score,
                description="Calculate similarity score between two texts"
            )
        ]

    def _setup_agents(self):
        """Setup the four subagents using LangChain"""
        # Common prompt template
        base_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert resume analyzer and career coach."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Subagent 1: Coaching Report Generator
        coaching_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a resume and job description alignment expert.

Your task is to analyze the provided resume and job description. Provide a detailed coaching report with the following sections:

1. **Summary Analysis**:
   - Evaluate the current professional summary
   - Suggest improvements for better recruiter engagement
   - Provide a revised 3-line summary

2. **Skills Alignment**:
   - Identify skills from JD present in resume
   - List missing critical skills
   - Suggest how to incorporate missing skills

3. **Experience Analysis**:
   - Evaluate current bullet points
   - Identify opportunities for better quantification
   - Suggest action verbs and job-specific terminology

4. **Project Relevance**:
   - Assess project alignment with JD
   - Suggest improvements to highlight relevant skills
   - Maintain authenticity of experience

5. **Format and ATS Optimization**:
   - Review current formatting
   - Suggest ATS-friendly improvements
   - Ensure clean, single-column format

6. **Recruiter's Perspective**:
   - List current weaknesses
   - Explain how suggested changes improve alignment
   - Highlight key improvements

7. **Hiring Manager's View**:
   - Compare with top candidate standards
   - Suggest strategic enhancements
   - Identify unique value propositions

8. **Scoring**:
   - Provide before/after scores (0-10)
   - Justify score changes
   - Highlight key improvements

Remember: Do not make up or hallucinate details. Focus on improving existing content."""),
            ("human", "Resume: {resume_text}\nJob Description: {jd_text}\nProvide a structured coaching report:")
        ])
        self.coaching_chain = LLMChain(llm=self.llm, prompt=coaching_prompt)

        # Subagent 2: Resume Reviser
        revision_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert resume writer. Revise the resume following these strict guidelines:

1. **Summary**:
   - Create a 3-line professional summary
   - Use industry-relevant keywords
   - Make it recruiter-friendly

2. **Skills**:
   - Include only skills mentioned in the job description
   - Remove unrelated skills
   - Maintain authenticity

3. **Experience**:
   - Rephrase bullets to emphasize quantifiable results
   - Use action verbs and job-specific terminology
   - Match job description phrasing
   - Keep original experience context

4. **Projects**:
   - Align with job description skills
   - Maintain original context and outcomes
   - Highlight relevant achievements

5. **Format**:
   - Use clean, single-column format
   - No tables, columns, or images
   - ATS-friendly formatting
   - Standardized structure

6. **Content Rules**:
   - No made-up details
   - Preserve original experience
   - Focus on relevant achievements
   - Use clear, concise language

Provide the complete revised resume in a clean, formatted text structure."""),
            ("human", "Original Resume: {resume_text}\nCoaching Report: {coaching_report}\nProvide the revised resume:")
        ])
        self.revision_chain = LLMChain(llm=self.llm, prompt=revision_prompt)

        # Subagent 3: Cover Letter Generator
        cover_letter_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert cover letter writer. Write a compelling cover letter following these guidelines:

1. **Structure**:
   - Professional greeting
   - Strong opening paragraph
   - 2-3 body paragraphs
   - Professional closing

2. **Content**:
   - Highlight relevant experience
   - Connect skills to job requirements
   - Show enthusiasm for the role
   - Keep it under 200 words

3. **Style**:
   - Professional tone
   - Clear and concise
   - No generic phrases
   - Specific to the role

4. **Focus**:
   - Address key job requirements
   - Highlight unique qualifications
   - Show cultural fit
   - End with call to action

Remember: Keep it concise, specific, and professional."""),
            ("human", "Resume: {resume_text}\nJob Description: {jd_text}\nWrite a concise cover letter:")
        ])
        self.cover_letter_chain = LLMChain(llm=self.llm, prompt=cover_letter_prompt)

        # Main agent setup
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=base_prompt
        )
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )

    def read_pdf(self, file_path: str) -> str:
        """Read PDF file and return its text content."""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text

    def read_txt(self, file_path: str) -> str:
        """Read text file and return its content."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def calculate_similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity score between two texts using text generation."""
        # Create a prompt for similarity analysis
        prompt = f"""Analyze the similarity between these two texts and provide a score from 0 to 10, where:
        0 means completely different
        10 means identical or extremely similar
        Only respond with the numerical score.

        Text 1: {text1}

        Text 2: {text2}

        Similarity score:"""

        # Get similarity score from the model
        response = self.client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct",
            messages=[
                {"role": "system", "content": "You are a text similarity analyzer. Respond only with a number between 0 and 10."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=5
        )
        
        # Extract and parse the score
        try:
            score = float(response.choices[0].message.content.strip())
            return min(max(score, 0), 10)  # Ensure score is between 0 and 10
        except (ValueError, IndexError):
            return 0.0  # Return 0 if parsing fails

    def create_pdf(self, content: str, title: str, output_path: Path) -> None:
        """Create a PDF file with the given content and title."""
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Create the content
        story = []
        
        # Add title
        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Spacer(1, 12))
        
        # Split content into paragraphs and add them
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            if para.strip():
                story.append(Paragraph(para.strip(), self.styles['CustomBody']))
                story.append(Spacer(1, 12))
        
        # Build the PDF
        doc.build(story)

    async def process_resume(self, resume_filename: str, jd_filename: str) -> Dict:
        """Main workflow to process resume and job description using LangChain agents."""
        # Read input files
        resume_path = str(self.resume_dir / resume_filename)
        jd_path = str(self.jd_dir / jd_filename)
        
        resume_text = self.read_pdf(resume_path)
        jd_text = self.read_txt(jd_path)

        # Calculate initial similarity score
        initial_similarity = self.calculate_similarity_score(resume_text, jd_text)

        # Generate coaching report using Subagent 1
        coaching_response = await self.coaching_chain.ainvoke({
            "resume_text": resume_text,
            "jd_text": jd_text
        })
        coaching_report = coaching_response.get('text', '') if isinstance(coaching_response, dict) else str(coaching_response)

        # Create revised resume using Subagent 2
        revised_response = await self.revision_chain.ainvoke({
            "resume_text": resume_text,
            "coaching_report": coaching_report
        })
        revised_resume = revised_response.get('text', '') if isinstance(revised_response, dict) else str(revised_response)

        # Generate cover letter using Subagent 3
        cover_letter_response = await self.cover_letter_chain.ainvoke({
            "resume_text": resume_text,
            "jd_text": jd_text
        })
        cover_letter = cover_letter_response.get('text', '') if isinstance(cover_letter_response, dict) else str(cover_letter_response)

        # Calculate final similarity score
        final_similarity = self.calculate_similarity_score(revised_resume, jd_text)

        # Save outputs
        output = {
            "coaching_report": coaching_report,
            "revised_resume": revised_resume,
            "cover_letter": cover_letter,
            "similarity_scores": {
                "initial": initial_similarity,
                "final": final_similarity,
            }
        }

        # Save to JSON file
        output_path = self.output_dir / f"{resume_filename.split('.')[0]}_analysis.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4)

        # Save individual PDFs
        base_name = resume_filename.split('.')[0]
        
        # Save coaching report as PDF
        coaching_pdf_path = self.output_dir / f"{base_name}_coaching_report.pdf"
        self.create_pdf(
            coaching_report,
            "Resume Coaching Report",
            coaching_pdf_path
        )
        
        # Save revised resume as PDF
        revised_pdf_path = self.output_dir / f"{base_name}_revised_resume.pdf"
        self.create_pdf(
            revised_resume,
            "Revised Resume",
            revised_pdf_path
        )
        
        # Save cover letter as PDF
        cover_letter_pdf_path = self.output_dir / f"{base_name}_cover_letter.pdf"
        self.create_pdf(
            cover_letter,
            "Cover Letter",
            cover_letter_pdf_path
        )

        return output

async def main():
    # Example usage
    agent = ResumeAnalyzerAgent()
    result = await agent.process_resume("Resume_Taduri.pdf", "JD1.txt")
    print("Analysis completed. Results saved to output directory.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
