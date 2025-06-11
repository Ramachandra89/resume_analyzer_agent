import streamlit as st
import asyncio
import tempfile
import os
from pathlib import Path
import sys

# Add the backend directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "backend"))
from agent_v2 import ResumeAnalyzerAgentV2

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = ResumeAnalyzerAgentV2()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Set page config
st.set_page_config(
    page_title="Resume Coach AI",
    page_icon="📝",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.assistant {
        background-color: #262730;
    }
    .chat-message .content {
        display: flex;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📝 Resume Coach AI")
st.markdown("""
Welcome to Resume Coach AI! This tool helps you:
- Analyze your resume against job descriptions
- Get personalized coaching and feedback
- Generate ATS-friendly resume revisions
- Create targeted cover letters
- Chat with an AI career advisor
""")

# Create tabs
tab1, tab2 = st.tabs(["Resume Analysis", "Chat with Coach"])

with tab1:
    st.header("Resume Analysis")
    
    # File upload
    uploaded_resume = st.file_uploader("Upload your resume (PDF)", type=['pdf'])
    jd_text = st.text_area("Paste the job description here", height=200)
    
    if uploaded_resume and jd_text:
        if st.button("Analyze Resume"):
            with st.spinner("Analyzing your resume..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_resume.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Read resume text
                    resume_text = st.session_state.agent.read_pdf(tmp_path)
                    
                    # Process resume
                    result = asyncio.run(st.session_state.agent.process_resume(resume_text, jd_text))
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    # Similarity Scores
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Initial Match Score", f"{result['similarity_scores']['initial']:.1f}/10")
                    with col2:
                        st.metric("Improved Match Score", f"{result['similarity_scores']['final']:.1f}/10")
                    
                    # Coaching Report
                    with st.expander("📊 Coaching Report", expanded=True):
                        st.markdown(result['coaching_report'])
                    
                    # Revised Resume
                    with st.expander("📝 Revised Resume", expanded=True):
                        st.markdown(result['revised_resume'])
                    
                    # Cover Letter
                    with st.expander("✉️ Cover Letter", expanded=True):
                        st.markdown(result['cover_letter'])
                    
                    # Download buttons
                    st.download_button(
                        "Download Coaching Report (PDF)",
                        data=result['coaching_report'],
                        file_name="coaching_report.txt",
                        mime="text/plain"
                    )
                    st.download_button(
                        "Download Revised Resume (PDF)",
                        data=result['revised_resume'],
                        file_name="revised_resume.txt",
                        mime="text/plain"
                    )
                    st.download_button(
                        "Download Cover Letter (PDF)",
                        data=result['cover_letter'],
                        file_name="cover_letter.txt",
                        mime="text/plain"
                    )
                    
                finally:
                    # Clean up temporary file
                    os.unlink(tmp_path)

with tab2:
    st.header("Chat with Resume Coach")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.container():
            st.markdown(f"""
            <div class="chat-message {message['role']}">
                <strong>{message['role'].title()}</strong>
                <div class="content">{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.text_input("Ask your question about resume writing or career development:")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Get AI response
        with st.spinner("Thinking..."):
            response = asyncio.run(st.session_state.agent.chat(user_input))
            
            # Add AI response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })
        
        # Rerun to update chat display
        st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit and LangChain</p>
</div>
""", unsafe_allow_html=True) 