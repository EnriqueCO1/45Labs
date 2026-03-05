##"""
##45Labs - IB Extended Essay Feedback Platform
##Streamlit interface for rubric-based feedback.
##"""
##
##import streamlit as st
##import sys
##import os
##from pathlib import Path
##import logging
##
### Add src to path for imports
##sys.path.append(str(Path(__file__).parent.parent.parent))
##
##from src.ingestion.pdf_processor import PDFProcessor
##from src.rag.vector_store import VectorStore, initialize_vectorstore
##from src.models.router import ModelRouter, SimpleRouter, check_api_keys
##from src.models.prompts import get_rubric_question_prompt, get_essay_feedback_prompt, get_no_context_message
##
### Configure logging
##logging.basicConfig(level=logging.INFO)
##logger = logging.getLogger(__name__)
##
### Page configuration
##st.set_page_config(
##    page_title="45Labs - IB Extended Essay Feedback",
##    page_icon="🎓",
##    layout="wide"
##)
##
### Initialize session state
##if "vectorstore" not in st.session_state:
##    st.session_state.vectorstore = None
##if "router" not in st.session_state:
##    st.session_state.router = ModelRouter()
##if "messages" not in st.session_state:
##    st.session_state.messages = []
##
##
##def initialize_system():
##    """Initialize the RAG system."""
##    with st.spinner("Initializing 45Labs system..."):
##        try:
##            # Check if vector store exists
##            vectorstore = VectorStore()
##            
##            if not vectorstore.load_index():
##                st.info("First time setup: Processing IB rubric documents...")
##                
##                # Process PDF if needed
##                pdf_path = "/mnt/okcomputer/output/45labs/data/rubrics/ib_extended_essay_guide.pdf"
##                chunks_path = "/mnt/okcomputer/output/45labs/data/rubrics/chunks.json"
##                
##                if os.path.exists(pdf_path):
##                    processor = PDFProcessor()
##                    chunks = processor.process_pdf(pdf_path, "Extended Essay")
##                    processor.save_chunks(chunks, chunks_path)
##                    
##                    # Create vector store
##                    vectorstore.create_index(chunks)
##                    vectorstore.save_index()
##                else:
##                    st.error("IB rubric PDF not found. Please ensure the rubric document is available.")
##                    return False
##            
##            st.session_state.vectorstore = vectorstore
##            st.success("45Labs system initialized successfully!")
##            return True
##            
##        except Exception as e:
##            st.error(f"Error initializing system: {str(e)}")
##            return False
##
##
##def get_feedback(input_text: str, input_type: str) -> str:
##    """
##    Get feedback for input text.
##    
##    Args:
##        input_text: User's input
##        input_type: Type of input ('question' or 'essay')
##        
##    Returns:
##        Generated feedback
##    """
##    if not st.session_state.vectorstore:
##        return "System not initialized. Please wait for the system to load."
##    
##    try:
##        # Get context from vector store
##        if input_type == "question":
##            context = st.session_state.vectorstore.get_context(input_text, max_chunks=2)
##        else:
##            context = st.session_state.vectorstore.get_context(input_text, max_chunks=5)
##        
##        if not context:
##            return get_no_context_message()
##        
##        # Create appropriate prompt
##        if input_type == "question":
##            prompt = get_rubric_question_prompt(input_text, context)
##        else:
##            prompt = get_essay_feedback_prompt(input_text, context)
##        
##        # Generate response
##        response_data = st.session_state.router.generate_response(prompt, input_type)
##        
##        return response_data["response"]
##        
##    except Exception as e:
##        logger.error(f"Error generating feedback: {e}")
##        return f"Error generating feedback: {str(e)}"
##
##
##def main():
##    """Main Streamlit app."""
##    
##    # Header
##    st.title("🎓 45Labs")
##    st.subheader("IB Extended Essay Feedback Platform")
##    st.markdown("Get feedback based on official IB rubrics. Strictly aligned with IB documentation.")
##    
##    # Check API keys
##    api_status = check_api_keys()
##    if not api_status["all_available"]:
##        st.warning("⚠️ API Keys Missing")
##        with st.expander("API Configuration Required"):
##            st.markdown("Please set the following environment variables:")
##            st.code("export OPENAI_API_KEY=your_key_here\nexport ANTHROPIC_API_KEY=your_key_here")
##            st.markdown("Available models:")
##            st.json(api_status)
##    
##    # Initialize system
##    if st.session_state.vectorstore is None:
##        if not initialize_system():
##            st.stop()
##    
##    # Input type selector
##    st.markdown("---")
##    input_type = st.radio(
##        "Select input type:",
##        options=["Pregunta sobre rúbrica", "Corregir ensayo"],
##        horizontal=True,
##        help="Choose whether you have a question about the rubric or want feedback on an essay"
##    )
##    
##    # Input area
##    if input_type == "Pregunta sobre rúbrica":
##        st.markdown("### Ask a question about the IB Extended Essay rubric")
##        input_text = st.text_area(
##            "Escribe tu pregunta:",
##            placeholder="Ejemplo: ¿Qué debe tener la conclusión de la Monografía?",
##            height=100,
##            key="question_input"
##        )
##        max_tokens = 500
##    else:
##        st.markdown("### Submit your Extended Essay for feedback")
##        input_text = st.text_area(
##            "Pega tu ensayo completo aquí:",
##            placeholder="Pega el texto completo de tu ensayo aquí...",
##            height=400,
##            key="essay_input"
##        )
##        max_tokens = 5000
##    
##    # Analyze button
##    if st.button("Analizar", type="primary", use_container_width=True):
##        if not input_text.strip():
##            st.error("Por favor ingresa algún texto para analizar.")
##        elif st.session_state.router.count_tokens(input_text) > max_tokens:
##            st.error(f"El texto es demasiado largo. Máximo {max_tokens} tokens permitidos.")
##        else:
##            # Get feedback
##            with st.spinner("Analizando con base en la rúbrica oficial del IB..."):
##                input_type_key = "question" if input_type == "Pregunta sobre rúbrica" else "essay"
##                feedback = get_feedback(input_text, input_type_key)
##                
##                # Store in session state
##                st.session_state.messages.append({
##                    "type": input_type_key,
##                    "input": input_text[:200] + "..." if len(input_text) > 200 else input_text,
##                    "response": feedback
##                })
##                
##                # Display feedback
##                st.markdown("---")
##                st.markdown("### Resultado del análisis")
##                st.markdown(feedback)
##    
##    # History section
##    if st.session_state.messages:
##        st.markdown("---")
##        with st.expander("Historial de consultas"):
##            for i, msg in enumerate(reversed(st.session_state.messages[-5:]), 1):
##                st.markdown(f"**{msg['type'].title()} {len(st.session_state.messages) - i + 1}:**")
##                st.markdown(f"_Input:_ {msg['input']}")
##                st.markdown(f"_Response preview:_ {msg['response'][:200]}...")
##                st.markdown("---")
##    
##    # Footer
##    st.markdown("---")
##    st.markdown("""
##    **45Labs** - Plataforma de feedback académico basado en rúbricas oficiales del IB.
##    
##    ✓ Usa SOLO documentación oficial del IB  
##    ✓ No inventa criterios ni alucina información  
##    ✓ Proporciona feedback criterio por criterio  
##    ✓ Respeta las normas de honestidad académica del IB
##    """)
##
##
##if __name__ == "__main__":
##    main()
"""
45Labs - IB Extended Essay Feedback Platform
ChatGPT-style Streamlit interface for rubric-based feedback.
"""

import streamlit as st
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import uuid

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ingestion.pdf_processor import PDFProcessor
from query.llamaindex_query import QueryPipeline, IBFeedbackEngine

# For simple queries
pipeline = QueryPipeline()
response = pipeline.query(user_question)

# For IB-specific feedback
engine = IBFeedbackEngine()
response = engine.answer_rubric_question(user_question)
response = engine.provide_essay_feedback(essay_text)from src.models.router import ModelRouter, SimpleRouter, check_api_keys
from src.models.prompts import get_rubric_question_prompt, get_essay_feedback_prompt, get_no_context_message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="45Labs - IB Extended Essay Feedback",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    /* Main chat container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
    }
    
    /* Chat message styling */
    .user-message {
        background-color: #f7f7f8;
        padding: 1rem 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background-color: transparent;
        padding: 1rem 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        margin-right: 2rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f7f7f8;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Chat history items */
    .chat-history-item {
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .chat-history-item:hover {
        background-color: #e5e5e5;
    }
    
    /* Input box styling */
    .stTextArea textarea {
        border-radius: 1rem;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 0.5rem;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "router" not in st.session_state:
    st.session_state.router = ModelRouter()
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "question"  # 'question' or 'essay'


def create_new_conversation():
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    st.session_state.conversations[conversation_id] = {
        "id": conversation_id,
        "title": "Nueva conversación",
        "messages": [],
        "created_at": datetime.now(),
        "mode": st.session_state.input_mode
    }
    st.session_state.current_conversation_id = conversation_id
    return conversation_id


def get_current_conversation():
    """Get the current conversation or create a new one."""
    if st.session_state.current_conversation_id is None:
        create_new_conversation()
    
    return st.session_state.conversations.get(st.session_state.current_conversation_id)


def update_conversation_title(conversation_id, first_message):
    """Update conversation title based on first message."""
    if conversation_id in st.session_state.conversations:
        # Use first 50 characters of the message as title
        title = first_message[:50] + "..." if len(first_message) > 50 else first_message
        st.session_state.conversations[conversation_id]["title"] = title


def initialize_system():
    """Initialize the RAG system."""
    try:
        # Check if vector store exists
        vectorstore = VectorStore()
        
        if not vectorstore.load_index():
            st.info("First time setup: Processing IB rubric documents...")
            
            # Process PDF if needed
            pdf_path = str(Path(__file__).parent.parent.parent / "data" / "rubrics" / "ib_extended_essay_guide.pdf")
            chunks_path = str(Path(__file__).parent.parent.parent / "data" / "rubrics" / "chunks.json")
            
            if os.path.exists(pdf_path):
                processor = PDFProcessor()
                chunks = processor.process_pdf(pdf_path, "Extended Essay")
                processor.save_chunks(chunks, chunks_path)
                
                # Create vector store
                vectorstore.create_index(chunks)
                vectorstore.save_index()
            else:
                st.error("IB rubric PDF not found. Please ensure the rubric document is available.")
                return False
        
        st.session_state.vectorstore = vectorstore
        return True
        
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return False


def get_feedback(input_text: str, input_type: str) -> str:
    """
    Get feedback for input text.
    
    Args:
        input_text: User's input
        input_type: Type of input ('question' or 'essay')
        
    Returns:
        Generated feedback
    """
    if not st.session_state.vectorstore:
        return "System not initialized. Please wait for the system to load."
    
    try:
        # Get context from vector store
        if input_type == "question":
            context = st.session_state.vectorstore.get_context(input_text, max_chunks=2)
        else:
            context = st.session_state.vectorstore.get_context(input_text, max_chunks=5)
        
        if not context:
            return get_no_context_message()
        
        # Create appropriate prompt
        if input_type == "question":
            prompt = get_rubric_question_prompt(input_text, context)
        else:
            prompt = get_essay_feedback_prompt(input_text, context)
        
        # Generate response
        response_data = st.session_state.router.generate_response(prompt, input_type)
        
        return response_data["response"]
        
    except Exception as e:
        logger.error(f"Error generating feedback: {e}")
        return f"Error generating feedback: {str(e)}"


def render_sidebar():
    """Render the sidebar with conversation history."""
    with st.sidebar:
        st.title("🎓 45Labs")
        
        # New conversation button
        if st.button("➕ Nueva conversación", use_container_width=True, type="primary"):
            create_new_conversation()
            st.rerun()
        
        st.markdown("---")
        
        # Mode selector
        mode = st.radio(
            "Modo:",
            options=["💬 Pregunta sobre rúbrica", "📝 Corregir ensayo"],
            key="mode_selector"
        )
        st.session_state.input_mode = "question" if "Pregunta" in mode else "essay"
        
        st.markdown("---")
        
        # Conversation history
        st.subheader("Historial")
        
        if st.session_state.conversations:
            # Sort conversations by creation date (newest first)
            sorted_conversations = sorted(
                st.session_state.conversations.values(),
                key=lambda x: x["created_at"],
                reverse=True
            )
            
            for conv in sorted_conversations:
                # Determine icon based on mode
                icon = "💬" if conv["mode"] == "question" else "📝"
                
                # Create a button for each conversation
                is_current = conv["id"] == st.session_state.current_conversation_id
                button_type = "primary" if is_current else "secondary"
                
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    if st.button(
                        f"{icon} {conv['title']}", 
                        key=f"conv_{conv['id']}",
                        use_container_width=True,
                        type=button_type
                    ):
                        st.session_state.current_conversation_id = conv["id"]
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"del_{conv['id']}", help="Eliminar"):
                        del st.session_state.conversations[conv["id"]]
                        if st.session_state.current_conversation_id == conv["id"]:
                            st.session_state.current_conversation_id = None
                        st.rerun()
        else:
            st.info("No hay conversaciones aún")
        
        st.markdown("---")
        
        # API status
        with st.expander("⚙️ Estado del sistema"):
            api_status = check_api_keys()
            if api_status["all_available"]:
                st.success("✅ APIs configuradas")
            else:
                st.warning("⚠️ Faltan algunas APIs")
            st.json(api_status)


def render_chat_interface():
    """Render the main chat interface."""
    conversation = get_current_conversation()
    
    if conversation is None:
        st.error("No conversation selected")
        return
    
    # Display chat messages
    for message in conversation["messages"]:
        if message["role"] == "user":
            with st.container():
                st.markdown(f"""
                <div class="user-message">
                    <strong>Tú:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>45Labs:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Input area at the bottom
    st.markdown("---")
    
    # Determine placeholder and max tokens based on mode
    if conversation["mode"] == "question":
        placeholder = "Escribe tu pregunta sobre la rúbrica del IB..."
        max_tokens = 500
        height = 100
    else:
        placeholder = "Pega tu ensayo completo aquí para obtener feedback..."
        max_tokens = 5000
        height = 300
    
    # Chat input
    user_input = st.text_area(
        "Mensaje:",
        placeholder=placeholder,
        height=height,
        key=f"input_{conversation['id']}",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        send_button = st.button("Enviar", type="primary", use_container_width=True)
    
    with col2:
        if st.button("Limpiar", use_container_width=True):
            st.session_state[f"input_{conversation['id']}"] = ""
            st.rerun()
    
    # Handle send button
    if send_button and user_input.strip():
        if st.session_state.router.count_tokens(user_input) > max_tokens:
            st.error(f"El texto es demasiado largo. Máximo {max_tokens} tokens permitidos.")
        else:
            # Add user message
            conversation["messages"].append({
                "role": "user",
                "content": user_input
            })
            
            # Update title if this is the first message
            if len(conversation["messages"]) == 1:
                update_conversation_title(conversation["id"], user_input)
            
            # Get AI response
            with st.spinner("Analizando con base en la rúbrica oficial del IB..."):
                response = get_feedback(user_input, conversation["mode"])
                
                # Add assistant message
                conversation["messages"].append({
                    "role": "assistant",
                    "content": response
                })
            
            st.rerun()


def main():
    """Main Streamlit app."""
    
    # Initialize system on first run
    if st.session_state.vectorstore is None:
        with st.spinner("Inicializando sistema 45Labs..."):
            if not initialize_system():
                st.stop()
    
    # Render sidebar
    render_sidebar()
    
    # Render main chat interface
    render_chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.85rem;">
    <strong>45Labs</strong> - Feedback académico basado en rúbricas oficiales del IB<br>
    ✓ Usa SOLO documentación oficial del IB | ✓ Feedback criterio por criterio | ✓ Honestidad académica
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
