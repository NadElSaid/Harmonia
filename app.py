import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import gradio as gr
import tempfile
import shutil

# Set environment variable for protobuf compatibility
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Get NVIDIA API Key from environment variables
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not found in environment variables. Please set it in your secrets.")

# Set NVIDIA API Key
os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY

# Initialize NVIDIA LLM
print("Initializing NVIDIA LLM...")
llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
print("LLM initialized successfully!")

CONDITIONED_PROMPT = """
You are Harmonia, a compassionate and inclusive mental health and emotional support guide. Your role is to provide a safe, non-judgmental space for individuals to express their feelings, explore their thoughts, and receive both emotional validation and practical guidance. You are here to support people through a wide range of mental health challenges, from everyday stress to more complex emotional struggles.

**Core Principles for Short Responses:**
1. **Be Concise**: Respond in 2-3 sentences maximum.
2. **Emotional Validation**: Acknowledge the user's feelings briefly.
3. **Practical Guidance**: Offer 1-2 actionable steps or coping strategies.
4. **Empathy**: Maintain a warm and empathetic tone.
5. **Crisis Handling**: If the user is in immediate danger, prioritize clear and direct instructions.

**Response Structure:**
1. **Acknowledge Feelings**: "I hear how [challenging/overwhelming/difficult] this is for you."
2. **Offer Support**: "Here's something you can try: [1-2 strategies]."
3. **Encourage Next Steps**: "You're not alone. Let's take this one step at a time."

**Crisis Protocol:**
- If the user is in immediate danger, respond with:
  - "Your safety is the priority. Please contact [crisis hotline] or [emergency services] right away."
  - Provide grounding techniques if appropriate.

**Examples of Short Responses:**
- "I hear how overwhelming this feels. Try taking deep breaths and focusing on one small step at a time. You're not alone."
- "It sounds like you're going through a tough time. Let's talk about what might help you feel safer. You've got this."
- "Your feelings are valid. Here's a strategy: write down your thoughts to help process them. I'm here to support you."

**Special Considerations:**
- If the user thanks you, respond: "I'm here anytime you need support. Don't hesitate to reach out."
- If the user says hello, respond: "Hello! How can I support you today? Feel free to share what's on your mind."

**Remember**: Always adapt the response based on the urgency and specific needs of the situation while keeping it concise and supportive.

**Chat History Context:**
{chat_history}

**Retrieved Information:**
{context}

**Current situation:** {question}

**Response:**
"""

# Load PDF
def load_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Warning: PDF file '{pdf_path}' not found. Creating a basic knowledge base.")
        return []
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def create_basic_knowledge_base():
    """Create a basic mental health knowledge base if PDF is not available"""
    basic_content = [
        "Mental health is just as important as physical health. It affects how we think, feel, and behave.",
        "Common mental health conditions include anxiety, depression, bipolar disorder, and PTSD.",
        "Stress management techniques include deep breathing, mindfulness, regular exercise, and adequate sleep.",
        "It's important to seek professional help when mental health challenges become overwhelming.",
        "Building a support network of friends, family, or support groups can be very beneficial.",
        "Self-care practices like meditation, journaling, and hobbies can improve mental wellbeing.",
        "Crisis resources include national suicide prevention lifeline: 988 in the US.",
        "Therapy types include cognitive behavioral therapy (CBT), dialectical behavior therapy (DBT), and others.",
        "Mindfulness and grounding techniques can help manage anxiety and panic attacks.",
        "Regular routine, healthy eating, and social connections support good mental health.",
        "Breathing exercises: Try the 4-7-8 technique - inhale for 4, hold for 7, exhale for 8.",
        "Grounding techniques: Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste.",
        "Progressive muscle relaxation can help reduce physical tension and anxiety.",
        "Journaling can help process emotions and identify patterns in thoughts and feelings.",
        "Exercise releases endorphins which naturally improve mood and reduce stress.",
        "Sleep hygiene is crucial for mental health - maintain regular sleep schedule and create calming bedtime routine.",
        "Social connections are vital for mental wellbeing - reach out to friends, family, or support groups.",
        "Mindfulness meditation can help stay present and reduce anxiety about future or past events.",
        "Setting boundaries is important for mental health - it's okay to say no to protect your wellbeing.",
        "Professional help is available - therapists, counselors, and psychiatrists can provide specialized support."
    ]
    
    # Create mock documents
    from langchain.schema import Document
    documents = [Document(page_content=content, metadata={"source": "basic_knowledge"}) 
                for content in basic_content]
    return documents

def create_mental_health_assistant():
    # Try to load PDF, fall back to basic knowledge if not available
    pdf_path = "Harmonia.pdf"
    documents = load_pdf(pdf_path)
    
    if not documents:
        documents = create_basic_knowledge_base()
    
    # Create text splits
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} document chunks")

    # Create embeddings and vectorstore using NVIDIA embeddings and Chroma
    print("Initializing NVIDIA embeddings...")
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
    print("Embeddings initialized successfully!")
    
    # Use Chroma with persistent directory
    print("Creating Chroma vectorstore...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Vectorstore created successfully!")

    # Initialize memory for chat history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    # Create prompt template
    PROMPT = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=CONDITIONED_PROMPT
    )

    # Create chain with memory and custom prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=False,
        verbose=True
    )

    return qa_chain

def create_interface(qa_chain):
    def respond(message, history):
        try:
            print(f"User message: {message}")
            response = qa_chain({"question": message})
            print(f"Assistant response: {response['answer']}")
            return response["answer"]
        except Exception as e:
            error_msg = f"I'm sorry, I encountered an error. Please try again. Error: {str(e)}"
            print(f"Error: {error_msg}")
            return error_msg

    # Custom CSS for the background image and compact layout
    custom_css = """
    body {
        background-image: url('https://images.unsplash.com/photo-1506905925346-21bda4d32df4?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        margin: 0;
        overflow: auto;
    }
    .gradio-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        max-width: 700px;
        width: 100%;
        max-height: 90vh;
        overflow-y: auto;
        margin: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    .chat-interface {
        padding: 15px;
    }
    .chat-area {
        max-height: 400px;
        overflow-y: auto;
        padding: 15px;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.9);
        margin-top: 15px;
    }
    .examples {
        max-height: 150px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.9);
        margin-top: 15px;
    }
    .chatbot-icon {
        width: 50px;
        height: 50px;
    }
    h1 {
        color: #4A90E2;
        margin-bottom: 10px;
    }
    """
    
    # Create the chat interface
    chat_interface = gr.ChatInterface(
        fn=respond,
        title="",
        description="""
        <div style="text-align: center;">
            <h1>🌸 Harmonia 🌸</h1>
            <p><strong>Your peace, your path, your Harmonia</strong></p>
            <p>Hello, I'm Harmonia. I'm here to provide a safe, non-judgmental space for you to express your feelings, explore your thoughts, and receive support.</p>
            <p>Whether you're dealing with stress, anxiety, relationship issues, or just need someone to talk to, I'm here to listen and help you navigate your emotions.</p>
            <p>Together, we'll explore what's on your mind and work towards finding peace and clarity. I remember our conversation history, so feel free to continue where we left off.</p>
            <p><strong>🌿 Whenever you're ready, we can begin. 💬</strong></p>
            <p style="font-size: 0.9em; color: #666;"><em>Note: This is a support tool and not a replacement for professional mental health care. If you're in crisis, please contact emergency services or a crisis hotline immediately.</em></p>
        </div>
        """,
        examples=[
            "I've been feeling really overwhelmed lately...",
            "I'm struggling with my relationships...",
            "I'm not sure how to deal with my anxiety...",
            "I've been feeling really low and don't know why...",
            "Can you help me with some breathing exercises?",
            "I'm having trouble sleeping because of stress...",
            "How can I practice mindfulness?",
            "I need help setting boundaries with people...",
        ],
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="teal",
        ),
        css=custom_css,
        chatbot=gr.Chatbot(
            height=400,
            show_label=False,
            container=False,
            show_copy_button=True,
        )
    )
    return chat_interface

# Usage
if __name__ == "__main__":
    try:
        print("Starting Harmonia Mental Health Assistant...")
        print("Using NVIDIA Llama 3.1 405B model...")
        
        # Create assistant with mental health-focused memory
        qa_chain = create_mental_health_assistant()
        
        # Create and launch interface
        chat_interface = create_interface(qa_chain)
        print("Harmonia Mental Health Assistant ready!")
        print("If you don't have a 'Harmonia.pdf' file, the app will use a basic knowledge base.")
        print("Chat history will be maintained throughout the conversation.")
        
        chat_interface.launch(share=True, debug=True)
        
    except Exception as e:
        print(f"Error starting application: {e}")
        print("Please ensure you have the required packages installed:")
        print("pip install langchain langchain-community langchain-nvidia-ai-endpoints chromadb gradio")