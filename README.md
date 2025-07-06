# Harmonia - Mental Health Support Chatbot🌸
Harmonia is a compassionate and inclusive mental health and emotional support chatbot. It provides a safe, non-judgmental space for individuals to express their feelings, explore their thoughts, and receive both emotional validation and practical guidance. Harmonia is designed to support users through a wide range of mental health challenges, from everyday stress to more complex emotional struggles.
# 📌Features
- Emotional Validation: Acknowledge and validate user emotions.
- Practical Guidance: Offer actionable steps and coping strategies.
- Crisis Support: Provide emergency contact information and grounding techniques.
- Personalized Responses: Tailored responses based on user input and context.
# 🚀How to Set Up and Run the Project
- **Prerequisites:**
  - Python 3.8 or higher: Ensure Python is installed on your system.
  - NVIDIA API Key: Obtain an NVIDIA API key to use the Llama 3.1 model
- **Step 1: Clone the Repository:**
  
  ``` git clone https://github.com/NadElSaid/Harmonia.git ```
  
     ``` cd Harmonia ```
- **Step 2: Install Dependencies:**
   Install the required Python packages using pip
  
    ```pip install -r requirements.txt```
- **Step 3: Set Up Google API Key:**
   replace "NVIDIA API key" in the code with your actual API key.
- **Step 4: Add the PDF File**
  
  Place the Harmonia.pdf file in the root directory of the project. This file is used to enhance the chatbot's knowledge base.
- **Step 5: Run the Application**
  
  Run the application locally using Gradio:

  ``` python app.py ```
# 📦Dependencies and Tools Used
- **Python Libraries:**
  - **langchain:** For building the conversational chain and integrating with the Gemini model.
  - **langchain-nvidia-ai-endpoints:** For using NVIDIA's AI models.
  - **gradio:** For creating the user interface.
  - **PyPDF2:** For loading and processing the PDF file.
  - **chromadb:** For vector storage and retrieval.
- **Tools:**
  - **NVIDIA AI Endpoints:** For accessing powerful AI models.
  - **Hugging Face Spaces:** For deploying the application (optional).
# 🤖Try it out
- [https://huggingface.co/spaces/Nadaazakaria/Harmonia](https://huggingface.co/spaces/Nadaazakaria/Harmonia)

