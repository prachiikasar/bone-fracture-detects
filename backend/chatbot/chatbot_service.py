import os
try:
    import google.generativeai as genai
except ImportError:
    genai = None
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
if genai:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class ChatbotService:
    @staticmethod
    def get_response(user_message, history=None):
        # Basic input sanitation
        if not isinstance(user_message, str) or not user_message.strip():
            return "Please enter a message. I can help with analysis steps, accuracy, or reports."
        user_message_lower = user_message.lower().strip()
        history = history or []
        
        # 1. Try Gemini AI first if API Key is available
        api_key = os.getenv("GEMINI_API_KEY")
        if genai and api_key and api_key != "your_gemini_api_key_here" and len(api_key) > 5:
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                
                # System context to guide the AI
                system_context = (
                    "You are Deep learning classification of fracture bones using ViT Assistant, a specialized AI for a bone fracture detection system. "
                    "The system uses ResNet50 and Vision Transformers to detect fractures in Elbow, Hand, and Shoulder X-rays. "
                    "Accuracy is 92%. You can answer questions about the system, its usage, and general bone health. "
                    "You should also be able to engage in normal conversation while maintaining your persona. "
                    "Always include a disclaimer that you are not a doctor and results should be verified by a professional."
                )
                # Build a simple conversation context from recent history
                # Expect history as list of dicts: [{sender: 'user'|'bot', text: str}]
                convo_snippets = []
                for turn in history[-6:]:  # Keep last 6 messages for brevity
                    sender = (turn.get("sender") or "").strip().lower()
                    text = (turn.get("text") or "").strip()
                    if not text:
                        continue
                    if sender == "user":
                        convo_snippets.append(f"User: {text}")
                    else:
                        convo_snippets.append(f"Assistant: {text}")
                convo_block = "\n".join(convo_snippets)
                prompt = (
                    f"{system_context}\n\n"
                    f"Conversation so far:\n{convo_block}\n\n"
                    f"User: {user_message}\n"
                    f"Assistant:"
                )
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"Gemini API Error: {e}")
                # Fall through to hardcoded rules if AI fails
        
        # 2. Intelligent Fallback Logic (Pseudo-AI)
        # This uses a comprehensive dictionary for "proper" replies without an LLM
        
        knowledge_base = {
            "greetings": {
                "keywords": ["hi", "hello", "hey", "hola", "greetings"],
                "response": "Hello! I'm your Deep learning classification of fracture bones using ViT Assistant. I can help you analyze X-rays, explain results, or chat about bone health. How can I assist you today?"
            },
            "wellbeing": {
                "keywords": ["how are you", "how's it going", "how are things"],
                "response": "I'm functioning perfectly! Ready to help you with your bone fracture analysis. How about you?"
            },
            "help": {
                "keywords": ["help", "what can you do", "capabilities", "guide", "instructions"],
                "response": "I can guide you through uploading X-rays, explain model confidence and accuracy, describe supported bones (Elbow, Hand, Shoulder), and help download a summary report. Ask me about accuracy, how to interpret results, or where to find features."
            },
            "disclaimer": {
                "keywords": ["disclaimer", "doctor", "medical", "diagnosis"],
                "response": "I provide research and educational information only. I am not a doctor, and this system does not give a clinical diagnosis. Always consult a qualified medical professional for decisions."
            },
            "identity": {
                "keywords": ["who are you", "what are you", "your name"],
                "response": "I am Deep learning classification of fracture bones using ViT Assistant, a specialized digital companion designed to help medical professionals detect and document bone fractures using deep learning."
            },
            "upload_process": {
                "keywords": ["upload", "how to start", "analyze", "image"],
                "response": "To start an analysis, go to the 'Overview' tab, drag and drop an X-ray image (JPG, PNG) into the box, and click 'Analyze Fracture'. Our ResNet50 models will then process the image."
            },
            "accuracy": {
                "keywords": ["accurate", "reliable", "precision", "accuracy"],
                "response": "Our system achieves a 92.4% accuracy rate on benchmark datasets like MURA. However, it is a diagnostic aid and not a final medical diagnosis. Clinical correlation is always required."
            },
            "models": {
                "keywords": ["model", "vit", "resnet", "architecture", "how it works"],
                "response": "Under the hood, we use ResNet50 and Vision Transformer (ViT) ensembles to classify bone regions and detect fracture patterns. Outputs include fracture status, confidence, and non-diagnostic guidance."
            },
            "results_info": {
                "keywords": ["confidence", "score", "percentage", "what does it mean"],
                "response": "The confidence score shows how certain the AI is. Scores above 75% indicate strong evidence, while lower scores (like 'Uncertain') suggest a manual review is strongly recommended."
            },
            "fracture_definition": {
                "keywords": ["what is a fracture", "broken bone", "bone break"],
                "response": "A fracture is a medical condition where there is a break in the continuity of the bone. It can range from subtle hairline fractures to complete displacements."
            },
            "supported_parts": {
                "keywords": ["which bones", "elbow", "hand", "shoulder", "parts"],
                "response": "Currently, Deep learning classification of fracture bones using ViT is optimized for Elbow, Hand, and Shoulder X-rays. We plan to add support for wrists and ankles in future updates."
            },
            "report_generation": {
                "keywords": ["report", "pdf", "generate", "download"],
                "response": "After analysis, click the 'Download Report' button in the results panel to get a professional PDF summing up the AI's findings and confidence scores."
            },
            "privacy": {
                "keywords": ["privacy", "data", "store", "saved"],
                "response": "Uploaded images are processed for analysis; avoid sharing sensitive personal data. Some metadata and results may be stored to improve the experience. Always follow local data policies."
            },
            "thanks": {
                "keywords": ["thanks", "thank you", "great", "awesome", "perfect"],
                "response": "You're very welcome! I'm glad I could help. Is there anything else you'd like to know about Deep learning classification of fracture bones using ViT?"
            }
        }

        # Find the best match
        best_match = None
        for category, data in knowledge_base.items():
            if any(keyword in user_message_lower for keyword in data["keywords"]):
                return data["response"]

        # If no specific match, try a more conversational generic response
        if len(user_message_lower) < 15:
            return "I'm here to help! Ask about uploading images, model accuracy, or interpreting results. How can I assist you right now?"

        return (
            "That's an interesting question. While I'm currently in 'Optimized Mode' (waiting for a full AI API key), "
            "I'm specifically trained on Deep learning classification of fracture bones using ViT documentation. I can tell you about our ResNet50/ViT architecture, "
            "how to generate reports, or explain confidence scores. Try asking something like 'How accurate is the system?'"
        )
