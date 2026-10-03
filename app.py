import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Configuration ⚙️
st.set_page_config(
    page_title="CodeAlpha FAQ Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS Styling 🎨
st.markdown("""
    <style>
    /* Main App Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Custom Header Styling */
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Tech Stack Badges */
    .badge {
        background-color: #ffffff22;
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 13px;
        margin: 0 4px;
        border: 1px solid #ffffff44;
    }
    
    /* Chat Message Bubbles */
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# FAQ Dataset 📝
FAQS = [
    {
        "question": "Is CodeAlpha internship real or fake? Does it ask for money?",
        "answer": "CodeAlpha is a genuine platform providing free virtual internships. We do not ask for any fees for internship, Certificate, or LOR."
    },
    {
        "question": "Will I get a valid Certificate and Letter of Recommendation (LOR)?",
        "answer": "Yes, after successfully completing and submitting your tasks, you will receive a verified Certificate and LOR with no hidden charges."
    },
    {
        "question": "How can I contact support if I get stuck?",
        "answer": "You can reach out through the official support email or community support groups mentioned in your internship offer letter."
    },
    {
        "question": "How does task selection work?",
        "answer": "Tasks are listed in your domain task document. You can select beginner-friendly tasks first and complete them at your own pace."
    },
    {
        "question": "Is it compulsory to complete all tasks?",
        "answer": "Completing all assigned tasks is recommended to receive a complete performance certificate."
    },
    {
        "question": "What is the deadline for task submission?",
        "answer": "Task submission deadlines are mentioned in your offer letter, usually 4 weeks from the start date."
    },
    {
        "question": "Can I use Python for all AI tasks?",
        "answer": "Yes, Python is the recommended language for AI and Machine Learning tasks."
    },
    {
        "question": "When will I receive my offer letter?",
        "answer": "Offer letters are issued via email within 2 to 3 days after selection."
    },
    {
        "question": "How should I submit my completed tasks?",
        "answer": "Upload your code to GitHub, share a video demo on LinkedIn tagging @CodeAlpha, and submit the links in the submission form."
    },
    {
        "question": "How are the submitted tasks evaluated?",
        "answer": "Submissions are evaluated based on code functionality, project structure, and video demonstration quality."
    }
]

# Extract Questions for Vectorization 🧠
questions = [faq["question"] for faq in FAQS]

# Initialize TF-IDF Vectorizer 📊
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

# Chatbot Response Logic 🎯
def get_bot_response(user_query):
    query_vector = vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]
    best_match_index = similarity_scores.argmax()
    best_score = similarity_scores[best_match_index]
    
    if best_score > 0.2:
        return FAQS[best_match_index]["answer"], best_score
    else:
        return "Maaf kijiye, mujhe iska jawab nahi mila. Kripya CodeAlpha support email par contact karein.", best_score

# Sidebar Navigation Panel 📌
with st.sidebar:
    st.title("📌 CodeAlpha Portal")
    st.image("https://img.icons8.com/color/96/bot.png", width=80)
    st.markdown("### **AI Internship Assistant**")
    st.write("Is chatbot ka upayog karke aap apne internship se jude sawal pooch sakte hain.")
    
    st.markdown("---")
    st.markdown("### 💡 **Quick Topics**")
    st.markdown("- 📜 Certificates & LOR")
    st.markdown("- 📅 Deadlines & Tasks")
    st.markdown("- 📤 Submission Guidelines")
    st.markdown("- 📧 Support Info")
    
    st.markdown("---")
    st.caption("Developed for CodeAlpha Internship | Task 2")

# Main Header Banner 🏷️️
st.markdown("""
    <div class="main-header">
        <h1>🤖 CodeAlpha FAQ Assistant</h1>
        <p>Your instant guide for AI Internship queries</p>
        <div>
            <span class="badge">Python</span>
            <span class="badge">TF-IDF</span>
            <span class="badge">Cosine Similarity</span>
            <span class="badge">Streamlit</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Chat History Setup 💬
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Existing Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User Input Field 📥
user_input = st.chat_input("Poochiye apna sawal (jaise: 'How to submit tasks?')...")

if user_input:
    # Display User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Compute Response
    response, score = get_bot_response(user_input)

    # Display Bot Response
    with st.chat_message("assistant"):
        st.write(response)
        if score > 0.2:
            st.caption(f"🎯 Match Confidence Score: {score*100:.1f}%")

    st.session_state.messages.append({"role": "assistant", "content": response})