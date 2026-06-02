"""
TASK 2: Chatbot for FAQs
CodeAlpha AI Internship
NLP-based FAQ chatbot using TF-IDF + Cosine Similarity (sklearn)
"""

import re
import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── FAQ Knowledge Base ────────────────────────────────────────────────────────
FAQ = {
    "What is artificial intelligence?":
        "Artificial Intelligence (AI) is the simulation of human intelligence by machines, "
        "enabling them to learn, reason, and solve problems.",

    "What is machine learning?":
        "Machine Learning is a subset of AI that allows systems to learn and improve from "
        "experience without being explicitly programmed, using algorithms and statistical models.",

    "What is deep learning?":
        "Deep Learning is a sub-field of machine learning that uses multi-layered neural networks "
        "to model complex patterns in large amounts of data.",

    "What is a neural network?":
        "A neural network is a computing model inspired by the human brain, made up of layers of "
        "interconnected nodes (neurons) that process data and learn patterns.",

    "What is natural language processing?":
        "NLP (Natural Language Processing) is a branch of AI that enables machines to understand, "
        "interpret, and respond to human language in a meaningful way.",

    "What is computer vision?":
        "Computer Vision is an AI field that trains machines to interpret and understand visual "
        "information from images and videos, like humans do with their eyes.",

    "What is Python?":
        "Python is a versatile, high-level programming language widely used in data science, "
        "AI, web development, and automation due to its simple syntax and rich libraries.",

    "What are the best Python libraries for AI?":
        "Popular AI libraries in Python include TensorFlow, PyTorch, scikit-learn, Keras, "
        "NumPy, Pandas, OpenCV, NLTK, and spaCy.",

    "What is overfitting?":
        "Overfitting occurs when a model learns the training data too well, including noise, "
        "causing it to perform poorly on new, unseen data.",

    "What is underfitting?":
        "Underfitting happens when a model is too simple to capture the patterns in the data, "
        "resulting in poor performance on both training and test data.",

    "What is a dataset?":
        "A dataset is a structured collection of data used to train, validate, and test "
        "machine learning models.",

    "What is supervised learning?":
        "Supervised learning is a type of ML where the model is trained on labeled data — "
        "the algorithm learns to map inputs to the correct outputs.",

    "What is unsupervised learning?":
        "Unsupervised learning involves training models on unlabeled data, letting the algorithm "
        "discover hidden patterns or groupings on its own.",

    "What is reinforcement learning?":
        "Reinforcement learning is a type of ML where an agent learns to make decisions by "
        "interacting with an environment and receiving rewards or penalties.",

    "What is a large language model?":
        "A Large Language Model (LLM) is an AI model trained on massive text datasets to "
        "understand and generate human-like text. Examples: GPT-4, Claude, Gemini.",

    "What is CodeAlpha?":
        "CodeAlpha is a leading software development company dedicated to driving innovation "
        "and excellence across emerging technologies, offering internship programs in AI, "
        "web development, and more.",

    "How do I submit my internship tasks?":
        "Submit your completed tasks through the submission form shared in your WhatsApp group. "
        "Upload source code to a GitHub repository named CodeAlpha_ProjectName and post a "
        "LinkedIn video explanation.",

    "How many tasks do I need to complete?":
        "You need to complete a minimum of 2 or 3 tasks out of 4 to be eligible for the "
        "internship completion certificate.",

    "What is YOLO?":
        "YOLO (You Only Look Once) is a real-time object detection algorithm that processes "
        "images in one pass, making it extremely fast and efficient.",

    "What is OpenCV?":
        "OpenCV (Open Source Computer Vision Library) is a powerful Python/C++ library used "
        "for image processing, video analysis, and computer vision tasks.",
}

questions = list(FAQ.keys())
answers   = list(FAQ.values())

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(questions)

GREETINGS = {"hi", "hello", "hey", "greetings", "good morning", "good evening"}
EXITS     = {"bye", "exit", "quit", "goodbye", "see you"}

# ── Chatbot engine ────────────────────────────────────────────────────────────
def get_response(user_input: str) -> str:
    clean = user_input.strip().lower()
    if clean in GREETINGS:
        return "👋 Hello! I'm your AI FAQ Bot. Ask me anything about AI, ML, or this internship!"
    if clean in EXITS:
        return "👋 Goodbye! Best of luck with your internship!"
    if not clean:
        return "Please type a question."

    query_vec = vectorizer.transform([user_input])
    scores    = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx  = int(np.argmax(scores))
    best_score = scores[best_idx]

    if best_score < 0.15:
        return ("❓ I'm not sure about that. Try asking about AI, ML, NLP, neural networks, "
                "Python, or your CodeAlpha internship.")

    confidence = f"{best_score * 100:.1f}%"
    return (f"🤖 **{questions[best_idx]}**\n\n{answers[best_idx]}\n\n"
            f"_(Confidence: {confidence})_")

# ── GUI ────────────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("🤖 AI FAQ Chatbot — CodeAlpha")
root.geometry("700x580")
root.configure(bg="#1e1e2e")
root.resizable(False, False)

BG     = "#1e1e2e"
FG     = "#cdd6f4"
ACCENT = "#89b4fa"
BOT_C  = "#a6e3a1"
USER_C = "#fab387"
FONT   = ("Segoe UI", 11)

tk.Label(root, text="🤖 AI FAQ Chatbot", font=("Segoe UI", 16, "bold"),
         bg=BG, fg=ACCENT).pack(pady=(14, 2))
tk.Label(root, text="Ask me about AI, ML, NLP, Python, or your CodeAlpha internship",
         font=("Segoe UI", 9), bg=BG, fg="#6c7086").pack()

chat_frame = tk.Frame(root, bg=BG)
chat_frame.pack(fill="both", expand=True, padx=20, pady=10)

chat_log = scrolledtext.ScrolledText(
    chat_frame, state="disabled", wrap="word",
    bg="#313244", fg=FG, font=FONT, relief="flat", padx=10, pady=10
)
chat_log.pack(fill="both", expand=True)
chat_log.tag_config("user",  foreground=USER_C, font=("Segoe UI", 11, "bold"))
chat_log.tag_config("bot",   foreground=BOT_C)
chat_log.tag_config("label", foreground="#6c7086", font=("Segoe UI", 9))


def append_message(sender, message):
    chat_log.config(state="normal")
    chat_log.insert(tk.END, f"{sender}\n", "label")
    tag = "user" if "You" in sender else "bot"
    chat_log.insert(tk.END, f"{message}\n\n", tag)
    chat_log.config(state="disabled")
    chat_log.see(tk.END)


def send_message(event=None):
    user_text = entry.get().strip()
    if not user_text:
        return
    entry.delete(0, tk.END)
    append_message("You:", user_text)
    response = get_response(user_text)
    append_message("Bot:", response)


entry_frame = tk.Frame(root, bg=BG)
entry_frame.pack(fill="x", padx=20, pady=(0, 14))

entry = tk.Entry(entry_frame, font=FONT, bg="#313244", fg=FG,
                 insertbackground=FG, relief="flat")
entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 8))
entry.bind("<Return>", send_message)

tk.Button(entry_frame, text="Send ➤", command=send_message, font=FONT,
          bg=ACCENT, fg="#1e1e2e", padx=14, pady=6, relief="flat",
          cursor="hand2").pack(side="right")

# Welcome message
append_message("Bot:", "👋 Hello! I'm your AI FAQ Chatbot. Ask me any question about "
               "Artificial Intelligence, Machine Learning, or your CodeAlpha internship!")

root.mainloop()
