from fastapi import FastAPI
from pydantic import BaseModel

from rag_chain import answer_question


# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(title="Priyems AI Assistant")


# =====================================================
# REQUEST MODEL
# =====================================================
class ChatRequest(BaseModel):
    question: str


# =====================================================
# HEALTH CHECK
# =====================================================
@app.get("/")
def health():
    return {"status": "ok"}


# =====================================================
# SIMPLE CONVERSATION / INTENT HANDLER
# =====================================================
def handle_conversation(user_input: str) -> str:
    q = user_input.strip().lower()

    # -------------------------------
    # CONFIRMATION INTENT
    # -------------------------------
    if q in ["yes", "yes please", "okay", "ok", "sure"]:
        return (
            "Great! 😊\n\n"
            "For further assistance, please contact our support team:\n"
            "📧 Email: contact@priyems.com\n"
            "📱 WhatsApp: +1 (248) 794-8293"
        )
    if q in ["no", "no thanks", "not now", "maybe later"]:
        return (
            "Thank you for visiting! If you have any questions later or need help with our products, "
            "feel free to reach out anytime."
        )

    # -------------------------------
    # QUANTITY / ORDER INTENT
    # -------------------------------
    if "dosa batter" in q and any(word in q for word in ["need", "want", "order", "buy"]):
        return (
            "Thank you for your interest in our dosa batter! 😊\n\n"
            "We offer a Regular Dosa Batter packet weighing 900g, "
            "which serves approximately 12–14 dosas.\n\n"
            "Would you like to proceed with this variant?"
        )

    # -------------------------------
    # DEFAULT → RAG KNOWLEDGE ANSWER
    # -------------------------------
    return answer_question(user_input)


# =====================================================
# CHAT ENDPOINT
# =====================================================
@app.post("/chat")
def chat(req: ChatRequest):
    answer = handle_conversation(req.question)
    return {
        "question": req.question,
        "answer": answer
    }
