from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


# =====================================================
# PROFESSIONAL CUSTOMER FALLBACK (NO INTERNAL LANGUAGE)
# =====================================================
FALLBACK_MESSAGE = (
    "I’m sorry, I don’t have that information at the moment.\n\n"
    "For further assistance, please contact our support team:\n"
    "📧 Email: contact@priyems.com\n"
    "📱 WhatsApp: +1 (248) 794-8293"
)


# =====================================================
# CUSTOMER-FACING PROMPT (NO META / NO SOURCE LEAK)
# =====================================================
PROMPT = """
You are a customer-facing assistant for Priyems.

Rules:
- Answer clearly and confidently as a brand representative.
- Do NOT mention documents, context, sources, or phrases like
  "according to the context", "based on the document", or similar.
- Do NOT explain how you know the answer.
- Use only the information provided to you.
- If the information is not known, respond politely without guessing.

<context>
{context}
</context>

Question: {question}

Answer:
"""


# =====================================================
# LOAD VECTOR STORE
# =====================================================
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = FAISS.load_local(
    "vector_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# =====================================================
# LLM (DETERMINISTIC)
# =====================================================
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0
)

prompt = ChatPromptTemplate.from_template(PROMPT)


# =====================================================
# QUESTION NORMALIZATION (INTENT AWARE)
# =====================================================
def normalize_question(question: str) -> str:
    q = question.lower()
    # Company location / address
    if any(word in q for word in ["where", "located", "location", "address", "head office", "headquarters"]):
        return "Priyems company address Wixom Michigan"

    # Catalog / existence intent
    if "product catalog" in q or "full catalog" in q:
        return "product catalog with ingredients"

    # Special idli
    if "special idli" in q:
        return "instructions for special idli recipe"

    # Generic idli
    if "idli" in q and "special" not in q:
        return "instructions for idli recipe"

    return question


# =====================================================
# MAIN ANSWER FUNCTION (SAFE + BRAND-CLEAN)
# =====================================================
def answer_question(question: str) -> str:
    normalized_question = normalize_question(question)

    # Retrieve relevant chunks
    results = vectorstore.similarity_search_with_score(
        normalized_question,
        k=5
    )

    # No retrieval → professional fallback
    if not results:
        return FALLBACK_MESSAGE

    # Sort by similarity (best first)
    results = sorted(results, key=lambda x: x[1])

    docs = [doc.page_content for doc, _ in results[:3]]

    if not docs:
        return FALLBACK_MESSAGE

    context = "\n\n".join(docs)

    # -----------------------------------------------
    # YES / NO – CATALOG EXISTENCE QUESTIONS
    # -----------------------------------------------
    if "catalog" in question.lower():
        return (
            "Yes. We have a comprehensive product catalog with detailed ingredient information. "
            "Our catalog includes ready-to-cook batters, ready-to-eat curries, ready-to-cook bases, "
            "and ready-to-eat bases, with ingredients clearly listed for each product."
        )

    # -----------------------------------------------
    # STANDARD CONTEXT-BASED ANSWER
    # -----------------------------------------------
    response = llm.invoke(
        prompt.format(
            context=context,
            question=question
        )
    )

    if not response or not response.content.strip():
        return FALLBACK_MESSAGE

    answer = response.content.strip()

    # -----------------------------------------------
    # SAFETY CLEAN-UP (META LANGUAGE REMOVAL)
    # -----------------------------------------------
    banned_phrases = [
        "according to the provided context",
        "based on the provided context",
        "based on the context",
        "according to the context",
        "the document states",
        "the documents state",
    ]

    for phrase in banned_phrases:
        answer = answer.replace(phrase, "").strip()

    return answer
