from openai import OpenAI
from src.config import OPENAI_API_KEY, OPENAI_MODEL, TOP_K
from src.vector_store import LocalVectorStore


def make_openai_client(api_key: str | None = None) -> OpenAI:
    key = (api_key or OPENAI_API_KEY).strip()
    if not key or key == "your_openai_api_key_here":
        raise ValueError(
            "OpenAI API key is missing. Add OPENAI_API_KEY to your .env file "
            "or enter it in the Streamlit sidebar."
        )
    return OpenAI(api_key=key)


def answer_question(
    question: str,
    store: LocalVectorStore,
    api_key: str | None = None,
    model_name: str = OPENAI_MODEL,
    top_k: int = TOP_K
) -> tuple[str, list[dict]]:
    question = question.strip()
    if not question:
        raise ValueError("Please enter a question.")

    retrieved = store.search(question, top_k=top_k)
    if not retrieved:
        return "I could not find relevant information in the uploaded PDFs.", []

    context_parts = []
    for number, item in enumerate(retrieved, start=1):
        context_parts.append(
            f"[Source {number}: {item['source']}, page {item['page']}]\n"
            f"{item['text']}"
        )
    context = "\n\n".join(context_parts)

    instructions = (
        "You are a careful PDF question-answering assistant. "
        "Answer using only the supplied context. "
        "If the context does not contain the answer, say: "
        "'I could not find this information in the uploaded documents.' "
        "Do not invent facts. Cite supporting material using "
        "[filename, page N]. Keep the answer clear and structured."
    )
    prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    client = make_openai_client(api_key)
    response = client.responses.create(
        model=model_name,
        instructions=instructions,
        input=prompt,
        temperature=0.2
    )
    return response.output_text, retrieved
