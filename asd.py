from __future__ import annotations

import glob
import os
import sys
from collections import deque
from pathlib import Path
from typing import AsyncIterator

from agents import Agent, Runner, function_tool, set_tracing_disabled, trace
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

from property_reco import (
    RecommendationResult,
    SessionConstraints,
    collection_count,
    load_property_catalog,
    recommend_properties_turn,
    seed_fake_properties,
)

MODEL = os.getenv("BROKER_MODEL", "gpt-5.2-2025-12-11")
KB_DB_NAME = os.getenv("KB_DB_NAME", "vector_db")
PROPERTY_DB_NAME = os.getenv("PROPERTY_DB_NAME", "property_vector_db")
PROPERTY_COLLECTION_NAME = os.getenv("PROPERTY_COLLECTION_NAME", "property_listings")
PROPERTY_SEED_CSV = os.getenv("PROPERTY_SEED_CSV", "data/properties_seed.csv")
PROPERTY_SEED_COUNT = int(os.getenv("PROPERTY_SEED_COUNT", "40"))
PROPERTY_SEED_RANDOM = int(os.getenv("PROPERTY_SEED_RANDOM", "42"))
KB_RETRIEVER_K = int(os.getenv("KB_RETRIEVER_K", "4"))

load_dotenv(override=True)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    print("OpenAI API key configured")
else:
    print("OpenAI API Key not set")

set_tracing_disabled(False)

LAST_USER_MESSAGES = deque(maxlen=5)
PROPERTY_SESSION_CONSTRAINTS = SessionConstraints()
LAST_RECOMMENDATION_RESULT: RecommendationResult | None = None

_kb_retriever = None


def _load_knowledge_docs():
    folders = glob.glob("knowledge-base/*")
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


def _build_knowledge_db_if_needed() -> None:
    force_reindex = os.getenv("FORCE_REINDEX", "0") == "1"
    if not force_reindex and os.path.exists(KB_DB_NAME):
        return

    documents = _load_knowledge_docs()
    print(f"Loaded {len(documents)} knowledge documents")
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("###", "section")])
    chunks = []
    for doc in documents:
        chunks.extend(header_splitter.split_text(doc.page_content))
    print(f"Divided into {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if os.path.exists(KB_DB_NAME):
        Chroma(persist_directory=KB_DB_NAME, embedding_function=embeddings).delete_collection()
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=KB_DB_NAME)


def _get_kb_retriever():
    global _kb_retriever
    if _kb_retriever is not None:
        return _kb_retriever

    _build_knowledge_db_if_needed()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory=KB_DB_NAME, embedding_function=embeddings)
    _kb_retriever = vectorstore.as_retriever(search_kwargs={"k": KB_RETRIEVER_K})
    return _kb_retriever


def _ensure_property_catalog_seeded():
    catalog = load_property_catalog(PROPERTY_DB_NAME, PROPERTY_COLLECTION_NAME)
    if collection_count(catalog) > 0:
        return catalog

    Path(PROPERTY_SEED_CSV).parent.mkdir(parents=True, exist_ok=True)
    seed_fake_properties(
        csv_path=PROPERTY_SEED_CSV,
        out_chroma_dir=PROPERTY_DB_NAME,
        n=PROPERTY_SEED_COUNT,
        seed=PROPERTY_SEED_RANDOM,
        collection_name=PROPERTY_COLLECTION_NAME,
    )
    return load_property_catalog(PROPERTY_DB_NAME, PROPERTY_COLLECTION_NAME)


PROPERTY_CATALOG = _ensure_property_catalog_seeded()


def get_last_recommendation_result() -> RecommendationResult | None:
    return LAST_RECOMMENDATION_RESULT


def reset_broker_session() -> None:
    global PROPERTY_SESSION_CONSTRAINTS, LAST_RECOMMENDATION_RESULT
    LAST_USER_MESSAGES.clear()
    PROPERTY_SESSION_CONSTRAINTS = SessionConstraints()
    LAST_RECOMMENDATION_RESULT = None


def prepare_recommendations(user_message: str) -> RecommendationResult:
    global PROPERTY_SESSION_CONSTRAINTS, LAST_RECOMMENDATION_RESULT
    result, updated_state = recommend_properties_turn(
        message=user_message,
        session_state=PROPERTY_SESSION_CONSTRAINTS,
        catalog=PROPERTY_CATALOG,
    )
    PROPERTY_SESSION_CONSTRAINTS = updated_state
    LAST_RECOMMENDATION_RESULT = result
    return result


@function_tool
def search_docs(question: str) -> str:
    """Search the knowledge base for visa rules, policy, and factual UAE guidance."""
    try:
        docs = _get_kb_retriever().invoke(question)
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as exc:
        return f"Knowledge-base lookup failed: {exc}"


@function_tool
def recommend_properties(requirements_text: str) -> str:
    """Return top matching properties as JSON for the provided user requirements."""
    preview_result, _ = recommend_properties_turn(
        message=requirements_text,
        session_state=PROPERTY_SESSION_CONSTRAINTS,
        catalog=PROPERTY_CATALOG,
    )
    return preview_result.model_dump_json()


BROKER_PROMPT = """
You are a knowledgeable UAE off-plan real estate broker assistant.
Use `recommend_properties` for property recommendations and `search_docs` for visa/regulatory facts.
When a recommendation snapshot is provided in system context, treat it as the source of truth and do not invent fields.
If no_match_reason exists, tell the user no properties currently match and ask which constraint to expand first.
Keep answers concise, practical, and investor-oriented.
"""

agent = Agent(
    name="Broker Agent",
    instructions=BROKER_PROMPT,
    tools=[search_docs, recommend_properties],
    model=MODEL,
)


def _build_contextual_input(
    user_message: str,
    recommendation_result: RecommendationResult,
) -> list[dict[str, str]]:
    history = [{"role": "user", "content": msg} for msg in list(LAST_USER_MESSAGES)]
    recommendation_json = recommendation_result.model_dump_json()
    system_context = (
        "Recommendation engine output for this turn:\n"
        f"{recommendation_json}\n"
        "Use these structured recommendations exactly for prices/metadata/cards."
    )
    return [{"role": "system", "content": system_context}, *history, {"role": "user", "content": user_message}]


def _trace_metadata(recommendation_result: RecommendationResult) -> dict[str, str]:
    return {
        "recent_user_messages_count": str(len(LAST_USER_MESSAGES)),
        "turn_index": str(recommendation_result.session_constraints.turn_index),
        "hard_constraints_count": str(len(recommendation_result.hard_filters_applied)),
        "candidate_count_pre_filter": str(recommendation_result.total_candidates),
        "candidate_count_post_filter": str(recommendation_result.filtered_candidates),
        "returned_card_count": str(len(recommendation_result.cards)),
        "no_match": str(bool(recommendation_result.no_match_reason)).lower(),
    }


def run_broker_agent(
    user_message: str,
    recommendation_result: RecommendationResult | None = None,
) -> str:
    clean_message = user_message.strip()
    if not clean_message:
        return ""

    result_for_turn = recommendation_result or prepare_recommendations(clean_message)
    contextual_input = _build_contextual_input(clean_message, result_for_turn)
    with trace(
        workflow_name="broker_agent_chat",
        metadata=_trace_metadata(result_for_turn),
    ):
        result = Runner.run_sync(agent, contextual_input)

    LAST_USER_MESSAGES.append(clean_message)
    return str(result.final_output)


def _extract_text_delta(raw_event_data) -> str:
    event_type = getattr(raw_event_data, "type", "") or ""
    delta = getattr(raw_event_data, "delta", None)
    text = getattr(raw_event_data, "text", None)

    if isinstance(raw_event_data, dict):
        event_type = raw_event_data.get("type", event_type) or ""
        delta = raw_event_data.get("delta", delta)
        text = raw_event_data.get("text", text)

    if isinstance(delta, str) and delta:
        if not event_type or "output_text" in event_type or "content_part" in event_type:
            return delta
    if isinstance(text, str) and text:
        if not event_type or "output_text" in event_type or "content_part" in event_type:
            return text
    return ""


async def stream_broker_agent(
    user_message: str,
    recommendation_result: RecommendationResult | None = None,
) -> AsyncIterator[str]:
    clean_message = user_message.strip()
    if not clean_message:
        return

    result_for_turn = recommendation_result or prepare_recommendations(clean_message)
    contextual_input = _build_contextual_input(clean_message, result_for_turn)
    collected = ""

    with trace(
        workflow_name="broker_agent_chat",
        metadata=_trace_metadata(result_for_turn),
    ):
        result = Runner.run_streamed(agent, contextual_input)
        async for event in result.stream_events():
            if getattr(event, "type", None) != "raw_response_event":
                continue
            chunk = _extract_text_delta(event.data)
            if chunk:
                collected += chunk
                yield chunk

    if not collected and result.final_output:
        fallback = str(result.final_output)
        if fallback:
            yield fallback

    LAST_USER_MESSAGES.append(clean_message)


if __name__ == "__main__":
    print("Broker Agent is ready. Type 'exit' to quit.")
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue
        print(f"Agent: {run_broker_agent(user_text)}")
