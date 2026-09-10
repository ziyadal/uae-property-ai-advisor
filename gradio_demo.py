from __future__ import annotations

from html import escape
import os

import gradio as gr

from asd import (
    get_last_recommendation_result,
    prepare_recommendations,
    reset_broker_session,
    run_broker_agent,
    stream_broker_agent,
)
from property_reco.types import RecommendationResult


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600&display=swap');

:root {
  --bg-a: #f3f5ef;
  --bg-b: #d5e3db;
  --ink: #0f2a24;
  --ink-soft: #3a5048;
  --card: #ffffffde;
  --accent: #007f6d;
  --accent-2: #e86f2f;
  --line: #c7d8cd;
}

body, .gradio-container {
  background:
    radial-gradient(circle at 12% 10%, #fff6de 0%, transparent 35%),
    radial-gradient(circle at 90% 80%, #d8fff3 0%, transparent 45%),
    linear-gradient(145deg, var(--bg-a), var(--bg-b));
  font-family: 'Space Grotesk', sans-serif;
  color: var(--ink);
}

#shell {
  max-width: 980px;
  margin: 24px auto;
  border-radius: 24px;
  border: 1px solid var(--line);
  background: var(--card);
  box-shadow: 0 18px 40px rgba(20, 50, 45, 0.12);
  overflow: hidden;
}

#header {
  padding: 26px 28px 14px;
  background: linear-gradient(120deg, #fef6e7 0%, #e7f7ef 52%, #e8f1fb 100%);
  border-bottom: 1px solid var(--line);
  animation: rise 500ms ease-out;
}

#title {
  font-family: 'Fraunces', serif;
  font-size: clamp(1.5rem, 2.6vw, 2.2rem);
  color: #173931;
  margin: 0;
}

#subtitle {
  margin-top: 6px;
  color: var(--ink-soft);
}

#chips { padding: 10px 22px 0; }
.chip button {
  border-radius: 999px !important;
  border: 1px solid #9cc9be !important;
  background: #f7fffc !important;
  color: #1e4d42 !important;
  transition: all .2s ease;
}
.chip button:hover {
  border-color: #2e8e7f !important;
  transform: translateY(-1px);
}

#chatwrap { padding: 12px 22px 6px; }
#chatbox {
  border-radius: 16px;
  border: 1px solid var(--line);
  background: #fcfdfb;
  min-height: 360px;
}

#controls {
  padding: 10px 22px 24px;
}

#reco-wrap {
  padding: 6px 22px 22px;
}

#reco-board {
  border: 1px solid var(--line);
  border-radius: 16px;
  background: #f8fbfa;
  padding: 14px;
}

.reco-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
}

.prop-card {
  border: 1px solid #c8ddd2;
  border-radius: 14px;
  overflow: hidden;
  background: #ffffff;
}

.prop-card img {
  width: 100%;
  height: 140px;
  object-fit: cover;
  display: block;
}

.prop-body {
  padding: 10px;
}

.prop-title {
  font-weight: 700;
  font-size: 0.96rem;
  margin-bottom: 6px;
}

.price-pill {
  display: inline-block;
  font-weight: 700;
  color: #004f44;
  background: #def8f2;
  padding: 3px 9px;
  border-radius: 999px;
  margin-bottom: 6px;
}

.meta-row {
  font-size: 0.83rem;
  color: #37544b;
  margin: 2px 0;
}

.no-match {
  border: 1px dashed #c18964;
  background: #fff5ef;
  border-radius: 12px;
  padding: 10px;
  color: #6d2f10;
}

#send button {
  background: linear-gradient(130deg, var(--accent), #0a6a5c) !important;
  color: #fff !important;
  border: none !important;
}
#clear button {
  background: #fff7f1 !important;
  color: #8a3f12 !important;
  border: 1px solid #f0b894 !important;
}

@keyframes rise {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 768px) {
  #shell { margin: 10px; border-radius: 18px; }
  #chatbox { min-height: 58vh; }
}
"""


def _currency(value: float) -> str:
    return f"AED {value:,.0f}"


def render_recommendation_cards(result: RecommendationResult | None) -> str:
    if result is None:
        return (
            "<div class='no-match'>No recommendations yet. "
            "Start by sharing your budget, location, and bedroom needs.</div>"
        )

    if not result.cards:
        suggestions = "".join(
            f"<li>{escape(item)}</li>" for item in result.next_relaxation_suggestions
        )
        return (
            "<div class='no-match'>"
            f"<div><strong>{escape(result.no_match_reason or 'No matches found.')}</strong></div>"
            "<div style='margin-top:6px'>Try expanding one of these:</div>"
            f"<ul>{suggestions}</ul>"
            "</div>"
        )

    cards_html = []
    for card in result.cards:
        amenities = ", ".join(card.amenities[:4]) if card.amenities else "No amenities listed"
        image_url = card.image_url or "https://picsum.photos/seed/property-fallback/960/640"
        score_rows = "".join(
            f"<li>{escape(metric.replace('_', ' ').title())}: {value:.2f}</li>"
            for metric, value in card.score_breakdown.items()
        )
        cards_html.append(
            f"""
            <article class="prop-card">
              <img src="{escape(image_url)}" alt="{escape(card.title)}"/>
              <div class="prop-body">
                <div class="prop-title">{escape(card.title)}</div>
                <div class="price-pill">{_currency(card.price_aed)}</div>
                <div class="meta-row">{card.beds} bed | {card.baths} bath | {card.area_sqft:,.0f} sqft</div>
                <div class="meta-row">{escape(card.community)}, {escape(card.city)}</div>
                <div class="meta-row">Handover: {escape(card.handover_date)} | {escape(card.status)}</div>
                <div class="meta-row"><strong>Why:</strong> {escape(card.match_reason)}</div>
                <details>
                  <summary>More details</summary>
                  <div class="meta-row">Developer: {escape(card.developer)}</div>
                  <div class="meta-row">Amenities: {escape(amenities)}</div>
                  <div class="meta-row"><a href="{escape(card.detail_url)}" target="_blank">Listing page</a></div>
                  <ul>{score_rows}</ul>
                </details>
              </div>
            </article>
            """
        )

    return "<div class='reco-grid'>" + "".join(cards_html) + "</div>"


async def respond_stream(user_message: str, history: list[dict]):
    if not user_message or not user_message.strip():
        yield "", history, render_recommendation_cards(get_last_recommendation_result())
        return

    clean_user_message = user_message.strip()
    recommendation_result = prepare_recommendations(clean_user_message)
    cards_html = render_recommendation_cards(recommendation_result)

    updated = history + [
        {"role": "user", "content": clean_user_message},
        {"role": "assistant", "content": ""},
    ]
    yield "", updated, cards_html

    try:
        async for chunk in stream_broker_agent(
            clean_user_message,
            recommendation_result=recommendation_result,
        ):
            updated[-1]["content"] += chunk
            yield "", updated, cards_html
        if not updated[-1]["content"]:
            updated[-1]["content"] = str(
                run_broker_agent(clean_user_message, recommendation_result=recommendation_result)
            )
            yield "", updated, cards_html
    except Exception:
        updated[-1]["content"] = (
            "Sorry, I could not complete that request. Please try again or adjust your criteria."
        )
        yield "", updated, cards_html


async def quick_prompt_1(history: list[dict]):
    async for output in respond_stream(
        "What are the key requirements for a UAE Golden Visa through property investment?",
        history,
    ):
        yield output


async def quick_prompt_2(history: list[dict]):
    async for output in respond_stream(
        "How do escrow accounts protect buyers in off-plan projects?", history
    ):
        yield output


async def quick_prompt_3(history: list[dict]):
    async for output in respond_stream(
        "Can off-plan purchases qualify for the Golden Visa, and what are the limits?",
        history,
    ):
        yield output


def clear_chat():
    reset_broker_session()
    return [], render_recommendation_cards(None)


with gr.Blocks(title="UAE Off-Plan Broker Demo") as demo:
    with gr.Column(elem_id="shell"):
        gr.HTML(
            """
            <div id="header">
              <h1 id="title">UAE Off-Plan Advisor</h1>
              <div id="subtitle">Client demo workspace: Golden Visa guidance + off-plan knowledge base answers</div>
            </div>
            """
        )

        with gr.Row(elem_id="chips"):
            p1 = gr.Button("Golden Visa basics", elem_classes=["chip"], size="sm")
            p2 = gr.Button("Escrow account protections", elem_classes=["chip"], size="sm")
            p3 = gr.Button("Off-plan eligibility limits", elem_classes=["chip"], size="sm")

        with gr.Column(elem_id="chatwrap"):
            chatbot = gr.Chatbot(
                elem_id="chatbox",
                value=[
                    {
                        "role": "assistant",
                        "content": "Hello. Ask me anything about UAE off-plan investing and Golden Visa rules.",
                    }
                ],
            )

        with gr.Column(elem_id="reco-wrap"):
            gr.HTML("<div style='font-weight:700;margin-bottom:8px'>Top Recommendations</div>")
            recommendations = gr.HTML(
                render_recommendation_cards(None),
                elem_id="reco-board",
            )

        with gr.Row(elem_id="controls"):
            msg = gr.Textbox(
                placeholder="Type your question...",
                container=False,
                scale=8,
            )
            send = gr.Button("Send", elem_id="send", scale=1, variant="primary")
            clear = gr.Button("Clear", elem_id="clear", scale=1)

    send.click(respond_stream, [msg, chatbot], [msg, chatbot, recommendations])
    msg.submit(respond_stream, [msg, chatbot], [msg, chatbot, recommendations])

    p1.click(quick_prompt_1, chatbot, [msg, chatbot, recommendations])
    p2.click(quick_prompt_2, chatbot, [msg, chatbot, recommendations])
    p3.click(quick_prompt_3, chatbot, [msg, chatbot, recommendations])

    clear.click(clear_chat, None, [chatbot, recommendations])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=8).launch(
        css=CSS,
        share=os.getenv("GRADIO_SHARE", "0") == "1",
    )
