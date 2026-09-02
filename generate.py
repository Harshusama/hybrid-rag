"""
Generation (ONLINE — final step of the query pipeline)
----------------------------------------------------------
Retrieved chunks + query -> Claude generates a cited answer -> a second,
cheap verification pass checks the answer is actually grounded in the
retrieved context before it's returned.

WHY A SEPARATE VERIFY STEP: the generation prompt already instructs the
model to stay grounded and cite sources, but instructions alone don't
guarantee it — models can still drift into unsupported claims, especially
on longer answers. A second pass that specifically checks "is this
supported by the context, yes/no" catches cases the first pass missed,
without the cost of a human reviewer. It's not perfect (it's still an
LLM judging another LLM), but it's a cheap, standard production guardrail.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

import config

_llm = ChatAnthropic(
    model=config.GENERATION_MODEL,
    anthropic_api_key=config.ANTHROPIC_API_KEY,
    max_tokens=config.MAX_ANSWER_TOKENS,
)

GENERATION_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the
provided context from internal documents. Rules:

1. Base your answer only on the context below — never use outside knowledge.
2. If the context doesn't contain enough information, say so explicitly
   instead of guessing.
3. Cite the source chunk for every claim, like [1], [2].
4. Be concise and direct.
"""

VERIFY_SYSTEM_PROMPT = """You are a fact-checker. You will be given a CONTEXT and an ANSWER that
claims to be based on that context. Check whether every factual claim in
the ANSWER is actually supported by the CONTEXT.

Respond with exactly one word on the first line: SUPPORTED or UNSUPPORTED.
If UNSUPPORTED, add a second line briefly stating which claim isn't backed
by the context.
"""


def _build_context_block(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        source = c["metadata"].get("source", "unknown")
        blocks.append(f"[{i}] (source: {source})\n{c['text']}")
    return "\n\n".join(blocks)


def generate_answer(query: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {
            "answer": "I don't have any relevant information in the knowledge base to answer that.",
            "verified": True,
            "verification_note": "No chunks retrieved — nothing to verify.",
        }

    context = _build_context_block(chunks)

    generation_prompt = f"""CONTEXT:
{context}

QUESTION:
{query}

Answer the question using only the context above, with citations like [1]."""

    gen_response = _llm.invoke(
        [
            SystemMessage(content=GENERATION_SYSTEM_PROMPT),
            HumanMessage(content=generation_prompt),
        ]
    )
    answer = gen_response.content

    # --- Verification pass ---
    verify_prompt = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
    verify_response = _llm.invoke(
        [
            SystemMessage(content=VERIFY_SYSTEM_PROMPT),
            HumanMessage(content=verify_prompt),
        ]
    )
    verify_text = verify_response.content.strip()
    verified = verify_text.upper().startswith("SUPPORTED")

    return {
        "answer": answer,
        "verified": verified,
        "verification_note": verify_text,
    }