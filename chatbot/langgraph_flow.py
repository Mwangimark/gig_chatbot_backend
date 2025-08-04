# chatbot/langgraph_flow.py  (HF standalone)

from typing import TypedDict, List, Optional
import os

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import (
    HumanMessage, AIMessage, BaseMessage, SystemMessage
)

# -------------------- HF client --------------------
from huggingface_hub import InferenceClient

# ✅ Provide a safe default model that supports the conversational/chat API
hf_model = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
hf_token = os.getenv("HF_TOKEN")

# ✅ Single reusable client
hf_client = InferenceClient(model=hf_model, token=hf_token, timeout=120)

# -------------------- State --------------------
class ChatState(TypedDict):
    messages: List[BaseMessage]

def _ensure_messages(state: ChatState) -> ChatState:
    if "messages" not in state or not isinstance(state["messages"], list):
        state["messages"] = []
    return state

# -------------------- Prompting (GiG-only) --------------------
GIG_SYSTEM_PROMPT = (
    "You are GIGBot, the official assistant for GiG Kenya (gig.co.ke). "
    "Only answer questions about GiG features, event creation, RSVP/ticketing, analytics, "
    "and using the GiG platform. If the user asks about anything unrelated to GiG, "
    "politely refuse and say: \"I’m here to assist with the GiG platform—please ask about gigs, "
    "events, or your GiG profile.\" Be concise, step-by-step, and accurate."
)

def _tail(messages: List[BaseMessage], n: int = 6) -> List[BaseMessage]:
    return messages[-n:] if messages else []


def hf_reply(state: ChatState, intent_hint: Optional[str] = None) -> AIMessage:
    """
    Calls Hugging Face Inference API via chat_completion (conversational task).
    We send: [system, latest user], and return the assistant content.
    """
    state = _ensure_messages(state)
    recent = _tail(state["messages"], n=6)

    system_extra = f"\nCurrent intent: {intent_hint}." if intent_hint else ""
    system_text = GIG_SYSTEM_PROMPT + system_extra

 
    user_text = recent[-1].content if recent else "Hello"   

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]

    try:
        resp = hf_client.chat_completion(
            model=hf_model,            # explicit
            messages=messages,
            max_tokens=256,
            temperature=0.2,
        )
        choice = resp.choices[0]
        msg = choice.message
        content = msg["content"] if isinstance(msg, dict) else getattr(msg, "content", "")
        return AIMessage(content=(content or "").strip())
    except Exception as e:
       
        import traceback
        print("HF chat_completion failed:", type(e).__name__, e)
        traceback.print_exc()
        try:
            resp = getattr(e, "response", None)
            if resp is not None:
                print("HTTP", resp.status_code, resp.text)
        except Exception:
            pass
        # Graceful user-facing fallback
        return AIMessage(content="Sorry, I couldn’t generate a response right now. Please try again.")

# -------------------- Classifier & Nodes --------------------
def classify_intent(state: ChatState) -> str:
    state = _ensure_messages(state)
    if not state["messages"]:
        return "fallback"

    last_message = state["messages"][-1].content.lower()

    if "event" in last_message or "register" in last_message or "create" in last_message:
        return "register_event"
    if "rsvp" in last_message:
        return "rsvp_help"
    if "analytics" in last_message or "insights" in last_message or "dashboard" in last_message:
        return "analytics_guide"
    if "what is gig" in last_message or "gig" in last_message:
        return "gig_intro"
    return "fallback"

def start_node(state: ChatState) -> ChatState:
    return _ensure_messages(state)

def router_node(state: ChatState) -> ChatState:
    return state 

def register_event(state: ChatState) -> ChatState:
    state = _ensure_messages(state)
    ai = hf_reply(state, intent_hint="register_event")
    state["messages"].append(ai)
    return state

def rsvp_help(state: ChatState) -> ChatState:
    state = _ensure_messages(state)
    ai = hf_reply(state, intent_hint="rsvp_help")
    state["messages"].append(ai)
    return state

def analytics_guide(state: ChatState) -> ChatState:
    state = _ensure_messages(state)
    ai = hf_reply(state, intent_hint="analytics_guide")
    state["messages"].append(ai)
    return state

def gig_intro(state: ChatState) -> ChatState:
    state = _ensure_messages(state)
    ai = hf_reply(state, intent_hint="gig_intro")
    state["messages"].append(ai)
    return state

def fallback(state: ChatState) -> ChatState:
    state = _ensure_messages(state)
    ai = hf_reply(state, intent_hint="fallback")
    state["messages"].append(ai)
    return state

# -------------------- Build & compile graph --------------------
def build_graph():
    builder = StateGraph(ChatState)

    builder.add_node("start", start_node)
    builder.add_node("router", router_node)
    builder.add_edge("start", "router")
    builder.set_entry_point("start")

    builder.add_node("register_event", register_event)
    builder.add_node("rsvp_help", rsvp_help)
    builder.add_node("analytics_guide", analytics_guide)
    builder.add_node("gig_intro", gig_intro)
    builder.add_node("fallback", fallback)

    builder.add_conditional_edges(
        "router",
        classify_intent,
        {
            "register_event": "register_event",
            "rsvp_help": "rsvp_help",
            "analytics_guide": "analytics_guide",
            "gig_intro": "gig_intro",
            "fallback": "fallback",
        },
    )

    for node in ["register_event", "rsvp_help", "analytics_guide", "gig_intro", "fallback"]:
        builder.add_edge(node, END)

    checkpointer = MemorySaver()  # keeps thread context across turns
    return builder.compile(checkpointer=checkpointer)

# Export compiled graph
gig_chatbot_graph = build_graph()
