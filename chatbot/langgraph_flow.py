from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# LangChain LLM setup
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Step 1: Define chatbot memory state
class ChatState(dict):
    pass

# Step 2: Intent Classifier
def classify_intent(state: ChatState):
    last_message = state["messages"][-1].content.lower()

    if "event" in last_message:
        return "register_event"
    elif "rsvp" in last_message:
        return "rsvp_help"
    elif "analytics" in last_message:
        return "analytics_guide"
    elif "what is gig" in last_message or "gig" in last_message:
        return "gig_intro"
    else:
        return "fallback"

# Step 3: Define functional nodes
def register_event(state: ChatState):
    response = (
        "To register an event on GiG:\n"
        "1. Log in.\n"
        "2. Go to the Events page.\n"
        "3. Click 'Create Event'.\n"
        "4. Fill in event name, location, time, and category.\n"
        "5. Save and publish.\n"
        "You can also add ticketing and RSVP options after that."
    )
    state["messages"].append(AIMessage(content=response))
    return state

def rsvp_help(state: ChatState):
    response = (
        "To track RSVPs:\n"
        "1. Open your event page.\n"
        "2. Go to the 'Attendees' tab.\n"
        "3. You’ll see a list of RSVPs, and you can export the list too."
    )
    state["messages"].append(AIMessage(content=response))
    return state

def analytics_guide(state: ChatState):
    response = (
        "The Analytics dashboard shows:\n"
        "- Number of views\n"
        "- RSVP trends\n"
        "- Attendee engagement\n"
        "- Device and location data"
    )
    state["messages"].append(AIMessage(content=response))
    return state

def gig_intro(state: ChatState):
    response = (
        "GiG is a platform that allows you to create, manage, and promote events. "
        "It supports RSVP, ticketing, audience engagement, and post-event analytics."
    )
    state["messages"].append(AIMessage(content=response))
    return state

def fallback(state: ChatState):
    response = (
        "I’m built to assist with GiG-related questions like event creation, RSVP, analytics, etc. "
        "Please ask something related to GiG features."
    )
    state["messages"].append(AIMessage(content=response))
    return state

def route_from_intent(state: ChatState) -> str:
    return classify_intent(state)

def start_node(state: ChatState) -> ChatState:
    print("▶️ Start Node State:", state)
    return state

def router_node(state: ChatState) -> ChatState:
    # no-op: just pass state along so conditional edges can decide
    return state

def build_graph():
    builder = StateGraph(ChatState)

    builder.add_node("start", start_node)
    builder.add_node("router", router_node)  # 👈 pass-through
    builder.add_edge("start", "router")
    builder.set_entry_point("start")

    # intent nodes
    builder.add_node("register_event", register_event)
    builder.add_node("rsvp_help", rsvp_help)
    builder.add_node("analytics_guide", analytics_guide)
    builder.add_node("gig_intro", gig_intro)
    builder.add_node("fallback", fallback)

    # conditional routing (route_from_intent returns a label string)
    builder.add_conditional_edges(
        "router",
        route_from_intent,
        {
            "register_event": "register_event",
            "rsvp_help": "rsvp_help",
            "analytics_guide": "analytics_guide",
            "gig_intro": "gig_intro",
            "fallback": "fallback",
        }
    )

    # terminate each intent node
    for node in ["register_event", "rsvp_help", "analytics_guide", "gig_intro", "fallback"]:
        builder.add_edge(node, END)

    return builder.compile()

gig_chatbot_graph = build_graph()