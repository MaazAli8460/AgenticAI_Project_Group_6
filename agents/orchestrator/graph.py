import os
import sqlite3
from typing import Annotated, Sequence, TypedDict, Literal

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq

from agents.edit_agent.intent_classifier import IntentClassifier
from agents.edit_agent.executor import EDIT_TOOLS
from state_manager.state_manager import StateManager


# Define Graph State
class EditState(TypedDict):
    project_id: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    intent: dict  # Stores parsed intent JSON


def classify_node(state: EditState):
    """Parses the user query into a structured intent."""
    # Find the last human message
    human_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not human_msg:
        return {}

    classifier = IntentClassifier()
    try:
        intent_obj = classifier.classify(human_msg.content)
        return {"intent": intent_obj.model_dump(mode="json")}
    except Exception as e:
        return {"intent": {"error": str(e)}}


def agent_node(state: EditState):
    """LLM agent that decides which tools to call based on the intent."""
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    llm = ChatGroq(model=model_name, temperature=0)
    llm_with_tools = llm.bind_tools(EDIT_TOOLS)
    
    sys_msg = SystemMessage(content=(
        f"You are an expert AI video editing assistant. The user wants to edit project '{state.get('project_id')}'.\n"
        f"Intent extracted: {state.get('intent', {})}\n\n"
        "Instructions:\n"
        "1. Act as a conversational assistant. If the user's request is ambiguous (e.g. 'change the character's hair' but there are multiple characters), use the `get_project_context` tool to find out who the characters are, then ASK the user to clarify. Do NOT guess.\n"
        "2. If the user wants to see an image of a character, use `get_asset_path` and provide the path to them.\n"
        "3. When updating a character's visual description or a scene's background, you MUST read the existing description (via `get_project_context`) and preserve all traits the user did not explicitly change. For example, if changing hair color, rewrite the existing description to replace ONLY the hair color while keeping the original clothing, eye color, etc. Do NOT wipe out the other details.\n"
        "4. If the user mentions a specific scene for a character appearance change, use `update_scene_character_visual` (scene-scoped) instead of `update_character_visual`.\n"
        "5. Once the user is clear, use the update tools to change the project state JSON.\n"
        "6. ONLY AFTER you update the state, ask the user if they want to run the pipeline to apply the changes. Do NOT run `run_pipeline_phase` without explicit permission.\n"
        "7. If the user asks to add or remove subtitles, use the `toggle_subtitles` tool and do NOT re-run Phase 3.\n"
        "8. For scene-level BGM changes, use `update_scene_bgm` only; it already regenerates the target scene audio and remuxes the final video. Do NOT run Phase 2 again.\n"
        "9. For scene-level dialogue re-voicing, use `update_scene_dialogue`; it regenerates TTS and remixes audio without re-rendering visuals.\n"
        "10. Explain what you are doing in the background (e.g. 'I am pulling up the project characters...', 'I have updated the state...')."
    ))
    
    msgs = [sys_msg] + list(state["messages"])
    try:
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response]}
    except Exception as exc:
        message = str(exc)
        if "rate limit" in message.lower() or "429" in message:
            reply = "Edit agent is rate limited by Groq. Please wait and try again."
        else:
            reply = f"Edit agent error: {message}"
        return {"messages": [AIMessage(content=reply)]}


def should_continue(state: EditState) -> Literal["tools", "__end__"]:
    """Routing function to call tools or end the turn."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


def build_edit_graph():
    """Builds and compiles the LangGraph orchestration."""
    builder = StateGraph(EditState)
    
    builder.add_node("classify", classify_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(EDIT_TOOLS))
    
    # Edges
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
    builder.add_edge("tools", "agent")
    
    # Checkpointer for memory and HITL multi-turn support
    db_path = Path("data/edit_history.db")
    db_path.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    memory = SqliteSaver(conn)
    
    # We compile the graph. 
    # By using `interrupt_before=["tools"]`, we can enforce Human-in-the-loop (HITL)
    # allowing the human to approve tool calls before they run.
    return builder.compile(checkpointer=memory)

from pathlib import Path
