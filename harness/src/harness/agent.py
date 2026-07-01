import asyncio
import datetime
from typing import Literal

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph, RunnableConfig
from langgraph.runtime import Runtime
from langgraph.types import StreamWriter

from harness.dtypes import AgentContext, AgentState, SkillContext, SkillStore
from harness.env_vars import (
    SUPERVISOR_MODEL,
    SYSTEM_SKILL,
    TOKEN_THRESHOLD_PREPARE,
)
from harness.tools import get_available_tools
from harness.utils import get_skills_xml, prepare_llm_messages, process_tool_output


async def initialize_agent_context(skill_store: SkillStore) -> AgentContext:
    skills_description = await skill_store.get_skills_description()
    del skills_description[SYSTEM_SKILL]
    return AgentContext(
        skill_store=skill_store,
        skill_context_variables=SkillContext(
            available_skills_xml=get_skills_xml(skills_description),
            today=str(datetime.date.today()),
        ),
    )


async def supervisor(state: AgentState, runtime: Runtime[AgentContext]):
    llm = ChatOpenRouter(model=SUPERVISOR_MODEL)
    llm_messages, tools = await prepare_llm_messages(
        state, tp=TOKEN_THRESHOLD_PREPARE, ctx=runtime
    )
    registered_tools = await get_available_tools()
    llm_with_tools = llm.bind_tools([registered_tools[tool] for tool in tools])
    resp = await llm_with_tools.ainvoke(llm_messages)
    return {"messages": [resp]}


async def tool_node(
    state: AgentState,
    writer: StreamWriter,
    config: RunnableConfig,
    runtime: Runtime[AgentContext],
):
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage):
        registered_tools = await get_available_tools()
        tool_messages = []
        for tool_call in last_msg.tool_calls:
            tool = registered_tools[tool_call["name"]]
            tool_res = await tool.ainvoke(
                {
                    **tool_call["args"],
                    "runtime": ToolRuntime(
                        context=runtime.context,
                        state=state,
                        stream_writer=writer,
                        config=config,
                        store=runtime.store,
                        tool_call_id=tool_call["id"],
                    ),
                }
            )
            content, metadata = process_tool_output(tool_call, tool_res)
            tool_messages.append(
                ToolMessage(
                    content=content, tool_call_id=tool_call["id"], artifact=metadata
                )
            )
        return {"messages": tool_messages}


async def tool_edge(state: AgentState) -> Literal["tool_node", END]:  # type: ignore
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tool_node"
    else:
        return END


async def responder(state: AgentState): ...


async def observer(state: AgentState):
    print("Start Observer")
    await asyncio.sleep(0.4)
    print("End Observer")
    return {"observations": ["o1", "o2"]}


async def reflector(state: AgentState):
    print("Start reflector")
    await asyncio.sleep(1)
    print("End reflector")
    return {"reflections": ["r1", "r2"]}


def create_supervisor_agent() -> CompiledStateGraph:
    graph = StateGraph(AgentState, context_schema=AgentContext)
    graph.add_node(supervisor)
    graph.add_node(tool_node)
    graph.add_node(observer)
    graph.add_node(reflector)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", tool_edge)
    graph.add_edge("tool_node", "supervisor")
    # graph.add_edge(START, "observer")
    # graph.add_edge(START, "reflector")
    graph.add_edge("supervisor", END)
    graph.add_edge("observer", END)
    graph.add_edge("reflector", END)
    agent = graph.compile()
    return agent
