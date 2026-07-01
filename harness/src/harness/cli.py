import asyncio
from pathlib import Path

import mlflow
from langchain.messages import AIMessage, HumanMessage

from harness.agent import create_supervisor_agent, initialize_agent_context
from harness.skill_file_store import SkillFileStore

mlflow.langchain.autolog()  # type: ignore


async def run_cli_chat(agent, ctx):
    state = {"messages": []}
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nGoodbye!")
                break
            if not user_input:
                continue
            state["messages"].append(HumanMessage(content=user_input))
            print("Agent: ", end="", flush=True)
            async for event in agent.astream(state, context=ctx):
                for node, output in event.items():
                    last_msg = output["messages"][-1]
                    if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                        print(last_msg.content, end="", flush=True)
            print("\n")

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            raise e


if __name__ == "__main__":
    skill_store = SkillFileStore(Path("./skills"))
    ctx = asyncio.run(initialize_agent_context(skill_store=skill_store))
    agent = create_supervisor_agent()
    asyncio.run(run_cli_chat(agent=agent, ctx=ctx))
