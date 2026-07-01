import json
from dataclasses import asdict
from string import Template

from langchain.tools import BaseTool, ToolRuntime, tool

from harness.dtypes import AgentContext, Skill, SkillStore


@tool
async def load_resource(uri: str, runtime: ToolRuntime[AgentContext]) -> str:
    """loads the resource using the `uri`"""
    skill_store: SkillStore = runtime.context.skill_store
    res = await skill_store.load_resource(uri)
    # TODO: Separate formatting logic for resource types
    if isinstance(res, Skill):
        res_dict = asdict(res)
        res_dict["content"] = Template(res_dict["content"]).substitute(
            **asdict(runtime.context.skill_context_variables)
        )
        return json.dumps(res_dict)
    else:
        return res


async def get_available_tools() -> dict[str, BaseTool]:
    # Add MCP tools dynamically
    # mcp_tools = ...
    harness_tools = (load_resource,)
    # tools = harness_tools.union(set(mcp_tools))
    tools = harness_tools
    tools_dict = {tl.name: tl for tl in tools}
    return tools_dict
