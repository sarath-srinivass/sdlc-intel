import json
import re
from dataclasses import asdict
from pathlib import Path
from string import Template
from typing import Tuple
from urllib.parse import urlparse

from jinja2 import Environment
from langchain.messages import AnyMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.runtime import Runtime

from harness.dtypes import (
    RESOURCE_TYPES,
    AgentContext,
    AgentState,
    Skill,
    SkillFrontMatter,
)
from harness.env_vars import SYSTEM_SKILL


def make_links_absolute(markdown_text, base_url, exclude_res_types=RESOURCE_TYPES):
    link_pattern = re.compile(r"(\[.*?\]\()([^)]+)(\))")

    def replace_link(match):
        prefix = match.group(1)  # [text](
        url = match.group(2).strip()  # url
        suffix = match.group(3)  # )

        if url.startswith(tuple(f"{res_typ}://" for res_typ in exclude_res_types)):
            return match.group(0)

        absolute_url = "file://" + str(Path(base_url).joinpath(url))
        return f"{prefix}{absolute_url}{suffix}"

    return link_pattern.sub(replace_link, markdown_text)


def get_skills_xml(skills_description: dict[str, SkillFrontMatter]) -> str:
    skills_uri = {nm: "skill://" + nm for nm in skills_description.keys()}
    xml_templ = """
    <available_skills>
    {% for skill, val in skills_description.items() %}
    <skill>
    <name>{{skill}}</name>
    <description>{{val.description}}</description>
    <location>{{skills_uri[skill]}}</location>
    </skill>
    {% endfor %}
    </available_skills>
    """
    env = Environment()
    tmpl = env.from_string(xml_templ)
    xml = tmpl.render(skills_description=skills_description, skills_uri=skills_uri)
    return xml


def process_tool_output(tool_call: ToolCall, tool_res) -> tuple[str, dict]:
    tool_name, tool_args = tool_call["name"], tool_call["args"]
    tool_metadata = {}
    if tool_name == "load_resource":
        res_type = urlparse(tool_args["uri"]).scheme
        tool_metadata["resource_type"] = res_type
        if res_type == "skill":
            tool_metadata["skill_tools"] = json.loads(tool_res)["tools"]
        return json.loads(tool_res)["content"], tool_metadata
    else:
        return tool_res, {}


async def prepare_llm_messages(
    state: AgentState, tp: int, ctx: Runtime[AgentContext]
) -> Tuple[list[AnyMessage], set[str]]:
    tokens = sum(count_tokens_approximately(msg.content) for msg in state["messages"])
    tools = set(["load_resource"])  # default tool that always stays in context
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage) and "skill_tools" in msg.artifact:
            tools.union(set(msg.artifact["skill_tools"]))

    messages = []
    system_skill_uri = "skill://" + SYSTEM_SKILL
    system_skill = await ctx.context.skill_store.load_resource(system_skill_uri)
    if isinstance(system_skill, Skill):
        system_msg = SystemMessage(
            Template(system_skill.content).substitute(
                **asdict(ctx.context.skill_context_variables)
            )
        )
        messages.append(system_msg)
        messages.extend(state["messages"])
        if system_skill.frontmatter.metadata:
            system_metadata_str = ", ".join(
                [
                    f"{k}:" + getattr(ctx.context.skill_context_variables, v)
                    for k, v in system_skill.frontmatter.metadata.items()
                ]
            )
            footer_msg = SystemMessage(content=system_metadata_str)
            messages.append(footer_msg)

    if tokens > tp:
        ## TODO: Implement Prepare Step for Observations/Reflections
        return messages, tools
    else:
        return messages, tools
