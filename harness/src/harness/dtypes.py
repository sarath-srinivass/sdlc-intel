from dataclasses import dataclass, field
from typing import Annotated, Any, List, Optional, Protocol, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

RESOURCE_TYPES = ["file", "skill"]


@dataclass
class SkillFrontMatter:
    name: str
    description: str
    location: str
    licence: Optional[str] = None
    compatibility: Optional[str] = None
    metadata: Optional[dict] = field(default_factory=dict)
    allowed_tools: List[str] = field(default_factory=list)


@dataclass
class Skill:
    name: str
    content: str
    frontmatter: SkillFrontMatter
    tools: list[str] = field(default_factory=list)


class SkillStore(Protocol):
    async def get_skills_description(
        self, skills: Optional[list[str]] = None
    ) -> dict[str, SkillFrontMatter]: ...
    async def load_resource(self, uri: str) -> Skill | str: ...


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    # observations: Annotated[List[str], add]
    # reflections: Annotated[List[str], add]


@dataclass
class SkillContext:
    # Has runtime variables available to be injected in skills using {var} format.
    available_skills_xml: str
    today: str


@dataclass
class AgentContext:
    skill_store: Any
    skill_context_variables: SkillContext
