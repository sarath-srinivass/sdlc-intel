from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import yaml

from harness.dtypes import RESOURCE_TYPES, Skill, SkillFrontMatter
from harness.utils import make_links_absolute


def get_frontmatter(skill_dir: str, skills_root: Path) -> SkillFrontMatter:
    skill_index_file = skills_root.joinpath(skill_dir).joinpath("index.md")
    with open(skill_index_file, "r") as f:
        skill_index = f.read()
    frontmatter_yml = skill_index.split("---")[1].strip()
    fm = yaml.safe_load(frontmatter_yml)
    if "allowed-tools" in fm:
        fm["allowed_tools"] = fm["allowed-tools"]
        del fm["allowed-tools"]
    fm["location"] = str(skill_index_file)
    return SkillFrontMatter(**fm)


class SkillFileStore:
    def __init__(self, skills_root_dir: Path) -> None:
        self.skills_root: Path = skills_root_dir
        skill_dirs = [
            str(skill_dir.stem)
            for skill_dir in self.skills_root.rglob("*")
            if skill_dir.is_dir()
        ]

        self.frontmatters: dict[str, SkillFrontMatter] = {}
        for skill_dir in skill_dirs:
            fm = get_frontmatter(skill_dir, skills_root=self.skills_root)
            self.frontmatters[fm.name] = fm

    async def get_skills_description(
        self, skills: Optional[list[str]] = None
    ) -> dict[str, SkillFrontMatter]:
        skills_descritpion = {fm.name: fm for _, fm in self.frontmatters.items()}
        skills_descritpion_filtered = {
            nm: desc
            for nm, desc in skills_descritpion.items()
            if nm in (skills or skills_descritpion.keys())
        }
        return skills_descritpion_filtered

    async def load_skill(self, uri: str) -> Skill:
        skill = urlparse(uri).netloc
        fm = self.frontmatters[skill]
        with open(fm.location, "r") as f:
            content = f.read().strip().split("---")[2].strip()
            content = make_links_absolute(content, base_url=fm.location)
        return Skill(
            name=skill, content=content, tools=fm.allowed_tools, frontmatter=fm
        )

    async def load_file(self, uri: str):
        url_parts = urlparse(uri)
        file_path = Path(url_parts.netloc).joinpath(url_parts.path.lstrip("/"))
        with open(file_path, "r") as f:
            content = f.read().strip()
        content = make_links_absolute(
            content, base_url=file_path.parent, exclude_res_types=RESOURCE_TYPES
        )
        return content

    async def load_resource(self, uri: str) -> Skill | str:
        url_parts = urlparse(uri)
        resource_type = url_parts.scheme
        if resource_type == "skill":
            return await self.load_skill(uri)
        elif resource_type == "file":
            return await self.load_file(uri)
        else:
            raise ValueError("Unknown Resource Type")
