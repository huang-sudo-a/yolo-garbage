"""Map detector item classes to the four garbage categories."""

from __future__ import annotations

from pathlib import Path

import yaml


class CategoryMap:
    def __init__(self, path: Path, model_names: dict[int, str]) -> None:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        groups = data["categories"]
        names = {str(name) for name in model_names.values()}
        mapping: dict[str, str] = {}
        for category, items in groups.items():
            for item in items:
                if item in mapping:
                    raise ValueError(f"重复映射类别：{item}")
                mapping[item] = category
        absent = names - mapping.keys()
        unknown = mapping.keys() - names
        if absent or unknown:
            raise ValueError(f"类别映射与模型不匹配；未映射={sorted(absent)}；模型无此类={sorted(unknown)}")
        self._mapping = mapping

    def category(self, item: str) -> str:
        return self._mapping[item]
