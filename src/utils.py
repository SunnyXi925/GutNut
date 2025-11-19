import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

def read_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    读取 YAML 配置文件并返回字典。
    支持相对项目根路径的默认位置 `config.yaml`。
    """
    root = Path(__file__).resolve().parents[1]
    cfg_path = Path(config_path) if config_path else root / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def project_root() -> Path:
    """返回项目根路径 MiNPS/"""
    return Path(__file__).resolve().parents[1]

def ensure_dir(path: Path) -> None:
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

def save_json(obj: Any, path: Path) -> None:
    """保存 JSON 文件（UTF-8）"""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: Path) -> Any:
    """读取 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_path(relative: str) -> Path:
    """将相对路径（相对项目根）解析为绝对路径"""
    return project_root() / relative