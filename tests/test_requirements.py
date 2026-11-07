"""The installable manifests match the non-notebook imports."""

import ast
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_REQUIREMENTS = ROOT / "google gemini LLM" / "requirements.txt"
APP_FILES = [
    ROOT / "google gemini LLM" / "llm_chat.py",
    ROOT / "Open Source LLM" / "mistral_llm.py",
]
DECLARED_FOR_IMPORT = {
    "streamlit": "streamlit",
    "PyPDF2": "pypdf2",
    "google.generativeai": "google-generativeai",
}


def requirement_lines(path):
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def normalize_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def declared_names():
    names = set()
    for line in requirement_lines(SCRIPT_REQUIREMENTS):
        assert not line.startswith("-"), line
        body = line.split(";", 1)[0].strip()
        assert not any(character.isspace() for character in body), line
        assert "--" not in line
        name = re.split(r"(===|==|!=|~=|>=|<=|>|<)", body, maxsplit=1)[0]
        names.add(normalize_name(name.split("[", 1)[0]))
    return names


def third_party_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module]
        else:
            continue
        for module in modules:
            top_level = module.split(".", 1)[0]
            if top_level not in sys.stdlib_module_names:
                found.add(module)
    return found


def test_root_manifest_includes_the_script_requirements():
    lines = requirement_lines(ROOT / "requirements.txt")
    assert lines == ['-r "google gemini LLM/requirements.txt"']
    flag, relative_path = shlex.split(lines[0])
    assert flag == "-r"
    assert (ROOT / relative_path).is_file()


def test_script_requirements_name_the_installed_distributions():
    assert declared_names() == {
        "streamlit",
        "pypdf2",
        "google-generativeai",
    }


def test_non_notebook_imports_are_declared():
    declared = declared_names()
    for path in APP_FILES:
        for module in third_party_imports(path):
            assert module in DECLARED_FOR_IMPORT, module
            assert DECLARED_FOR_IMPORT[module] in declared
