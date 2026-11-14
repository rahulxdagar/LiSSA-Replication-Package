"""Java structure for RQ2 (code summaries) and RQ3 (identifier check, code of a cited element).

Uses tree-sitter (pinned in requirements.txt), which also recovers the declarations of files with syntax errors.
A file is parsed from LiSSA's content of it, without the UTF-8 byte-order mark.
"""

import re
from dataclasses import dataclass, field

import tree_sitter_java
from tree_sitter import Language, Parser

_PARSER = Parser(Language(tree_sitter_java.language()))
TYPE_NODES = {"class_declaration": "class", "interface_declaration": "interface", "enum_declaration": "enum",
              "annotation_type_declaration": "annotation"}
MEMBER_NODES = {"method_declaration": "method", "constructor_declaration": "constructor",
                "field_declaration": "field", "constant_declaration": "field"}
MAX_SUMMARY_CHARS = 4000  # keeps the longest summary prompt inside the 16,384-token context (prepare_prompts.py checks)


@dataclass
class Member:
    kind: str  # method, constructor or field
    names: list  # a field declaration can declare several names
    signature: str
    javadoc: str
    code: str


@dataclass
class TypeDecl:
    kind: str
    name: str
    header: str
    javadoc: str
    members: list = field(default_factory=list)


@dataclass
class JavaFile:
    types: list
    has_errors: bool


def _text(source, node):
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _squash(text):
    return re.sub(r"\s+", " ", text).strip()


def _javadoc(source, node):
    """The first sentence of the /** ... */ comment right before a declaration, without tags."""
    comment = node.prev_named_sibling
    if comment is None or comment.type != "block_comment" or not _text(source, comment).startswith("/**"):
        return ""
    description = []
    for line in _text(source, comment)[3:-2].splitlines():
        line = re.sub(r"^\s*\*\s?", "", line)
        if line.lstrip().startswith("@"):
            break  # the description ends at the first tag (@param, @return, ...)
        description.append(line)
    body = _squash(" ".join(description))
    sentence = re.split(r"(?<=\.)\s", body, maxsplit=1)[0]
    return sentence[:200]


def _with_javadoc(source, node):
    """The declaration's code, preceded by its Javadoc comment if it has one."""
    comment = node.prev_named_sibling
    if comment is not None and comment.type == "block_comment" and _text(source, comment).startswith("/**"):
        return source[comment.start_byte:node.end_byte].decode("utf-8", errors="replace")
    return _text(source, node)


def _head(source, node, body):
    """A declaration up to its body (or the whole declaration if it has none), on one line."""
    end = body.start_byte if body is not None else node.end_byte
    return _squash(source[node.start_byte:end].decode("utf-8", errors="replace")).rstrip(";").strip()


def parse(content):
    source = content.lstrip("﻿").encode("utf-8")
    tree = _PARSER.parse(source)
    types = []

    def visit(node):
        if node.type in TYPE_NODES and node.child_by_field_name("name") is not None:
            body = node.child_by_field_name("body")
            decl = TypeDecl(TYPE_NODES[node.type], _text(source, node.child_by_field_name("name")),
                            _head(source, node, body), _javadoc(source, node))
            types.append(decl)
            containers = [body] if body is not None else []
            containers += [c for c in (body.named_children if body is not None else [])
                           if c.type == "enum_body_declarations"]
            for container in containers:
                for child in container.named_children:
                    if child.type in MEMBER_NODES:
                        decl.members.append(_member(source, child))
        for child in node.named_children:
            visit(child)

    visit(tree.root_node)
    return JavaFile(types, tree.root_node.has_error)


def _member(source, node):
    kind = MEMBER_NODES[node.type]
    if kind == "field":
        names = [_text(source, d.child_by_field_name("name")) for d in node.named_children
                 if d.type == "variable_declarator" and d.child_by_field_name("name") is not None]
        signature = _squash(_text(source, node)).rstrip(";")[:200]
    else:
        names = [_text(source, node.child_by_field_name("name"))]
        signature = _head(source, node, node.child_by_field_name("body"))
    return Member(kind, names, signature, _javadoc(source, node), _with_javadoc(source, node))


# --- RQ2: structural summary ---------------------------------------------------------------------------------------


def summary(java):
    """Types with their Javadoc, then one line per field, constructor and method (signature and Javadoc)."""
    if not java.types:
        return "(no Java declarations found)"
    lines = []
    for decl in java.types:
        lines.append(decl.header + (f"  // {decl.javadoc}" if decl.javadoc else ""))
        for member in decl.members:
            lines.append(f"  {member.kind} {member.signature}" + (f"  // {member.javadoc}" if member.javadoc else ""))
    kept, size = [], 0
    for index, line in enumerate(lines):
        if size + len(line) + 1 > MAX_SUMMARY_CHARS:
            kept.append(f"  ... ({len(lines) - index} more lines)")
            break
        kept.append(line)
        size += len(line) + 1
    return "\n".join(kept)


# --- RQ3: identifiers and the code of a cited element --------------------------------------------------------------


def declared_identifiers(java):
    names = set()
    for decl in java.types:
        names.add(decl.name)
        for member in decl.members:
            names.update(member.names)
    return names


def normalize_identifier(identifier):
    """'Manager.delete(int)' -> 'delete'; 'DBTourist' -> 'DBTourist'. Parameters and qualifiers are dropped."""
    name = re.sub(r"\(.*", "", identifier.strip()).strip().rstrip(".")
    return name.split(".")[-1].strip()


def element_code(java, identifier):
    """The code a claim about 'identifier' is checked against: all methods or constructors of that name (overloads
    included), else the field declaration, else for a type its header, Javadoc and member signatures.
    None if the file declares no such identifier (the claim then fails the identifier check)."""
    name = normalize_identifier(identifier)
    if not name:
        return None
    for kinds in (("method", "constructor"), ("field",)):
        found = [m.code for d in java.types for m in d.members if m.kind in kinds and name in m.names]
        if found:
            return "\n\n".join(found)
    for decl in java.types:
        if decl.name == name:
            return summary(JavaFile([decl], False))
    return None
