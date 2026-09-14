from __future__ import annotations

from openai.lib._pydantic import to_strict_json_schema

from cv_agent.schemas import DocumentPackage, GroundednessAudit, JobRequirements, MasterProfile


def _assert_objects_are_strict(node: object) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False
            properties = node.get("properties", {})
            assert set(node.get("required", [])) == set(properties)
        for value in node.values():
            _assert_objects_are_strict(value)
    elif isinstance(node, list):
        for value in node:
            _assert_objects_are_strict(value)


def test_all_llm_output_schemas_are_openai_strict_compatible() -> None:
    for model in (MasterProfile, JobRequirements, DocumentPackage, GroundednessAudit):
        _assert_objects_are_strict(to_strict_json_schema(model))
