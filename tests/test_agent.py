from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

from cv_agent.agent import (
    GeminiStructuredClient,
    HedgedStructuredClient,
    OpenAIStructuredClient,
    OpenRouterStructuredClient,
    ProviderAPIError,
    ProviderPoolError,
    StructuredOutputError,
    _alignment_profile_payload,
    _compact_candidate_sources,
    _repair_evidence_quotes,
    build_reliable_structured_client,
    build_structured_client,
)
import pytest
from cv_agent.ingestion import ingest_request
from cv_agent.schemas import (
    GroundednessAudit,
    GroundednessFinding,
    IngestionRequest,
    ModelSettings,
    PipelineSnapshot,
)


ROOT = Path(__file__).resolve().parents[1]


class TransientError(RuntimeError):
    status_code = 429


class ExhaustedCreditsError(RuntimeError):
    status_code = 429
    body = {"error": {"type": "insufficient_quota", "code": "credit_balance_exhausted"}}


class ServiceUnavailableError(RuntimeError):
    status_code = 503


class FakeResponses:
    def __init__(self) -> None:
        self.calls = 0
        self.last_options: dict[str, object] = {}

    def parse(self, **options: object) -> SimpleNamespace:
        self.calls += 1
        self.last_options = options
        if self.calls == 1:
            raise TransientError("rate limited")
        return SimpleNamespace(
            id="resp_test",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
            output_parsed=GroundednessAudit(
                passed=True,
                findings=[
                    GroundednessFinding(
                        location="test",
                        supported=True,
                        explanation="supported",
                        supporting_achievement_ids=["a1"],
                    )
                ],
            ),
        )


class FakeGeminiCompletions:
    def __init__(self) -> None:
        self.calls = 0
        self.last_options: dict[str, object] = {}

    def parse(self, **options: object) -> SimpleNamespace:
        self.calls += 1
        self.last_options = options
        return SimpleNamespace(
            id="chatcmpl_test",
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=7, total_tokens=19),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        parsed=GroundednessAudit(
                            passed=True,
                            findings=[
                                GroundednessFinding(
                                    location="test",
                                    supported=True,
                                    explanation="supported",
                                    supporting_achievement_ids=["a1"],
                                )
                            ],
                        )
                    )
                )
            ],
        )


class FakeOpenRouterCompletions:
    def __init__(self) -> None:
        self.calls = 0
        self.last_options: dict[str, object] = {}
        self.options: list[dict[str, object]] = []

    def create(self, **options: object) -> SimpleNamespace:
        self.calls += 1
        self.last_options = options
        self.options.append(options)
        arguments = GroundednessAudit(
            passed=True,
            findings=[
                GroundednessFinding(
                    location="test",
                    supported=True,
                    explanation="supported",
                    supporting_achievement_ids=["a1"],
                )
            ],
        ).model_dump_json()
        return SimpleNamespace(
            id="or_test",
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=9, total_tokens=29),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        reasoning_details=[{"type": "reasoning.text", "text": "checked"}],
                        tool_calls=[
                            SimpleNamespace(
                                id=f"call_{self.calls}",
                                function=SimpleNamespace(
                                    name="submit_groundednessaudit",
                                    arguments=arguments,
                                )
                            )
                        ]
                    )
                )
            ],
        )


class FakeOpenRouterContentCompletions(FakeOpenRouterCompletions):
    def create(self, **options: object) -> SimpleNamespace:
        self.calls += 1
        self.last_options = options
        self.options.append(options)
        content = GroundednessAudit(
            passed=True,
            findings=[
                GroundednessFinding(
                    location="test",
                    supported=True,
                    explanation="supported",
                    supporting_achievement_ids=["a1"],
                )
            ],
        ).model_dump_json()
        return SimpleNamespace(
            id="or_content_test",
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=9, total_tokens=29),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=f"```json\n{content}\n```",
                        reasoning_details=[],
                        tool_calls=[],
                    )
                )
            ],
        )


class FakeOpenRouterRawCompletions:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.last_options: dict[str, object] = {}
        self.with_raw_response = self

    def create(self, **options: object) -> SimpleNamespace:
        self.last_options = options
        return SimpleNamespace(text=json.dumps(self.payload))


def test_transient_openai_errors_use_exponential_retry() -> None:
    responses = FakeResponses()
    sleeps: list[float] = []
    client = OpenAIStructuredClient(
        ModelSettings(
            model="test-model",
            temperature=0.2,
            max_correction_retries=2,
            keyword_coverage_threshold=0.65,
            metric_density_threshold=0.60,
            groundedness_threshold=1.0,
        ),
        client=SimpleNamespace(responses=responses),
        sleep=sleeps.append,
    )
    result = client.parse(
        GroundednessAudit,
        system_prompt="audit",
        user_prompt="payload",
        operation="test",
    )
    assert result.passed
    assert responses.calls == 2
    assert len(sleeps) == 1
    assert 1.0 <= sleeps[0] <= 1.5
    assert responses.last_options["temperature"] == 0.2


def test_reasoning_model_omits_unsupported_temperature() -> None:
    responses = FakeResponses()
    responses.calls = 1
    client = OpenAIStructuredClient(
        ModelSettings(
            model="gpt-5.6-sol",
            temperature=0.2,
            max_correction_retries=2,
            keyword_coverage_threshold=0.65,
            metric_density_threshold=0.60,
            groundedness_threshold=1.0,
        ),
        client=SimpleNamespace(responses=responses),
    )
    client.parse(
        GroundednessAudit,
        system_prompt="audit",
        user_prompt="payload",
        operation="test",
    )
    assert "temperature" not in responses.last_options


def test_gemini_uses_openai_compatible_chat_parse() -> None:
    completions = FakeGeminiCompletions()
    client = GeminiStructuredClient(
        ModelSettings(
            provider="gemini",
            model="gemini-3.8-flash",
            temperature=0.2,
            max_correction_retries=2,
            keyword_coverage_threshold=0.65,
            metric_density_threshold=0.60,
            groundedness_threshold=1.0,
        ),
        client=SimpleNamespace(
            beta=SimpleNamespace(
                chat=SimpleNamespace(completions=completions),
            ),
        ),
    )
    result = client.parse(
        GroundednessAudit,
        system_prompt="audit",
        user_prompt="payload",
        operation="test",
    )
    assert result.passed
    assert completions.calls == 1
    assert completions.last_options["model"] == "gemini-3.8-flash"
    assert completions.last_options["response_format"] is GroundednessAudit
    assert "temperature" not in completions.last_options


def test_openrouter_uses_reasoning_and_forced_schema_tool() -> None:
    completions = FakeOpenRouterCompletions()
    client = OpenRouterStructuredClient(
        ModelSettings(
            provider="openrouter",
            model="nvidia/nemotron-3.5-lightning:free",
            temperature=0.2,
            max_correction_retries=2,
            keyword_coverage_threshold=0.65,
            metric_density_threshold=0.60,
            groundedness_threshold=1.0,
        ),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    result = client.parse(
        GroundednessAudit,
        system_prompt="audit",
        user_prompt="payload",
        operation="test",
    )

    assert result.passed
    assert completions.calls == 1
    assert completions.last_options["model"] == "nvidia/nemotron-3.5-lightning:free"
    assert completions.last_options["extra_body"] == {
        "reasoning": {"effort": "none", "exclude": True},
        "provider": {"allow_fallbacks": True, "require_parameters": True},
    }
    assert completions.last_options["max_tokens"] == 4_000
    tools = completions.last_options["tools"]
    assert isinstance(tools, list)
    assert tools[0]["function"]["name"] == "submit_groundednessaudit"
    assert completions.last_options["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_groundednessaudit"},
    }

    client.parse(
        GroundednessAudit,
        system_prompt="audit",
        user_prompt="correct the previous result",
        operation="test",
    )
    second_messages = completions.options[1]["messages"]
    assert second_messages == [
        {"role": "system", "content": "audit"},
        {"role": "user", "content": "correct the previous result"},
    ]


def test_openrouter_accepts_strict_json_content_when_tool_call_is_omitted() -> None:
    completions = FakeOpenRouterContentCompletions()
    client = OpenRouterStructuredClient(
        ModelSettings(
            provider="openrouter",
            model="nvidia/nemotron-3.5-lightning:free",
            temperature=0.2,
            max_correction_retries=2,
            keyword_coverage_threshold=0.65,
            metric_density_threshold=0.60,
            groundedness_threshold=1.0,
        ),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    result = client.parse(
        GroundednessAudit,
        system_prompt="audit",
        user_prompt="payload",
        operation="test",
    )

    assert result.passed


def test_openrouter_parses_raw_response_before_sdk_materialization() -> None:
    arguments = GroundednessAudit(
        passed=True,
        findings=[
            GroundednessFinding(
                location="test",
                supported=True,
                explanation="supported",
                supporting_achievement_ids=["a1"],
            )
        ],
    ).model_dump_json()
    completions = FakeOpenRouterRawCompletions(
        {
            "id": "raw_test",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_raw",
                                "type": "function",
                                "function": {
                                    "name": "submit_groundednessaudit",
                                    "arguments": arguments,
                                },
                            }
                        ],
                    }
                }
            ],
        }
    )
    client = OpenRouterStructuredClient(
        ModelSettings(
            provider="openrouter",
            model="nvidia/nemotron-3.5-lightning:free",
        ),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    result = client.parse(
        GroundednessAudit,
        system_prompt="audit",
        user_prompt="payload",
        operation="test",
    )

    assert result.passed


def test_openrouter_raw_null_choices_raise_structured_error() -> None:
    completions = FakeOpenRouterRawCompletions({"id": "raw_test", "choices": None})
    client = OpenRouterStructuredClient(
        ModelSettings(
            provider="openrouter",
            model="nvidia/nemotron-3.5-lightning:free",
        ),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    with pytest.raises(StructuredOutputError, match="no usable choices"):
        client.parse(
            GroundednessAudit,
            system_prompt="audit",
            user_prompt="payload",
            operation="test",
        )


def test_openrouter_uses_later_matching_tool_call(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_MAX_API_ATTEMPTS", "1")
    arguments = GroundednessAudit(
        passed=True,
        findings=[
            GroundednessFinding(
                location="test",
                supported=True,
                explanation="supported",
                supporting_achievement_ids=["a1"],
            )
        ],
    ).model_dump_json()
    completions = FakeOpenRouterRawCompletions(
        {
            "id": "multiple_tools",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "tool_calls": [
                            {
                                "id": "wrong",
                                "type": "function",
                                "function": {"name": "unrelated_tool", "arguments": "{}"},
                            },
                            {
                                "id": "right",
                                "type": "function",
                                "function": {
                                    "name": "submit_groundednessaudit",
                                    "arguments": arguments,
                                },
                            },
                        ]
                    },
                }
            ],
        }
    )
    client = OpenRouterStructuredClient(
        ModelSettings(provider="openrouter", model="test-model"),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    assert client.parse(
        GroundednessAudit,
        system_prompt="audit",
        user_prompt="payload",
        operation="test",
    ).passed


def test_openrouter_rejects_truncated_tool_output(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_MAX_API_ATTEMPTS", "1")
    completions = FakeOpenRouterRawCompletions(
        {
            "id": "truncated",
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "{\"passed\": true", "tool_calls": []},
                }
            ],
        }
    )
    client = OpenRouterStructuredClient(
        ModelSettings(provider="openrouter", model="test-model"),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    with pytest.raises(StructuredOutputError, match="incomplete/refused"):
        client.parse(
            GroundednessAudit,
            system_prompt="audit",
            user_prompt="payload",
            operation="test",
        )


def test_openrouter_preserves_upstream_timeout_classification(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_MAX_API_ATTEMPTS", "1")
    completions = FakeOpenRouterRawCompletions(
        {
            "error": {
                "code": "504",
                "message": "Upstream idle timeout exceeded",
            }
        }
    )
    client = OpenRouterStructuredClient(
        ModelSettings(provider="openrouter", model="test-model"),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    with pytest.raises(ProviderAPIError) as captured:
        client.parse(
            GroundednessAudit,
            system_prompt="audit",
            user_prompt="payload",
            operation="test",
        )

    assert captured.value.status_code == 504
    assert not isinstance(captured.value, StructuredOutputError)


def test_hedged_client_returns_other_provider_when_one_times_out() -> None:
    barrier = Barrier(2)
    result = GroundednessAudit(
        passed=True,
        findings=[
            GroundednessFinding(
                location="test",
                supported=True,
                explanation="supported",
                supporting_achievement_ids=["a1"],
            )
        ],
    )

    class StubClient:
        def __init__(self, provider: str, failure: Exception | None = None) -> None:
            self.settings = ModelSettings(provider=provider, model=f"{provider}-model")
            self.failure = failure
            self.calls = 0

        def parse(self, *_: object, **__: object) -> GroundednessAudit:
            self.calls += 1
            barrier.wait(timeout=2)
            if self.failure is not None:
                time.sleep(0.02)
                raise self.failure
            return result

    gemini = StubClient("gemini")
    openrouter = StubClient(
        "openrouter",
        ProviderAPIError(
            "OpenRouter",
            {"code": 504, "message": "Upstream idle timeout exceeded"},
        ),
    )
    client = HedgedStructuredClient(
        ModelSettings(provider="openrouter", model="ensemble"),
        [gemini, openrouter],  # type: ignore[list-item]
    )

    assert client.parse(
        GroundednessAudit,
        system_prompt="audit",
        user_prompt="payload",
        operation="test",
    ).passed
    assert gemini.calls == 1
    assert openrouter.calls == 1


def test_hedged_client_reports_when_all_providers_fail() -> None:
    class FailingClient:
        def __init__(self, provider: str, status: int) -> None:
            self.settings = ModelSettings(provider=provider, model=f"{provider}-model")
            self.status = status

        def parse(self, *_: object, **__: object) -> GroundednessAudit:
            raise ProviderAPIError(
                self.settings.provider,
                {"code": self.status, "message": "unavailable"},
            )

    client = HedgedStructuredClient(
        ModelSettings(provider="openrouter", model="ensemble"),
        [
            FailingClient("gemini", 504),
            FailingClient("openrouter", 504),
        ],  # type: ignore[list-item]
    )

    with pytest.raises(ProviderPoolError) as captured:
        client.parse(
            GroundednessAudit,
            system_prompt="audit",
            user_prompt="payload",
            operation="test",
        )
    assert captured.value.status_code == 504


def test_hedged_client_classifies_sdk_timeouts_without_status_codes() -> None:
    class APITimeoutError(RuntimeError):
        pass

    failure = ProviderPoolError(
        [
            ("gemini", APITimeoutError("timed out")),
            ("openrouter", APITimeoutError("timed out")),
        ]
    )

    assert failure.status_code == 504


def test_gemini_switches_models_after_first_503(monkeypatch) -> None:
    class FallbackCompletions(FakeGeminiCompletions):
        def __init__(self) -> None:
            super().__init__()
            self.models: list[str] = []

        def parse(self, **options: object) -> SimpleNamespace:
            model = str(options["model"])
            self.models.append(model)
            if model == "gemini-3.8-flash":
                raise ServiceUnavailableError("temporarily unavailable")
            return super().parse(**options)

    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.7-flash")
    completions = FallbackCompletions()
    sleeps: list[float] = []
    client = GeminiStructuredClient(
        ModelSettings(
            provider="gemini",
            model="gemini-3.8-flash",
            temperature=0.2,
            max_correction_retries=2,
            keyword_coverage_threshold=0.65,
            metric_density_threshold=0.60,
            groundedness_threshold=1.0,
        ),
        client=SimpleNamespace(
            beta=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        ),
        sleep=sleeps.append,
        max_api_attempts=5,
    )

    result = client.parse(
        GroundednessAudit,
        system_prompt="audit",
        user_prompt="payload",
        operation="test",
    )

    assert result.passed
    assert completions.models == ["gemini-3.8-flash", "gemini-3.7-flash"]
    assert sleeps == []


def test_openrouter_switches_to_configured_fallback_after_timeout(monkeypatch) -> None:
    arguments = GroundednessAudit(
        passed=True,
        findings=[
            GroundednessFinding(
                location="test",
                supported=True,
                explanation="supported",
                supporting_achievement_ids=["a1"],
            )
        ],
    ).model_dump_json()

    class FallbackRawCompletions:
        def __init__(self) -> None:
            self.with_raw_response = self
            self.models: list[str] = []

        def create(self, **options: object) -> SimpleNamespace:
            model = str(options["model"])
            self.models.append(model)
            if model == "nvidia/nemotron-3.5-lightning:free":
                return SimpleNamespace(
                    text=json.dumps(
                        {"error": {"code": 504, "message": "upstream timeout"}}
                    )
                )
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "id": "or_fallback",
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                        "choices": [
                            {
                                "finish_reason": "tool_calls",
                                "message": {
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_fallback",
                                            "type": "function",
                                            "function": {
                                                "name": "submit_groundednessaudit",
                                                "arguments": arguments,
                                            },
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                )
            )

    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "openrouter/free")
    monkeypatch.setenv("OPENROUTER_MAX_API_ATTEMPTS", "2")
    completions = FallbackRawCompletions()
    client = OpenRouterStructuredClient(
        ModelSettings(
            provider="openrouter",
            model="nvidia/nemotron-3.5-lightning:free",
        ),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    result = client.parse(
        GroundednessAudit,
        system_prompt="audit",
        user_prompt="payload",
        operation="test",
    )

    assert result.passed
    assert completions.models == [
        "nvidia/nemotron-3.5-lightning:free",
        "openrouter/free",
    ]


def test_reliable_factory_uses_both_configured_providers(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PARALLEL_PROVIDERS", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3-flash-preview")
    monkeypatch.setenv(
        "OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"
    )

    client = build_reliable_structured_client(
        ModelSettings(provider="openrouter", model="configured-primary")
    )

    assert isinstance(client, HedgedStructuredClient)
    assert [item.settings.provider for item in client.clients] == [
        "gemini",
        "openrouter",
    ]
    assert client.clients[0].settings.model == "gemini-3-flash-preview"


def test_profile_evidence_quotes_snap_to_exact_source(snapshot: PipelineSnapshot) -> None:
    payload = deepcopy(snapshot.master_profile.model_dump())
    payload["evidence_catalog"][3]["quote"] = (
        "Reduced pipeline latency by 42% with Spark partitioning and Airflow."
    )
    profile = snapshot.master_profile.model_validate(payload)
    bundle = ingest_request(
        IngestionRequest(
            jd_path=ROOT / "examples" / "job_description.md",
            jd_text=None,
            cv_paths=[
                ROOT / "examples" / "candidate_primary.md",
                ROOT / "examples" / "candidate_history.txt",
            ],
            max_chars_per_document=120_000,
        )
    )

    repaired = _repair_evidence_quotes(profile, bundle)

    assert repaired.evidence_catalog[3].quote == snapshot.master_profile.evidence_catalog[3].quote


def test_candidate_source_compaction_removes_repeated_long_lines() -> None:
    bundle = ingest_request(
        IngestionRequest(
            jd_path=ROOT / "examples" / "job_description.md",
            jd_text=None,
            cv_paths=[
                ROOT / "examples" / "candidate_primary.md",
                ROOT / "examples" / "candidate_history.txt",
            ],
            max_chars_per_document=120_000,
        )
    )
    repeated = "Reduced processing latency by 42% through careful pipeline optimization."
    first = bundle.candidate_documents[0].model_copy(
        update={"text": f"Candidate one\n{repeated}", "character_count": len(repeated) + 14}
    )
    second_text = f"Candidate two\n{repeated}\nAdded a unique platform achievement."
    second = bundle.candidate_documents[1].model_copy(
        update={"text": second_text, "character_count": len(second_text)}
    )
    compact_bundle = bundle.model_copy(update={"candidate_documents": [first, second]})

    payload = _compact_candidate_sources(compact_bundle)

    assert repeated in payload[0]["text"]
    assert repeated not in payload[1]["text"]
    assert "Added a unique platform achievement." in payload[1]["text"]


def test_alignment_payload_omits_raw_provenance_but_keeps_claim_ids(
    snapshot: PipelineSnapshot,
) -> None:
    payload = _alignment_profile_payload(snapshot.master_profile)

    assert "evidence_catalog" not in payload
    assert "contact_evidence_ids" not in payload
    assert "evidence_ids" not in payload["summary_facts"][0]
    assert "evidence_ids" not in payload["experience"][0]
    assert "evidence_ids" not in payload["experience"][0]["achievements"][0]
    assert payload["summary_facts"][0]["fact_id"] == "f_summary"
    assert payload["experience"][0]["achievements"][0]["achievement_id"] == "a1"


def test_structured_client_factory_selects_gemini() -> None:
    client = build_structured_client(
        ModelSettings(
            provider="gemini",
            model="gemini-3.8-flash",
            temperature=0.2,
            max_correction_retries=2,
            keyword_coverage_threshold=0.65,
            metric_density_threshold=0.60,
            groundedness_threshold=1.0,
        ),
        client=SimpleNamespace(),
    )
    assert isinstance(client, GeminiStructuredClient)


def test_structured_client_factory_selects_openrouter() -> None:
    client = build_structured_client(
        ModelSettings(
            provider="openrouter",
            model="nvidia/nemotron-3.5-lightning:free",
            temperature=0.2,
            max_correction_retries=2,
            keyword_coverage_threshold=0.65,
            metric_density_threshold=0.60,
            groundedness_threshold=1.0,
        ),
        client=SimpleNamespace(),
    )
    assert isinstance(client, OpenRouterStructuredClient)


def test_exhausted_credits_are_not_retried() -> None:
    assert not OpenAIStructuredClient._is_transient(ExhaustedCreditsError())
