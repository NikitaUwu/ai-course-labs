# -*- coding: utf-8 -*-
from pathlib import Path
import sys
from unittest.mock import Mock, patch

import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import agent_core
from guardrails.input_validator import RiskLevel, SecurityCheck
from memory.working_memory import WorkingMemory


class FakeExecutor:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def run(self, query: str):
        self.calls.append(query)
        if self.error is not None:
            raise self.error
        return self.result


class FakeSemanticMemory:
    def __init__(self):
        self.calls = []

    def add_document(self, content, metadata=None, doc_id=None):
        self.calls.append(
            {
                "content": content,
                "metadata": metadata or {},
                "doc_id": doc_id,
            }
        )
        return doc_id or "fake-doc-id"


@pytest.fixture
def console_reporter(capfd):
    def _report(test_name: str, details: str):
        with capfd.disabled():
            print(f"[PASS] {test_name}: {details}", flush=True)

    return _report


def test_init_creates_agent_with_memory_and_guardrails(console_reporter):
    fake_llm = object()
    fake_agent_executor = Mock()
    fake_semantic_memory = FakeSemanticMemory()
    config = agent_core.AgentConfig(
        memory_enabled=True,
        guardrails_enabled=True,
        verbose=False,
    )

    with (
        patch.object(agent_core.AIAgent, "_init_llm", return_value=fake_llm),
        patch.object(agent_core.AIAgent, "_init_tools", return_value=["tool-a", "tool-b"]),
        patch.object(agent_core.AIAgent, "_init_agent", return_value=fake_agent_executor),
        patch.object(agent_core, "SemanticMemory", return_value=fake_semantic_memory),
    ):
        agent = agent_core.AIAgent(config)

    assert agent.llm is fake_llm
    assert agent.tools == ["tool-a", "tool-b"]
    assert isinstance(agent.working_memory, WorkingMemory)
    assert agent.semantic_memory is fake_semantic_memory
    assert isinstance(agent.guardrails, agent_core.InputGuardrails)
    assert agent.agent is fake_agent_executor
    assert agent.request_count == 0
    assert agent.total_tokens == 0
    console_reporter(
        "test_init_creates_agent_with_memory_and_guardrails",
        "агент инициализируется с памятью, guardrails и executor",
    )


def test_init_disables_semantic_memory_when_initialization_fails(console_reporter):
    config = agent_core.AgentConfig(
        memory_enabled=True,
        guardrails_enabled=False,
        verbose=False,
    )

    with (
        patch.object(agent_core.AIAgent, "_init_llm", return_value=object()),
        patch.object(agent_core.AIAgent, "_init_tools", return_value=[]),
        patch.object(agent_core.AIAgent, "_init_agent", return_value=Mock()),
        patch.object(agent_core, "SemanticMemory", side_effect=RuntimeError("chroma failed")),
    ):
        agent = agent_core.AIAgent(config)

    assert isinstance(agent.working_memory, WorkingMemory)
    assert agent.semantic_memory is None
    assert agent.guardrails is None
    console_reporter(
        "test_init_disables_semantic_memory_when_initialization_fails",
        "при ошибке SemanticMemory агент продолжает работу без семантической памяти",
    )


def test_init_agent_uses_structured_chat_and_memory(console_reporter):
    fake_config = agent_core.AgentConfig(max_iterations=7, verbose=False)
    fake_object = agent_core.AIAgent.__new__(agent_core.AIAgent)
    fake_object.tools = ["tool-1"]
    fake_object.llm = "fake-llm"
    fake_object.config = fake_config
    fake_object.working_memory = object()

    with (
        patch.object(agent_core, "ConversationBufferMemory", return_value="memory-object") as memory_mock,
        patch.object(agent_core, "initialize_agent", return_value="executor-object") as init_mock,
    ):
        result = agent_core.AIAgent._init_agent(fake_object)

    assert result == "executor-object"
    memory_mock.assert_called_once_with(
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=4000,
    )
    init_mock.assert_called_once()

    kwargs = init_mock.call_args.kwargs
    assert kwargs["tools"] == ["tool-1"]
    assert kwargs["llm"] == "fake-llm"
    assert kwargs["agent"] == agent_core.AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
    assert kwargs["memory"] == "memory-object"
    assert kwargs["verbose"] is False
    assert kwargs["max_iterations"] == 7
    assert kwargs["handle_parsing_errors"] is True
    console_reporter(
        "test_init_agent_uses_structured_chat_and_memory",
        "initialize_agent вызывается в structured-chat режиме с памятью",
    )


def test_run_success_updates_memories_and_stats(console_reporter):
    fake_semantic_memory = FakeSemanticMemory()
    fake_executor = FakeExecutor(result="готовый ответ")
    config = agent_core.AgentConfig(
        memory_enabled=True,
        guardrails_enabled=True,
        verbose=False,
    )

    with (
        patch.object(agent_core.AIAgent, "_init_llm", return_value=object()),
        patch.object(agent_core.AIAgent, "_init_tools", return_value=["tool"]),
        patch.object(agent_core.AIAgent, "_init_agent", return_value=fake_executor),
        patch.object(agent_core, "SemanticMemory", return_value=fake_semantic_memory),
    ):
        agent = agent_core.AIAgent(config)

    response = agent.run("привет", session_id="session-123")

    assert response.success is True
    assert response.answer == "готовый ответ"
    assert response.error is None
    assert response.tokens_used == agent._estimate_tokens("привет", "готовый ответ")
    assert agent.request_count == 1
    assert agent.total_tokens == response.tokens_used
    assert fake_executor.calls == ["привет"]

    messages = agent.working_memory.get_messages()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "привет"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "готовый ответ"

    assert len(fake_semantic_memory.calls) == 1
    saved = fake_semantic_memory.calls[0]
    assert saved["metadata"] == {"session_id": "session-123", "type": "interaction"}
    assert "Query: привет" in saved["content"]
    assert "Answer: готовый ответ" in saved["content"]
    console_reporter(
        "test_run_success_updates_memories_and_stats",
        "успешный run обновляет память, статистику и семантическое хранилище",
    )


def test_run_rejects_request_when_guardrails_block_input(console_reporter):
    fake_executor = FakeExecutor(result="не должен вызваться")
    config = agent_core.AgentConfig(
        memory_enabled=False,
        guardrails_enabled=True,
        verbose=False,
    )

    with (
        patch.object(agent_core.AIAgent, "_init_llm", return_value=object()),
        patch.object(agent_core.AIAgent, "_init_tools", return_value=["tool"]),
        patch.object(agent_core.AIAgent, "_init_agent", return_value=fake_executor),
    ):
        agent = agent_core.AIAgent(config)

    agent.guardrails.validate_input = Mock(
        return_value=SecurityCheck(
            is_safe=False,
            reason="Запрос содержит опасный паттерн",
            risk_level=RiskLevel.HIGH,
        )
    )

    response = agent.run("вредоносный запрос")

    assert response.success is False
    assert response.answer == "Запрос отклонён системой безопасности."
    assert response.error == "Запрос содержит опасный паттерн"
    assert response.duration_ms == 0
    assert response.tokens_used == 0
    assert fake_executor.calls == []
    assert agent.request_count == 0
    console_reporter(
        "test_run_rejects_request_when_guardrails_block_input",
        "guardrails блокируют опасный запрос до вызова executor",
    )


def test_run_returns_error_response_when_executor_fails(console_reporter):
    fake_executor = FakeExecutor(error=RuntimeError("executor crashed"))
    config = agent_core.AgentConfig(
        memory_enabled=False,
        guardrails_enabled=False,
        verbose=False,
    )

    with (
        patch.object(agent_core.AIAgent, "_init_llm", return_value=object()),
        patch.object(agent_core.AIAgent, "_init_tools", return_value=["tool"]),
        patch.object(agent_core.AIAgent, "_init_agent", return_value=fake_executor),
    ):
        agent = agent_core.AIAgent(config)

    response = agent.run("рабочий запрос")

    assert response.success is False
    assert response.answer == "Произошла ошибка при обработке запроса."
    assert response.error == "executor crashed"
    assert response.duration_ms == 0
    assert response.tokens_used == 0
    assert agent.request_count == 0
    assert agent.total_tokens == 0
    console_reporter(
        "test_run_returns_error_response_when_executor_fails",
        "ошибка executor преобразуется в безопасный AgentResponse",
    )


def test_get_stats_returns_current_agent_state(console_reporter):
    config = agent_core.AgentConfig(
        memory_enabled=False,
        guardrails_enabled=False,
        verbose=False,
    )

    with (
        patch.object(agent_core.AIAgent, "_init_llm", return_value=object()),
        patch.object(agent_core.AIAgent, "_init_tools", return_value=["a", "b", "c"]),
        patch.object(agent_core.AIAgent, "_init_agent", return_value=Mock()),
    ):
        agent = agent_core.AIAgent(config)

    agent.request_count = 4
    agent.total_tokens = 128
    stats = agent.get_stats()

    assert stats == {
        "name": config.name,
        "version": config.version,
        "tools_count": 3,
        "memory_enabled": False,
        "guardrails_enabled": False,
        "request_count": 4,
        "total_tokens": 128,
    }
    console_reporter(
        "test_get_stats_returns_current_agent_state",
        "get_stats возвращает актуальное состояние агента",
    )


def test_save_session_persists_full_session_to_semantic_memory(console_reporter):
    fake_semantic_memory = FakeSemanticMemory()
    config = agent_core.AgentConfig(
        memory_enabled=True,
        guardrails_enabled=False,
        verbose=False,
    )

    with (
        patch.object(agent_core.AIAgent, "_init_llm", return_value=object()),
        patch.object(agent_core.AIAgent, "_init_tools", return_value=[]),
        patch.object(agent_core.AIAgent, "_init_agent", return_value=Mock()),
        patch.object(agent_core, "SemanticMemory", return_value=fake_semantic_memory),
    ):
        agent = agent_core.AIAgent(config)

    agent.working_memory.add_message("user", "привет")
    agent.working_memory.add_message("assistant", "ответ")

    saved = agent.save_session("session-save")

    assert saved is True
    assert len(fake_semantic_memory.calls) == 1
    saved_call = fake_semantic_memory.calls[0]
    assert saved_call["metadata"] == {
        "session_id": "session-save",
        "type": "full_session",
    }
    assert "привет" in saved_call["content"]
    assert "ответ" in saved_call["content"]
    console_reporter(
        "test_save_session_persists_full_session_to_semantic_memory",
        "save_session сохраняет полную сессию в семантическую память",
    )


def test_save_session_returns_false_without_semantic_memory(console_reporter):
    config = agent_core.AgentConfig(
        memory_enabled=False,
        guardrails_enabled=False,
        verbose=False,
    )

    with (
        patch.object(agent_core.AIAgent, "_init_llm", return_value=object()),
        patch.object(agent_core.AIAgent, "_init_tools", return_value=[]),
        patch.object(agent_core.AIAgent, "_init_agent", return_value=Mock()),
    ):
        agent = agent_core.AIAgent(config)

    assert agent.save_session("session-save") is False
    console_reporter(
        "test_save_session_returns_false_without_semantic_memory",
        "save_session корректно возвращает False без семантической памяти",
    )


def test_estimate_tokens_uses_simple_character_based_rule(console_reporter):
    fake_object = agent_core.AIAgent.__new__(agent_core.AIAgent)
    tokens = agent_core.AIAgent._estimate_tokens(fake_object, "12345678", "1234")
    assert tokens == 3
    console_reporter(
        "test_estimate_tokens_uses_simple_character_based_rule",
        "оценка токенов использует простое правило по числу символов",
    )
