from pathlib import Path
import sys
from typing import Dict, List, Optional

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from agents.base_agent import AgentConfig, BaseAgent
from agents.researcher_agent import ResearcherAgent
from agents.analyst_agent import AnalystAgent
from agents.writer_agent import WriterAgent
from agents.special_agent import SpecialAgent, SpecialtyAgent
from crew.research_crew import ResearchCrew, run_demo


class DummyAgent(BaseAgent):
    def execute_task(self, task_description: str, context: Optional[Dict] = None) -> Dict:
        self.state.current_task = task_description
        self.state.completed_tasks.append(task_description)
        self.statistics["tasks_completed"] += 1
        return {"task": task_description, "status": "completed", "context": context or {}}

    def get_capabilities(self) -> List[str]:
        return ["dummy"]


@pytest.fixture
def graph_context() -> Dict:
    return {
        "chart_type": "line",
        "image_quality": "high",
        "axis_info": {"x_scale": 0.1, "y_scale": 0.2, "origin": (0, 240)},
        "graph_points": [(0, 10), (1, 8), (2, 6), (3, 4)],
    }


def test_base_agent_tracks_messages_and_statistics():
    agent = DummyAgent(AgentConfig(role="Dummy"))

    outgoing = agent.send_message("receiver-1", {"payload": "hello"})
    agent.receive_message({"sender_id": "receiver-1", "content": "world"})

    stats = agent.get_statistics()

    assert outgoing["receiver_id"] == "receiver-1"
    assert stats["role"] == "Dummy"
    assert stats["statistics"]["messages_sent"] == 1
    assert stats["statistics"]["messages_received"] == 1
    assert stats["message_history_length"] == 2


def test_special_agent_alias_points_to_current_implementation():
    assert SpecialtyAgent is SpecialAgent


def test_researcher_agent_builds_research_payload():
    agent = ResearcherAgent()
    result = agent.execute_task(
        "Изучи методы цифровизации графиков функций",
        context={"keywords": ["OCR", "графики"]},
    )

    assert result["status"] == "completed"
    assert len(result["sources_found"]) == 2
    assert len(result["key_facts"]) == 3
    assert len(result["recommendations"]) == 3
    assert agent.sources_searched == result["sources_found"]
    assert agent.facts_collected == result["key_facts"]


def test_analyst_agent_uses_research_data_from_context():
    agent = AnalystAgent()
    result = agent.execute_task(
        "Проанализируй результаты исследования",
        context={
            "research_data": {
                "sources_found": [{"title": "s1"}, {"title": "s2"}],
                "key_facts": ["f1", "f2", "f3", "f4"],
            }
        },
    )

    assert result["status"] == "completed"
    assert result["structured_data"]["sources_count"] == 2
    assert result["structured_data"]["facts_count"] == 4
    assert result["metrics"]["reliability_score"] == 0.92
    assert len(result["insights"]) == 3


def test_special_agent_analyzes_graph_data_and_updates_state(graph_context: Dict):
    agent = SpecialAgent()
    context = {
        **graph_context,
        "research_data": {"key_facts": ["f1", "f2"]},
        "analysis_data": {"insights": ["i1", "i2", "i3"]},
    }

    result = agent.execute_task(
        "Оцени пригодность извлечённых точек к числовому преобразованию",
        context=context,
    )

    assert result["status"] == "completed"
    assert result["data"]["domain"] == "graph_function_digitization"
    assert result["data"]["points_detected"] == 4
    assert result["data"]["pipeline_stage"] == "готово к числовому преобразованию"
    assert result["analysis"]["trend"] == "монотонно убывает"
    assert result["analysis"]["suitability_for_transformation"] == "готово"
    assert agent.processed_cases == 1
    assert agent.last_quality_score == result["data"]["quality_score"]
    assert "line" in agent.detected_chart_types


def test_writer_agent_embeds_specialized_analysis_into_document():
    agent = WriterAgent()
    result = agent.execute_task(
        "Сформируй итоговый отчёт",
        context={
            "research_data": {
                "key_facts": ["Факт 1", "Факт 2"],
                "recommendations": ["Общая рекомендация"],
                "sources_found": [{"title": "Источник", "url": "https://example.com"}],
            },
            "analysis_data": {"insights": ["Инсайт 1", "Инсайт 2"]},
            "special_data": {
                "data": {"quality_score": 0.87},
                "analysis": {
                    "trend": "монотонно возрастает",
                    "suitability_for_transformation": "готово",
                },
                "recommendations": [
                    "Проверить корректность осей.",
                    "Выполнить интерполяцию.",
                ],
            },
        },
    )

    document = result["document"]

    assert result["status"] == "completed"
    assert "Выполнен специализированный анализ данных графика функции" in document
    assert "Оценка качества извлечённых данных: 0.87" in document
    assert "Характер изменения функции: монотонно возрастает" in document
    assert "Готовность данных к числовому преобразованию: готово" in document
    assert "Проверить корректность осей." in document
    assert "Источник - https://example.com" in document


def test_research_crew_runs_full_pipeline(graph_context: Dict):
    crew = ResearchCrew()

    result = crew.execute(
        "Исследование методов извлечения данных с графиков функций",
        context=graph_context,
    )

    assert result.success is True
    assert set(result.agent_results.keys()) == {"researcher", "analyst", "special", "writer"}
    assert "Выполнен специализированный анализ данных графика функции" in result.final_output
    assert "Оценка качества извлечённых данных" in result.final_output
    assert crew.statistics["successful_executions"] == 1


def test_research_crew_returns_statistics_for_all_agents(graph_context: Dict):
    crew = ResearchCrew()
    crew.execute("Проведи исследование и подготовь отчёт", context=graph_context)

    stats = crew.get_statistics()

    assert stats["crew_name"] == "ResearchCrew"
    assert stats["statistics"]["crews_executed"] == 1
    assert set(stats["agent_statistics"].keys()) == {"researcher", "analyst", "special", "writer"}
    assert stats["agent_statistics"]["special"]["role"] == "Специалист по обработке графиков"


def test_research_crew_returns_failed_result_when_agent_raises(
    graph_context: Dict,
    monkeypatch: pytest.MonkeyPatch,
):
    crew = ResearchCrew()

    def broken_execute_task(task_description: str, context: Optional[Dict] = None) -> Dict:
        raise RuntimeError("special agent failed")

    monkeypatch.setattr(crew.special_agent, "execute_task", broken_execute_task)

    result = crew.execute("Тест ошибки конвейера", context=graph_context)

    assert result.success is False
    assert "special agent failed" in result.final_output
    assert crew.statistics["failed_executions"] == 1
    assert crew.statistics["crews_executed"] == 1


def test_run_demo_returns_success_and_prints_sections(capsys: pytest.CaptureFixture[str]):
    result = run_demo()
    captured = capsys.readouterr()

    assert result.success is True
    assert "ЗАДАЧА" in captured.out
    assert "РЕЗУЛЬТАТ ИССЛЕДОВАТЕЛЯ" in captured.out
    assert "РЕЗУЛЬТАТ АНАЛИТИКА" in captured.out
    assert "РЕЗУЛЬТАТ СПЕЦИАЛИЗИРОВАННОГО АГЕНТА" in captured.out
    assert "ИТОГОВЫЙ ОТЧЁТ ПИСАТЕЛЯ" in captured.out
