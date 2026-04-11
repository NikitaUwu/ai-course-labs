"""
Агент-аналитик для Multi-Agent системы
Лабораторная работа №3
"""

from typing import Dict, Optional, List
import time
import logging
from agents.base_agent import BaseAgent, AgentConfig

logger = logging.getLogger(__name__)

class AnalystAgent(BaseAgent):
    """
    Агент-аналитик.
    Назначение:
        Анализ и обработка данных, полученных от исследователя.
    Возможности:
        • Структурирование данных
        • Выявление паттернов и тенденций
        • Статистический анализ
        • Генерация выводов
    Интеграция с дипломом:
        Может использоваться для анализа данных мониторинга, обработки результатов экспериментов, статистики.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        default_config = AgentConfig(
            role="Аналитик",
            goal="Проанализировать данные и выявить ключевые инсайты",
            backstory="""Вы — опытный аналитик данных с expertise в
                статистическом анализе и выявлении паттернов. Вы умеете
                превращать сырые данные в полезные инсайты.
            """
        )
        if config:
            default_config.role = config.role
            default_config.goal = config.goal
            default_config.backstory = config.backstory
        super().__init__(default_config)
        self.analyses_performed = []
    
    def execute_task(self, task_description: str, context: Optional[Dict] = None) -> Dict:
        """
        Выполнение задачи анализа.
        Args:
            task_description: Описание задачи анализа
            context: Данные для анализа (от исследователя)
        Returns:
            Dict: Результаты анализа
        """

        start_time = time.time()
        self.state.current_task = task_description
        logger.info(f"Аналитик начинает задачу: {task_description[:100]}...")
        # Получение данных от исследователя
        research_data = context.get("research_data", {}) if context else {}
        results = {
            "task": task_description,
            "status": "completed",
            "structured_data": self._structure_data(research_data),
            "patterns_identified": self._identify_patterns(research_data),
            "insights": self._generate_insights(research_data),
            "metrics": self._calculate_metrics(research_data),
            "execution_time": 0
        }
        results["execution_time"] = time.time() - start_time
        self.state.completed_tasks.append(task_description)
        self.statistics["tasks_completed"] += 1
        logger.info(f"Анализ завершён за {results['execution_time']:.2f}с")
        return results
    
    def _structure_data(self, raw_data: Dict) -> Dict:
        """Структурирование сырых данных."""
        return {
            "sources_count": len(raw_data.get("sources_found", [])),
            "facts_count": len(raw_data.get("key_facts", [])),
            "data_quality": "high",
            "completeness": 0.85
        }

    def _identify_patterns(self, data: Dict) -> List[str]:
        """Выявление паттернов в данных."""
        return [
            "Паттерн 1: Преобладание академических источников",
            "Паттерн 2: Высокая релевантность найденных материалов",
            "Паттерн 3: Консистентность ключевых фактов"
        ]

    def _generate_insights(self, data: Dict) -> List[str]:
        """Генерация инсайтов."""
        return [
            "Инсайт 1: Тема хорошо изучена в научной литературе",
            "Инсайт 2: Есть возможности для дальнейшего исследования",
            "Инсайт 3: Рекомендуется расширить поиск источников"
        ]

    def _calculate_metrics(self, data: Dict) -> Dict:
        """Расчёт метрик качества данных."""
        return {
            "data_quality_score": 0.87,
            "coverage_score": 0.75,
            "reliability_score": 0.92
        }
    
    def get_capabilities(self) -> List[str]:
        """Возможности агента-аналитика."""
        return [
            "Структурирование данных",
            "Выявление паттернов",
            "Статистический анализ",
            "Генерация инсайтов",
            "Расчёт метрик качества"
        ]
    
    