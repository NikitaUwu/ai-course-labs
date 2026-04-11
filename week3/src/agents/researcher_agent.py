"""
Агент-исследователь для Multi-Agent системы
Лабораторная работа №3
"""

from typing import Dict, Optional, List
import time
import logging
from agents.base_agent import BaseAgent, AgentConfig


logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """
    Агент-исследователь.
    Назначение:
        Сбор и первичная обработка информации по заданной теме.
    Возможности:
        • Поиск информации в источниках
        • Фильтрация релевантных данных
        • Структурирование найденной информации
        • Выявление ключевых фактов
    Интеграция с дипломом:
        Может использоваться для автоматического сбора литературы, анализа патентов, мониторинга источников.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        default_config = AgentConfig(
            role="Исследователь",
            goal="Найти и собрать актуальную информацию по заданной теме",
            backstory="""
                Вы — опытный исследователь с expertise в поиске
                и анализе информации. Вы умеете находить релевантные источники,
                проверять достоверность данных и выделять ключевые факты.
            """
        )
        if config:
            default_config.role = config.role
            default_config.goal = config.goal
            default_config.backstory = config.backstory
        
        super().__init__(default_config)

        # Специфичные для исследователя атрибуты
        self.sources_searched = []
        self.facts_collected = []
    
    def execute_task(self, task_description: str, context: Optional[Dict] = None) -> Dict:
        """
        Выполнение задачи исследования.
        Args:
            task_description: Описание темы исследования
            context: Дополнительный контекст (ключевые слова, ограничения)
        Returns:
            Dict: Результаты исследования
        """

        start_time = time.time()
        self.state.current_task = task_description
        logger.info(f"Исследователь начинает задачу: {task_description[:100]}...")

        # Имитация процесса исследования
        # В production: подключить реальные инструменты поиска
        results = {
            "task": task_description,
            "status": "completed",
            "sources_found": self._search_sources(task_description, context),
            "key_facts": self._extract_facts(task_description, context),
            "recommendations": self._generate_recommendations(task_description),
            "execution_time": 0
        }
        results["execution_time"] = time.time() - start_time
        self.state.completed_tasks.append(task_description)
        self.statistics["tasks_completed"] += 1
        self.statistics["total_execution_time"] += results["execution_time"]
        logger.info(f"Исследование завершено за {results['execution_time']:.2f}с")
        return results
    
    def _search_sources(self, topic: str, context: Optional[Dict]) -> List[Dict]:
        """Поиск источников информации."""
        # Учебная реализация
        sources = [
            {
                "title": f"Источник по теме: {topic[:50]}",
                "type": "academic",
                "relevance": 0.85,
                "url": "https://example.com/source1"
            },
            {
                "title": f"Актуальные данные: {topic[:50]}",
                "type": "news",
                "relevance": 0.72,
                "url": "https://example.com/source2"
            }
        ]
        self.sources_searched = sources
        return sources
    
    def _extract_facts(self, topic: str, context: Optional[Dict]) -> List[str]:
        """Извлечение ключевых фактов."""
        facts = [
            f"Ключевой факт 1 по теме: {topic[:30]}",
            f"Ключевой факт 2 по теме: {topic[:30]}",
            f"Ключевой факт 3 по теме: {topic[:30]}"
        ]
        self.facts_collected = facts
        return facts
    
    def _generate_recommendations(self, topic: str) -> List[str]:
        """Генерация рекомендаций для дальнейшей работы."""
        return [
            "Рекомендуется углубить исследование в направлении...",
            "Следует проверить следующие источники...",
            "Обратить внимание на следующие аспекты..."
        ]
    
    def get_capabilities(self) -> List[str]:
        """Возможности агента-исследователя."""
        return [
            "Поиск информации в источниках",
            "Фильтрация по релевантности",
            "Извлечение ключевых фактов",
            "Генерация рекомендаций",
            "Оценка достоверности источников"
        ]