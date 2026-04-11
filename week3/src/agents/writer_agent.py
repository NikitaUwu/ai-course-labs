"""
Агент-писатель для Multi-Agent системы
Лабораторная работа №3
"""

from typing import Dict, Optional, List
import time
import logging
from agents.base_agent import BaseAgent, AgentConfig


logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """
    Агент-писатель.
    Назначение:
        Генерация финального отчёта на основе исследований и анализа.
    Возможности:
        • Структурирование контента
        • Генерация текста в заданном стиле
        • Форматирование по требованиям
        • Проверка согласованности
    Интеграция с дипломом:
        Может использоваться для автоматической генерации разделов дипломной работы, отчётов, документации.
    """
    def __init__(self, config: Optional[AgentConfig] = None):
        default_config = AgentConfig(
            role="Писатель",
            goal="Создать структурированный и качественный отчёт",
            backstory="""
                Вы — опытный технический писатель с expertise
                в создании документации и отчётов. Вы умеете превращать
                сложные данные в понятный и структурированный текст.
            """
        )
        if config:
            default_config.role = config.role
            default_config.goal = config.goal
            default_config.backstory = config.backstory
        super().__init__(default_config)
        self.documents_created = []
    
    def execute_task(self, task_description: str, context: Optional[Dict] = None) -> Dict:
        """
        Выполнение задачи написания.
        Args:
            task_description: Описание документа для создания
            context: Данные от исследователя и аналитика
        Returns:
            Dict: Созданный документ
        """
        
        start_time = time.time()
        self.state.current_task = task_description
        logger.info(f"Писатель начинает задачу: {task_description[:100]}...")

        # Получение данных от предыдущих агентов
        research_data = context.get("research_data", {}) if context else {}
        analysis_data = context.get("analysis_data", {}) if context else {}
        special_data = context.get("special_data", {}) if context else {}
        results = {
            "task": task_description,
            "status": "completed",
            "document": self._generate_document(
                task_description,
                research_data,
                analysis_data,
                special_data
            ),
            "structure": self._define_structure(task_description),
            "quality_metrics": self._assess_quality(),
            "execution_time": 0
        }
        results["execution_time"] = time.time() - start_time
        self.state.completed_tasks.append(task_description)
        self.statistics["tasks_completed"] += 1
        logger.info(f"Документ создан за {results['execution_time']:.2f}с")
        return results

    def _generate_document(
        self,
        topic: str,
        research: Dict,
        analysis: Dict,
        special: Optional[Dict] = None
    ) -> str:
        """Генерация документа."""
        document = f"""
            # Отчёт по теме: {topic}
            ## 1. Введение
            {self._write_introduction(topic)}

            ## 2. Основные findings
            {self._write_findings(research)}

            ## 3. Анализ данных
            {self._write_analysis(analysis, special or {})}

            ## 4. Выводы и рекомендации
            {self._write_conclusions(research, analysis, special or {})}

            ## 5. Источники
            {self._write_sources(research)}
        """
        self.documents_created.append({"topic": topic, "timestamp": time.time()})
        return document
    
    def _write_introduction(self, topic: str) -> str:
        return f"Данный отчёт посвящён исследованию темы: {topic}. " \
        f"В работе представлены результаты комплексного анализа..."
    
    def _write_findings(self, research: Dict) -> str:
        facts = research.get("key_facts", [])
        return "\n".join([f"• {fact}" for fact in facts])
    
    def _write_analysis(self, analysis: Dict, special: Optional[Dict] = None) -> str:
        insights = analysis.get("insights", [])
        analysis_lines = [f"• {insight}" for insight in insights]

        special = special or {}
        special_payload = special.get("analysis", {})
        special_data = special.get("data", {})

        if special_payload or special_data:
            analysis_lines.append("• Выполнен специализированный анализ данных графика функции")

        quality_score = special_data.get("quality_score")
        if quality_score is not None:
            analysis_lines.append(f"• Оценка качества извлечённых данных: {quality_score}")

        trend = special_payload.get("trend")
        if trend:
            analysis_lines.append(f"• Характер изменения функции: {trend}")

        readiness = special_payload.get("suitability_for_transformation")
        if readiness:
            analysis_lines.append(
                f"• Готовность данных к числовому преобразованию: {readiness}"
            )

        return "\n".join(analysis_lines)
    
    def _write_conclusions(
        self,
        research: Dict,
        analysis: Dict,
        special: Optional[Dict] = None
    ) -> str:
        recommendations = research.get("recommendations", [])
        conclusion_lines = [f"• {rec}" for rec in recommendations]

        for recommendation in (special or {}).get("recommendations", [])[:3]:
            conclusion_lines.append(f"• {recommendation}")

        return "\n".join(conclusion_lines)
    
    def _write_sources(self, research: Dict) -> str:
        sources = research.get("sources_found", [])
        return "\n".join([f"{i+1}. {s['title']} - {s['url']}" for i, s in enumerate(sources)])
    
    def _define_structure(self, topic: str) -> List[str]:
        return [
            "Введение",
            "Основные findings",
            "Анализ данных",
            "Выводы и рекомендации",
            "Источники"
        ]
    
    def _assess_quality(self) -> Dict:
        return {
            "readability_score": 0.88,
            "structure_score": 0.92,
            "completeness_score": 0.85
        }
    
    def get_capabilities(self) -> List[str]:
        """Возможности агента-писателя."""
        return [
            "Генерация структурированных документов",
            "Адаптация стиля под требования",
            "Форматирование по стандартам",
            "Проверка согласованности",
            "Создание различных типов отчётов"
        ]
    
    
