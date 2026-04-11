"""
Специализированный агент для дипломной работы
Лабораторная работа №3
Автор: Жданов Никита Валерьевич
Специальность: Фундаментальная информатика и информационные технологии
Тема диплома: Информационная система по извлечению и преобразованию числовых данных с графиков функций
"""

from typing import Any, Dict, List, Optional, Tuple
import time
import logging

from agents.base_agent import BaseAgent, AgentConfig


logger = logging.getLogger(__name__)

__all__ = ["SpecialAgent", "SpecialtyAgent"]


class SpecialAgent(BaseAgent):
    """
    Специализированный агент по анализу данных графиков функций.

    Назначение:
        Агент предназначен для обработки предметного контекста дипломной работы,
        связанного с оцифровкой графиков функций, анализом извлечённых точек,
        оценкой качества данных и формированием рекомендаций для последующих
        этапов обработки.

    Как связан с темой дипломной работы:
        Агент моделирует интеллектуальный слой системы, которая получает
        результаты извлечения данных с графика и помогает определить,
        насколько эти данные пригодны для дальнейшего преобразования,
        аппроксимации, экспорта и визуального контроля.

    Возможности:
        • Анализ контекста задачи по обработке графиков
        • Оценка набора извлечённых точек
        • Выявление диапазонов и характера изменения функции
        • Формирование рекомендаций для дальнейшей цифровизации графика

    Интеграция с дипломом:
        Агент может использоваться как вспомогательный модуль в практической
        части диплома для анализа качества промежуточных результатов после
        детекции осей, шкал, подписей и точек функции.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        default_config = AgentConfig(
            role="Специалист по обработке графиков",
            goal=(
                "Оценивать данные, извлечённые с графиков функций, и "
                "формировать рекомендации по их преобразованию"
            ),
            backstory="""
                Вы — специалист по цифровой обработке графиков и извлечению
                числовых данных. Вы понимаете, как переходить от изображения
                графика к структурам данных, как выявлять ошибки в точках,
                оценивать качество распознавания и готовить данные для
                последующего анализа, интерполяции и экспорта.
            """
        )
        if config:
            default_config.role = config.role
            default_config.goal = config.goal
            default_config.backstory = config.backstory
        super().__init__(default_config)

        self.processed_cases = 0
        self.detected_chart_types: List[str] = []
        self.last_quality_score = 0.0

    def execute_task(self, task_description: str, context: Optional[Dict] = None) -> Dict:
        """
        Выполнение специализированной задачи.
        Args:
            task_description: Описание задачи
            context: Контекст выполнения
        Returns:
            Dict: Результат выполнения
        """

        start_time = time.time()
        self.state.current_task = task_description
        logger.info(f"Специализированный агент начинает задачу: {task_description[:100]}...")

        processed_data = self._process_specialty_data(task_description, context)
        analysis = self._perform_analysis(task_description, context)
        recommendations = self._generate_recommendations(task_description, context)

        results = {
            "task": task_description,
            "status": "completed",
            "data": processed_data,
            "analysis": analysis,
            "recommendations": recommendations,
            "execution_time": 0
        }
        results["execution_time"] = time.time() - start_time

        self.processed_cases += 1
        self.state.completed_tasks.append(task_description)
        self.statistics["tasks_completed"] += 1
        self.statistics["total_execution_time"] += results["execution_time"]
        logger.info(f"Задача завершена за {results['execution_time']:.2f}с")
        return results

    def _process_specialty_data(self, task: str, context: Optional[Dict]) -> Dict:
        """
        Обработка специализированных данных по графику.
        """
        context = context or {}
        points = self._extract_points(context)
        chart_type = context.get("chart_type", "line")
        image_quality = context.get("image_quality", "unknown")
        axis_info = context.get("axis_info", {})

        if chart_type not in self.detected_chart_types:
            self.detected_chart_types.append(chart_type)

        x_range, y_range = self._calculate_ranges(points)
        quality_score = self._estimate_quality_score(points, context)
        self.last_quality_score = quality_score

        research_data = context.get("research_data", {})
        analysis_data = context.get("analysis_data", {})

        return {
            "domain": "graph_function_digitization",
            "task_focus": task,
            "chart_type": chart_type,
            "image_quality": image_quality,
            "points_detected": len(points),
            "x_range": x_range,
            "y_range": y_range,
            "axis_info_present": bool(axis_info),
            "quality_score": quality_score,
            "pipeline_stage": self._infer_pipeline_stage(context, points),
            "research_facts_count": len(research_data.get("key_facts", [])),
            "analysis_insights_count": len(analysis_data.get("insights", [])),
        }

    def _perform_analysis(self, task: str, context: Optional[Dict]) -> Dict:
        """
        Выполнение специализированного анализа.
        """
        context = context or {}
        points = self._extract_points(context)
        trend = self._detect_trend(points)
        repeated_x = self._count_repeated_x(points)
        missing_axis_data = not bool(context.get("axis_info"))
        risks = self._identify_risks(context, points)

        return {
            "trend": trend,
            "repeated_x_values": repeated_x,
            "axis_metadata_missing": missing_axis_data,
            "suitability_for_transformation": self._assess_transform_readiness(context, points),
            "detected_risks": risks,
            "expected_next_steps": self._get_expected_next_steps(context, points),
        }

    def _generate_recommendations(self, task: str, context: Optional[Dict] = None) -> List[str]:
        """
        Генерация рекомендаций.
        """
        context = context or {}
        points = self._extract_points(context)
        recommendations = [
            "Проверить корректность определения начала координат и масштаба по осям.",
            "Сохранить извлечённые точки в табличном формате для последующей верификации.",
        ]

        if len(points) < 3:
            recommendations.append(
                "Увеличить количество извлечённых точек, так как текущего набора недостаточно для уверенного анализа функции."
            )
        else:
            recommendations.append(
                "Выполнить интерполяцию или аппроксимацию функции после валидации точек."
            )

        if not context.get("axis_info"):
            recommendations.append(
                "Добавить метаданные об осях графика, чтобы перейти от пиксельных координат к числовым значениям."
            )

        if context.get("image_quality", "unknown") in {"low", "poor"}:
            recommendations.append(
                "Перед повторным извлечением точек улучшить качество изображения: удалить шум, сетку и повысить контраст."
            )

        if self._count_repeated_x(points) > 0:
            recommendations.append(
                "Проверить повторяющиеся значения X: они могут указывать на ошибки распознавания или на график неявной зависимости."
            )

        return recommendations

    def get_capabilities(self) -> List[str]:
        """Возможности специализированного агента."""
        return [
            "Анализ извлечённых точек графика",
            "Оценка готовности данных к преобразованию",
            "Выявление проблем качества цифровизации",
            "Формирование рекомендаций по обработке графиков",
            "Поддержка задач дипломной системы по оцифровке графиков функций",
        ]

    def _extract_points(self, context: Dict[str, Any]) -> List[Tuple[float, float]]:
        """Извлечение и нормализация списка точек из контекста."""
        raw_points = (
            context.get("graph_points")
            or context.get("extracted_points")
            or context.get("points")
            or []
        )

        normalized: List[Tuple[float, float]] = []

        for item in raw_points:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                normalized.append((float(item[0]), float(item[1])))
            elif isinstance(item, dict) and "x" in item and "y" in item:
                normalized.append((float(item["x"]), float(item["y"])))

        return normalized

    def _calculate_ranges(
        self,
        points: List[Tuple[float, float]]
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """Расчёт диапазонов X и Y."""
        if not points:
            return None, None

        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        return (min(x_values), max(x_values)), (min(y_values), max(y_values))

    def _estimate_quality_score(self, points: List[Tuple[float, float]], context: Dict[str, Any]) -> float:
        """Упрощённая оценка качества набора данных."""
        score = 0.4

        if points:
            score += min(len(points), 10) * 0.04

        if context.get("axis_info"):
            score += 0.15

        image_quality = context.get("image_quality")
        if image_quality == "high":
            score += 0.15
        elif image_quality == "medium":
            score += 0.08
        elif image_quality in {"low", "poor"}:
            score -= 0.05

        return round(max(0.0, min(score, 1.0)), 2)

    def _infer_pipeline_stage(self, context: Dict[str, Any], points: List[Tuple[float, float]]) -> str:
        """Определение текущего этапа обработки графика."""
        if points and context.get("axis_info"):
            return "готово к числовому преобразованию"
        if points:
            return "точки извлечены, но не хватает параметров осей"
        if context.get("image_quality"):
            return "подготовка к извлечению графических элементов"
        return "постановка задачи"

    def _detect_trend(self, points: List[Tuple[float, float]]) -> str:
        """Простейшее определение характера изменения функции."""
        if len(points) < 2:
            return "недостаточно данных"

        sorted_points = sorted(points, key=lambda point: (point[0], point[1]))
        deltas = []

        for left, right in zip(sorted_points, sorted_points[1:]):
            if right[0] == left[0]:
                continue
            deltas.append(right[1] - left[1])

        if not deltas:
            return "не удаётся определить тренд"
        if all(delta > 0 for delta in deltas):
            return "монотонно возрастает"
        if all(delta < 0 for delta in deltas):
            return "монотонно убывает"
        if all(delta >= 0 for delta in deltas):
            return "неубывающая зависимость"
        if all(delta <= 0 for delta in deltas):
            return "невозрастающая зависимость"
        return "смешанное поведение"

    def _count_repeated_x(self, points: List[Tuple[float, float]]) -> int:
        """Подсчёт повторяющихся значений X."""
        if not points:
            return 0

        seen = {}
        repeated = 0

        for x_value, y_value in points:
            if x_value in seen and seen[x_value] != y_value:
                repeated += 1
            seen[x_value] = y_value

        return repeated

    def _identify_risks(self, context: Dict[str, Any], points: List[Tuple[float, float]]) -> List[str]:
        """Определение основных рисков обработки."""
        risks = []

        if not points:
            risks.append("Не обнаружены точки графика для анализа.")
        elif len(points) < 3:
            risks.append("Слишком мало точек для уверенного анализа поведения функции.")

        if not context.get("axis_info"):
            risks.append("Отсутствуют данные о шкалах и начале координат.")

        if context.get("image_quality", "unknown") in {"low", "poor"}:
            risks.append("Низкое качество изображения может приводить к ошибкам извлечения.")

        if self._count_repeated_x(points) > 0:
            risks.append("Есть повторяющиеся X со спорными значениями Y.")

        if not risks:
            risks.append("Критических рисков не обнаружено.")

        return risks

    def _assess_transform_readiness(self, context: Dict[str, Any], points: List[Tuple[float, float]]) -> str:
        """Оценка готовности данных к числовому преобразованию."""
        if not points:
            return "не готово"
        if not context.get("axis_info"):
            return "частично готово"
        if len(points) < 3:
            return "ограниченно готово"
        return "готово"

    def _get_expected_next_steps(self, context: Dict[str, Any], points: List[Tuple[float, float]]) -> List[str]:
        """Следующие ожидаемые этапы конвейера."""
        steps = []

        if not points:
            steps.append("детекция осей и линии функции")
            steps.append("извлечение опорных точек")
            return steps

        if not context.get("axis_info"):
            steps.append("определение шкал и подписей осей")

        steps.append("преобразование координат в числовые значения")
        steps.append("валидация и фильтрация точек")
        steps.append("экспорт в таблицу или использование для интерполяции")
        return steps


SpecialtyAgent = SpecialAgent


if __name__ == "__main__":
    agent = SpecialAgent()
    result = agent.execute_task(
        "Оцени пригодность извлечённых данных с графика функции к дальнейшему преобразованию",
        context={
            "chart_type": "line",
            "image_quality": "medium",
            "axis_info": {"x_scale": 0.1, "y_scale": 0.2},
            "graph_points": [(10, 120), (20, 110), (30, 95), (40, 80)],
        }
    )
    print(result)
