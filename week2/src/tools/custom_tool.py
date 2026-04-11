# -*- coding: utf-8 -*-
"""
Специализированный инструмент для дипломной темы:
"Информационная система по извлечению и преобразованию числовых данных
с графиков функций".

Инструмент принимает уже извлеченные точки графика в пиксельных координатах,
преобразует их в значения осей графика и возвращает краткий анализ результата.
"""

import logging
import re
from typing import ClassVar, List, Literal, Pattern, Tuple, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class CustomToolInput(BaseModel):
    """
    Входные параметры инструмента преобразования точек графика.

    Формат points_text:
        "(120,340); (180,300); (240,260)"
        или
        "120,340
         180,300
         240,260"

    Для дробных значений используйте точку: 12.5
    """

    points_text: str = Field(
        description=(
            "Список точек графика в пиксельных координатах. "
            "Каждая точка задается как x,y. "
            "Точки можно разделять ';' или переводом строки."
        ),
        min_length=3,
        max_length=5000,
    )
    x_origin: float = Field(
        description=(
            "Пиксельная координата X начала осей или точки отсчета "
            "для преобразования."
        ),
        default=0.0,
    )
    y_origin: float = Field(
        description=(
            "Пиксельная координата Y начала осей. "
            "Используется для перевода координат изображения в систему графика."
        ),
        default=0.0,
    )
    x_scale: float = Field(
        description="Масштаб по оси X: сколько единиц графика соответствует одному пикселю.",
        default=1.0,
        gt=0,
    )
    y_scale: float = Field(
        description="Масштаб по оси Y: сколько единиц графика соответствует одному пикселю.",
        default=1.0,
        gt=0,
    )
    precision: int = Field(
        description="Количество знаков после запятой в результате.",
        default=3,
        ge=0,
        le=10,
    )
    output_mode: Literal["summary", "table", "csv"] = Field(
        description=(
            "Формат ответа: "
            "summary - краткий анализ и таблица, "
            "table - только таблица, "
            "csv - CSV-представление."
        ),
        default="summary",
    )


class CustomTool(BaseTool):
    """
    Инструмент для преобразования точек графика функций.

    Назначение:
        Используется после этапа извлечения координат с изображения графика.
        Инструмент переводит пиксельные координаты в числовые значения осей,
        сортирует точки по X и формирует удобный для анализа результат.

    Как связан с темой диплома:
        Это упрощенная реализация этапа преобразования данных в системе
        извлечения числовых значений с графиков функций.

    Когда использовать:
        - когда пользователь уже получил набор точек с графика;
        - когда нужно перевести пиксели в реальные значения осей;
        - когда нужен краткий анализ диапазона и поведения функции.

    Ограничения:
        - инструмент не распознает изображение сам;
        - работает только с уже переданными координатами;
        - предполагает линейное преобразование координат;
        - для дробных чисел нужно использовать точку, а не запятую.
    """

    name: str = "transform_graph_points"
    description: str = """
        Преобразует извлеченные с графика пиксельные координаты точек
        в числовые значения осей и возвращает краткий анализ.
        Используйте этот инструмент, когда у вас уже есть набор точек графика
        и известны начало координат и масштаб по осям.
        Инструмент полезен для задач цифровизации графиков функций,
        подготовки таблицы значений и проверки диапазонов данных.
    """
    args_schema: Type[BaseModel] = CustomToolInput

    _number_pattern: ClassVar[Pattern[str]] = re.compile(r"[-+]?\d+(?:\.\d+)?")

    def _run(
        self,
        points_text: str,
        x_origin: float = 0.0,
        y_origin: float = 0.0,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        precision: int = 3,
        output_mode: str = "summary",
    ) -> str:
        """
        Основная логика инструмента.

        Преобразование выполняется по формулам:
            x_value = (x_pixel - x_origin) * x_scale
            y_value = (y_origin - y_pixel) * y_scale

        Это соответствует типичной системе координат изображения,
        где ось Y направлена вниз.
        """

        logger.info(
            "Вызов transform_graph_points: x_origin=%s, y_origin=%s, x_scale=%s, y_scale=%s",
            x_origin,
            y_origin,
            x_scale,
            y_scale,
        )

        try:
            pixel_points = self._parse_points(points_text)
            rows = self._transform_points(
                pixel_points=pixel_points,
                x_origin=x_origin,
                y_origin=y_origin,
                x_scale=x_scale,
                y_scale=y_scale,
            )

            if output_mode == "csv":
                return self._format_csv(rows, precision)
            if output_mode == "table":
                return self._format_table(rows, precision)

            return self._build_summary(
                rows=rows,
                precision=precision,
                x_origin=x_origin,
                y_origin=y_origin,
                x_scale=x_scale,
                y_scale=y_scale,
            )
        except ValueError as exc:
            return f"Ошибка обработки точек графика: {exc}"
        except Exception as exc:
            logger.exception("Непредвиденная ошибка transform_graph_points")
            return f"Внутренняя ошибка инструмента: {exc}"

    def _parse_points(self, points_text: str) -> List[Tuple[float, float]]:
        """Парсинг списка точек из строки."""
        cleaned_text = points_text.strip()
        if not cleaned_text:
            raise ValueError("Список точек пуст.")

        chunks = [
            chunk.strip()
            for chunk in re.split(r"[;\n]+", cleaned_text)
            if chunk.strip()
        ]

        points: List[Tuple[float, float]] = []

        if len(chunks) > 1:
            for index, chunk in enumerate(chunks, start=1):
                numbers = self._extract_numbers(chunk)
                if len(numbers) != 2:
                    raise ValueError(
                        f"Точка #{index} должна содержать ровно две координаты, получено: {chunk}"
                    )
                points.append((numbers[0], numbers[1]))
        else:
            numbers = self._extract_numbers(cleaned_text)
            if len(numbers) < 4 or len(numbers) % 2 != 0:
                raise ValueError(
                    "Не удалось разобрать точки. "
                    "Используйте формат '(x,y); (x,y)' или по одной точке на строку."
                )
            for index in range(0, len(numbers), 2):
                points.append((numbers[index], numbers[index + 1]))

        if len(points) < 2:
            raise ValueError("Для анализа требуется минимум две точки.")

        return points

    def _extract_numbers(self, text: str) -> List[float]:
        """Извлечение чисел из текстового фрагмента."""
        matches = self._number_pattern.findall(text)
        return [float(value) for value in matches]

    def _transform_points(
        self,
        pixel_points: List[Tuple[float, float]],
        x_origin: float,
        y_origin: float,
        x_scale: float,
        y_scale: float,
    ) -> List[dict]:
        """Преобразование пиксельных координат в координаты графика."""
        rows = []

        for index, (pixel_x, pixel_y) in enumerate(pixel_points, start=1):
            graph_x = (pixel_x - x_origin) * x_scale
            graph_y = (y_origin - pixel_y) * y_scale
            rows.append(
                {
                    "index": index,
                    "pixel_x": pixel_x,
                    "pixel_y": pixel_y,
                    "graph_x": graph_x,
                    "graph_y": graph_y,
                }
            )

        rows.sort(key=lambda row: (row["graph_x"], row["graph_y"], row["index"]))
        return rows

    def _build_summary(
        self,
        rows: List[dict],
        precision: int,
        x_origin: float,
        y_origin: float,
        x_scale: float,
        y_scale: float,
    ) -> str:
        """Формирование краткого анализа преобразованных точек."""
        graph_x_values = [row["graph_x"] for row in rows]
        graph_y_values = [row["graph_y"] for row in rows]

        trend = self._detect_trend(rows)
        is_function, function_comment = self._check_function(rows)
        preview = self._format_table(rows[:10], precision)

        lines = [
            "Преобразование точек графика выполнено.",
            f"Количество точек: {len(rows)}",
            (
                "Параметры преобразования: "
                f"x_origin={self._format_number(x_origin, precision)}, "
                f"y_origin={self._format_number(y_origin, precision)}, "
                f"x_scale={self._format_number(x_scale, precision)}, "
                f"y_scale={self._format_number(y_scale, precision)}"
            ),
            (
                "Диапазон X: "
                f"{self._format_number(min(graph_x_values), precision)} .. "
                f"{self._format_number(max(graph_x_values), precision)}"
            ),
            (
                "Диапазон Y: "
                f"{self._format_number(min(graph_y_values), precision)} .. "
                f"{self._format_number(max(graph_y_values), precision)}"
            ),
            f"Характер изменения Y при росте X: {trend}",
            f"Проверка на корректность как функции: {function_comment}",
            "",
            "Первые точки после сортировки по X:",
            preview,
        ]

        if len(rows) > 10:
            lines.append(f"\nПоказаны первые 10 из {len(rows)} точек.")

        if not is_function:
            lines.append(
                "Рекомендация: проверьте дубликаты по X или качество извлечения точек с графика."
            )

        return "\n".join(lines)

    def _check_function(self, rows: List[dict]) -> Tuple[bool, str]:
        """
        Простая проверка:
        если одному X соответствуют разные Y, то набор точек может не задавать функцию.
        """
        grouped = {}
        for row in rows:
            x_key = round(row["graph_x"], 10)
            y_key = round(row["graph_y"], 10)
            grouped.setdefault(x_key, set()).add(y_key)

        conflicting_x = [x for x, y_values in grouped.items() if len(y_values) > 1]
        if conflicting_x:
            preview = ", ".join(str(x) for x in conflicting_x[:3])
            return False, f"обнаружены повторяющиеся X с разными Y: {preview}"

        return True, "дубликатов X с разными Y не обнаружено"

    def _detect_trend(self, rows: List[dict]) -> str:
        """Определение общего характера изменения Y."""
        deltas = []

        for left, right in zip(rows, rows[1:]):
            if right["graph_x"] == left["graph_x"]:
                continue
            deltas.append(right["graph_y"] - left["graph_y"])

        if not deltas:
            return "недостаточно данных для оценки"

        if all(delta > 0 for delta in deltas):
            return "монотонно возрастает"
        if all(delta < 0 for delta in deltas):
            return "монотонно убывает"
        if all(delta == 0 for delta in deltas):
            return "почти постоянная зависимость"
        if all(delta >= 0 for delta in deltas):
            return "неубывающая зависимость"
        if all(delta <= 0 for delta in deltas):
            return "невозрастающая зависимость"
        return "смешанное поведение"

    def _format_table(self, rows: List[dict], precision: int) -> str:
        """Форматирование результата в виде простой таблицы."""
        header = "index | x_pixel | y_pixel | x_value | y_value"
        separator = "-" * len(header)
        lines = [header, separator]

        for row in rows:
            lines.append(
                " | ".join(
                    [
                        str(row["index"]),
                        self._format_number(row["pixel_x"], precision),
                        self._format_number(row["pixel_y"], precision),
                        self._format_number(row["graph_x"], precision),
                        self._format_number(row["graph_y"], precision),
                    ]
                )
            )

        return "\n".join(lines)

    def _format_csv(self, rows: List[dict], precision: int) -> str:
        """Форматирование результата в CSV."""
        lines = ["index,x_pixel,y_pixel,x_value,y_value"]

        for row in rows:
            lines.append(
                ",".join(
                    [
                        str(row["index"]),
                        self._format_number(row["pixel_x"], precision),
                        self._format_number(row["pixel_y"], precision),
                        self._format_number(row["graph_x"], precision),
                        self._format_number(row["graph_y"], precision),
                    ]
                )
            )

        return "\n".join(lines)

    def _format_number(self, value: float, precision: int) -> str:
        """Компактное форматирование чисел."""
        formatted = f"{value:.{precision}f}"
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted

    async def _arun(
        self,
        points_text: str,
        x_origin: float = 0.0,
        y_origin: float = 0.0,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        precision: int = 3,
        output_mode: str = "summary",
    ) -> str:
        """Асинхронная версия инструмента."""
        return self._run(
            points_text=points_text,
            x_origin=x_origin,
            y_origin=y_origin,
            x_scale=x_scale,
            y_scale=y_scale,
            precision=precision,
            output_mode=output_mode,
        )

    def to_langchain_tool(self) -> BaseTool:
        """Конвертация в формат LangChain."""
        return self


if __name__ == "__main__":
    tool = CustomTool()
    result = tool.run(
        points_text="(120,340); (180,300); (240,260); (300,220)",
        x_origin=120,
        y_origin=400,
        x_scale=0.05,
        y_scale=0.1,
        precision=2,
        output_mode="summary",
    )
    print(result)
