from typing import Any, Dict, List

from symbolic.rule_engine import Rule, RulePriority


def _num(facts: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """Safely read numeric facts that can arrive from UI forms as strings."""
    try:
        return float(facts.get(key, default))
    except (TypeError, ValueError):
        return default


def _is_set(value: Any) -> bool:
    return value not in (None, "", [], {})


def get_graph_digitization_rules() -> List[Rule]:
    """
    Правила для дипломной темы:
    "Информационная система по извлечению и преобразованию числовых данных
    с графиков функций".

    Ожидаемые факты описывают состояние обработки изображения графика:
    качество изображения, калибровку осей, тип шкалы, найденные кривые,
    уверенность извлечения и готовность результата к экспорту.
    """
    return [
        Rule(
            rule_id="IMG_LOW_RESOLUTION",
            name="Недостаточное разрешение изображения",
            condition=lambda f: _num(f, "image_width") < 600 or _num(f, "image_height") < 400,
            conclusion=(
                "Изображение имеет низкое разрешение: рекомендуется загрузить "
                "более качественный график перед извлечением числовых данных"
            ),
            priority=RulePriority.HIGH,
            description=(
                "Для устойчивого распознавания осей, подписей и кривой размер "
                "изображения должен быть не менее 600x400 пикселей"
            ),
            domain="graph_digitization",
        ),
        Rule(
            rule_id="IMG_PREPROCESSING_REQUIRED",
            name="Требуется предварительная обработка",
            condition=lambda f: _num(f, "contrast_score", 1.0) < 0.35 or _num(f, "noise_level") > 0.6,
            conclusion=(
                "Перед выделением кривой нужно применить предобработку: "
                "повышение контраста, подавление шума и бинаризацию"
            ),
            priority=RulePriority.HIGH,
            description=(
                "Низкий контраст или высокий шум ухудшают поиск осей, сетки "
                "и пикселей кривой"
            ),
            domain="graph_digitization",
        ),
        Rule(
            rule_id="CALIBRATION_INCOMPLETE",
            name="Неполная калибровка осей",
            condition=lambda f: _num(f, "calibration_x_points") < 2 or _num(f, "calibration_y_points") < 2,
            conclusion=(
                "Нельзя корректно преобразовать пиксели в числовые значения: "
                "нужно задать минимум две опорные точки для оси X и две для оси Y"
            ),
            priority=RulePriority.CRITICAL,
            description=(
                "Без полной калибровки система может найти кривую на изображении, "
                "но не сможет восстановить значения функции"
            ),
            domain="graph_digitization",
        ),
        Rule(
            rule_id="CALIBRATION_POINTS_TOO_CLOSE",
            name="Опорные точки калибровки расположены слишком близко",
            condition=lambda f: (
                _num(f, "calibration_x_points") >= 2
                and _num(f, "calibration_y_points") >= 2
                and (_num(f, "calibration_x_span_px") < 120 or _num(f, "calibration_y_span_px") < 120)
            ),
            conclusion=(
                "Калибровка может быть неточной: выберите более удаленные "
                "деления шкалы по X и Y"
            ),
            priority=RulePriority.MEDIUM,
            description=(
                "Чем меньше расстояние между опорными точками, тем сильнее "
                "ошибка клика влияет на восстановленные координаты"
            ),
            domain="graph_digitization",
        ),
        Rule(
            rule_id="LOG_SCALE_CONFIRMATION_REQUIRED",
            name="Нужно подтвердить логарифмическую шкалу",
            condition=lambda f: (
                (
                    f.get("axis_scale_x") == "log"
                    or f.get("axis_scale_y") == "log"
                    or bool(f.get("detected_log_scale"))
                )
                and not bool(f.get("log_scale_confirmed"))
            ),
            conclusion=(
                "Обнаружен или выбран логарифмический масштаб: пользователь "
                "должен подтвердить тип шкалы перед преобразованием координат"
            ),
            priority=RulePriority.HIGH,
            description=(
                "Логарифмическая шкала требует другого преобразования пиксельных "
                "координат в реальные значения"
            ),
            domain="graph_digitization",
        ),
        Rule(
            rule_id="MULTIPLE_CURVES_SELECTION_REQUIRED",
            name="Не выбрана кривая при наличии нескольких серий",
            condition=lambda f: _num(f, "curve_count", 1) > 1 and not _is_set(f.get("selected_curve_id")),
            conclusion=(
                "На графике найдено несколько кривых: перед экспортом нужно "
                "выбрать целевую серию данных"
            ),
            priority=RulePriority.HIGH,
            description=(
                "При нескольких линиях автоматическое усреднение пикселей может "
                "смешать разные функции в один набор точек"
            ),
            domain="graph_digitization",
        ),
        Rule(
            rule_id="EXTRACTION_REVIEW_REQUIRED",
            name="Требуется ручная проверка извлеченных точек",
            condition=lambda f: (
                _num(f, "extraction_confidence", 1.0) < 0.7
                or _num(f, "curve_gaps_count") > 3
                or _num(f, "outlier_ratio") > 0.15
            ),
            conclusion=(
                "Результат извлечения ненадежен: нужно показать точки пользователю "
                "и разрешить ручную корректировку"
            ),
            priority=RulePriority.MEDIUM,
            description=(
                "Низкая уверенность, разрывы кривой или много выбросов указывают "
                "на возможные ошибки сегментации"
            ),
            domain="graph_digitization",
        ),
        Rule(
            rule_id="EXPORT_READY",
            name="Результат готов к экспорту",
            condition=lambda f: (
                bool(f.get("calibrated"))
                and _num(f, "points_count") >= 10
                and _num(f, "extraction_confidence", 0.0) >= 0.75
                and f.get("export_format", "csv") in {"csv", "json", "xlsx"}
            ),
            conclusion=(
                "Данные можно экспортировать: сохраните точки x,y, параметры "
                "калибровки, метод извлечения и предупреждения обработки"
            ),
            priority=RulePriority.LOW,
            description=(
                "Для воспроизводимости результата экспорт должен включать не "
                "только точки, но и метаданные обработки"
            ),
            domain="graph_digitization",
        ),
    ]
