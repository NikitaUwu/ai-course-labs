"""
Модуль безопасности для AI-агента (Guardrails)
Лабораторная работа №2
"""
import re
import logging
from typing import Tuple, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Уровень риска."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityCheck:
    """Результат проверки безопасности."""
    is_safe: bool
    reason: str
    risk_level: RiskLevel

class InputGuardrails:
    """
    Система защиты AI-агента от нежелательных действий.
    Реализует многоуровневую проверку:
        1. Валидация длины ввода
        2. Детекция опасных паттернов
        3. Проверка на инъекции
        4. Фильтрация запрещённых слов
    """
    def __init__(self):
        self.blocked_patterns = [
            r"ignore\s+(previous|all)\s+instructions",
            r"bypass\s+(security|restrictions)",
            r"execute\s+(code|command|script)",
            r"delete\s+(all|database|table)",
            r"drop\s+table",
            r";\s*--", # SQL injection
            r"<script>", # XSS
            r"eval\s*\(",
            r"exec\s*\(",
        ]
        self.max_tokens = 10000
        self.allowed_tools = ["search", "calculate", "custom_specialty_tool"]
    
    def validate_input(self, user_input: str) -> SecurityCheck:
        """
        Проверка входных данных пользователя.
        Args:
            user_input: Текст запроса от пользователя
        Returns:
            SecurityCheck: Результат проверки
        """
        # Проверка длины
        if len(user_input) > self.max_tokens * 4:
            logger.warning("Превышена максимальная длина ввода")
            return SecurityCheck(
                is_safe=False,
                reason="Превышена максимальная длина ввода",
                risk_level=RiskLevel.MEDIUM
            )
        
        # Проверка на опасные паттерны
        for pattern in self.blocked_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                logger.warning(f"Обнаружен опасный паттерн: {pattern}")
                return SecurityCheck(
                    is_safe=False,
                    reason=f"Обнаружена попытка инъекции (паттерн: {pattern})",
                    risk_level=RiskLevel.HIGH
                )

        return SecurityCheck(
            is_safe=True,
            reason="Входные данные прошли проверку",
            risk_level=RiskLevel.LOW
        )

    def validate_tool_call(self, tool_name: str, parameters: dict) -> SecurityCheck:
        """
        Проверка вызова инструмента.
        Args:
            tool_name: Имя вызываемого инструмента
            parameters: Параметры вызова
        Returns:
            SecurityCheck: Результат проверки
        """
        if tool_name not in self.allowed_tools:
            return SecurityCheck(
                is_safe=False,
                reason=f"Инструмент {tool_name} не в whitelist",
                risk_level=RiskLevel.HIGH
            )
        # Проверка параметров на SQL injection
        for key, value in parameters.items():
            if isinstance(value, str) and re.search(r";\s*--|DROP|DELETE", value, re.IGNORECASE):
                return SecurityCheck(
                    is_safe=False,
                    reason="Обнаружена потенциальная SQL-инъекция в параметрах",
                    risk_level=RiskLevel.CRITICAL
                )

        return SecurityCheck(
            is_safe=True,
            reason="Инструмент разрешён к вызову",
            risk_level=RiskLevel.LOW
        )
    