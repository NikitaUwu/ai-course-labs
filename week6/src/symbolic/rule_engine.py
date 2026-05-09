import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


logger = logging.getLogger(__name__)

class RulePriority(Enum):
    """Приоритеты правил."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class Rule:
    """
    Правило в формате IF-THEN.
    Атрибуты:
    rule_id: Уникальный идентификатор правила
    name: Название правила
    condition: Функция условия (принимает факты, возвращает bool)
    conclusion: Вывод при истинности условия
    priority: Приоритет правила
    description: Описание правила для объяснимости
    domain: Область применения (специальность)
    """
    rule_id: str
    name: str
    condition: Callable[[Dict], bool]
    conclusion: str
    priority: RulePriority = RulePriority.MEDIUM
    description: str = ""
    domain: str = "general"

    def evaluate(self, facts: Dict[str, Any]) -> bool:
        """
        Оценка истинности условия.
        Args:
        facts: Факты для оценки
        Returns:
        bool: Истинность условия
        """
        try:
            return self.condition(facts)
        except Exception as e:
            logger.error(f"Ошибка оценки правила {self.rule_id}: {e}")
            return False
    
    def to_dict(self) -> Dict:
        """Сериализация правила."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "conclusion": self.conclusion,
            "priority": self.priority.value,
            "description": self.description,
            "domain": self.domain
        }
    
@dataclass
class InferenceResult:
    """Результат логического вывода."""
    success: bool
    conclusions: List[str]
    triggered_rules: List[Rule]
    explanation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class RuleEngine:
    """
    Движок логического вывода на правилах.
    Атрибуты:
    rules: Список правил
    facts: Текущие факты
    inference_history: История выводов
    """
    def __init__(self):
        self.rules: List[Rule] = []
        self.facts: Dict[str, Any] = {}
        self.inference_history: List[InferenceResult] = []
        logger.info("RuleEngine инициализирован")

    def add_rule(self, rule: Rule) -> None:
        """
        Добавление правила.
        Args:
        rule: Правило для добавления
        """
        self.rules.append(rule)
        # Сортировка по приоритету (критичные первыми)
        self.rules.sort(key=lambda r: r.priority.value)
        logger.info(f"Добавлено правило: {rule.rule_id} ({rule.name})")
    
    def add_rules(self, rules: List[Rule]) -> None:
        """Добавление списка правил."""
        for rule in rules:
            self.add_rule(rule)
    
    def set_facts(self, facts: Dict[str, Any]) -> None:
        """Установка фактов для вывода."""
        self.facts = facts
        logger.debug(f"Установлено {len(facts)} фактов")
    
    def infer(self, facts: Optional[Dict[str, Any]] = None) -> InferenceResult:
        """
        Логический вывод на основе фактов.
        Args:
        facts: Факты для вывода (опционально, использует self.facts)
        Returns:
        InferenceResult: Результат вывода
        """
        if facts:
            self.set_facts(facts)
        
        triggered_rules = []
        conclusions = []
        explanation_parts = []
        logger.info(f"Начало логического вывода ({len(self.rules)} правил)")
        
        for rule in self.rules:
            if rule.evaluate(self.facts):
                triggered_rules.append(rule)
                conclusions.append(rule.conclusion)
                explanation_parts.append(f"• Правило '{rule.name}': {rule.description} → {rule.conclusion}")
                logger.debug(f"Сработало правило: {rule.rule_id}")
        # Формирование объяснения
        if triggered_rules:
            explanation = "Логический вывод:\n" + "\n".join(explanation_parts)
        else:
            explanation = "Ни одно правило не сработало"
        
        result = InferenceResult(
            success=len(triggered_rules) > 0,
            conclusions=conclusions,
            triggered_rules=triggered_rules,
            explanation=explanation
        )
        self.inference_history.append(result)
        logger.info(f"Вывод завершён: {len(conclusions)} заключений")
        return result
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Статистика по правилам."""
        return {
            "total_rules": len(self.rules),
            "rules_by_priority": {
                priority.name: sum(1 for r in self.rules if r.priority == priority)
                for priority in RulePriority
            },
            "rules_by_domain": {},
            "total_inferences": len(self.inference_history)
        }

    def clear_facts(self) -> None:
        """Очистка фактов."""
        self.facts = {}
        logger.debug("Факты очищены")

    def export_rules(self) -> List[Dict]:
        """Экспорт правил в формат JSON."""
        return [rule.to_dict() for rule in self.rules]
    

# Пример использования
if __name__ == "__main__":
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ RULE ENGINE")
    print("=" * 80)
    engine = RuleEngine()
    # Пример правил для технической диагностики
    engine.add_rule(Rule(
        rule_id="TEMP_HIGH",
        name="Высокая температура",
        condition=lambda f: f.get("temperature", 0) > 80,
        conclusion="Требуется проверка системы охлаждения",
        priority=RulePriority.HIGH,
        description="Температура превышает критический порог 80°C",
        domain="technical"
    ))
    engine.add_rule(Rule(
        rule_id="PRESSURE_LOW",
        name="Низкое давление",
        condition=lambda f: f.get("pressure", 100) < 50,
        conclusion="Требуется проверка герметичности",
        priority=RulePriority.HIGH,
        description="Давление ниже минимального порога 50",
        domain="technical"
    ))
    engine.add_rule(Rule(
        rule_id="VIBRATION_NORMAL",
        name="Нормальная вибрация",
        condition=lambda f: 0.1 < f.get("vibration", 0) < 5.0,
        conclusion="Вибрация в норме",
        priority=RulePriority.LOW,
        description="Вибрация в допустимых пределах",
        domain="technical"
    ))

    # Тестовые факты
    test_facts = {
        "temperature": 85,
        "pressure": 45,
        "vibration": 2.5
    }

    print(f"\nФакты: {test_facts}")
    result = engine.infer(test_facts)
    print(f"\nРезультат вывода:")
    print(f" Успех: {result.success}")
    print(f" Заключений: {len(result.conclusions)}")
    print(f"\nЗаключения:")

    for c in result.conclusions:
        print(f" • {c}")
    print(f"\nОбъяснение:\n{result.explanation}")
    print(f"\nСтатистика: {engine.get_rule_statistics()}")