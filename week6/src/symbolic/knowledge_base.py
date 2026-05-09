import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


logger = logging.getLogger(__name__)

@dataclass
class KnowledgeFact:
    """Факт в базе знаний."""
    fact_id: str
    subject: str
    predicate: str
    object: Any
    confidence: float = 1.0
    source: str = "manual"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_triple(self) -> tuple:
        """Представление в виде тройки (subject, predicate, object)."""
        return (self.subject, self.predicate, self.object)

class KnowledgeBase:
    """
    База знаний для хранения фактов и онтологий.
    Атрибуты:
    facts: Словарь фактов по ID
    index: Индекс для быстрого поиска
    """
    def __init__(self):
        self.facts: Dict[str, KnowledgeFact] = {}
        self.index: Dict[str, List[str]] = {} # subject -> [fact_ids]
        logger.info("KnowledgeBase инициализирована")

    def add_fact(self, fact: KnowledgeFact) -> str:
        """Добавление факта."""
        self.facts[fact.fact_id] = fact
        # Индексация по subject
        if fact.subject not in self.index:
            self.index[fact.subject] = []
        self.index[fact.subject].append(fact.fact_id)
        logger.debug(f"Добавлен факт: {fact.fact_id}")
        return fact.fact_id
    
    def get_facts_by_subject(self, subject: str) -> List[KnowledgeFact]:
        """Получение фактов по subject."""
        fact_ids = self.index.get(subject, [])
        return [self.facts[fid] for fid in fact_ids if fid in self.facts]

    def query(self, subject: Optional[str] = None,
        predicate: Optional[str] = None) -> List[KnowledgeFact]:
        """Запрос к базе знаний."""
        results = []
        for fact in self.facts.values():
            match = True
            if subject and fact.subject != subject:
                match = False
            if predicate and fact.predicate != predicate:
                match = False
            if match:
                results.append(fact)
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Статистика базы знаний."""
        return {
            "total_facts": len(self.facts),
            "subjects": len(self.index),
            "avg_confidence": sum(f.confidence for f in self.facts.values()) / max(len(self.facts), 1)
        }

# Пример использования
if __name__ == "__main__":
    kb = KnowledgeBase()
    kb.add_fact(KnowledgeFact(
        fact_id="F001",
        subject="двигатель",
        predicate="макс_температура",
        object=90,
        source="ГОСТ Р 12345-2020"
    ))
    kb.add_fact(KnowledgeFact(
        fact_id="F002",
        subject="двигатель",
        predicate="мин_давление",
        object=50,
        source="Технический регламент"
    ))
    print(f"Статистика БЗ: {kb.get_statistics()}")
    facts = kb.query(subject="двигатель")
    print(f"\nФакты о двигателе:")
    for f in facts:
        print(f" {f.subject}.{f.predicate} = {f.object}")