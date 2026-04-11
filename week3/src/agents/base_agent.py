from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """
    Конфигурация агента.
    Атрибуты:
        role: Роль агента в системе
        goal: Цель агента
        backstory: Контекст и история агента
        verbose: Режим подробного логирования
        allow_delegation: Может ли делегировать задачи
        max_iter: Максимальное количество итераций
    """
    role: str = "Assistant"
    goal: str = "Help users with their tasks"
    backstory: str = "You are a helpful AI assistant"
    verbose: bool = True
    allow_delegation: bool = False
    max_iter: int = 10


@dataclass
class AgentState:
    """
    Состояние агента.
    Атрибуты:
        agent_id: Уникальный идентификатор
        current_task: Текущая задача
        completed_tasks: Выполненные задачи
        message_history: История сообщений
        created_at: Время создания
        last_active: Последняя активность
    """
    agent_id: str = field(default_factory=lambda: f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    current_task: Optional[str] = None
    completed_tasks: List[str] = field(default_factory=list)
    message_history: List[Dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)

class BaseAgent:
    """
    Базовый класс для всех агентов в системе.
    Предоставляет общую функциональность:
        • Управление состоянием
        • Логирование действий
        • Обработка сообщений
        • Статистика работы
    Наследники должны реализовать:
        • execute_task() — выполнение задачи
        • get_capabilities() — возможности агента
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Инициализация агента.
        Args:
            config: Конфигурация агента
        """
        self.config = config or AgentConfig()
        self.state = AgentState()
        self.tools = []
        self.statistics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0,
            "messages_sent": 0,
            "messages_received": 0
        }
        logger.info(f"Агент инициализирован: {self.config.role}")
    
    def execute_task(self, task_description: str, context: Optional[Dict] = None) -> Dict:
        """
        Выполнение задачи.
        Args:
            task_description: Описание задачи
            context: Контекст выполнения
        Returns:
            Dict: Результат выполнения
        Raises:
            NotImplementedError: Метод должен быть реализован в наследнике
        """
        raise NotImplementedError("Метод execute_task должен быть реализован в наследнике")
    
    def get_capabilities(self) -> List[str]:
        """
        Получение списка возможностей агента.
        Returns:
            List[str]: Список возможностей
        Raises:
            NotImplementedError: Метод должен быть реализован в наследнике
        """
        raise NotImplementedError("Метод get_capabilities должен быть реализован в наследнике")
    
    def receive_message(self, message: Dict) -> None:
        """
        Получение сообщения от другого агента.
        Args:
            message: Сообщение в формате dict
        """
        self.state.message_history.append({
            "direction": "incoming",
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        self.statistics["messages_received"] += 1
        self.state.last_active = datetime.now()
        logger.debug(f"Агент {self.state.agent_id} получил сообщение")
    
    def send_message(self, receiver_id: str, content: Dict) -> Dict:
        """
        Отправка сообщения другому агенту.
        Args:
            receiver_id: ID агента-получателя
            content: Содержание сообщения
        Returns:
            Dict: Отправленное сообщение
        """
        message = {
            "sender_id": self.state.agent_id,
            "receiver_id": receiver_id,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "type": "agent_message"
        }
        self.state.message_history.append({
            "direction": "outgoing",
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        self.statistics["messages_sent"] += 1
        logger.debug(f"Агент {self.state.agent_id} отправил сообщение агенту {receiver_id}")
        return message
    
    def get_statistics(self) -> Dict:
        """Получение статистики работы агента."""
        return {
            "agent_id": self.state.agent_id,
            "role": self.config.role,
            "statistics": self.statistics,
            "completed_tasks_count": len(self.state.completed_tasks),
            "message_history_length": len(self.state.message_history)
        }

    def reset_state(self) -> None:
        """Сброс состояния агента."""
        self.state.current_task = None
        self.state.message_history = []
        logger.info(f"Агент {self.state.agent_id} сбросил состояние")

