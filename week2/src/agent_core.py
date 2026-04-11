import os
import sys
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import time
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage

# Локальные импорты
from tools.search_tool import SearchTool
from tools.calc_tool import CalculateTool
from tools.custom_tool import CustomTool # Адаптируемый инструмент
from memory.working_memory import WorkingMemory
from memory.semantic_memory import SemanticMemory
from guardrails.input_validator import InputGuardrails


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """
    Конфигурация AI-агента.

    Атрибуты:
        name: Имя агента
        version: Версия агента
        max_iterations: Максимальное количество итераций ReAct
        temperature: Параметр креативности LLM
        memory_enabled: Флаг включения памяти
        guardrails_enabled: Флаг включения защиты
        verbose: Режим подробного логирования
    """
    name: str = "TechnicalAssistant"
    version: str = "2.0"
    max_iterations: int = 10
    temperature: float = 0.7
    memory_enabled: bool = True
    guardrails_enabled: bool = True
    verbose: bool = True

@dataclass
class AgentResponse:
    """
    Структурированный ответ агента.

    Атрибуты:
        success: Флаг успешного выполнения
        answer: Текстовый ответ
        steps: Список выполненных шагов
        duration_ms: Время выполнения в миллисекундах
        tokens_used: Оценка использованных токенов
        error: Сообщение об ошибке (если есть)
    """
    success: bool
    answer: str
    steps: List[Dict]
    duration_ms: int
    tokens_used: int
    error: Optional[str] = None

class AIAgent:
    """
    Основной класс AI-агента.
    Реализует архитектуру с поддержкой:
        • Множественных инструментов
        • Краткосрочной и долгосрочной памяти
        • Механизмов безопасности (guardrails)
        • Паттерна ReAct для принятия решений
    Пример использования:
        >>> agent = AIAgent()
        >>> response = agent.run("Найди информацию о Python 3.12")
        >>> print(response.answer)
    """
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Инициализация AI-агента.
        Args:
            config: Конфигурация агента (опционально, используется default)
        Raises:
            ValueError: Если не найдены необходимые переменные окружения
        """
        self.config = config or AgentConfig()
        logger.info(f"Инициализация агента: {self.config.name} v{self.config.version}")

        # Инициализация LLM
        self.llm = self._init_llm()
        logger.info("LLM инициализирован")

        # Инициализация инструментов
        self.tools = self._init_tools()
        logger.info(f"Доступно инструментов: {len(self.tools)}")

        # Инициализация памяти
        self.working_memory = None
        self.semantic_memory = None

        if self.config.memory_enabled:
            self.working_memory = WorkingMemory()
            try:
                self.semantic_memory = SemanticMemory()
            except Exception as e:
                logger.warning(
                    "Semantic memory disabled because initialization failed: %s",
                    e
                )
                self.semantic_memory = None
            logger.info("Память активирована")
        
        # Инициализация guardrails
        self.guardrails = None
        if self.config.guardrails_enabled:
            self.guardrails = InputGuardrails()
            logger.info("Guardrails активированы")
        
        # Инициализация LangChain-агента
        self.agent = self._init_agent()
        logger.info("Агент готов к работе")
        
        # Статистика
        self.request_count = 0
        self.total_tokens = 0
    
    def _init_llm(self):
        """
        Инициализация LLM-клиента (YandexGPT).
        Returns:
            YandexGPTLangChain: Настроенный клиент LLM
        Raises:
            ValueError: Если не найдены учётные данные
        """
        from langchain_community.llms import YandexGPT

        iam_token = os.getenv("YANDEX_IAM_TOKEN")
        folder_id = os.getenv("YANDEX_FOLDER_ID")

        if not iam_token or not folder_id:
            raise ValueError(
                "Не найдены YANDEX_IAM_TOKEN или YANDEX_FOLDER_ID. "
                "Проверьте файл .env"
            )
        
        return YandexGPT(
            iam_token=iam_token,
            folder_id=folder_id,
            temperature=self.config.temperature,
            max_tokens=1000
        )
    
    def _init_tools(self) -> List[Tool]:
        """
        Инициализация инструментов агента.
        Returns:
            List[Tool]: Список доступных инструментов
        """
        tools = [
            SearchTool().to_langchain_tool(),
            CalculateTool().to_langchain_tool(),
            CustomTool().to_langchain_tool(), # Адаптируемый инструмент
        ]
        return tools
    
    def _init_agent(self):
        """
        Инициализация LangChain-агента.
        Returns:
            AgentExecutor: Настроенный агент LangChain
        """
        memory = None
        if self.working_memory:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                max_token_limit=4000
            )
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=memory,
            verbose=self.config.verbose,
            max_iterations=self.config.max_iterations,
            handle_parsing_errors=True
        )
    
    def run(self, query: str, session_id: Optional[str] = None) -> AgentResponse:
        """
        Выполнение запроса к агенту.
        Args:
            query: Текстовый запрос пользователя
            session_id: Идентификатор сессии (опционально)
        Returns:
            AgentResponse: Структурированный ответ агента
        """
        start_time = time.time()
        session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Проверка безопасности входа
        if self.guardrails:
            security_check = self.guardrails.validate_input(query)
            if not security_check.is_safe:
                logger.warning(f"Запрос отклонён guardrails: {security_check.reason}")
                return AgentResponse(
                    success=False,
                    answer="Запрос отклонён системой безопасности.",
                    steps=[],
                    duration_ms=0,
                    tokens_used=0,
                    error=security_check.reason
                )
        try:
            # Выполнение запроса
            logger.info(f"Обработка запроса: {query[:100]}...")
            result = self.agent.run(query)

            # Сохранение в память
            if self.working_memory:
                self.working_memory.add_message("user", query)
                self.working_memory.add_message("assistant", result)

            if self.semantic_memory and session_id:
                self.semantic_memory.add_document(
                    content=f"Query: {query}\nAnswer: {result}",
                    metadata={"session_id": session_id, "type": "interaction"}
                )
            
            # Формирование ответа
            end_time = time.time()
            duration_ms = int((end_time - start_time) * 1000)

            self.request_count += 1
            tokens_used = self._estimate_tokens(query, result)
            self.total_tokens += tokens_used

            response = AgentResponse(
                success=True,
                answer=result,
                steps=self._extract_steps(),
                duration_ms=duration_ms,
                tokens_used=tokens_used
            )

            logger.info(f"Запрос выполнен за {duration_ms}мс")
            return response

        except Exception as e:
            logger.error(f"Ошибка выполнения: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                answer="Произошла ошибка при обработке запроса.",
                steps=[],
                duration_ms=0,
                tokens_used=0,
                error=str(e)
            )
    
    def _extract_steps(self) -> List[Dict]:
        """
        Извлечение шагов выполнения из истории агента.
        Returns:
            List[Dict]: Список шагов ReAct
        """
        # В production: парсинг логов LangChain
        return []
    
    def _estimate_tokens(self, input_text: str, output_text: str) -> int:
        """
        Оценка количества использованных токенов.
        Args:
            input_text: Входной текст
            output_text: Выходной текст
        Returns:
            int: Приблизительное количество токенов
        """
        # Приблизительно: 1 токен ≈ 4 символа для русского
        return (len(input_text) + len(output_text)) // 4
    
    def get_stats(self) -> Dict:
        """
        Получение статистики агента.
        Returns:
            Dict: Статистика использования
        """
        return {
            "name": self.config.name,
            "version": self.config.version,
            "tools_count": len(self.tools),
            "memory_enabled": self.config.memory_enabled,
            "guardrails_enabled": self.config.guardrails_enabled,
            "request_count": self.request_count,
            "total_tokens": self.total_tokens
        }

    def save_session(self, session_id: str) -> bool:
        """
        Сохранение текущей сессии в долговременную память.
        Args:
            session_id: Идентификатор сессии
        Returns:
            bool: Успешность сохранения
        """
        if self.working_memory and self.semantic_memory:
            content = str(self.working_memory.get_messages())
            self.semantic_memory.add_document(
                content=content,
                metadata={"session_id": session_id, "type": "full_session"}
            )
            return True
        return False


# Точка входа для тестирования
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА №2")
    print("Демонстрация AI-агента, инструментов и памяти")
    print("=" * 80)

    print("\nДЕМОНСТРАЦИЯ ИНСТРУМЕНТОВ")
    print("=" * 80)

    search_tool = SearchTool()
    calc_tool = CalculateTool()
    custom_tool = CustomTool()

    print("\n[1] Инструмент search_web")
    print("-" * 80)
    search_result = search_tool._run(
        query="извлечение числовых данных с графиков функций",
        num_results=3
    )
    print(search_result)

    print("\n[2] Инструмент calculate")
    print("-" * 80)
    calc_result = calc_tool._run(
        expression="(12.5 - 2.5) / 2",
        precision=2
    )
    print(calc_result)

    print("\n[3] Инструмент transform_graph_points")
    print("-" * 80)
    custom_result = custom_tool._run(
        points_text="(120,340); (180,300); (240,260); (300,220)",
        x_origin=120,
        y_origin=400,
        x_scale=0.05,
        y_scale=0.1,
        precision=2,
        output_mode="summary"
    )
    print(custom_result)

    print("\n" + "=" * 80)
    print("ИНТЕГРАЦИОННЫЙ ЗАПУСК АГЕНТА")
    print("=" * 80)

    try:
        agent = AIAgent()

        test_query = (
            "Найди краткую информацию о методах извлечения данных с графиков, "
            "затем вычисли значение выражения (12.5 - 2.5) / 2 и поясни результат."
        )

        print(f"\nЗапрос: {test_query}\n")

        response = agent.run(test_query)

        print("\n" + "=" * 80)
        print("ОТВЕТ АГЕНТА")
        print("=" * 80)
        print(response.answer)
        print("=" * 80)
        print(f"Время выполнения: {response.duration_ms}мс")
        print(f"Использовано токенов: {response.tokens_used}")
        print(f"Статус: {'Успех' if response.success else 'Ошибка'}")

        if response.error:
            print(f"Ошибка: {response.error}")

    except Exception as e:
        print("Интеграционный запуск агента не выполнен.")
        print(f"Причина: {e}")

    print("=" * 80)
