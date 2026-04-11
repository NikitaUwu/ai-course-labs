# -*- coding: utf-8 -*-
"""
Инструмент поиска в интернете
Лабораторная работа №2
"""


from langchain.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup
import logging


logger = logging.getLogger(__name__)

class SearchInput(BaseModel):
    """Схема входных параметров для поиска."""
    query: str = Field(
        description="Поисковый запрос",
        min_length=1,
        max_length=500
    )
    num_results: int = Field(
        description="Количество результатов (1-10)",
        default=5,
        ge=1,
        le=10
    )

class SearchTool(BaseTool):
    """
    Инструмент для поиска информации в интернете.
    Назначение:
        Получение актуальной информации из открытых источников.
    Ограничения:
        • Учебная версия использует mock-данные
        • В production подключить Yandex Search API или аналог
    """
    name: str = "search_web"
    description: str = """
        Поиск актуальной информации в интернете по запросу.
        Используйте для получения свежих данных, новостей, документации.
        Возвращает до 10 результатов поиска с описанием.
    """
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str, num_results: int = 5) -> str:
        """
        Выполнение поиска.
        Args:
            query: Поисковый запрос
            num_results: Количество результатов
        Returns:
            str: Форматированные результаты поиска
        """
        logger.info(f"Поиск: {query} (результатов: {num_results})")
        # Учебная реализация (mock)
        # В production: подключить Yandex Search API
        results = [
            f'''Результат {i+1}: Информация по запросу '{query}',
            Источник: Открытые данные,
            Актуальность: 2026'''
            for i in range(num_results)
        ]
        formatted = "\n".join(results)
        
        return f"Поиск по запросу: {query}\n\n{formatted}"
    
    async def _arun(self, query: str, num_results: int = 5) -> str:
        """Асинхронная версия."""
        return self._run(query, num_results)
    
    def to_langchain_tool(self) -> BaseTool:
        """Конвертация в формат LangChain."""
        return self
