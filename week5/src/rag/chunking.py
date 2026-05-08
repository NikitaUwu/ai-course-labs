"""
Модуль разбиения документов на чанки
Лабораторная работа №5
"""
import logging
import sys
from pathlib import Path
from typing import List, Optional

WEEK5_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = WEEK5_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
from langchain_core.documents import Document


logger = logging.getLogger(__name__)

class ChunkingStrategy:
    """
    Стратегии разбиения текста на чанки.
    Поддерживаемые методы:
    • Recursive — рекурсивное по разделителям (рекомендуется)
    • Character — по символам фиксированного размера
    • Token — по токенам LLM
    """
    @staticmethod
    def get_splitter(
        strategy: str = "recursive",
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Получение сплиттера текста.
        Args:
        strategy: Стратегия разбиения (recursive/character/token)
        chunk_size: Размер чанка в символах/токенах
        chunk_overlap: Перекрытие между чанками
        Returns:
        TextSplitter: Настроенный сплиттер
        """
        logger.info(f"Создание сплиттера: {strategy}, chunk_size={chunk_size}")
        if strategy == "recursive":
            return RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False
            )
        elif strategy == "character":
            return CharacterTextSplitter(
                separator="\n",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
        elif strategy == "token":
            return TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            raise ValueError(f"Неизвестная стратегия: {strategy}")
    
    @staticmethod
    def split_documents(
        documents: List[Document],
        strategy: str = "recursive",
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> List[Document]:
        """
        Разбиение документов на чанки.
        Args:
        documents: Список документов для разбиения
        strategy: Стратегия разбиения
        chunk_size: Размер чанка
        chunk_overlap: Перекрытие
        Returns:
        List[Document]: Список чанков
        """
        splitter = ChunkingStrategy.get_splitter(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        logger.info(f"Разбиение {len(documents)} документов на чанки")
        chunks = splitter.split_documents(documents)
        logger.info(f"Получено {len(chunks)} чанков")
        # Добавление метаданных о чанках
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['total_chunks'] = len(chunks)
        return chunks
    
    @staticmethod
    def get_statistics(chunks: List[Document]) -> dict:
        """Статистика чанков."""
        if not chunks:
            return {"count": 0}
        sizes = [len(chunk.page_content) for chunk in chunks]
        return {
            "count": len(chunks),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "avg_size": round(sum(sizes) / len(sizes), 2),
            "total_characters": sum(sizes)
        }
    

# Тестирование
if __name__ == "__main__":
    from rag.document_loader import DocumentLoader


    print("=" * 80)
    print("ТЕСТИРОВАНИЕ CHUNKING")
    print("=" * 80)
    loader = DocumentLoader(source_directory=str(WEEK5_DIR / "data" / "documents"))
    documents = loader.load_directory()

    if documents:
        chunks = ChunkingStrategy.split_documents(
            documents,
            strategy="recursive",
            chunk_size=512,
            chunk_overlap=50
        )
        stats = ChunkingStrategy.get_statistics(chunks)
        print(f"\nСтатистика чанков:")
        print(f" Количество: {stats['count']}")
        print(f" Мин. размер: {stats['min_size']} символов")
        print(f" Макс. размер: {stats['max_size']} символов")
        print(f" Сред. размер: {stats['avg_size']} символов")
