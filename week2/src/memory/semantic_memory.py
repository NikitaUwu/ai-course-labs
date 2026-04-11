"""
Семантическая память на основе ChromaDB
Лабораторная работа №2
"""

from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions
import uuid
from datetime import datetime
import logging
import os


logger = logging.getLogger(__name__)


class SemanticMemory:
    """
    Семантическая память для долгосрочного хранения знаний агента.
    Использует векторные embeddings для семантического поиска.
    Поддерживает мультиязычные запросы (русский/английский).
    Атрибуты:
        client: ChromaDB клиент
        collection: Коллекция для хранения
        embedding_function: Функция embeddings
    """
    def __init__(
        self,
        collection_name: str = "agent_knowledge",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        Инициализация семантической памяти.
        Args:
            collection_name: Имя коллекции
            persist_directory: Путь для сохранения
            embedding_model: Модель для embeddings
        """
        # Создание директории для хранения
        os.makedirs(persist_directory, exist_ok=True)

        # Инициализация ChromaDB с сохранением на диск
        self.client = chromadb.Client(
            Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
            )
        )

        # Мультиязычная embedding модель (поддержка русского)
        model_path = f"sentence-transformers/{embedding_model}"
        try:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_path
            )
            logger.info(
                "Using local SentenceTransformer embedding model: %s",
                model_path
            )
        except Exception as e:
            logger.warning(
                "Failed to initialize SentenceTransformer embeddings (%s). "
                "Falling back to Chroma default embeddings.",
                e
            )
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        # Создание коллекции
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"} # Косинусное сходство
        )

        logger.info(f"Семантическая память инициализирована: {collection_name}")
    
    def add_document(
        self,
        content: str,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Добавление документа в память.
        Args:
            content: Текст документа
            metadata: Метаданные (источник, дата, теги)
            doc_id: Уникальный ID
        Returns:
            str: ID документа
        """
        doc_id = doc_id or str(uuid.uuid4())
        meta = metadata or {}
        meta["created_at"] = datetime.now().isoformat()

        self.collection.add(
            documents=[content],
            metadatas=[meta],
            ids=[doc_id]
        )

        logger.debug(f"Документ добавлен: {doc_id}")
        return doc_id
    
    def search_knowledge(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Поиск релевантных знаний.
        Args:
            query: Поисковый запрос
            k: Количество результатов
            filter_metadata: Фильтр по метаданным
        Returns:
            List[Dict]: Найденные документы
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity": 1 - results["distances"][0][i] if results["distances"] else 1
            })
        
        logger.debug(f"Найдено {len(formatted)} документов")
        return formatted
    
    def delete_document(self, doc_id: str) -> bool:
        """Удаление документа по ID."""
        try:
            self.collection.delete(ids=[doc_id])
            logger.debug(f"Документ удалён: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Ошибка удаления: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Статистика памяти."""
        embedding_model = getattr(
            self.embedding_function,
            "model_name",
            self.embedding_function.__class__.__name__
        )
        return {
            "collection": self.collection.name,
            "documents": self.collection.count(),
            "embedding_model": embedding_model
        }
