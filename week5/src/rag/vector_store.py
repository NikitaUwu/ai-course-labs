"""
Модуль работы с векторной базой данных ChromaDB
Лабораторная работа №5
"""
import os
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

WEEK5_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = WEEK5_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import chromadb


logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Менеджер векторной базы данных ChromaDB.
    Атрибуты:
    persist_directory: Путь для сохранения данных
    collection_name: Имя коллекции
    embedding_model: Модель для embeddings
    """
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "rag_documents",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        Инициализация векторного хранилища.
        Args:
        persist_directory: Путь для сохранения данных
        collection_name: Имя коллекции
        embedding_model: Модель embeddings
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        # Создание директории
        os.makedirs(persist_directory, exist_ok=True)
        # Инициализация embedding модели
        logger.info(f"Загрузка embedding модели: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=f"sentence-transformers/{embedding_model}",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        # Инициализация ChromaDB
        logger.info(f"Инициализация ChromaDB: {collection_name}")
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        logger.info("VectorStoreManager инициализирован")

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Добавление документов в векторную базу.
        Args:
        documents: Список документов для добавления
        batch_size: Размер пакета для добавления
        Returns:
        Dict: Статистика добавления
        """
        logger.info(f"Добавление {len(documents)} документов в векторную базу")
        # Добавление по пакетам для стабильности
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.vectorstore.add_documents(batch)
            logger.info(f"Добавлен пакет {i//batch_size + 1}")
        stats = {
            "documents_added": len(documents),
            "collection_name": self.collection_name,
            "total_documents": self.vectorstore._collection.count()
        }
        logger.info(f"Всего документов в базе: {stats['total_documents']}")
        return stats
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов.
        Args:
        query: Поисковый запрос
        k: Количество результатов
        filter_meta Фильтр по метаданным
        Returns:
        List[Dict]: Найденные документы с метаданными
        """
        logger.info(f"Поиск по запросу: {query[:100]}... (k={k})")
        docs = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter_metadata
        )
        results = []
        for i, doc in enumerate(docs):
            results.append({
                "rank": i + 1,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": doc.metadata.get('score', None)
            })
        logger.info(f"Найдено {len(results)} документов")
        return results

    def search_with_scores(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Поиск с оценками сходства.
        Args:
        query: Поисковый запрос
        k: Количество результатов
        Returns:
        List[Dict]: Документы с оценками
        """
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k
        )
        results = []
        for i, (doc, score) in enumerate(docs_and_scores):
            results.append({
                "rank": i + 1,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": 1 - score # Преобразование в сходство
            })
        return results

    def delete_collection(self) -> bool:
        """Удаление коллекции."""
        try:
            client = chromadb.Client(
                chromadb.config.Settings(
                    persist_directory=self.persist_directory
                )
            )
            client.delete_collection(self.collection_name)
            logger.info(f"Коллекция {self.collection_name} удалена")
            return True
        except Exception as e:
            logger.error(f"Ошибка удаления: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Статистика векторного хранилища."""
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_model_name,
            "total_documents": self.vectorstore._collection.count(),
            "embedding_dimension": 384
        }

    def clear(self) -> bool:
        """Очистка хранилища."""
        try:
            existing = self.vectorstore._collection.get()
            ids = existing.get("ids", [])
            if ids:
                self.vectorstore._collection.delete(ids=ids)
            logger.info("Хранилище очищено")
            return True
        except Exception as e:
            logger.error(f"Ошибка очистки: {e}")
            return False
    

if __name__ == "__main__":
    from dotenv import load_dotenv
    from rag.document_loader import DocumentLoader
    from rag.chunking import ChunkingStrategy


    load_dotenv(WEEK5_DIR / ".env")
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ VECTOR STORE")
    print("=" * 80)
    
    # Загрузка и чанкинг
    loader = DocumentLoader(source_directory=str(WEEK5_DIR / "data" / "documents"))
    documents = loader.load_directory()
    if documents:
        chunks = ChunkingStrategy.split_documents(documents)

        # Векторное хранилище
        store = VectorStoreManager(
            persist_directory=str(WEEK5_DIR / "data" / "chroma_db"),
            collection_name="rag_documents"
        )
        store.clear()

        # Добавление
        stats = store.add_documents(chunks)
        print(f"\nДобавлено документов: {stats['documents_added']}")

        # Поиск
        query = "калибровка осей и извлечение числовых данных с графика функции"
        results = store.search(query, k=3)
        print(f"\nРезультаты поиска по запросу: {query}")
        for r in results:
            print(f"\n[{r['rank']}] Score: {r['similarity_score']:.3f}")
            print(f"Источник: {r['metadata'].get('source', 'N/A')}")
            print(f"Текст: {r['content'][:200]}...")
        
        # Статистика
        store_stats = store.get_statistics()
        print(f"\nСтатистика хранилища:")
        print(f" Всего документов: {store_stats['total_documents']}")
        print(f" Размерность embeddings: {store_stats['embedding_dimension']}")
