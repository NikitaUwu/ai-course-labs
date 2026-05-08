"""
Модуль загрузки документов из различных источников
Лабораторная работа №5
Дисциплина: Искусственный интеллект
Автор: [ФИО]
Группа: [НОМЕР ГРУППЫ]
Дата: 2026
"""
import os
import logging
import sys
from typing import List, Optional, Dict, Any
from pathlib import Path

WEEK5_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = WEEK5_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader
)
from langchain_core.documents import Document


logger = logging.getLogger(__name__)
class DocumentLoader:
    """
    Загрузчик документов из различных форматов.
    Поддерживаемые форматы:
    • PDF — технические документы, мануалы
    • TXT — простые текстовые файлы
    • MD — Markdown документация
    • DOCX — документы Word
    Атрибуты:
    source_directory: Путь к директории с документами
    supported_extensions: Поддерживаемые расширения файлов
    """
    
    def __init__(self, source_directory: str):
        """
        Инициализация загрузчика документов.
        Args:
        source_directory: Путь к директории с документами
        """
        self.source_directory = Path(source_directory)
        self.supported_extensions = ['.pdf', '.txt', '.md', '.docx']
        if not self.source_directory.exists():
            logger.warning(f"Директория не существует: {source_directory}")
            self.source_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"DocumentLoader инициализирован: {source_directory}")
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Загрузка单个 документа.
        Args:
        file_path: Путь к файлу
        Returns:
        List[Document]: Список загруженных документов
        Raises:
        ValueError: Если формат файла не поддерживается
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        extension = path.suffix.lower()
        logger.info(f"Загрузка документа: {file_path}")
        
        try:
            if extension == '.pdf':
                loader = PyPDFLoader(str(path))
            elif extension == '.txt':
                loader = TextLoader(str(path), encoding='utf-8')
            elif extension == '.md':
                loader = UnstructuredMarkdownLoader(str(path))
            elif extension == '.docx':
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(str(path))
            else:
                raise ValueError(f"Неподдерживаемый формат: {extension}")
            documents = loader.load()
            
            # Добавление метаданных
            for doc in documents:
                doc.metadata['source'] = str(path)
                doc.metadata['file_name'] = path.name
                doc.metadata['file_type'] = extension
            logger.info(f"Загружено {len(documents)} чанков из {file_path}")
            
            return documents
        
        except Exception as e:
            logger.error(f"Ошибка загрузки {file_path}: {e}")
            raise
    
    def load_directory(self, pattern: str = "**/*") -> List[Document]:
        """
        Загрузка всех документов из директории.
        Args:
        pattern: Шаблон поиска файлов
        Returns:
        List[Document]: Список всех загруженных документов
        """
        all_documents = []
        for ext in self.supported_extensions:
            file_pattern = f"{pattern}{ext}"
            files = list(self.source_directory.glob(file_pattern))
            logger.info(f"Найдено {len(files)} файлов с расширением {ext}")
            for file_path in files:
                try:
                    documents = self.load_document(str(file_path))
                    all_documents.extend(documents)
                except Exception as e:
                    logger.warning(f"Пропущен файл {file_path}: {e}")
        
        logger.info(f"Всего загружено {len(all_documents)} документов")
        
        return all_documents
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики по документам."""
        stats = {
            "directory": str(self.source_directory),
            "files": {},
            "total_size_bytes": 0
        }
        for ext in self.supported_extensions:
            files = list(self.source_directory.glob(f"**/*{ext}"))
            stats["files"][ext] = len(files)
            stats["total_size_bytes"] += sum(f.stat().st_size for f in files)
        
        stats["total_size_mb"] = round(stats["total_size_bytes"] / (1024 * 1024), 2)
        
        return stats
    

# Точка входа для тестирования
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(WEEK5_DIR / ".env")

    print("=" * 80)
    print("ТЕСТИРОВАНИЕ DOCUMENT LOADER")
    print("=" * 80)
    loader = DocumentLoader(source_directory=str(WEEK5_DIR / "data" / "documents"))
    stats = loader.get_statistics()
    print(f"\nСтатистика директории:")
    print(f" Директория: {stats['directory']}")
    print(f" Общий размер: {stats['total_size_mb']} MB")
    print(f" Файлы по типам: {stats['files']}")
    documents = loader.load_directory()
    print(f"\nЗагружено документов: {len(documents)}")
    
    if documents:
        print(f"\nПример первого документа:")
        print(f" Источник: {documents[0].metadata.get('source')}")
        print(f" Размер: {len(documents[0].page_content)} символов")
        print(f" Предпросмотр: {documents[0].page_content[:200]}...")
