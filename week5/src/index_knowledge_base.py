import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

WEEK5_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = WEEK5_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag.chunking import ChunkingStrategy
from rag.document_loader import DocumentLoader
from rag.vector_store import VectorStoreManager


def build_knowledge_base(reset: bool = True) -> None:
    load_dotenv(WEEK5_DIR / ".env")

    documents_dir = WEEK5_DIR / "data" / "documents"
    chroma_dir = WEEK5_DIR / "data" / "chroma_db"

    loader = DocumentLoader(source_directory=str(documents_dir))
    documents = loader.load_directory()
    if not documents:
        raise RuntimeError(f"В папке {documents_dir} нет документов для индексации.")

    chunks = ChunkingStrategy.split_documents(
        documents,
        strategy="recursive",
        chunk_size=700,
        chunk_overlap=100,
    )

    store = VectorStoreManager(
        persist_directory=str(chroma_dir),
        collection_name="rag_documents",
    )
    if reset:
        store.clear()

    stats = store.add_documents(chunks)
    chunk_stats = ChunkingStrategy.get_statistics(chunks)

    print("База знаний week5 обновлена")
    print(f"Документов загружено: {len(documents)}")
    print(f"Чанков добавлено: {stats['documents_added']}")
    print(f"Всего в коллекции: {stats['total_documents']}")
    print(f"Средний размер чанка: {chunk_stats.get('avg_size', 0)} символов")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Индексация базы знаний week5 по теме диплома."
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Не очищать коллекцию перед добавлением документов.",
    )
    args = parser.parse_args()
    build_knowledge_base(reset=not args.append)


if __name__ == "__main__":
    main()
