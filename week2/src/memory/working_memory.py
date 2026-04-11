"""
Модуль работы с различными типами памяти агента
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import sqlite3
import json

@dataclass
class Message:
    """Сообщение в диалоге."""
    role: str # system, user, assistant
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

class WorkingMemory:
    """
    Оперативная память агента (краткосрочная).
    Реализует буфер сообщений в контексте LLM.
    """
    def __init__(self, max_tokens: int = 4000):
        self.messages: List[Message] = []
        self.max_tokens = max_tokens
        self.current_tokens = 0

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Добавление сообщения в память."""
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        self.current_tokens += len(content) // 4 # Приблизительный подсчёт

        # Trim при превышении лимита
        while self.current_tokens > self.max_tokens and len(self.messages) > 1:
            removed = self.messages.pop(0)
            self.current_tokens -= len(removed.content) // 4

    def get_messages(self) -> List[Dict]:
        """Получение всех сообщений в формате для LLM."""
        return [
            {"role": msg.role, "content": msg.content, "metadata": msg.metadata}
            for msg in self.messages
        ]
    
    def clear(self) -> None:
        """Очистка памяти."""
        self.messages = []
        self.current_tokens = 0
    
    def get_stats(self) -> Dict:
        """Статистика использования памяти."""
        return {
            "message_count": len(self.messages),
            "current_tokens": self.current_tokens,
            "max_tokens": self.max_tokens,
            "usage_percent": (self.current_tokens / self.max_tokens) * 100
        }

class EpisodicMemory:
    """
    Эпизодическая память агента (долгосрочная).
    Хранит историю взаимодействий в SQLite.
    """
    def __init__(self, db_path: str = "./memory/episodes.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Инициализация базы данных."""
        import os

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_input TEXT,
            agent_output TEXT,
            tools_used TEXT,
            duration_ms INTEGER,
            success BOOLEAN
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_session ON episodes(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(timestamp)')
        conn.commit()
        conn.close()
    
    def save_episode(
        self,
        session_id: str,
        user_input: str,
        agent_output: str,
        tools_used: List[str],
        duration_ms: int,
        success: bool
    ) -> int:
        """Сохранение эпизода взаимодействия."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO episodes
            (session_id, user_input, agent_output, tools_used, duration_ms, success)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            user_input,
            agent_output,
            json.dumps(tools_used),
            duration_ms,
            success
        ))
        episode_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return episode_id
    
    def get_episodes(
        self,
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Получение истории эпизодов."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if session_id:
            cursor.execute('''
                SELECT * FROM episodes
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (session_id, limit))
        else:
            cursor.execute('''
                SELECT * FROM episodes
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            episodes = [dict(row) for row in cursor.fetchall()]
            conn.close()
        return episodes
    
    def search_episodes(self, keyword: str, limit: int = 10) -> List[Dict]:
        """Поиск эпизодов по ключевому слову."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM episodes
            WHERE user_input LIKE ? OR agent_output LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (f'%{keyword}%', f'%{keyword}%', limit))

        episodes = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return episodes