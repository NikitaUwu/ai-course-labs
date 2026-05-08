import os
import json
import requests
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()
load_dotenv(Path(__file__).resolve().parents[1] / "docker" / ".env")

DEFAULT_BASE_URL = "http://localhost:5678"
DEFAULT_WEBHOOK_PATH = "application"

class WorkflowClient:
    """
    Клиент для взаимодействия с n8n workflow.
    Атрибуты:
    base_url: URL n8n сервера
    webhook_path: Путь webhook
    secret: Секретный ключ для аутентификации
    """
    def __init__(
        self,
        base_url: Optional[str] = None,
        webhook_path: Optional[str] = None,
        secret: Optional[str] = None
    ):
        self.base_url = self._normalize_base_url(
            base_url or os.getenv("N8N_BASE_URL") or os.getenv("WEBHOOK_URL") or DEFAULT_BASE_URL
        )
        self.webhook_path = self._normalize_webhook_path(
            webhook_path if webhook_path is not None else os.getenv("WEBHOOK_PATH")
        )
        self.secret = secret or os.getenv("WEBHOOK_SECRET")
        self.webhook_url = self._build_webhook_url(self.webhook_path)
        logger.info(f"WorkflowClient инициализирован: {self.webhook_url}")

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        return (base_url or DEFAULT_BASE_URL).strip().rstrip("/")

    @staticmethod
    def _normalize_webhook_path(webhook_path: Optional[str]) -> str:
        path = (webhook_path or "").strip().strip("/")
        return path or DEFAULT_WEBHOOK_PATH

    def _build_webhook_url(self, webhook_path: str) -> str:
        if self.base_url.endswith("/webhook"):
            return f"{self.base_url}/{webhook_path}"
        if "/webhook/" in self.base_url:
            return self.base_url
        return f"{self.base_url}/webhook/{webhook_path}"

    @staticmethod
    def _format_request_error(error: requests.exceptions.RequestException, url: str) -> str:
        response = getattr(error, "response", None)
        if response is not None and response.status_code == 404:
            return (
                f"{error}. URL: {url}. "
                "For n8n, make sure the workflow is imported and active. "
                "The default endpoint for this lab is /webhook/application."
            )
        return str(error)
    
    def send_application(
        self,
        message: str,
        contact: str,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """
        Отправка заявки в workflow.
        Args:
        message: Текст заявки
        contact: Контактная информация
        priority: Приоритет (low, normal, high)
        Returns:
        Dict: Ответ от workflow
        """
        payload = {
            "message": message,
            "contact": contact,
            "priority": priority,
            "timestamp": datetime.now().isoformat()
        }
        headers = {
            "Content-Type": "application/json"
        }
        if self.secret:
            headers["X-Webhook-Secret"] = self.secret
        
        logger.info(f"Отправка заявки: {message[:50]}...")

        try:
            response = requests.post(
                self.webhook_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Workflow выполнил обработку")
            return {
                "success": True,
                "response": result,
                "status_code": response.status_code
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка отправки: {e}")
            return {
                "success": False,
                "error": self._format_request_error(e, self.webhook_url),
                "status_code": getattr(getattr(e, "response", None), "status_code", None)
            }
    
    def send_security_incident(
        self,
        log_entry: str,
        source_ip: str,
        event_type: str
    ) -> Dict[str, Any]:
        """
        Отправка инцидента безопасности (для ИБ-специальности).
        Args:
        log_entry: Лог события
        source_ip: IP-адрес источника
        event_type: Тип события
        Returns:
        Dict: Ответ от workflow
        """
        payload = {
            "log_entry": log_entry,
            "source_ip": source_ip,
            "event_type": event_type,
            "timestamp": datetime.now().isoformat()
        }
        headers = {
            "Content-Type": "application/json"
        }
        if self.secret:
            headers["X-Webhook-Secret"] = self.secret
        
        webhook_url = self._build_webhook_url("security-incident")
        logger.info(f"Отправка инцидента ИБ: {event_type}")
        
        try:
            response = requests.post(
                webhook_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Инцидент обработан")
            return {
                "success": True,
                "response": result,
                "status_code": response.status_code
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка отправки: {e}")
            return {
                "success": False,
                "error": self._format_request_error(e, webhook_url),
                "status_code": getattr(getattr(e, "response", None), "status_code", None)
            }
    
    def check_workflow_status(self) -> Dict[str, Any]:
        """Проверка доступности n8n."""
        try:
            response = requests.get(
                f"{self.base_url}/healthz",
                timeout=5
            )
            return {
                "available": response.status_code == 200,
                "status_code": response.status_code
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }


# Точка входа для тестирования
if __name__ == "__main__":
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА №4")
    print("Тестирование workflow automation")
    print("=" * 80)
    client = WorkflowClient()

    # Проверка доступности
    print("\nПроверка доступности n8n...")
    status = client.check_workflow_status()
    if status.get("available"):
        print("✅ n8n доступен")
    else:
        print(f"❌ n8n недоступен: {status.get('error')}")
        exit(1)
    
    # Тестовая заявка
    print("\n" + "=" * 80)
    print("ТЕСТОВАЯ ЗАЯВКА")
    print("=" * 80)
    result = client.send_application(
        message="Не работает вход в систему, ошибка 403",
        contact="user@example.com",
        priority="high"
    )
    if result["success"]:
        print("✅ Заявка отправлена успешно")
        print(f"Статус код: {result['status_code']}")
    else:
        print(f"❌ Ошибка: {result['error']}")
    print("=" * 80)
    
