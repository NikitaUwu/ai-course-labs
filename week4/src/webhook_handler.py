import argparse
import json
import logging
import os
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:5678"
DEFAULT_WEBHOOK_PATH = "application"
GRAPH_ANALYSIS_WEBHOOK_PATH = "graph-digitization-analysis"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file(Path(".env"))
load_env_file(Path(__file__).resolve().parents[1] / "docker" / ".env")


class WorkflowClient:
    """
    Клиент для взаимодействия с n8n workflow.

    По умолчанию работает с базовым workflow `/webhook/application`.
    Для дипломного workflow используйте `send_graph_analysis`.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        webhook_path: Optional[str] = None,
        secret: Optional[str] = None,
    ):
        self.base_url = self._normalize_base_url(
            base_url or os.getenv("N8N_BASE_URL") or os.getenv("WEBHOOK_URL") or DEFAULT_BASE_URL
        )
        self.webhook_path = self._normalize_webhook_path(
            webhook_path if webhook_path is not None else os.getenv("WEBHOOK_PATH")
        )
        self.secret = secret or os.getenv("WEBHOOK_SECRET")
        self.webhook_url = self._build_webhook_url(self.webhook_path)
        logger.info("WorkflowClient инициализирован: %s", self.webhook_url)

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        return (base_url or DEFAULT_BASE_URL).strip().rstrip("/")

    @staticmethod
    def _normalize_webhook_path(webhook_path: Optional[str]) -> str:
        path = (webhook_path or "").strip().strip("/")
        return path or DEFAULT_WEBHOOK_PATH

    def _build_webhook_url(self, webhook_path: str, test_mode: bool = False) -> str:
        webhook_prefix = "webhook-test" if test_mode else "webhook"

        if self.base_url.endswith("/webhook") or self.base_url.endswith("/webhook-test"):
            return f"{self.base_url.rsplit('/', 1)[0]}/{webhook_prefix}/{webhook_path}"
        if "/webhook/" in self.base_url or "/webhook-test/" in self.base_url:
            return self.base_url
        return f"{self.base_url}/{webhook_prefix}/{webhook_path}"

    @staticmethod
    def _format_request_error(error: Exception, url: str, status_code: Optional[int] = None) -> str:
        if status_code == 404:
            return (
                f"{error}. URL: {url}. "
                "Проверьте, что workflow импортирован, активен и используется правильный путь webhook. "
                "Базовый workflow: /webhook/application. "
                "Специальный workflow: /webhook/graph-digitization-analysis."
            )
        return str(error)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.secret:
            headers["X-Webhook-Secret"] = self.secret
        return headers

    def _post_json(self, webhook_path: str, payload: Dict[str, Any], test_mode: bool = False) -> Dict[str, Any]:
        webhook_url = self._build_webhook_url(webhook_path, test_mode=test_mode)
        logger.info("Отправка данных в webhook: %s", webhook_url)

        try:
            body = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                webhook_url,
                data=body,
                headers=self._headers(),
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                response_body = response.read().decode("utf-8")
                status_code = response.status

            try:
                result: Any = json.loads(response_body)
            except ValueError:
                result = response_body

            return {
                "success": True,
                "response": result,
                "status_code": status_code,
                "url": webhook_url,
            }
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            error = f"{e.reason}. {error_body}".strip()
            logger.error("Ошибка отправки: %s", error)
            return {
                "success": False,
                "error": self._format_request_error(Exception(error), webhook_url, e.code),
                "status_code": e.code,
                "url": webhook_url,
            }
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            logger.error("Ошибка отправки: %s", e)
            return {
                "success": False,
                "error": self._format_request_error(e, webhook_url),
                "status_code": None,
                "url": webhook_url,
            }

    def send_application(
        self,
        message: str,
        contact: str,
        priority: str = "normal",
        test_mode: bool = False,
    ) -> Dict[str, Any]:
        """Отправка заявки в базовый workflow."""
        payload = {
            "message": message,
            "contact": contact,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
        }
        return self._post_json(DEFAULT_WEBHOOK_PATH, payload, test_mode=test_mode)

    def send_graph_analysis(
        self,
        graph_payload: Optional[Dict[str, Any]] = None,
        test_mode: bool = False,
    ) -> Dict[str, Any]:
        """Отправка данных графика функции в специализированный дипломный workflow."""
        payload = graph_payload or build_sample_graph_payload()
        payload.setdefault("timestamp", datetime.now().isoformat())
        return self._post_json(GRAPH_ANALYSIS_WEBHOOK_PATH, payload, test_mode=test_mode)

    def send_security_incident(
        self,
        log_entry: str,
        source_ip: str,
        event_type: str,
        test_mode: bool = False,
    ) -> Dict[str, Any]:
        """Отправка инцидента безопасности, оставлена для совместимости с шаблоном лабораторной."""
        payload = {
            "log_entry": log_entry,
            "source_ip": source_ip,
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
        }
        return self._post_json("security-incident", payload, test_mode=test_mode)

    def check_workflow_status(self) -> Dict[str, Any]:
        """Проверка доступности n8n."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/healthz", timeout=5) as response:
                return {
                    "available": response.status == 200,
                    "status_code": response.status,
                }
        except urllib.error.HTTPError as e:
            return {
                "available": False,
                "status_code": e.code,
                "error": e.reason,
            }
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return {
                "available": False,
                "error": str(e),
            }


def build_sample_graph_payload() -> Dict[str, Any]:
    """Тестовый payload для workflow по теме диплома."""
    return {
        "task": "Оцени пригодность извлечённых точек графика функции к числовому преобразованию",
        "chart_type": "line",
        "image_quality": "high",
        "axis_info": {
            "x_scale": 0.1,
            "y_scale": 0.2,
            "origin": [0, 240],
        },
        "graph_points": [
            {"x": 10, "y": 220},
            {"x": 20, "y": 200},
            {"x": 30, "y": 185},
            {"x": 40, "y": 168},
            {"x": 50, "y": 150},
        ],
        "notes": "Точки получены после детекции линии функции на изображении графика.",
    }


def load_payload(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("JSON-файл должен содержать объект верхнего уровня.")
    return data


def print_result(result: Dict[str, Any]) -> None:
    print(f"URL: {result.get('url')}")
    if result.get("success"):
        print("Статус: успешно")
        print(f"Код ответа: {result.get('status_code')}")
        print("Ответ n8n:")
        print(json.dumps(result.get("response"), ensure_ascii=False, indent=2))
    else:
        print("Статус: ошибка")
        print(f"Код ответа: {result.get('status_code')}")
        print(f"Ошибка: {result.get('error')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Удобный запуск webhook для лабораторной работы №4.",
    )
    parser.add_argument(
        "workflow",
        nargs="?",
        default="basic",
        choices=("basic", "special", "status"),
        help="Что проверить: basic, special или status. По умолчанию: basic.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Отправить запрос в /webhook-test/... для режима Listen for test event в n8n.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Базовый URL n8n. По умолчанию берётся из env или http://localhost:5678.",
    )
    parser.add_argument(
        "--payload",
        default=None,
        help="Путь к JSON-файлу с payload для специального workflow.",
    )
    parser.add_argument("--message", default="Не работает вход в систему, ошибка 403")
    parser.add_argument("--contact", default="user@example.com")
    parser.add_argument("--priority", default="high")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = WorkflowClient(base_url=args.base_url)

    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА №4: тестирование n8n workflow")
    print("=" * 80)

    status = client.check_workflow_status()
    if not status.get("available"):
        print(f"n8n недоступен: {status.get('error') or status.get('status_code')}")
        return 1

    print("n8n доступен")

    if args.workflow == "status":
        return 0

    if args.workflow == "basic":
        print("Запуск базового workflow: /application")
        result = client.send_application(
            message=args.message,
            contact=args.contact,
            priority=args.priority,
            test_mode=args.test,
        )
    else:
        print("Запуск специального workflow: /graph-digitization-analysis")
        result = client.send_graph_analysis(
            graph_payload=load_payload(args.payload),
            test_mode=args.test,
        )

    print_result(result)
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
