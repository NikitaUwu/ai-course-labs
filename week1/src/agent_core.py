# -*- coding: utf-8 -*-
"""
Модуль для работы с YandexGPT API.
Лабораторная работа №1.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
DEFAULT_MODEL_NAME = "yandexgpt/latest"
SUPPORTED_ENV_KEYS = {
    "YANDEX_IAM_TOKEN",
    "YANDEX_API_KEY",
    "YANDEX_FOLDER_ID",
    "YANDEX_GPT_API_URL",
}


def _clean_env_value(value: Optional[str]) -> Optional[str]:
    """Убирает пробелы и внешние кавычки у значения переменной окружения."""
    if value is None:
        return None

    cleaned = value.strip().strip('"').strip("'")
    return cleaned or None


def _preview_response_text(response: requests.Response, limit: int = 300) -> str:
    """Возвращает короткий фрагмент ответа API для логов."""
    text = response.text.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _parse_env_file(dotenv_path: Path) -> Tuple[Dict[str, str], List[str], Set[str]]:
    """
    Разбирает .env вручную, чтобы заметить токены, случайно перенесенные на новую строку.

    python-dotenv читает только строку вида KEY=value и игнорирует последующие bare lines.
    Для длинных токенов это часто приводит к тихому обрезанию значения и ошибке 401.
    """
    parsed: Dict[str, str] = {}
    warnings: List[str] = []
    multiline_keys: Set[str] = set()

    if not dotenv_path.exists():
        return parsed, warnings, multiline_keys

    try:
        lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = dotenv_path.read_text(encoding="utf-8-sig").splitlines()

    last_key: Optional[str] = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            last_key = None
            continue

        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = _clean_env_value(value)
            if key in SUPPORTED_ENV_KEYS and value:
                parsed[key] = value
            last_key = key
            continue

        if last_key in {"YANDEX_IAM_TOKEN", "YANDEX_API_KEY"}:
            multiline_keys.add(last_key)
            parsed[last_key] = f"{parsed.get(last_key, '')}{line}"
            warnings.append(
                f"В .env значение {last_key} перенесено на новую строку. "
                "Токены должны храниться в одну строку без переносов."
            )
        else:
            warnings.append(
                "В .env обнаружена строка без имени переменной. "
                "Проверьте формат файла окружения."
            )

        last_key = None

    return parsed, warnings, multiline_keys


def load_yandex_credentials(
    dotenv_path: Optional[Path] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Загружает IAM token/API key, folder_id и необязательный URL API."""
    env_path = dotenv_path or Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=env_path)

    parsed_env, warnings, multiline_keys = _parse_env_file(env_path)
    for warning in warnings:
        logger.warning(warning)

    iam_token = _clean_env_value(os.getenv("YANDEX_IAM_TOKEN"))
    api_key = _clean_env_value(os.getenv("YANDEX_API_KEY"))
    folder_id = _clean_env_value(os.getenv("YANDEX_FOLDER_ID"))
    api_url = _clean_env_value(os.getenv("YANDEX_GPT_API_URL"))

    if "YANDEX_IAM_TOKEN" in multiline_keys:
        iam_token = parsed_env.get("YANDEX_IAM_TOKEN")
    elif not iam_token:
        iam_token = parsed_env.get("YANDEX_IAM_TOKEN")

    if "YANDEX_API_KEY" in multiline_keys:
        api_key = parsed_env.get("YANDEX_API_KEY")
    elif not api_key:
        api_key = parsed_env.get("YANDEX_API_KEY")

    if not folder_id:
        folder_id = parsed_env.get("YANDEX_FOLDER_ID")

    if not api_url:
        api_url = parsed_env.get("YANDEX_GPT_API_URL")

    return iam_token, api_key, folder_id, api_url


def _normalize_api_url(api_url: Optional[str]) -> str:
    """Проверяет endpoint и подменяет его на стандартный, если строка повреждена."""
    candidate = (_clean_env_value(api_url) or DEFAULT_API_URL).rstrip("/")

    host = "llm.api.cloud.yandex.net"
    expected_path = "/foundationModels/v1/completion"
    parsed = urlsplit(candidate)

    is_invalid = (
        parsed.scheme not in {"http", "https"}
        or host not in parsed.netloc
        or candidate.count(host) > 1
        or "foundationModellm.api.cloud.yandex.net" in candidate
        or not parsed.path.endswith(expected_path)
    )

    if is_invalid:
        logger.warning(
            "Обнаружен некорректный URL API '%s'. Используется стандартный endpoint '%s'.",
            candidate,
            DEFAULT_API_URL,
        )
        return DEFAULT_API_URL

    return candidate


class YandexGPTClient:
    """Клиент для взаимодействия с YandexGPT API."""

    def __init__(
        self,
        auth_token: str,
        folder_id: str,
        auth_scheme: str = "Bearer",
        api_url: Optional[str] = None,
    ) -> None:
        if not auth_token or not folder_id:
            raise ValueError("Необходимо указать auth_token и folder_id")

        if auth_scheme not in {"Bearer", "Api-Key"}:
            raise ValueError("auth_scheme должен быть 'Bearer' или 'Api-Key'")

        self.auth_token = auth_token.strip()
        self.auth_scheme = auth_scheme
        self.folder_id = folder_id.strip()
        self.model_uri = f"gpt://{self.folder_id}/{DEFAULT_MODEL_NAME}"
        self.api_url = _normalize_api_url(api_url)

        logger.info(
            "Клиент инициализирован. Folder ID: %s, auth scheme: %s",
            self.folder_id,
            self.auth_scheme,
        )

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"{self.auth_scheme} {self.auth_token}",
            "x-folder-id": self.folder_id,
        }

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Dict:
        """Генерация ответа модели."""
        headers = self._build_headers()

        payload = {
            "modelUri": self.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
            "messages": [
                {
                    "role": "system",
                    "text": "Вы полезный ассистент. Отвечайте точно и по делу. Используйте русский язык.",
                },
                {
                    "role": "user",
                    "text": prompt,
                },
            ],
        }

        logger.info("Отправка запроса к API. Длина промпта: %s символов", len(prompt))

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30,
            )

            if response.status_code == 401:
                raise requests.exceptions.HTTPError(
                    "Ошибка авторизации Yandex Cloud (401). Проверьте, что используется "
                    "действующий IAM token или API key, токен в .env записан в одну строку "
                    "без переносов, а folder_id соответствует каталогу с доступом к модели. "
                    f"Endpoint: {response.request.url}. Ответ API: {_preview_response_text(response)}",
                    response=response,
                    request=response.request,
                )

            response.raise_for_status()
            result = response.json()

            if "result" not in result:
                raise ValueError("Некорректный формат ответа API: отсутствует ключ 'result'")

            alternatives = result["result"].get("alternatives", [])
            if not alternatives:
                raise ValueError("Пустой ответ от модели")

            generated_text = alternatives[0]["message"]["text"]
            tokens_info = result["result"].get("usage", {})

            response_data = {
                "text": generated_text,
                "tokens_input": tokens_info.get("inputTextTokens", 0),
                "tokens_output": tokens_info.get("completionTokens", 0),
                "raw_response": result,
            }

            logger.info(
                "Запрос выполнен. Выходных токенов: %s",
                response_data["tokens_output"],
            )
            return response_data

        except requests.exceptions.Timeout:
            logger.error("Превышено время ожидания ответа от API")
            raise
        except requests.exceptions.HTTPError as error:
            logger.error("HTTP-ошибка: %s", error)
            raise
        except requests.exceptions.RequestException as error:
            logger.error("Ошибка запроса: %s", error)
            raise
        except ValueError as error:
            logger.error("Ошибка парсинга ответа: %s", error)
            raise

    def test_connection(self) -> bool:
        """Проверяет подключение к API простым тестовым запросом."""
        try:
            test_prompt = "Ответь одним словом: работает"
            response = self.generate(test_prompt, temperature=0.1)
            return "работает" in response["text"].lower()
        except Exception as error:
            logger.error("Тест подключения не пройден: %s", error)
            return False


def main() -> None:
    """Точка входа для локального тестирования клиента."""
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА №1")
    print("Тестирование YandexGPT API")
    print("=" * 80)

    dotenv_path = Path(__file__).resolve().parents[1] / ".env"
    iam_token, api_key, folder_id, api_url = load_yandex_credentials(dotenv_path)

    if iam_token and api_key:
        logger.info("Найдены и IAM token, и API key. Используется IAM token.")

    auth_token = iam_token or api_key
    auth_scheme = "Bearer" if iam_token else "Api-Key"

    if not auth_token:
        print("\nОШИБКА: Не найден YANDEX_IAM_TOKEN или YANDEX_API_KEY")
        print("Создайте файл .env и добавьте одну из переменных:")
        print(" - YANDEX_IAM_TOKEN")
        print(" - YANDEX_API_KEY")
        sys.exit(1)

    if not folder_id:
        print("\nОШИБКА: Не найден YANDEX_FOLDER_ID")
        print("Создайте файл .env и добавьте переменную YANDEX_FOLDER_ID")
        sys.exit(1)

    print("\nПеременные окружения загружены")

    try:
        client = YandexGPTClient(
            auth_token=auth_token,
            folder_id=folder_id,
            auth_scheme=auth_scheme,
            api_url=api_url,
        )
        print("Клиент инициализирован")
    except Exception as error:
        print(f"\nОШИБКА инициализации: {error}")
        sys.exit(1)

    print("\nПроверка подключения...")
    if client.test_connection():
        print("Подключение успешно")
    else:
        print("Подключение не удалось")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("ТЕСТОВЫЙ ЗАПРОС")
    print("=" * 80)

    test_prompt = "Объясни кратко, что такое искусственный интеллект, не более 100 слов."
    print(f"\nЗапрос: {test_prompt}\n")

    try:
        response = client.generate(test_prompt, temperature=0.5)

        print("ОТВЕТ МОДЕЛИ:")
        print("-" * 80)
        print(response["text"])
        print("-" * 80)
        print("\nСтатистика:")
        print(f" • Входные токены: {response['tokens_input']}")
        print(f" • Выходные токены: {response['tokens_output']}")
        print(f" • Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as error:
        print(f"\nОШИБКА выполнения запроса: {error}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА №1 ВЫПОЛНЕНА УСПЕШНО")
    print("=" * 80)


if __name__ == "__main__":
    main()
