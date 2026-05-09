import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import requests
from dotenv import load_dotenv


logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

class LLMClient:
    """
    Клиент для работы с YandexGPT API.
    Атрибуты:
    iam_token: IAM-токен для аутентификации
    folder_id: Идентификатор каталога
    model_uri: URI модели
    """
    def __init__(
        self,
        iam_token: Optional[str] = None,
        folder_id: Optional[str] = None
    ):
        self.iam_token = iam_token or os.getenv("YANDEX_IAM_TOKEN")
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        if not self.iam_token or not self.folder_id:
            logger.warning("YANDEX_IAM_TOKEN или YANDEX_FOLDER_ID не настроены")
        self.model_uri = f"gpt://{self.folder_id}/yandexgpt/latest"
        self.api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        logger.info("LLMClient инициализирован")
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "Вы — полезный ассистент.",
        temperature: float = 0.5,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Генерация ответа от LLM.
        Args:
        prompt: Пользовательский запрос
        system_prompt: Системный промпт
        temperature: Параметр креативности
        max_tokens: Максимальное количество токенов
        Returns:
        Dict: Ответ с текстом и метаданными
        """
        if not self.iam_token or not self.folder_id:
            return {
                "success": False,
                "text": "LLM не настроен (отсутствуют учётные данные)",
                "tokens_used": 0
            }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.iam_token}",
            "x-folder-id": self.folder_id
        }
        payload = {
            "modelUri": self.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": max_tokens
            },
            "messages": [
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": prompt}
            ]
        }
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            if "result" not in result:
                raise ValueError("Некорректный ответ API")
            alternatives = result["result"].get("alternatives", [])
            if not alternatives:
                raise ValueError("Пустой ответ от модели")
            text = alternatives[0]["message"]["text"]
            tokens_info = result["result"].get("usage", {})
            return {
                "success": True,
                "text": text,
                "tokens_input": tokens_info.get("inputTextTokens", 0),
                "tokens_output": tokens_info.get("completionTokens", 0),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Ошибка LLM: {e}")
            return {
                "success": False,
                "text": f"Ошибка генерации: {e}",
                "tokens_used": 0
            }

    def classify(
        self,
        text: str,
        categories: list,
        prompt_template: str = None
    ) -> Dict[str, Any]:
        """
        Классификация текста.
        💻 ЧАСТЬ 4: ИНТЕГРАЦИЯ В ГИБРИДНЫЙ ПАЙПЛАЙН
        Шаг 4.1: Основной пайплайн
        Создайте файл week6/src/neuro_symbolic/pipeline.py :
        Args:
        text: Текст для классификации
        categories: Список категорий
        prompt_template: Шаблон промпта
        Returns:
        Dict: Результат классификации
        """
        if prompt_template is None:
            prompt_template = """
                Классифицируй следующий текст в одну из категорий: {categories}.
                Ответь только названием категории.
                Текст: {text}
                Категория:
            """
        prompt = prompt_template.format(
            categories=", ".join(categories),
            text=text
        )
        result = self.generate(
            prompt=prompt,
            system_prompt="Ты — классификатор. Отвечай точно и кратко.",
            temperature=0.1 # Низкая температура для точности
        )
        # Извлечение категории из ответа
        predicted_category = result.get("text", "").strip()

        return {
            "success": result.get("success", False),
            "predicted_category": predicted_category,
            "confidence": 0.8, # В production: рассчитать из logits
            "llm_response": result
        }
    

# Тестирование
if __name__ == "__main__":
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ LLM CLIENT")
    print("=" * 80)
    client = LLMClient()
    # Тест генерации
    result = client.generate("Что такое искусственный интеллект?")
    print(f"\nГенерация: {result.get('success', False)}")
    print(f"Ответ: {result.get('text', '')[:200]}...")
    # Тест классификации
    categories = ["техника", "медицина", "право", "экономика"]
    result = client.classify("Двигатель перегрелся, температура 95°C", categories)
    print(f"\nКлассификация: {result.get('predicted_category', 'N/A')}")
