# -*- coding: utf-8 -*-
"""
Адаптированный запрос для специальности.
Лабораторная работа №1.
"""

import sys
from pathlib import Path

try:
    from agent_core import YandexGPTClient, load_yandex_credentials
except ImportError:
    from .agent_core import YandexGPTClient, load_yandex_credentials


BASE_DIR = Path(__file__).resolve().parent.parent


def get_specialty_prompt() -> str:
    """Возвращает промпт, адаптированный под тему дипломной работы."""
    prompt = """
Я студент технической специальности и работаю над дипломной темой:
«Информационная система по извлечению и преобразованию числовых данных с графиков функций».

Прошу предоставить информацию по следующим вопросам:
1. Какие методы и технологии используются для извлечения числовых данных с изображений графиков функций?
2. Как выполняется преобразование координат графика в реальные числовые значения с учётом масштаба осей?
3. Какие алгоритмы применяются для обнаружения осей, подписей, делений шкалы и линии функции?
4. Какие готовые открытые и бесплатные решения можно использовать как основу для такой информационной системы?
5. Какие основные проблемы возникают при обработке графиков (шум, сетка, искажения, низкое качество изображения, несколько линий на одном графике) и как их решать?

Требования к ответу:
• Ответ должен быть структурирован
• Используй технические термины
• Приведи конкретные примеры библиотек, алгоритмов и программных средств
• Укажи, какие подходы лучше подходят для растровых изображений графиков
• Объём: 300-500 слов
"""
    return prompt


def main() -> None:
    """Выполняет адаптированный запрос к YandexGPT."""
    print("=" * 80)
    print("АДАПТИРОВАННЫЙ ЗАПРОС ПО СПЕЦИАЛЬНОСТИ")
    print("=" * 80)

    dotenv_path = BASE_DIR / ".env"
    iam_token, api_key, folder_id, api_url = load_yandex_credentials(dotenv_path)

    if iam_token and api_key:
        print("\nНайдены и IAM token, и API key. Используется IAM token.")

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

    try:
        client = YandexGPTClient(
            auth_token=auth_token,
            folder_id=folder_id,
            auth_scheme=auth_scheme,
            api_url=api_url,
        )
    except Exception as error:
        print(f"\nОШИБКА инициализации клиента: {error}")
        sys.exit(1)

    prompt = get_specialty_prompt()

    print("\nЗАПРОС:")
    print("-" * 80)
    print(prompt)
    print("-" * 80)

    print("\nОТВЕТ МОДЕЛИ:")
    print("-" * 80)

    try:
        response = client.generate(prompt, temperature=0.5)
    except Exception as error:
        print(f"\nОШИБКА выполнения запроса: {error}")
        sys.exit(1)

    print(response["text"])
    print("-" * 80)
    print(f"\nТокены: вход={response['tokens_input']}, выход={response['tokens_output']}")

    output_path = BASE_DIR / "docs" / "graph_data_extraction_response.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("ЗАПРОС:\n")
        file.write(prompt)
        file.write("\n\nОТВЕТ:\n")
        file.write(response["text"])

    print(f"\nРезультат сохранён в {output_path}")


if __name__ == "__main__":
    main()
