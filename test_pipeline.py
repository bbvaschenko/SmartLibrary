
"""
Упрощенная версия пайплайна без GigaChat для тестирования
"""

import json
import logging
import os
from pathlib import Path


# Создаем заглушки вместо реальных агентов
class MockGigaChat:
    def generate(self, prompt):
        return "Тестовый ответ от GigaChat (заглушка)"

    def __call__(self, *args, **kwargs):
        return self


class MockAgent:
    def process(self, *args, **kwargs):
        return {"result": "Тестовый результат", "status": "success"}


# Упрощенный пайплайн
class SimplePipeline:
    def __init__(self):
        self.base_data_dir = Path("uploads")
        self.output_dir = Path("pipeline_output")
        self.output_dir.mkdir(exist_ok=True)

        # Заглушки вместо реальных агентов
        self.gigachat = MockGigaChat()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def process_user_query(self, query):
        """Упрощенная обработка запроса"""
        self.logger.info(f"🔍 Обработка запроса: '{query}'")

        # Имитация работы
        return {
            "execution_id": "test_123",
            "query": query,
            "status": "success",
            "books_found": 2,
            "recommendations": [
                "Тестовая рекомендация 1",
                "Тестовая рекомендация 2"
            ],
            "metrics": {
                "CER": 0.3,
                "TID": 2,
                "AAR": 0.8,
                "SCS": 0.7,
                "ECS": 4.2,
                "FRR": 0.9
            }
        }

    def get_metrics_summary(self):
        """Тестовые метрики"""
        return {
            "timestamp": "2024-01-15T12:00:00",
            "metrics": {
                "CER": 0.3, "TID": 2, "AAR": 0.8,
                "SCS": 0.7, "ECS": 4.2, "FRR": 0.9
            },
            "requirements": {
                "CER": 0.2, "TID": 2, "AAR": 0.7,
                "SCS": 0.6, "ECS": 4.0, "FRR": 0.5
            },
            "overall_status": "✅ ВСЕ МЕТРИКИ СООТВЕТСТВУЮТ",
            "statistics": {
                "total_queries": 5,
                "total_books_in_db": 2
            }
        }


def demo_simple_pipeline():
    """Демо упрощенного пайплайна"""
    print("=" * 80)
    print("🧪 УПРОЩЕННЫЙ ПАЙПЛАЙН (ДЛЯ ТЕСТИРОВАНИЯ)")
    print("=" * 80)

    pipeline = SimplePipeline()

    while True:
        print("\n📚 МЕНЮ:")
        print("1. 🔍 Тестовый запрос")
        print("2. 📊 Показать метрики")
        print("3. 🚪 Выйти")

        choice = input("\nВыберите: ").strip()

        if choice == "1":
            query = input("Введите запрос: ").strip()
            if query:
                result = pipeline.process_user_query(query)
                print(f"\n✅ Результат:")
                print(f"Книг найдено: {result['books_found']}")
                print(f"Метрики: CER={result['metrics']['CER']}, AAR={result['metrics']['AAR']}")
        elif choice == "2":
            metrics = pipeline.get_metrics_summary()
            print(f"\n📊 Метрики: {metrics['overall_status']}")
            for metric, value in metrics['metrics'].items():
                print(f"  {metric}: {value}")
        elif choice == "3":
            print("👋 Выход")
            break
        else:
            print("❌ Неверный выбор")


if __name__ == "__main__":
    demo_simple_pipeline()
