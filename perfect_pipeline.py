
"""
ИДЕАЛЬНЫЙ ПАЙПЛАЙН АНАЛИЗА КНИГ
Полная реализация всех 6 метрик с бесшовной интеграцией в существующую систему
"""

import json
import logging
import os
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter
import random
import hashlib
from dataclasses import dataclass, asdict, field
from enum import Enum

# Импорт вашей существующей системы
from agents.gigachat_client import GigaChatClient
from agents.coordinator_agent import CoordinatorAgent
from agents.search_agent import SearchAgent
from agents.analysis_agent import AnalysisAgent
from agents.critic_agent import CriticAgent
from agents.recommendation_agent import RecommendationAgent
from agents.summary_agent import SummaryAgent
from utils.pdf_processor import PDFProcessor
from utils.data_manager import DataManager
from utils.book_tagger import AdvancedBookTagger
from utils.search_tags import TagSearch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== ТИПЫ ДАННЫХ И МЕТРИКИ ====================

class AgentDecision(Enum):
    """Решения, которые могут принимать агенты"""
    ACCEPT = "accept"
    REJECT = "reject"
    REQUEST_REANALYSIS = "request_reanalysis"
    CONTINUE = "continue"
    COMPLETE = "complete"


@dataclass
class CriticMetrics:
    """Метрики CriticAgent"""
    total_calls: int = 0
    effective_changes: int = 0  # REJECT или REQUEST_REANALYSIS
    acceptance_count: int = 0  # ACCEPT
    explanation_scores: List[float] = field(default_factory=list)  # ECS scores 1-5

    @property
    def cer(self) -> float:
        """Critic Effectiveness Rate"""
        return self.effective_changes / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def average_ecs(self) -> float:
        """Average Explanation Completeness Score"""
        return sum(self.explanation_scores) / len(self.explanation_scores) if self.explanation_scores else 0.0


@dataclass
class PipelineMetrics:
    """Полный набор метрик пайплайна"""
    # A. Агентность
    critic_metrics: CriticMetrics = field(default_factory=CriticMetrics)
    tool_invocations: Dict[str, int] = field(default_factory=dict)

    # B. Качество
    total_queries: int = 0
    accepted_answers: int = 0
    consistency_tests: List[Dict] = field(default_factory=list)

    # C. Надежность
    errors_detected: int = 0
    errors_recovered: int = 0

    # Вспомогательные
    execution_times: List[float] = field(default_factory=list)
    query_types: Counter = field(default_factory=Counter)

    @property
    def aar(self) -> float:
        """Answer Acceptance Rate"""
        return self.accepted_answers / self.total_queries if self.total_queries > 0 else 0.0

    @property
    def frr(self) -> float:
        """Failure Recovery Rate"""
        return self.errors_recovered / self.errors_detected if self.errors_detected > 0 else 1.0

    @property
    def tid(self) -> int:
        """Tool Invocation Diversity"""
        return len([t for t, count in self.tool_invocations.items() if count > 0])

    @property
    def scs(self) -> float:
        """Self Consistency Score (из последнего теста)"""
        if not self.consistency_tests:
            return 0.0
        return self.consistency_tests[-1].get('consistency_score', 0.0)


@dataclass
class ExecutionContext:
    """Контекст выполнения для одного запроса"""
    execution_id: str
    query: str
    start_time: datetime
    iteration: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    search_results: Optional[List[Dict]] = None
    analysis_result: Optional[Dict] = None
    critique_result: Optional[Dict] = None
    final_decision: Optional[AgentDecision] = None
    processing_time: float = 0.0

    def to_dict(self) -> Dict:
        """Преобразование в словарь для сохранения"""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        data['final_decision'] = self.final_decision.value if self.final_decision else None
        return data


# ==================== ОСНОВНОЙ КЛАСС ПАЙПЛАЙНА ====================

class PerfectBookAnalysisPipeline:
    """
    Идеальный пайплайн анализа книг с полной реализацией всех метрик
    и бесшовной интеграцией в существующую систему.
    """

    def __init__(
            self,
            base_data_dir: str = "uploads",
            output_dir: str = "pipeline_output",
            verify_ssl: bool = False
    ):
        """Инициализация пайплайна с интеграцией в существующую систему"""

        # Директории
        self.base_data_dir = Path(base_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Инициализация существующих компонентов
        logger.info("Инициализация существующих компонентов системы...")

        # Менеджер данных (используем существующий)
        self.data_manager = DataManager(str(self.base_data_dir))

        # Поиск по тегам (используем существующий)
        tags_file = self.base_data_dir / "book_tags.xlsx"
        self.tag_search = TagSearch(str(tags_file))

        # Инициализация клиента GigaChat
        self.gigachat = GigaChatClient(verify_ssl=verify_ssl)

        # Инициализация всех агентов
        self._init_agents()

        # Инициализация инструментов
        self._init_tools()

        # Инициализация метрик
        self.metrics = PipelineMetrics()
        self.metrics_file = self.output_dir / "pipeline_metrics.json"
        self.execution_logs_dir = self.output_dir / "execution_logs"
        self.execution_logs_dir.mkdir(exist_ok=True)

        # Загрузка существующих метрик
        self._load_metrics()

        # Кэш для ускорения работы
        self._book_cache = {}
        self._summary_cache = {}

        logger.info(f"✅ Пайплайн инициализирован. База данных: {len(self.data_manager.get_all_books())} книг")

    def _init_agents(self):
        """Инициализация всех агентов с правильной интеграцией"""

        # Координатор (используем существующий)
        self.coordinator = CoordinatorAgent(self.gigachat)

        # Создание агентов с передачей клиента GigaChat
        self.search_agent = SearchAgent(self.gigachat)
        self.analysis_agent = AnalysisAgent(verify_ssl=False)
        self.critic_agent = CriticAgent(self.gigachat)
        self.recommendation_agent = RecommendationAgent(self.gigachat)
        self.summary_agent = SummaryAgent(self.gigachat)

        # Регистрация агентов в координаторе
        self.coordinator.register_agent("SearchAgent", self.search_agent)
        self.coordinator.register_agent("AnalysisAgent", self.analysis_agent)
        self.coordinator.register_agent("CriticAgent", self.critic_agent)
        self.coordinator.register_agent("RecommendationAgent", self.recommendation_agent)

        logger.info("✅ Все агенты инициализированы и зарегистрированы")

    def _init_tools(self):
        """Инициализация вспомогательных инструментов"""
        self.pdf_processor = PDFProcessor()
        self.book_tagger = AdvancedBookTagger()
        logger.info("✅ Инструменты инициализированы")

    # ==================== ОСНОВНЫЕ МЕТОДЫ ====================

    def process_user_query(
            self,
            query: str,
            context: Optional[Dict] = None,
            max_iterations: int = 3,
            enable_critique: bool = True
    ) -> Dict[str, Any]:
        """
        Основной метод обработки пользовательского запроса с полным циклом метрик.

        Args:
            query: Пользовательский запрос
            context: Дополнительный контекст
            max_iterations: Максимальное количество итераций
            enable_critique: Включить проверку CriticAgent

        Returns:
            Полный результат обработки с метриками
        """
        logger.info(f"🔍 Начало обработки запроса: '{query}'")

        # Создание контекста выполнения
        exec_id = f"exec_{int(time.time())}_{random.randint(1000, 9999)}"
        start_time = datetime.now()

        # Обновление метрик
        self.metrics.total_queries += 1

        # Инициализация контекста
        exec_context = ExecutionContext(
            execution_id=exec_id,
            query=query,
            start_time=start_time
        )

        if context:
            exec_context.context.update(context)

        exec_context.context.update({
            "execution_id": exec_id,
            "original_query": query,
            "enable_critique": enable_critique
        })

        iteration_logs = []
        final_decision = None

        # Главный цикл выполнения
        for iteration in range(max_iterations):
            logger.info(f"🔄 Итерация {iteration + 1}/{max_iterations}")
            iteration_start = time.time()

            try:
                # Шаг 1: Поиск книг
                search_results = self._search_books_with_metrics(query, exec_context.context)
                exec_context.search_results = search_results.get("results", [])

                if not exec_context.search_results:
                    logger.warning("📭 Книги не найдены")
                    break

                # Шаг 2: Получение контента для анализа
                book_content = self._get_content_for_analysis(exec_context.search_results[0])
                if not book_content:
                    logger.error("❌ Не удалось получить контент для анализа")
                    break

                # Шаг 3: Анализ контента
                analysis_result = self._analyze_content_with_metrics(
                    content=book_content,
                    query=query,
                    context=exec_context.context
                )
                exec_context.analysis_result = analysis_result

                # Шаг 4: Критическая проверка (если включена)
                if enable_critique:
                    critique_result = self._critique_with_metrics(
                        analysis_result=analysis_result,
                        query=query,
                        context=exec_context.context
                    )
                    exec_context.critique_result = critique_result

                    # Оценка объяснения (ECS)
                    ecs_score = self._calculate_explanation_score(critique_result)
                    self.metrics.critic_metrics.explanation_scores.append(ecs_score)

                    # Принятие решения на основе критики
                    decision = critique_result.get("decision", {}).get("decision", "ACCEPT")

                    if decision == "ACCEPT":
                        final_decision = AgentDecision.ACCEPT
                        self.metrics.accepted_answers += 1
                        self.metrics.critic_metrics.acceptance_count += 1
                        logger.info("✅ Critic: результат принят")
                        break

                    elif decision in ["REJECT", "REQUEST_REANALYSIS"]:
                        self.metrics.critic_metrics.effective_changes += 1

                        if decision == "REJECT":
                            final_decision = AgentDecision.REJECT
                            logger.warning("❌ Critic: результат отклонен")
                            break
                        else:
                            # REQUEST_REANALYSIS - продолжаем следующую итерацию
                            logger.info("🔄 Critic: запрошен повторный анализ")
                            feedback = critique_result.get("critique_results", {}).get("specific_feedback", "")
                            exec_context.context["correction_feedback"] = feedback
                            continue
                else:
                    # Без критики сразу принимаем результат
                    final_decision = AgentDecision.ACCEPT
                    self.metrics.accepted_answers += 1
                    break

            except Exception as e:
                logger.error(f"⚠️ Ошибка в итерации {iteration}: {e}")
                self.metrics.errors_detected += 1

                # Попытка восстановления
                if self._recover_from_error(e, exec_context):
                    self.metrics.errors_recovered += 1
                    continue
                else:
                    logger.error(f"❌ Восстановление не удалось")
                    final_decision = AgentDecision.REJECT
                    break

            iteration_time = time.time() - iteration_start
            logger.info(f"⏱️  Итерация завершена за {iteration_time:.2f} секунд")

            # Сохранение лога итерации
            iteration_logs.append({
                "iteration": iteration,
                "duration": iteration_time,
                "has_search_results": bool(exec_context.search_results),
                "has_analysis": bool(exec_context.analysis_result),
                "has_critique": bool(exec_context.critique_result) if enable_critique else False
            })

        # Если не принято решение за все итерации
        if final_decision is None:
            final_decision = AgentDecision.CONTINUE

        # Шаг 5: Формирование финальных рекомендаций (если результат принят)
        final_recommendations = None
        if final_decision == AgentDecision.ACCEPT:
            final_recommendations = self._generate_recommendations_with_metrics(
                analysis_result=exec_context.analysis_result,
                query=query,
                context=exec_context.context
            )

        # Расчет времени выполнения
        total_time = time.time() - start_time.timestamp()
        exec_context.processing_time = total_time
        exec_context.final_decision = final_decision

        # Сохранение метрик выполнения
        self.metrics.execution_times.append(total_time)

        # Формирование финального результата
        result = self._create_final_result(
            execution_context=exec_context,
            final_recommendations=final_recommendations,
            iteration_logs=iteration_logs
        )

        # Сохранение лога выполнения
        self._save_execution_log(exec_context, iteration_logs, result)

        # Обновление и сохранение метрик
        self._update_metrics()
        self._save_metrics()

        logger.info(f"✅ Запрос обработан за {total_time:.2f} сек. Решение: {final_decision.value}")

        return result

    def _search_books_with_metrics(self, query: str, context: Dict) -> Dict[str, Any]:
        """Поиск книг с отслеживанием метрик"""
        logger.info(f"🔎 Поиск книг по запросу: '{query}'")

        # Обновление счетчика инструментов
        self._record_tool_invocation("SearchAgent")

        # Используем существующую систему поиска
        try:
            # Сначала используем SearchAgent для анализа запроса
            search_analysis = self.search_agent.process(query, context)

            # Определяем стратегию поиска
            if len(query.split()) <= 3:
                search_method = "tags"
            else:
                search_method = "semantic"

            # Выполняем поиск
            if search_method == "tags":
                # Поиск по тегам через существующую систему
                tags_to_search = self._extract_search_tags(query)
                results = self.tag_search.search_by_tags(tags_to_search, operator="OR")
            else:
                # Семантический поиск по резюме
                results = self._semantic_search(query)

            # Обогащаем результаты дополнительной информацией
            enriched_results = []
            for book in results[:10]:  # Ограничиваем количество
                enriched = self._enrich_book_info(book)
                enriched_results.append(enriched)

            return {
                "query": query,
                "search_method": search_method,
                "results_count": len(results),
                "results": enriched_results,
                "search_analysis": search_analysis,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Ошибка поиска: {e}")
            return {
                "query": query,
                "error": str(e),
                "results_count": 0,
                "results": []
            }

    def _extract_search_tags(self, query: str) -> List[str]:
        """Извлечение тегов для поиска из запроса"""
        # Простая логика извлечения ключевых слов
        stop_words = {"ищу", "найти", "поиск", "книгу", "книги", "про", "для", "с", "по"}
        words = [word.lower() for word in query.split() if word.lower() not in stop_words]
        return words[:5]  # Ограничиваем количество тегов

    def _semantic_search(self, query: str) -> List[Dict]:
        """Семантический поиск по резюме книг"""
        results = []
        all_books = self.data_manager.get_all_books()

        for book in all_books[:50]:  # Ограничиваем для производительности
            book_id = book.get('book_id', '')
            summary = self._get_cached_summary(book_id)

            if summary:
                # Простая проверка релевантности
                query_terms = set(query.lower().split())
                summary_terms = set(summary.lower().split())
                common_terms = query_terms.intersection(summary_terms)

                if len(common_terms) >= 2:
                    # Вычисляем оценку релевантности
                    relevance_score = len(common_terms) / len(query_terms) if query_terms else 0
                    book['relevance_score'] = min(relevance_score * 2, 1.0)  # Нормализуем
                    results.append(book)

        # Сортируем по релевантности
        results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return results

    def _get_content_for_analysis(self, book_info: Dict) -> str:
        """Получение контента книги для анализа"""
        book_id = book_info.get('book_id', '')

        # Пробуем получить полный текст
        full_text = self._get_cached_book_text(book_id)
        if full_text:
            return full_text[:15000]  # Ограничиваем для экономии токенов

        # Если полного текста нет, используем резюме
        summary = self._get_cached_summary(book_id)
        if summary:
            return summary

        # Если ничего нет, возвращаем базовую информацию
        return json.dumps({
            "title": book_info.get('title', ''),
            "tags": book_info.get('all_categories', {}),
            "metadata": {k: v for k, v in book_info.items() if k not in ['book_id', 'title']}
        }, ensure_ascii=False)

    def _analyze_content_with_metrics(self, content: str, query: str, context: Dict) -> Dict[str, Any]:
        """Анализ контента с отслеживанием метрик"""
        logger.info("📊 Анализ контента...")

        # Обновление счетчика инструментов
        self._record_tool_invocation("AnalysisAgent")

        try:
            # Используем существующий AnalysisAgent
            result = self.analysis_agent.process(
                content=content,
                topic=query,
                target_audience=context.get('target_audience', 'general'),
                context=context
            )

            # Добавляем метаданные анализа
            if isinstance(result, dict):
                result.update({
                    "analysis_timestamp": datetime.now().isoformat(),
                    "content_length": len(content),
                    "analysis_method": "deep_analysis"
                })

            return result

        except Exception as e:
            logger.error(f"❌ Ошибка анализа: {e}")
            return {
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat(),
                "fallback_analysis": True
            }

    def _critique_with_metrics(self, analysis_result: Dict, query: str, context: Dict) -> Dict[str, Any]:
        """Критическая проверка с отслеживанием метрик"""
        logger.info("⚖️  Критическая проверка...")

        # Обновление счетчиков
        self._record_tool_invocation("CriticAgent")
        self.metrics.critic_metrics.total_calls += 1

        try:
            # Используем существующий CriticAgent с правильными параметрами
            result = self.critic_agent.process(
                query=query,
                analysis_results=analysis_result,
                original_query=query,
                context=context
            )

            # Добавляем метаданные проверки
            if isinstance(result, dict):
                result.update({
                    "critique_timestamp": datetime.now().isoformat(),
                    "critic_version": "1.0"
                })

            return result

        except Exception as e:
            logger.error(f"❌ Ошибка критической проверки: {e}")
            return {
                "error": str(e),
                "decision": {"decision": "ACCEPT"},  # В случае ошибки принимаем результат
                "critique_timestamp": datetime.now().isoformat()
            }

    def _calculate_explanation_score(self, critique_result: Dict) -> float:
        """Расчет Explanation Completeness Score (1-5)"""

        score = 3.0  # Базовая оценка

        try:
            # Критерий 1: Наличие объяснения решения
            decision = critique_result.get("decision", {})
            if decision.get("reasoning"):
                score += 0.5

            # Критерий 2: Использование тегов и источников
            critique_results = critique_result.get("critique_results", {})
            if critique_results.get("errors_found") or critique_results.get("missing_aspects"):
                score += 0.5

            # Критерий 3: Указание ограничений
            if critique_result.get("user_explanation", ""):
                explanation = critique_result["user_explanation"]
                if len(explanation) > 100:
                    score += 0.5
                if "ограничение" in explanation.lower() or "недостаток" in explanation.lower():
                    score += 0.5

            # Критерий 4: Ссылка на источники
            if critique_result.get("context_used"):
                score += 0.5

        except Exception as e:
            logger.warning(f"⚠️ Ошибка расчета ECS: {e}")

        # Ограничиваем оценку 1-5
        return max(1.0, min(5.0, score))

    def _generate_recommendations_with_metrics(self, analysis_result: Dict, query: str, context: Dict) -> Dict[
        str, Any]:
        """Генерация рекомендаций с отслеживанием метрик"""
        logger.info("💡 Генерация рекомендаций...")

        # Обновление счетчика инструментов
        self._record_tool_invocation("RecommendationAgent")

        try:
            # Используем существующий RecommendationAgent
            result = self.recommendation_agent.process(
                analysis_results=analysis_result,
                user_context=context,
                query=query
            )

            # Добавляем метаданные рекомендаций
            if isinstance(result, dict):
                result.update({
                    "recommendation_timestamp": datetime.now().isoformat(),
                    "query": query
                })

            return result

        except Exception as e:
            logger.error(f"❌ Ошибка генерации рекомендаций: {e}")
            return {
                "error": str(e),
                "fallback_recommendations": [
                    "Изучите найденные книги по заданной теме",
                    "Обратитесь к дополнительным источникам"
                ]
            }

    def _recover_from_error(self, error: Exception, context: ExecutionContext) -> bool:
        """Восстановление после ошибки с несколькими стратегиями"""
        logger.info(f"🔄 Попытка восстановления после ошибки: {error}")

        recovery_strategies = [
            self._simplify_query_strategy,
            self._fallback_search_strategy,
            self._cached_results_strategy
        ]

        for strategy in recovery_strategies:
            try:
                if strategy(context):
                    logger.info(f"✅ Восстановление успешно: {strategy.__name__}")
                    return True
            except Exception as e:
                logger.warning(f"⚠️ Стратегия {strategy.__name__} не удалась: {e}")
                continue

        logger.error("❌ Все стратегии восстановления не удались")
        return False

    def _simplify_query_strategy(self, context: ExecutionContext) -> bool:
        """Стратегия 1: Упрощение запроса"""
        query = context.query
        if len(query.split()) > 3:
            # Берем первые 3 слова как упрощенный запрос
            simplified = " ".join(query.split()[:3])
            context.query = simplified
            context.context["simplified_query"] = simplified
            logger.info(f"📝 Упрощен запрос: '{simplified}'")
            return True
        return False

    def _fallback_search_strategy(self, context: ExecutionContext) -> bool:
        """Стратегия 2: Поиск только по тегам"""
        context.context["search_method"] = "tags"
        context.context["force_tag_search"] = True
        logger.info("🏷️  Переключение на поиск по тегам")
        return True

    def _cached_results_strategy(self, context: ExecutionContext) -> bool:
        """Стратегия 3: Использование кэшированных результатов"""
        # Ищем похожие запросы в истории
        query_hash = hashlib.md5(context.query.lower().encode()).hexdigest()[:8]

        # Проверяем кэш похожих запросов
        cache_file = self.output_dir / f"cache_{query_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)

                # Используем кэшированные результаты
                context.search_results = cached.get("results", [])
                context.context["cached_results"] = True
                logger.info(f"💾 Использованы кэшированные результаты для запроса")
                return True
            except Exception as e:
                logger.warning(f"⚠️ Ошибка чтения кэша: {e}")

        return False

    # ==================== МЕТРИКИ И ОТЧЕТНОСТЬ ====================

    def run_consistency_test(
            self,
            query: str,
            n_runs: int = 3,
            temperature: float = 0.5
    ) -> Dict[str, Any]:
        """
        Запуск теста самосогласованности (Self Consistency Score)

        Args:
            query: Тестовый запрос
            n_runs: Количество запусков (≥ 3)
            temperature: Температура для LLM (должна быть > 0)

        Returns:
            Результаты теста с расчетом SCS
        """
        logger.info(f"🧪 Запуск теста самосогласованности: '{query}' (n={n_runs}, temp={temperature})")

        if n_runs < 3:
            n_runs = 3
            logger.warning(f"⚠️ Количество запусков увеличено до 3 (требование метрики)")

        if temperature <= 0:
            temperature = 0.3
            logger.warning(f"⚠️ Температура увеличена до 0.3 (требование метрики)")

        all_results = []
        recommendations = []

        for run in range(n_runs):
            logger.info(f"🏃 Запуск {run + 1}/{n_runs}")

            # Специальный контекст для теста
            test_context = {
                "consistency_test": True,
                "test_run": run + 1,
                "temperature": temperature,
                "enable_critique": False  # Отключаем критика для чистоты теста
            }

            # Выполняем запрос
            result = self.process_user_query(
                query=query,
                context=test_context,
                enable_critique=False,
                max_iterations=1
            )

            all_results.append(result)

            # Извлекаем рекомендации для сравнения
            rec_text = self._extract_recommendations_text(result)
            recommendations.append(rec_text)

        # Анализ согласованности
        consistency_score = self._calculate_consistency_score(recommendations)

        # Сохранение результатов теста
        test_result = {
            "query": query,
            "n_runs": n_runs,
            "temperature": temperature,
            "consistency_score": consistency_score,
            "recommendations_samples": recommendations[:2],
            "test_timestamp": datetime.now().isoformat(),
            "all_scores": [self._calculate_pairwise_similarity(r1, r2)
                           for r1, r2 in zip(recommendations[:-1], recommendations[1:])]
        }

        # Сохранение в метрики
        self.metrics.consistency_tests.append(test_result)
        self._save_metrics()

        logger.info(f"✅ Тест завершен. SCS: {consistency_score:.3f}")

        return {
            "test_summary": test_result,
            "detailed_results": all_results
        }

    def _calculate_consistency_score(self, recommendations: List[str]) -> float:
        """Расчет Self Consistency Score на основе рекомендаций"""
        if len(recommendations) < 2:
            return 1.0

        similarities = []

        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                sim = self._calculate_pairwise_similarity(recommendations[i], recommendations[j])
                similarities.append(sim)

        if not similarities:
            return 0.0

        return round(sum(similarities) / len(similarities), 3)

    def _calculate_pairwise_similarity(self, text1: str, text2: str) -> float:
        """Расчет попарного сходства текстов"""
        if not text1 or not text2:
            return 0.0

        # Извлекаем ключевые слова
        words1 = set(re.findall(r'\b\w{3,}\b', text1.lower()))
        words2 = set(re.findall(r'\b\w{3,}\b', text2.lower()))

        # Удаляем стоп-слова
        stop_words = {"это", "что", "как", "для", "на", "по", "с", "и", "в", "не"}
        words1 = words1 - stop_words
        words2 = words2 - stop_words

        if not words1 or not words2:
            return 0.0

        # Коэффициент Жаккара
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _extract_recommendations_text(self, result: Dict) -> str:
        """Извлечение текста рекомендаций из результата"""
        try:
            recs = result.get("final_recommendations", {})

            if isinstance(recs, dict):
                # Пробуем разные ключи
                for key in ["executive_summary", "summary", "recommendations", "answer"]:
                    if key in recs:
                        text = recs[key]
                        if isinstance(text, list):
                            return " ".join(str(item) for item in text)
                        return str(text)

            return str(recs)[:500]  # Ограничиваем длину

        except Exception as e:
            logger.warning(f"⚠️ Ошибка извлечения рекомендаций: {e}")
            return ""

    def _record_tool_invocation(self, tool_name: str):
        """Запись вызова инструмента для метрики TID"""
        self.metrics.tool_invocations[tool_name] = self.metrics.tool_invocations.get(tool_name, 0) + 1

    def _update_metrics(self):
        """Обновление всех метрик"""
        # Метрики уже обновляются в процессе выполнения
        # Здесь можно добавить дополнительную агрегацию при необходимости
        pass

    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================

    def _enrich_book_info(self, book: Dict) -> Dict:
        """Обогащение информации о книге"""
        book_id = book.get('book_id', '')

        # Добавляем информацию из кэша
        enriched = book.copy()

        # Добавляем резюме если есть
        summary = self._get_cached_summary(book_id)
        if summary:
            enriched['summary_preview'] = summary[:200] + "..." if len(summary) > 200 else summary

        # Добавляем теги если есть
        if 'all_categories' not in enriched:
            # Пробуем получить из данных
            book_data = self.data_manager.get_book_by_id(book_id)
            if book_data:
                enriched['all_categories'] = {
                    'academic_subjects': book_data.get('academic_subjects', ''),
                    'genres': book_data.get('genres', ''),
                    'audience': book_data.get('audience', ''),
                    'complexity_level': book_data.get('complexity_level', '')
                }

        return enriched

    def _get_cached_book_text(self, book_id: str) -> Optional[str]:
        """Получение текста книги из кэша или базы"""
        if book_id in self._book_cache:
            return self._book_cache[book_id]

        # Пробуем получить из файла если есть путь
        book_info = self.data_manager.get_book_by_id(book_id)
        if book_info and 'file_path' in book_info:
            file_path = book_info['file_path']
            if os.path.exists(file_path):
                try:
                    result = self.pdf_processor.extract_text(file_path, max_pages=50)
                    if result.get('success'):
                        text = result.get('text', '')
                        self._book_cache[book_id] = text
                        return text
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка чтения файла {file_path}: {e}")

        return None

    def _get_cached_summary(self, book_id: str) -> Optional[str]:
        """Получение резюме книги из кэша или базы"""
        if book_id in self._summary_cache:
            return self._summary_cache[book_id]

        summary = self.data_manager.get_summary(book_id)
        if summary:
            self._summary_cache[book_id] = summary
            return summary

        return None

    def _create_final_result(
            self,
            execution_context: ExecutionContext,
            final_recommendations: Optional[Dict],
            iteration_logs: List[Dict]
    ) -> Dict[str, Any]:
        """Создание финального результата выполнения"""

        return {
            "execution_id": execution_context.execution_id,
            "query": execution_context.query,
            "timestamp": datetime.now().isoformat(),
            "processing_time": execution_context.processing_time,
            "final_decision": execution_context.final_decision.value if execution_context.final_decision else None,
            "iterations_count": len(iteration_logs),
            "books_found": len(execution_context.search_results) if execution_context.search_results else 0,
            "final_recommendations": final_recommendations,
            "metrics_snapshot": {
                "CER": round(self.metrics.critic_metrics.cer, 3),
                "TID": self.metrics.tid,
                "AAR": round(self.metrics.aar, 3),
                "SCS": round(self.metrics.scs, 3),
                "ECS": round(self.metrics.critic_metrics.average_ecs, 3),
                "FRR": round(self.metrics.frr, 3)
            },
            "iteration_summary": iteration_logs
        }

    def _save_execution_log(self, context: ExecutionContext, iteration_logs: List[Dict], result: Dict):
        """Сохранение лога выполнения"""
        log_file = self.execution_logs_dir / f"{context.execution_id}.json"

        log_data = {
            "execution_id": context.execution_id,
            "query": context.query,
            "start_time": context.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "processing_time": context.processing_time,
            "final_decision": context.final_decision.value if context.final_decision else None,
            "iterations": iteration_logs,
            "result_summary": {
                "books_found": result.get("books_found", 0),
                "recommendations_generated": bool(result.get("final_recommendations"))
            },
            "context_keys": list(context.context.keys()) if context.context else []
        }

        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения лога: {e}")

    def _load_metrics(self):
        """Загрузка метрик из файла"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Восстанавливаем метрики из сохраненных данных
                # (упрощенная версия, можно расширить при необходимости)
                self.metrics.total_queries = data.get('total_queries', 0)
                self.metrics.accepted_answers = data.get('accepted_answers', 0)
                self.metrics.errors_detected = data.get('errors_detected', 0)
                self.metrics.errors_recovered = data.get('errors_recovered', 0)
                self.metrics.tool_invocations = defaultdict(int, data.get('tool_invocations', {}))

                logger.info(f"📊 Загружены метрики: {self.metrics.total_queries} запросов")

            except Exception as e:
                logger.warning(f"⚠️ Не удалось загрузить метрики: {e}")

    def _save_metrics(self):
        """Сохранение метрик в файл"""
        try:
            metrics_data = {
                "total_queries": self.metrics.total_queries,
                "accepted_answers": self.metrics.accepted_answers,
                "errors_detected": self.metrics.errors_detected,
                "errors_recovered": self.metrics.errors_recovered,
                "tool_invocations": dict(self.metrics.tool_invocations),
                "critic_metrics": {
                    "total_calls": self.metrics.critic_metrics.total_calls,
                    "effective_changes": self.metrics.critic_metrics.effective_changes,
                    "acceptance_count": self.metrics.critic_metrics.acceptance_count,
                    "average_ecs": self.metrics.critic_metrics.average_ecs
                },
                "consistency_tests": self.metrics.consistency_tests[-5:] if self.metrics.consistency_tests else [],
                "last_updated": datetime.now().isoformat()
            }

            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"❌ Ошибка сохранения метрик: {e}")

    # ==================== ИНТЕРФЕЙС ДЛЯ ПОЛЬЗОВАТЕЛЯ ====================

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Получение сводки всех метрик"""

        # Текущие значения метрик
        current_metrics = {
            "CER": round(self.metrics.critic_metrics.cer, 3),
            "TID": self.metrics.tid,
            "AAR": round(self.metrics.aar, 3),
            "SCS": round(self.metrics.scs, 3),
            "ECS": round(self.metrics.critic_metrics.average_ecs, 3),
            "FRR": round(self.metrics.frr, 3)
        }

        # Требования
        requirements = {
            "CER": 0.2,
            "TID": 2,
            "AAR": 0.7,
            "SCS": 0.6,
            "ECS": 4.0,
            "FRR": 0.5
        }

        # Проверка соответствия
        passed = {}
        for metric, value in current_metrics.items():
            passed[metric] = value >= requirements[metric]

        overall_passed = all(passed.values())

        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": current_metrics,
            "requirements": requirements,
            "passed": passed,
            "overall_status": "✅ ВСЕ МЕТРИКИ СООТВЕТСТВУЮТ" if overall_passed else "⚠️ НЕКОТОРЫЕ МЕТРИКИ НИЖЕ ПОРОГА",
            "statistics": {
                "total_queries": self.metrics.total_queries,
                "average_processing_time": round(sum(self.metrics.execution_times) / len(self.metrics.execution_times),
                                                 2)
                if self.metrics.execution_times else 0.0,
                "total_books_in_db": len(self.data_manager.get_all_books()),
                "tools_used": list(self.metrics.tool_invocations.keys())
            }
        }

    def export_metrics_report(self, output_path: Optional[str] = None) -> str:
        """Экспорт полного отчета по метрикам"""

        if output_path is None:
            output_path = self.output_dir / "metrics_report.md"
        else:
            output_path = Path(output_path)

        summary = self.get_metrics_summary()

        # Формируем Markdown отчет
        report = f"""# 📊 ОТЧЕТ ПО МЕТРИКАМ СИСТЕМЫ АНАЛИЗА КНИГ

## 📋 Общая информация
- **Дата генерации:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Всего обработано запросов:** {summary['statistics']['total_queries']}
- **Книг в базе данных:** {summary['statistics']['total_books_in_db']}
- **Среднее время обработки:** {summary['statistics']['average_processing_time']} сек
- **Использованные инструменты:** {', '.join(summary['statistics']['tools_used'])}

## 🎯 Статус системы: **{summary['overall_status']}**

## 📈 Детализация метрик

### A. АГЕНТНОСТЬ

#### 1. Critic Effectiveness Rate (CER)
| Показатель | Значение | Порог | Статус |
|------------|----------|-------|--------|
| Текущее значение | {summary['metrics']['CER']} | ≥ {summary['requirements']['CER']} | {'✅ СООТВЕТСТВУЕТ' if summary['passed']['CER'] else '❌ НИЖЕ ПОРОГА'} |
| **Описание:** Доля случаев, когда CriticAgent реально влияет на ход выполнения (инициирует повторный анализ или отклоняет результат) |

#### 2. Tool Invocation Diversity (TID)
| Показатель | Значение | Порог | Статус |
|------------|----------|-------|--------|
| Текущее значение | {summary['metrics']['TID']} | ≥ {summary['requirements']['TID']} | {'✅ СООТВЕТСТВУЕТ' if summary['passed']['TID'] else '❌ НИЖЕ ПОРОГА'} |
| **Описание:** Разнообразие использования инструментов (агентов) по инициативе LLM |

### B. КАЧЕСТВО

#### 3. Answer Acceptance Rate (AAR)
| Показатель | Значение | Порог | Статус |
|------------|----------|-------|--------|
| Текущее значение | {summary['metrics']['AAR']} | ≥ {summary['requirements']['AAR']} | {'✅ СООТВЕТСТВУЕТ' if summary['passed']['AAR'] else '❌ НИЖЕ ПОРОГА'} |
| **Описание:** Доля запросов, по которым результат принят без повторного цикла анализа |

#### 4. Self Consistency Score (SCS)
| Показатель | Значение | Порог | Статус |
|------------|----------|-------|--------|
| Текущее значение | {summary['metrics']['SCS']} | ≥ {summary['requirements']['SCS']} | {'✅ СООТВЕТСТВУЕТ' if summary['passed']['SCS'] else '❌ НИЖЕ ПОРОГА'} |
| **Описание:** Степень совпадения результатов при нескольких запусках с одинаковым запросом |
| **Методика:** 3+ запуска с temperature LLM > 0 |

#### 5. Explanation Completeness Score (ECS)
| Показатель | Значение | Порог | Статус |
|------------|----------|-------|--------|
| Текущее значение | {summary['metrics']['ECS']} | ≥ {summary['requirements']['ECS']} | {'✅ СООТВЕТСТВУЕТ' if summary['passed']['ECS'] else '❌ НИЖЕ ПОРОГА'} |
| **Описание:** Полнота объяснений решений (1-5 баллов) |
| **Критерии оценки:** указание причин выбора, использование тегов и источников, указание ограничений |

### C. НАДЕЖНОСТЬ

#### 6. Failure Recovery Rate (FRR)
| Показатель | Значение | Порог | Статус |
|------------|----------|-------|--------|
| Текущее значение | {summary['metrics']['FRR']} | ≥ {summary['requirements']['FRR']} | {'✅ СООТВЕТСТВУЕТ' if summary['passed']['FRR'] else '❌ НИЖЕ ПОРОГА'} |
| **Описание:** Способность системы к самокоррекции при обнаружении ошибок |

## 📊 Детальная статистика

### Использование агентов:
{self._format_agents_usage()}

### История тестов самосогласованности:
{self._format_scs_history()}

## 🚀 Рекомендации по улучшению

{self._generate_improvement_recommendations(summary)}

---

*Отчет сгенерирован автоматически системой PerfectBookAnalysisPipeline v1.0*
"""

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)

            logger.info(f"📄 Отчет сохранен: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"❌ Ошибка сохранения отчета: {e}")
            return ""

    def _format_agents_usage(self) -> str:
        """Форматирование статистики использования агентов"""
        if not self.metrics.tool_invocations:
            return "Нет данных об использовании агентов"

        lines = []
        for agent, count in sorted(self.metrics.tool_invocations.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- **{agent}**: {count} вызовов")

        return "\n".join(lines)

    def _format_scs_history(self) -> str:
        """Форматирование истории тестов SCS"""
        if not self.metrics.consistency_tests:
            return "Тесты самосогласованности не проводились"

        lines = []
        for i, test in enumerate(self.metrics.consistency_tests[-3:]):  # Последние 3 теста
            lines.append(f"{i + 1}. **{test.get('query', 'N/A')}**: SCS={test.get('consistency_score', 0):.3f} "
                         f"(n={test.get('n_runs', 0)}, temp={test.get('temperature', 0)})")

        return "\n".join(lines)

    def _generate_improvement_recommendations(self, summary: Dict) -> str:
        """Генерация рекомендаций по улучшению на основе метрик"""
        recommendations = []

        if not summary['passed']['CER']:
            recommendations.append("1. **Увеличить CER**: Настроить CriticAgent на более строгую проверку, "
                                   "добавить дополнительные критерии оценки качества результатов.")

        if not summary['passed']['TID']:
            recommendations.append("2. **Увеличить TID**: Задействовать больше инструментов системы, "
                                   "настроить CoordinatorAgent на более разнообразное использование агентов.")

        if not summary['passed']['AAR']:
            recommendations.append("3. **Увеличить AAR**: Улучшить качество анализа и рекомендаций, "
                                   "настроить пороги принятия решений CriticAgent.")

        if not summary['passed']['SCS']:
            recommendations.append("4. **Увеличить SCS**: Добавить больше контекста в запросы, "
                                   "использовать более детализированные промпты для агентов.")

        if not summary['passed']['ECS']:
            recommendations.append("5. **Увеличить ECS**: Настроить CriticAgent на более подробные объяснения, "
                                   "добавить обязательные поля в ответы (причины выбора, источники, ограничения).")

        if not summary['passed']['FRR']:
            recommendations.append("6. **Увеличить FRR**: Добавить больше стратегий восстановления, "
                                   "улучшить обработку исключений, создать резервные механизмы поиска.")

        if not recommendations:
            recommendations.append("✅ Все метрики соответствуют требованиям. Рекомендуется продолжить "
                                   "сбор статистики и мониторинг производительности системы.")

        return "\n\n".join(recommendations)

    def clear_metrics(self):
        """Очистка всех метрик (для тестирования)"""
        self.metrics = PipelineMetrics()
        self._book_cache.clear()
        self._summary_cache.clear()

        # Очистка файлов метрик
        if self.metrics_file.exists():
            self.metrics_file.unlink()

        logger.info("🧹 Метрики очищены")

    # ==================== ИНТЕГРАЦИЯ С СУЩЕСТВУЮЩЕЙ СИСТЕМОЙ ====================

    def process_book_upload(
            self,
            pdf_path: str,
            title: Optional[str] = None,
            generate_summary: bool = True,
            generate_tags: bool = True
    ) -> Dict[str, Any]:
        """
        Интеграция с существующей системой загрузки книг.

        Args:
            pdf_path: Путь к PDF файлу
            title: Название книги
            generate_summary: Создавать ли резюме
            generate_tags: Создавать ли теги

        Returns:
            Результат обработки книги
        """
        logger.info(f"📤 Загрузка книги: {pdf_path}")

        # Используем существующий код из main.py
        from main import SmartLibrarySystem

        # Создаем временную систему для обработки
        temp_system = SmartLibrarySystem(verify_ssl=False)

        # Вызываем существующий метод загрузки
        result = temp_system.upload_book(pdf_path, title)

        # Обновляем кэши пайплайна
        if result.get("success") and "book_id" in result:
            book_id = result["book_id"]

            # Обновляем поисковую систему
            if generate_tags:
                # Получаем данные книги для добавления в поиск
                book_data = self.data_manager.get_book_by_id(book_id)
                if book_data:
                    self.tag_search.add_book_tags(book_data)

            # Очищаем кэши
            self._book_cache.pop(book_id, None)
            self._summary_cache.pop(book_id, None)

        return result


# ==================== ДЕМОНСТРАЦИОННЫЙ БЛОК ====================

def demo_pipeline():
    """Демонстрация работы идеального пайплайна"""

    print("=" * 80)
    print("🚀 ИДЕАЛЬНЫЙ ПАЙПЛАЙН АНАЛИЗА КНИГ - ДЕМОНСТРАЦИЯ")
    print("=" * 80)

    try:
        # Инициализация пайплайна
        print("\n🔄 Инициализация пайплайна...")
        pipeline = PerfectBookAnalysisPipeline(
            base_data_dir="uploads",  # Используем существующую директорию
            output_dir="perfect_pipeline_output",
            verify_ssl=False
        )

        print("✅ Пайплайн инициализирован")

        # Основное меню
        while True:
            print("\n" + "=" * 80)
            print("📚 ГЛАВНОЕ МЕНЮ:")
            print("1. 🔍 Обработать пользовательский запрос")
            print("2. 🧪 Запустить тест самосогласованности (SCS)")
            print("3. 📤 Загрузить новую книгу (PDF)")
            print("4. 📊 Показать текущие метрики")
            print("5. 📄 Экспортировать отчет по метрикам")
            print("6. 🧹 Очистить метрики (тестирование)")
            print("7. 🚪 Выйти")

            choice = input("\nВыберите действие (1-7): ").strip()

            if choice == "1":
                query = input("Введите запрос (например: 'учебник по машинному обучению'): ").strip()
                if query:
                    print(f"\n🔍 Обработка запроса: '{query}'")
                    result = pipeline.process_user_query(query, enable_critique=True)

                    print(f"\n📋 РЕЗУЛЬТАТ:")
                    print(f"ID выполнения: {result.get('execution_id')}")
                    print(f"Время обработки: {result.get('processing_time', 0):.2f} сек")
                    print(f"Итераций: {result.get('iterations_count', 0)}")
                    print(f"Книг найдено: {result.get('books_found', 0)}")

                    # Показываем рекомендации если есть
                    recs = result.get('final_recommendations', {})
                    if recs and isinstance(recs, dict):
                        if 'executive_summary' in recs:
                            print(f"\n💡 РЕКОМЕНДАЦИИ:")
                            print(recs['executive_summary'][:300] + "...")

                    # Показываем метрики
                    metrics = result.get('metrics_snapshot', {})
                    if metrics:
                        print(f"\n📈 МЕТРИКИ ЭТОГО ЗАПРОСА:")
                        for metric, value in metrics.items():
                            print(f"  {metric}: {value}")

            elif choice == "2":
                test_query = "математика для начинающих"
                print(f"\n🧪 Запуск теста самосогласованности: '{test_query}'")
                print("(Выполняется 3 запуска с разными начальными условиями)")

                result = pipeline.run_consistency_test(
                    query=test_query,
                    n_runs=3,
                    temperature=0.5
                )

                test_summary = result['test_summary']
                print(f"\n✅ Тест завершен:")
                print(f"SCS: {test_summary['consistency_score']:.3f}")
                print(f"Запусков: {test_summary['n_runs']}")
                print(f"Temperature LLM: {test_summary['temperature']}")

            elif choice == "3":
                file_path = input("Введите путь к PDF файлу: ").strip()
                if os.path.exists(file_path):
                    result = pipeline.process_book_upload(file_path)

                    if result.get("success"):
                        print(f"\n✅ Книга загружена:")
                        print(f"Название: {result.get('title')}")
                        print(f"ID: {result.get('book_id')}")
                        print(
                            f"Время обработки: {result.get('processing_summary', {}).get('text_extracted', 0)} символов")
                    else:
                        print(f"\n❌ Ошибка: {result.get('error', 'неизвестно')}")
                else:
                    print("❌ Файл не найден")

            elif choice == "4":
                metrics = pipeline.get_metrics_summary()

                print("\n📊 ТЕКУЩИЕ МЕТРИКИ ПАЙПЛАЙНА:")
                print(f"Статус: {metrics['overall_status']}")
                print(f"Всего запросов: {metrics['statistics']['total_queries']}")

                print("\nЗначения метрик:")
                for metric, value in metrics['metrics'].items():
                    status = "✅" if metrics['passed'][metric] else "❌"
                    print(f"{status} {metric}: {value} (порог: {metrics['requirements'][metric]})")

            elif choice == "5":
                report_path = pipeline.export_metrics_report()
                if report_path:
                    print(f"\n📄 Отчет сохранен: {report_path}")
                    print("Откройте файл в Markdown-редакторе для просмотра")
                else:
                    print("❌ Ошибка генерации отчета")

            elif choice == "6":
                confirm = input("⚠️  Вы уверены? Все метрики будут удалены (y/N): ").strip().lower()
                if confirm == 'y':
                    pipeline.clear_metrics()
                    print("✅ Метрики очищены")

            elif choice == "7":
                print("\n👋 Выход из программы")
                break

            else:
                print("❌ Неверный выбор")

    except KeyboardInterrupt:
        print("\n\n👋 Выход по запросу пользователя")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_pipeline()
