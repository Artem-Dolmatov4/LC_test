PYTHON  ?= python3
VENV    ?= .venv
PY      := $(VENV)/bin/python
ARCHIVE ?= MLAW_dataset.tar.zst
ACTS    ?= 800
SEED    ?= 20260815
MODEL   ?= voyage
QDRANT  ?= http://localhost:6333

export PYTHONPATH := src

# .env — ключи API. Не обязателен для setup/test/inventory/slice/chunk/lexical,
# нужен для index/basket/eval/ablation/answers. Отсутствие файла не ошибка —
# include молча пропускает то, чего нет; шаги, которым нужен ключ, упадут
# сами с понятным сообщением из mlaw.embed / mlaw.llm.
-include .env
export VOYAGE_API_KEY DEEPSEEK_API_KEY DASHSCOPE_API_KEY

.DEFAULT_GOAL := help
.PHONY: help setup test test-slow qdrant qdrant-stop gate inventory slice chunk \
        lexical index basket eval ablation answers all clean

help:
	@echo "MLAW RAG — пайплайн"
	@echo "  setup      venv + зависимости"
	@echo "  test       быстрые тесты"
	@echo "  test-slow  + проверки на настоящем архиве"
	@echo "  qdrant     поднять векторную базу в docker"
	@echo ""
	@echo "  gate       шаг 0.5 — замер пропускной способности эмбеддеров"
	@echo "  inventory  шаг 1   — сплошной проход по банку"
	@echo "  slice      шаг 2   — срез по актам с полными цепочками"
	@echo "  chunk      шаг 3   — нарезка"
	@echo "  index      шаг 4   — эмбеддинги и плотный индекс"
	@echo "  lexical    шаг 4   — индекс BM25"
	@echo "  basket     шаг 5   — корзина запросов"
	@echo "  eval       шаг 5   — метрики и контроли"
	@echo "  ablation   шаг 6   — вклад стадий конвейера"
	@echo "  answers    шаг 6   — ответы с проверяемыми цитатами"
	@echo "  all        весь пайплайн с нуля"
	@echo ""
	@echo "Нужны ключи в .env (см. .env.example): VOYAGE_API_KEY, DEEPSEEK_API_KEY"

$(PY):
	$(PYTHON) -m venv $(VENV)

setup: $(PY)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e '.[zstd,dev,pipeline]'

test: $(PY)
	$(PY) -m pytest -q

test-slow: $(PY)
	MLAW_SLOW_TESTS=1 $(PY) -m pytest -q

qdrant:
	@docker start mlaw-qdrant 2>/dev/null || \
	 docker run -d --name mlaw-qdrant -p 6333:6333 \
	   -v "$(PWD)/qdrant_storage:/qdrant/storage" qdrant/qdrant:latest
	@until curl -s --max-time 2 $(QDRANT)/collections >/dev/null; do sleep 2; done
	@echo "Qdrant готов"

qdrant-stop:
	-docker stop mlaw-qdrant

# --- шаги пайплайна ------------------------------------------------------- #

gate: $(PY)
	$(PY) -m mlaw.gate --archive $(ARCHIVE) --chunks 10
	$(PY) -m mlaw.gate --voyage --chunks 6

reports/inventory.json inventory: $(PY)
	$(PY) -m mlaw.inventory --archive $(ARCHIVE)

data/slice.jsonl slice: $(PY)
	$(PY) -m mlaw.slice_build --archive $(ARCHIVE) --acts $(ACTS) --seed $(SEED)

data/chunks.jsonl chunk: data/slice.jsonl
	$(PY) -m mlaw.chunk

data/bm25 lexical: data/chunks.jsonl
	$(PY) -m mlaw.lexical

index: data/chunks.jsonl qdrant
	$(PY) -m mlaw.index --model $(MODEL) --qdrant $(QDRANT)

queries/all.jsonl basket: data/chunks.jsonl
	$(PY) -m mlaw.basket --seed $(SEED)

eval: queries/all.jsonl data/bm25
	$(PY) -m mlaw.evaluate --split dev --k 20
	$(PY) -m mlaw.evaluate --split test --k 20

ablation: queries/all.jsonl data/bm25
	$(PY) -m mlaw.search --split test --k 10

answers: queries/all.jsonl data/bm25
	$(PY) -m mlaw.answer --split test --limit 20

all: inventory slice chunk lexical index basket eval ablation answers
	@echo "Пайплайн пройден. Отчёты в reports/"

# reports/*.json — трекаемые артефакты сдачи, их сносить нельзя;
# make all перезапишет их сам.
clean:
	rm -rf data/* reports/*.log .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
