PYTHON  ?= python3
VENV    ?= .venv
PY      := $(VENV)/bin/python
ARCHIVE ?= MLAW_dataset.tar.zst

.DEFAULT_GOAL := help
.PHONY: help setup test test-slow clean

help:
	@echo "MLAW RAG"
	@echo "  setup      venv + зависимости"
	@echo "  test       быстрые тесты"
	@echo "  test-slow  + проверки на настоящем архиве"
	@echo "  clean      убрать производные артефакты"
	@echo ""
	@echo "Шаги пайплайна (inventory/slice/chunk/index/eval) добавляются по мере"
	@echo "реализации — пустых целей-заглушек здесь нет намеренно."

$(PY):
	$(PYTHON) -m venv $(VENV)

setup: $(PY)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e '.[zstd,dev]'
	@echo "Готово. Тяжёлый стек — отдельно: $(PY) -m pip install -e '.[pipeline]'"

test: $(PY)
	$(PY) -m pytest -q

test-slow: $(PY)
	MLAW_SLOW_TESTS=1 $(PY) -m pytest -q

clean:
	rm -rf data/* reports/* .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
