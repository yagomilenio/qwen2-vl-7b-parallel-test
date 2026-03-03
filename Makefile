# ─────────────────────────────────────────────────────────────────
#  Makefile — uso manual y desarrollo
#  El servidor de paralelización llama a run_vision.py directamente.
# ─────────────────────────────────────────────────────────────────

PYTHON := python3
PACK   ?= general
START  ?= 0
END    ?= 9
OUTPUT ?= outputs/results_$(START)_$(END).json



ifeq ($(USER),root)
    USER_HOME := /root
else
    USER_HOME := /home/$(USER)
endif

LMS_BIN_DIR := $(USER_HOME)/.lmstudio/bin

export PATH := $(LMS_BIN_DIR):$(PATH)

.PHONY: help setup install-lms install-model run test list clean

help:
	@echo ""
	@echo "  LM Studio Vision Runner"
	@echo "  ──────────────────────────────────────────────────────"
	@echo "  make setup                    Instala lms + descarga modelo"
	@echo "  make list                     Lista imágenes con índices"
	@echo "  make run START=0 END=9        Procesa rango, pack 'general'"
	@echo "  make run START=0 END=9 PACK=ocr"
	@echo "  make test                     Prueba rápida: imagen 0"
	@echo "  make clean                    Borra outputs"
	@echo ""
	@echo "  Comando que lanza el servidor por cada worker:"
	@echo "    python run_vision.py --start 0 --end 9 --pack general --output outputs/results_0_9.json"
	@echo ""

install-lms:
	curl -fsSL https://lmstudio.ai/install.sh | bash

install-model:
	$(LMS_BIN_DIR)/lms get Qwen2-VL-7B-Instruct-GGUF@Q4_K_M --gguf -y

install-library:
	pip install lmstudio --break-system-packages

setup: install-lms install-model install-library
	@echo "✓ Listo. Añade imágenes a inputs/images/ y ejecuta: make list"

list:
	@echo ""; \
	i=0; \
	for f in $$(ls inputs/images/ 2>/dev/null | sort); do \
	  printf "  [%3d]  %s\n" $$i "$$f"; i=$$((i+1)); \
	done; \
	echo ""

run:
	PATH=$(LMS_BIN_PATH):$$PATH $(PYTHON) run_vision.py \
		--start     $(START) \
		--end       $(END) \
		--pack      $(PACK) \
		--input-dir inputs/images \
		--output    $(OUTPUT)

test:
	$(PYTHON) run_vision.py \
		--start     0 \
		--end       0 \
		--pack      general \
		--input-dir inputs/images \
		--output    outputs/results_test.json

clean:
	rm -f outputs/*.json outputs/*.log
	@echo "✓ Limpiado"
