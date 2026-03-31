.PHONY: venv install backend start clean

VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
BACKEND_DIR := backend

venv:
	python3 -m venv $(VENV_DIR)
	$(PYTHON) -m pip install --upgrade pip

install: venv
	$(PYTHON) -m pip install -r $(BACKEND_DIR)/requirements.txt

backend:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --app-dir $(BACKEND_DIR)

start:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir $(BACKEND_DIR)

clean:
	rm -rf $(VENV_DIR)
	rm -rf frontend/dist
