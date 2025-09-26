PYTHON=python3
VENV_DIR=venv

venv:
	$(PYTHON) -m venv $(VENV_DIR)

install: 
	source $(VENV_DIR)/bin/activate && pip3 install -r requirements.txt

run:
	source $(VENV_DIR)/bin/activate && clear && $(PYTHON) -m server.server

clean:
	rm -rf $(VENV_DIR) __pycache__
