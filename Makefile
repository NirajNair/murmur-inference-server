PYTHON=python3
VENV_DIR=venv

venv:
	$(PYTHON) -m venv $(VENV_DIR)

install: 
	source $(VENV_DIR)/bin/activate && pip3 install -r requirements.txt

run:
	source $(VENV_DIR)/bin/activate && clear && $(PYTHON) main.py

run-docker:
	docker compose up -d
	
stop-docker:
	docker compose down

restart-docker: stop-docker run-docker
	
clean:
	rm -rf $(VENV_DIR) __pycache__
