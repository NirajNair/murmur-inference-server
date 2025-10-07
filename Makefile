PYTHON=python3
VENV_DIR=venv

venv:
	$(PYTHON) -m venv $(VENV_DIR)

install: 
	source $(VENV_DIR)/bin/activate && pip3 install -r requirements.txt

run:
	source $(VENV_DIR)/bin/activate && clear && $(PYTHON) main.py

build-image:
	docker rmi murmur-inference-server:slim
	docker build -t murmur-inference-server:latest .
	slim build --target murmur-inference-server:latest \
		--tag murmur-inference-server:slim \
		--continue-after 10 \
		--include-path /app \
		--include-path /usr/local \
		--include-path /usr/lib
	docker rmi murmur-inference-server:latest
	
build-image-prod:
	docker rmi murmur-inference-server:slim
	docker buildx build --platform linux/amd64 -t murmur-inference-server-prod:latest .
	slim build --target murmur-inference-server-prod:latest \
		--tag murmur-inference-server-prod:slim \
		--continue-after 10 \
		--include-path /app \
		--include-path /usr/local \
		--include-path /usr/lib
	docker rmi murmur-inference-server-prod:latest
	
	
start-container:
	docker compose up -d
	
stop-container:
	docker compose down

restart-container: stop-container start-container
	
clean:
	rm -rf $(VENV_DIR) __pycache__
