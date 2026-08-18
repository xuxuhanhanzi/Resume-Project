.PHONY: install format-check lint typecheck unit integration test smoke check

install:
	python -m pip install -e . -r requirements-dev.lock

format-check:
	python scripts/dev.py format-check

lint:
	python scripts/dev.py lint

typecheck:
	python scripts/dev.py typecheck

unit:
	python scripts/dev.py unit

integration:
	python scripts/dev.py integration

test:
	python scripts/dev.py test

smoke:
	python scripts/dev.py smoke

check:
	python scripts/dev.py check
