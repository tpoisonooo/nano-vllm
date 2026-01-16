.PHONY: format check test

format:
	ruff check --fix --unsafe-fixes
	ruff format

check:
	ruff check
	ruff format --check

test:
	pytest --doctest-modules -vv
