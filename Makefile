SRC := src
MYPY_FLAGS := --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
UV := uv
install:
	${UV} sync

run:
	${UV} run python -m src

debug:
	${UV} run main.py

clean:
	rm -rf __pycache__ \
		   .pytest_cache \
		   .mypy_cache \

lint: flake
	mypy ${SRC} ${MYPY_FLAGS}

lint-strtict: flake
	mypy ${SRC} --strict

flake:
	flake8 ${SRC}

.PHONY: install run debug clean lint lint-strtict flake
