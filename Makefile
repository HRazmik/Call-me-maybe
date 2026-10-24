SRC := src
MYPY_FLAGS := --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
UV := uv
install:
	${UV} sync

run:
	${UV} run python -m src

debug:
	${UV} run python -m pdb main.py

clean:
	rm -rf __pycache__ \
		   .pytest_cache \
		   .mypy_cache \
	rm -rf src/__pycache__ \
		   src/.mypy_cache \
		   src/*.egg-info

lint: flake
	mypy ${SRC} ${MYPY_FLAGS}

lint-strict: flake
	mypy ${SRC} --strict

flake:
	flake8 ${SRC}

.PHONY: install run debug clean lint lint-strict flake
