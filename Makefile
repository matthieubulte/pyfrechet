init:
	pip install -r requirements.txt

test:
	py.test tests

test_fast:
	py.test tests -x

build:
	python -m build

install:
	python3 -m pip install .[all]

lint:
	ruff --format=github --select=E9,F63,F7,F82 --target-version=py37 .

all: init lint test build install

.PHONY: init lint test build install
