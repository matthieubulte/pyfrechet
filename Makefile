init:
	pip install -r requirements.txt

test:
	py.test tests

test_fast:
	py.test tests -x

lint:
	ruff --format=github --select=E9,F63,F7,F82 --target-version=py37 .

.PHONY: init lint test
