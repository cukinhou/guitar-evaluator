init:
	pip install -r requirements.txt

install:
	python setup.py build
	python setup.py install
test:
	nosetests tests/

.PHONY: init tests
