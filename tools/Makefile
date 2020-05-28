.PHONY: all clean

all: venv crank test
test: venv crank

venv:
	test -d venv || python3 -m venv venv
	. venv/bin/activate; pip install -U -q pip
	. venv/bin/activate; pip install -q numpy
	. venv/bin/activate; pip install -q -r requirements.txt

crank:
	. venv/bin/activate; cd .. && pip install -e .

test: venv
	. venv/bin/activate; pip install -q -r test_requirements.txt

clean:
	rm -fr venv
