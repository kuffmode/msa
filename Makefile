test:
	python -m pytest tests

release:
	rm dist/*
	python setup.py sdist bdist_wheel
	twine upload dist/*

docs:
	mkdocs gh-deploy