.PHONY: install app

app:
	poetry run streamlit run app.py

install:
	poetry install


