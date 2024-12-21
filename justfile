# format by ruff
fmt:
    uv run ruff format .

# lint by ruff
lint:
    uv run ruff check --fix .

# type-check by mypy
type-check:
    uv run mypy .

# run pytest in tests directory
test:
    uv run pytest tests

# upload to kaggle dataset
upload:
	uv run python src/tools/upload_code.py
	uv run python src/tools/upload_model.py

# run streamlit app
streamlit:
	uv run streamlit run visualizer.py --server.address 0.0.0.0
