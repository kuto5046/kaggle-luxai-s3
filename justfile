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

# cdコマンドの効果は次のコマンドには引き継がれないので()で囲む
make_subfile exp_name:
    (cd /home/user/work/exp/{{exp_name}}/ && \
    cp -r /home/user/work/.venv/lib/python3.10/site-packages/lightning ./ && \
    tar -czf submission.tar.gz --exclude="*.tar.gz" *)
    uv run kaggle competitions submit -c lux-ai-season-3 -f /home/user/work/exp/{{exp_name}}/submission.tar.gz -m "{{exp_name}}"
