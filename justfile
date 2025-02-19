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
vis exp_name:
	uv run streamlit run /home/user/work/exp/{{exp_name}}/visualizer.py --server.address 0.0.0.0

# cdコマンドの効果は次のコマンドには引き継がれないので()で囲む
sub exp_name:
    (cd /home/user/work/exp/{{exp_name}}/ && \
    cp -r /home/user/work/.venv/lib/python3.10/site-packages/lightning ./ && \
    tar -czf submission.tar.gz --exclude="*.tar.gz" --exclude="*.json" --exclude="*.pkl"  *)
    uv run kaggle competitions submit -c lux-ai-season-3 -f /home/user/work/exp/{{exp_name}}/submission.tar.gz -m "{{exp_name}}"

game exp_name:
    uv run luxai-s3 /home/user/work/agents/exp017/main.py /home/user/work/exp/{{exp_name}}/main.py --output replay.json

game2 exp_name:
    uv run luxai-s3 /home/user/work/agents/exp017/main.py /home/user/work/exp/{{exp_name}}/main.py --tournament --tournament-cfg-concurrent 2
