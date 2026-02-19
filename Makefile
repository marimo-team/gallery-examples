sessions:
	uv run scripts/create-sessions.py notebooks/*/*.py
	rm *.parquet
	rm -rf inputs
