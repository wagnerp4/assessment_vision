# Contributing

## Setup

1. Clone the repository
2. Install dependencies with `uv sync --all-extras` (or `pip install -e ".[dev]"`)
3. Run tests: `uv run pytest`
4. Check code quality: `uv run black src/ && uv run ruff check src/`

## Development Workflow

1. Create a branch: `git checkout -b feature/my-feature`
2. Make changes
3. Run tests and linting: `./scripts/test.sh`
4. Commit: `git commit -m "Description"`
5. Push and open a PR

## Code Style

- Use `black` for formatting
- Use `ruff` for linting
- Add type hints where possible
- Write docstrings for public functions

## Testing

- Add tests in `tests/` for new features
- Ensure tests pass: `uv run pytest --cov`
- Aim for >80% coverage

## Dev Scripts

Located in `scripts/`:

- `dev_grab.py` - Capture frames from camera
- `dev_live_app.py` - Live test through app
- `dev_record.py` - Record video from camera
- `dev_replay.py` - Replay recorded video through detector
- `dev_metric.py` - Measure detection accuracy

Example:
```bash
python scripts/dev_grab.py 1          # Get 1 frame
python scripts/dev_live_app.py 150    # Test 150 frames live
```
