.PHONY: demo test dashboard

# Zero-friction reproduction: no API key, prints verified metrics in
# single-digit seconds. See run_demo.sh for the Windows/no-make fallback.
demo:
	pip install -q -r requirements.txt
	python scripts/print_verified_metrics.py

test:
	pip install -q -r requirements.txt
	pytest tests/ -v

dashboard:
	pip install -q -r requirements-dashboard.txt
	streamlit run dashboard/app.py
