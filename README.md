# 🎈 Blank app template

A simple Streamlit app template for you to modify!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### How to run it on your own machine

Prerequisite: install `uv` if you don't already have it.

```
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. Sync the dependencies

   ```
   $ uv sync
   ```

2. Run the app

   ```
   $ uv run streamlit run streamlit_app.py
   ```

The Sprint 4 simulator currently trains on synthetic data because the project dataset is not included. Its output is illustrative and has no validated performance metrics. Add the real, appropriately prepared dataset before using it to train or evaluate a predictive model.

For Streamlit Community Cloud, keep dependencies in the root-level `requirements.txt` file. For local development with `uv`, use `uv sync` and the `pyproject.toml` configuration.
