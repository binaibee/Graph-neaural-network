# Cloud deployment

This project is a Streamlit app. Streamlit Community Cloud is the simplest initial host because the demo already uses Streamlit and Community Cloud deploys directly from a GitHub repository.

## App settings

- Repository: `binaibee/Graph-neaural-network`
- Branch: `main`
- App file: `app/app.py`
- Python version: `3.12` (the project requires Python 3.11 or newer)
- Dependencies: `requirements.txt`

The app needs the complete project source and these runtime artifacts in the deployed checkout:

- `config.json`
- `results/model.pt`
- `results/prepared.pkl`
- `results/predictions.csv`
- `results/results.json`
- `data/public/` files referenced by `config.json`

`results/prepared.pkl` is ignored by Git in the current project configuration, so it must be explicitly included in the deployment commit (or moved to an artifact store with a download step). It is about 26 MB. Verify the repository's visibility and review the exact files before pushing: the datasets include public research email/message records, and a public repository makes every pushed file downloadable.

## Deploy

1. Commit the intended app source, dependencies, configuration, and required runtime artifacts to the GitHub branch selected above.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with the GitHub account that has admin access to the repository.
3. Choose **Create app**, then select `binaibee/Graph-neaural-network`, branch `main`, and `app/app.py`.
4. In advanced settings, select Python 3.12 and deploy.
5. Confirm the app loads, then try one SMS, one email, and one web link in the demo.

Community Cloud installs dependencies from `requirements.txt`; a deployment failure may require pinning compatible package versions. Keep secrets out of the repository. This demo does not need secrets.
