# Streamlit Cloud deployment

This project runs directly from the public GitHub repository. It uses the existing datasets and saved model artifacts; it does not require a second private repository, access token, Streamlit secret, or artifact download step.

## Required repository files

Keep these files committed alongside the app:

- `config.json`
- `data/public/uci_sms_spam_collection.csv`
- `data/public/phiusiil_url_collection.csv`
- `data/public/spamassassin_email.csv`
- `results/prepared.pkl`
- `results/model.pt`
- `results/predictions.csv`
- `results/results.json`

The saved graph, weights, prediction index, and email dataset are present locally. `results/prepared.pkl` is normally ignored, so it must be explicitly added to Git for the cloud build.

## Deploy

1. Push the deployment commit to the existing public GitHub repository using a normal push.
2. In Streamlit Community Cloud, choose **Create app** and select `binaibee/Graph-neaural-network`, branch `main`, and `app/app.py`.
3. Select Python 3.12 and deploy. Community Cloud installs the packages listed in the root `requirements.txt`.
4. No app secrets or private artifact settings are needed.

The application loads the saved graph, model, and prediction index through the existing project configuration. The GNN architecture, evaluation results, and SMS, email, and web-link channels are unchanged.

See Streamlit's [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).
