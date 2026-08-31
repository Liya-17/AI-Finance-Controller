# Deploying the dashboard to Streamlit Community Cloud

Takes about 2 minutes. Requires your GitHub login (Streamlit Cloud auths
via GitHub OAuth) — this is the one step that has to happen in your
browser, nothing here can be scripted around that.

## Steps

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in
   with GitHub (click "Continue with GitHub", authorize Streamlit if
   asked — it only needs read access to your public repos).

2. Click **"Create app"** (top right, or the button on the empty-state
   screen).

3. Choose **"Deploy a public app from GitHub"**.

4. Fill in the three fields:
   - **Repository:** `Liya-17/AI-Finance-Controller`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`

5. (Optional) Click "Advanced settings" if you want to set a custom app
   URL slug — otherwise Streamlit assigns one automatically
   (something like `ai-finance-controller-<random>.streamlit.app`).

6. Click **"Deploy"**. First deploy takes 2-3 minutes (installing
   `requirements.txt`). You'll see a live build log.

7. Once it's up, the dashboard loads directly from the repo's committed
   `reports/audit_log.csv` and `reports/exception_queue.csv` — no
   `.env`, no API key, nothing else to configure. It's read-only, so
   there's no live LLM cost from anyone visiting the page.

8. Copy the live URL (shown at the top of the app once deployed, also
   visible on your Streamlit Cloud dashboard) and paste it into the top
   of `README.md` where noted.

## If it fails to build

The most likely failure mode is a `requirements.txt` install timeout or
version conflict on Streamlit Cloud's Python version (they run 3.9-3.12
depending on settings). If the build log shows a pip error:

- Check the "Python version" under Advanced settings when deploying —
  set it to 3.11 to match what this project was developed and tested
  against.
- If a specific package fails to resolve, that's worth reporting back —
  it would mean `requirements.txt`'s pins need loosening for a package
  that doesn't ship a wheel for Streamlit Cloud's platform, which hasn't
  been an issue for any package in this list when checked.

## Keeping it updated

Streamlit Cloud auto-redeploys on every push to `main` by default — no
action needed after the first deploy. If you push a change and the live
app doesn't reflect it within a minute or two, check the app's "Manage
app" menu (bottom right when viewing it) for the redeploy status/logs.
