"""The Hugging Face Space: the project's cell explorer on the released data (dataset milosand1/sec-filings-tone).
explore.py is a link to src/sec_filings_tone/explore.py, which upload_folder uploads as a file.

Usage (locally, from the repo root; RELEASE_DIR reads a local copy instead of downloading):
    RELEASE_DIR=data/release python space/app.py
"""

import os

import explore

explore.D = explore.release(os.environ.get("RELEASE_DIR"))
explore.build().launch()
