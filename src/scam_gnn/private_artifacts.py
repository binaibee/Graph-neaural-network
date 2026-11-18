"""Load private deployment artifacts from a private GitHub repository.

The dataset and prepared graph remain local during development. In a cloud
deployment, the app downloads only missing files from a separate private
artifact repository using a read-only token stored in Streamlit secrets.
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
MAX_ARTIFACT_BYTES = 100 * 1024 * 1024

# These digests identify the local files that were reviewed for this deploy.
# Update them when intentionally replacing either private artifact.
ARTIFACTS = {
    "data/public/spamassassin_email.csv": {
        "sha256": "3bfe8f8abff89f69a98456be50413c2fc b20a476141116dd84ee1507b980c00e".replace(" ", ""),
        "repo_path": "data/spamassassin_email.csv",
    },
    "results/prepared.pkl": {
        "sha256": "9eaa6d028ddef1d7130b89251352669478856bd6fd54f754ce47c4c16ab0c4b5",
        "repo_path": "results/prepared.pkl",
    },
}
REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class PrivateArtifactError(RuntimeError):
    """The private dataset or prepared graph could not be loaded safely."""


def _matches_digest(path: Path, expected: str) -> bool:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def _private_artifact_settings() -> tuple[str, str, str]:
    try:
        import streamlit as st

        settings = st.secrets["private_artifacts"]
        repository = str(settings["repository"])
        reference = str(settings.get("ref", "main"))
        token = str(settings["token"])
    except Exception as error:  # noqa: BLE001 - secrets backends expose different errors
        raise PrivateArtifactError(
            "Private deployment files are missing. Add the private_artifacts settings "
            "to Streamlit secrets, or keep the two files beside the app for local use."
        ) from error

    if not REPOSITORY_PATTERN.fullmatch(repository) or not token:
        raise PrivateArtifactError("The private_artifacts repository or token setting is invalid.")
    return repository, reference, token


def _download(repository: str, reference: str, token: str, repo_path: str, destination: Path) -> None:
    url = (
        f"https://api.github.com/repos/{repository}/contents/"
        f"{quote(repo_path, safe='/')}?ref={quote(reference, safe='')}"
    )
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github.raw+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "scam-gnn-private-artifact-loader",
        },
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with urlopen(request, timeout=90) as response:
            length = response.headers.get("Content-Length")
            if length and int(length) > MAX_ARTIFACT_BYTES:
                raise PrivateArtifactError(f"Private artifact is larger than the allowed size: {repo_path}")
            digest = hashlib.sha256()
            size = 0
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".artifact-", delete=False) as tmp:
                temporary_path = Path(tmp.name)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_ARTIFACT_BYTES:
                        raise PrivateArtifactError(f"Private artifact is larger than the allowed size: {repo_path}")
                    digest.update(chunk)
                    tmp.write(chunk)
        expected = next(item["sha256"] for item in ARTIFACTS.values() if item["repo_path"] == repo_path)
        if digest.hexdigest() != expected:
            raise PrivateArtifactError(f"The downloaded artifact failed its integrity check: {repo_path}")
        os.replace(temporary_path, destination)
        temporary_path = None
    except HTTPError as error:
        raise PrivateArtifactError(
            f"Could not access the private artifact repository (HTTP {error.code}). "
            "Check the repository name, file path, and read-only token permissions."
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise PrivateArtifactError(f"Could not download a required private artifact: {repo_path}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def ensure_private_artifacts() -> None:
    """Verify local copies and download missing files from private storage."""
    missing: list[tuple[Path, dict[str, str]]] = []
    for relative_path, metadata in ARTIFACTS.items():
        destination = ROOT / relative_path
        if destination.is_file():
            if not _matches_digest(destination, metadata["sha256"]):
                raise PrivateArtifactError(
                    f"A local deployment artifact does not match its reviewed checksum: {relative_path}"
                )
        else:
            missing.append((destination, metadata))

    if not missing:
        return

    repository, reference, token = _private_artifact_settings()
    for destination, metadata in missing:
        _download(repository, reference, token, metadata["repo_path"], destination)
