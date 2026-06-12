"""Google Drive sync backend — alternative to Hugging Face Hub.

Uses a service account key (stored as GDRIVE_KEY env var — no browser login ever).
All artifacts go into a single Drive folder identified by GDRIVE_FOLDER_ID.

Size reality check (no replay buffer synced):
  best.pt / latest.pt  ~10–80 MB each (small profile ≈ 10 MB)
  ckpt_N.pt            same
  ouroboros.db         <100 MB
  Total                well under 1 GB — 15 GB free tier is more than enough.
"""
import base64
import io
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/drive"]
_MIME_BINARY = "application/octet-stream"
_LOCK = threading.Lock()


def _get_credentials() -> tuple[Optional[str], Optional[str]]:
    """Return (service_account_json_str, folder_id) from env vars."""
    raw = os.environ.get("GDRIVE_KEY", "")
    folder = os.environ.get("GDRIVE_FOLDER_ID", "")
    if not raw or not folder:
        return None, None
    # Accept raw JSON or base64-encoded JSON
    try:
        json.loads(raw)
        return raw, folder
    except json.JSONDecodeError:
        try:
            decoded = base64.b64decode(raw).decode()
            json.loads(decoded)
            return decoded, folder
        except Exception:
            log.warning("GDRIVE_KEY is not valid JSON or base64-encoded JSON")
            return None, None


def is_enabled() -> bool:
    key, folder = _get_credentials()
    return bool(key and folder)


def _build_service(key_json: str):
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    info = json.loads(key_json)
    creds = service_account.Credentials.from_service_account_info(info, scopes=_SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _find_file_id(service, name: str, folder_id: str) -> Optional[str]:
    q = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    result = service.files().list(q=q, fields="files(id,name)").execute()
    files = result.get("files", [])
    return files[0]["id"] if files else None


def push(local_path: Path, remote_name: Optional[str] = None) -> bool:
    """Upload a local file to Google Drive folder. Overwrites if exists."""
    key, folder_id = _get_credentials()
    if not key or not folder_id:
        return False
    if not local_path.exists():
        return False
    name = remote_name or local_path.name
    with _LOCK:
        try:
            from googleapiclient.http import MediaFileUpload
            service = _build_service(key)
            existing_id = _find_file_id(service, name, folder_id)
            media = MediaFileUpload(str(local_path), mimetype=_MIME_BINARY, resumable=True)
            if existing_id:
                service.files().update(fileId=existing_id, media_body=media).execute()
            else:
                meta = {"name": name, "parents": [folder_id]}
                service.files().create(body=meta, media_body=media, fields="id").execute()
            log.info("GDrive push: %s → %s", local_path, name)
            return True
        except Exception as e:
            log.warning("GDrive push failed (%s): %s", name, e)
            return False


def pull(remote_name: str, local_path: Path) -> bool:
    """Download a file from Google Drive to local_path."""
    key, folder_id = _get_credentials()
    if not key or not folder_id:
        return False
    with _LOCK:
        try:
            from googleapiclient.http import MediaIoBaseDownload
            service = _build_service(key)
            file_id = _find_file_id(service, remote_name, folder_id)
            if not file_id:
                log.debug("GDrive pull: %s not found in folder", remote_name)
                return False
            local_path.parent.mkdir(parents=True, exist_ok=True)
            request = service.files().get_media(fileId=file_id)
            buf = io.BytesIO()
            downloader = MediaIoBaseDownload(buf, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            local_path.write_bytes(buf.getvalue())
            log.info("GDrive pull: %s → %s", remote_name, local_path)
            return True
        except Exception as e:
            log.warning("GDrive pull failed (%s): %s", remote_name, e)
            return False
