"""Verify document upload encoding without provider dependencies."""

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from langgraph_agent_toolkit.ui.utils import message
from langgraph_agent_toolkit.ui.utils.constants import MULTIMODAL_FILE_TYPES


DOCUMENT_TYPES = [
    ("pdf", "application/pdf"),
    ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ("pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ("csv", "text/csv"),
    ("json", "application/json"),
    ("md", "text/markdown"),
]


def upload(name, mime_type="application/octet-stream", data=b"Uploaded file bytes."):
    return SimpleNamespace(name=name, type=mime_type, getvalue=lambda: data)


@pytest.mark.parametrize("extension,mime_type", DOCUMENT_TYPES)
def test_document_upload_keeps_filename_and_uses_document_mime_type(extension, mime_type):
    """Use the document MIME type when the browser sends a generic type."""
    filename = f"Report.{extension.upper()}"
    block = message.file_to_content_block(upload(filename))

    assert extension in MULTIMODAL_FILE_TYPES
    assert block == {
        "type": "file",
        "base64": base64.b64encode(b"Uploaded file bytes.").decode("ascii"),
        "mime_type": mime_type,
        "extras": {"filename": filename},
    }


def test_document_upload_accepts_missing_browser_mime_and_keeps_explicit_mime():
    """Infer a missing MIME type and preserve an explicit browser MIME type."""
    assert message.file_to_content_block(upload("report.csv", None))["mime_type"] == "text/csv"
    assert message.file_to_content_block(upload("report.csv", "text/plain"))["mime_type"] == "text/plain"


@pytest.mark.parametrize(
    "mime_type,block_type", [("image/png", "image"), ("audio/wav", "audio"), ("video/mp4", "video")]
)
def test_media_upload_keeps_existing_content_block_contract(mime_type, block_type):
    """Keep the image, audio, and video content block fields unchanged."""
    block = message.file_to_content_block(upload("media", mime_type))
    assert block["type"] == block_type
    assert block["mime_type"] == mime_type
    assert "extras" not in block


def test_document_caption_uses_filename_and_accepts_old_history(monkeypatch):
    """Show file names and keep old or incomplete history readable."""
    streamlit = MagicMock()
    monkeypatch.setattr(message, "st", streamlit)
    message.render_human_message(
        [
            {"type": "file", "extras": {"filename": "report.docx"}},
            {"type": "file", "mime_type": "application/pdf"},
            {"type": "file", "mime_type": "application/pdf", "extras": "invalid metadata"},
        ]
    )
    assert [call.args[0] for call in streamlit.caption.call_args_list] == [
        "📎 report.docx attachment",
        "📎 application/pdf attachment",
        "📎 application/pdf attachment",
    ]
