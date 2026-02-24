"""Tests for CorpusDocument schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared_types import CorpusDocument


class TestCorpusDocument:
    def test_creates_with_required_fields(self):
        """CorpusDocument can be created with just id and text."""
        doc = CorpusDocument(id="doc1", text="Sample text content")

        assert doc.id == "doc1"
        assert doc.text == "Sample text content"
        assert doc.metadata == {}

    def test_creates_with_metadata(self):
        """CorpusDocument includes metadata when provided."""
        doc = CorpusDocument(
            id="doc1",
            text="Sample text",
            metadata={"title": "Test Document", "author": "Test Author"},
        )

        assert doc.metadata["title"] == "Test Document"
        assert doc.metadata["author"] == "Test Author"

    def test_requires_id(self):
        """CorpusDocument requires an id field."""
        with pytest.raises(ValidationError):
            CorpusDocument(text="Sample text")

    def test_requires_text(self):
        """CorpusDocument requires a text field."""
        with pytest.raises(ValidationError):
            CorpusDocument(id="doc1")

    def test_serializes_to_dict(self):
        """CorpusDocument can be serialized to dict."""
        doc = CorpusDocument(
            id="doc1",
            text="Sample text",
            metadata={"key": "value"},
        )

        data = doc.model_dump()

        assert data == {
            "id": "doc1",
            "text": "Sample text",
            "metadata": {"key": "value"},
        }

    def test_deserializes_from_dict(self):
        """CorpusDocument can be created from dict."""
        data = {
            "id": "doc1",
            "text": "Sample text",
            "metadata": {"key": "value"},
        }

        doc = CorpusDocument(**data)

        assert doc.id == "doc1"
        assert doc.text == "Sample text"
        assert doc.metadata == {"key": "value"}

    def test_metadata_defaults_to_empty_dict(self):
        """Metadata defaults to empty dict if not provided."""
        doc = CorpusDocument(id="doc1", text="text")

        assert isinstance(doc.metadata, dict)
        assert len(doc.metadata) == 0
