"""File type detection and content hashing.

Detection combines three signals, in this order of trust:

1. **magic bytes** -- what the file actually is;
2. **extension** -- what the caller says it is;
3. ``mimetypes`` -- the platform's registry, used only as a fallback.

Magic-byte sniffing is done in-process with a small table rather than through
``python-magic``, which needs a libmagic binary that is awkward to install on
Windows -- the exact platform this server has to work on first.
"""

from __future__ import annotations

import hashlib
import mimetypes
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from .errors import UnsupportedMediaTypeError
from .models import MediaCategory, MediaInfo

_HASH_CHUNK = 1024 * 1024

# extension -> (mime type, category)
_EXTENSION_MAP: dict[str, tuple[str, MediaCategory]] = {
    ".png": ("image/png", MediaCategory.IMAGE),
    ".jpg": ("image/jpeg", MediaCategory.IMAGE),
    ".jpeg": ("image/jpeg", MediaCategory.IMAGE),
    ".webp": ("image/webp", MediaCategory.IMAGE),
    ".bmp": ("image/bmp", MediaCategory.IMAGE),
    ".tif": ("image/tiff", MediaCategory.IMAGE),
    ".tiff": ("image/tiff", MediaCategory.IMAGE),
    ".gif": ("image/gif", MediaCategory.IMAGE),
    ".pdf": ("application/pdf", MediaCategory.PDF),
    ".docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        MediaCategory.OFFICE,
    ),
    ".pptx": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        MediaCategory.OFFICE,
    ),
    ".xlsx": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        MediaCategory.OFFICE,
    ),
    ".xls": ("application/vnd.ms-excel", MediaCategory.OFFICE),
    ".csv": ("text/csv", MediaCategory.DATA),
    ".tsv": ("text/tab-separated-values", MediaCategory.DATA),
    ".json": ("application/json", MediaCategory.TEXT),
    ".txt": ("text/plain", MediaCategory.TEXT),
    ".md": ("text/markdown", MediaCategory.TEXT),
    ".markdown": ("text/markdown", MediaCategory.TEXT),
    ".rst": ("text/x-rst", MediaCategory.TEXT),
    ".log": ("text/plain", MediaCategory.TEXT),
    ".html": ("text/html", MediaCategory.HTML),
    ".htm": ("text/html", MediaCategory.HTML),
    ".xhtml": ("application/xhtml+xml", MediaCategory.HTML),
    ".msg": ("application/vnd.ms-outlook", MediaCategory.EMAIL),
    ".epub": ("application/epub+zip", MediaCategory.EBOOK),
    ".ipynb": ("application/x-ipynb+json", MediaCategory.NOTEBOOK),
    ".mp3": ("audio/mpeg", MediaCategory.AUDIO),
    ".wav": ("audio/wav", MediaCategory.AUDIO),
    ".m4a": ("audio/mp4", MediaCategory.AUDIO),
    ".flac": ("audio/flac", MediaCategory.AUDIO),
    ".zip": ("application/zip", MediaCategory.ARCHIVE),
}

# Formats that are recognised but deliberately refused, with the reason.
_REFUSED: dict[MediaCategory, tuple[str, str]] = {
    MediaCategory.ARCHIVE: (
        "Archives are not processed.",
        "Extract the archive yourself and analyse the individual file you need. "
        "Recursive archive processing is disabled because it is an amplification "
        "vector (a small zip can expand into unbounded work).",
    ),
    MediaCategory.AUDIO: (
        "Audio transcription is not implemented in this release.",
        "Transcribe the audio with a dedicated tool and pass the transcript as a "
        "text file. An audio processor is a planned extension point.",
    ),
}

_MAGIC_SIGNATURES: list[tuple[bytes, str, str]] = [
    (b"\x89PNG\r\n\x1a\n", "png", "image/png"),
    (b"\xff\xd8\xff", "jpeg", "image/jpeg"),
    (b"GIF87a", "gif", "image/gif"),
    (b"GIF89a", "gif", "image/gif"),
    (b"BM", "bmp", "image/bmp"),
    (b"II*\x00", "tiff", "image/tiff"),
    (b"MM\x00*", "tiff", "image/tiff"),
    (b"%PDF-", "pdf", "application/pdf"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "ole2", "application/x-ole-storage"),
]

_OOXML_MEMBER_HINTS: list[tuple[str, str, MediaCategory]] = [
    ("word/", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
     MediaCategory.OFFICE),
    ("ppt/", "application/vnd.openxmlformats-officedocument.presentationml.presentation",
     MediaCategory.OFFICE),
    ("xl/", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
     MediaCategory.OFFICE),
]


def sha256_file(path: Path) -> str:
    """Streamed content hash. This, not the path, is the identity of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _read_header(path: Path, size: int = 16) -> bytes:
    with path.open("rb") as handle:
        return handle.read(size)


def sniff_magic(path: Path) -> tuple[str | None, str | None]:
    """Return ``(sniffed_kind, mime)`` from the file's leading bytes."""
    try:
        header = _read_header(path)
    except OSError:
        return None, None

    # WEBP is 'RIFF' + 4 size bytes + 'WEBP'; check it before the generic table.
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "webp", "image/webp"

    for signature, kind, mime in _MAGIC_SIGNATURES:
        if header.startswith(signature):
            return kind, mime

    if header[:2] == b"PK":
        return _sniff_zip_container(path)

    return None, None


def _sniff_zip_container(path: Path) -> tuple[str | None, str | None]:
    """Distinguish OOXML documents, EPUB and plain zips by their member names.

    Only the central directory is read -- no member is ever extracted, so this
    cannot itself be a decompression bomb.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()[:200]
    except (zipfile.BadZipFile, OSError):
        return "zip", "application/zip"

    if any(name == "mimetype" for name in names):
        return "epub", "application/epub+zip"
    for prefix, mime, _category in _OOXML_MEMBER_HINTS:
        if any(name.startswith(prefix) for name in names):
            return "ooxml", mime
    return "zip", "application/zip"


def _category_for_mime(mime: str) -> MediaCategory:
    if mime.startswith("image/"):
        return MediaCategory.IMAGE
    if mime == "application/pdf":
        return MediaCategory.PDF
    if mime.startswith("audio/"):
        return MediaCategory.AUDIO
    if mime in {"application/zip"}:
        return MediaCategory.ARCHIVE
    if mime == "application/epub+zip":
        return MediaCategory.EBOOK
    if "officedocument" in mime or mime == "application/vnd.ms-excel":
        return MediaCategory.OFFICE
    if mime in {"text/html", "application/xhtml+xml"}:
        return MediaCategory.HTML
    if mime.startswith("text/"):
        return MediaCategory.TEXT
    return MediaCategory.UNKNOWN


def detect_media(path: Path) -> MediaInfo:
    """Build a :class:`MediaInfo` for an already-sandboxed path."""
    stat_result = path.stat()
    extension = path.suffix.lower()

    mime, category = _EXTENSION_MAP.get(extension, ("", MediaCategory.UNKNOWN))
    sniffed_kind, sniffed_mime = sniff_magic(path)

    if sniffed_mime:
        sniffed_category = _category_for_mime(sniffed_mime)
        # Trust the bytes when they contradict the extension: a '.txt' that is really
        # a PDF should be processed as a PDF, and a '.png' that is really a zip must
        # not reach the image decoder.
        contradicts = category is MediaCategory.UNKNOWN or (
            sniffed_category is not MediaCategory.UNKNOWN and sniffed_category is not category
        )
        # OLE2 covers .xls, .msg and .doc alike; the extension disambiguates it,
        # so keep the extension's answer when we already have one.
        if contradicts and not (sniffed_kind == "ole2" and category is MediaCategory.OFFICE):
            mime, category = sniffed_mime, sniffed_category

    if not mime:
        guessed, _ = mimetypes.guess_type(path.name)
        if guessed:
            mime = guessed
            category = _category_for_mime(guessed)

    if not mime:
        mime = "application/octet-stream"

    return MediaInfo(
        path=path,
        name=path.name,
        extension=extension,
        mime_type=mime,
        category=category,
        size_bytes=stat_result.st_size,
        modified_at=datetime.fromtimestamp(stat_result.st_mtime, tz=UTC),
        sha256=sha256_file(path),
        sniffed_type=sniffed_kind,
    )


def ensure_supported(info: MediaInfo) -> None:
    """Refuse categories we knowingly do not handle, with an actionable reason."""
    if info.category in _REFUSED:
        message, hint = _REFUSED[info.category]
        raise UnsupportedMediaTypeError(
            f"{message} (detected: {info.mime_type})",
            hint=hint,
            details={"mime_type": info.mime_type, "category": info.category.value},
        )
    if info.category is MediaCategory.UNKNOWN:
        raise UnsupportedMediaTypeError(
            f"Unrecognised file type '{info.extension or info.name}' "
            f"(detected MIME: {info.mime_type}).",
            hint=(
                "Supported: images (png/jpg/webp/bmp/tiff/gif), pdf, docx, pptx, xlsx, "
                "xls, csv, html, txt, md, ipynb, epub, msg. Convert the file to one of "
                "these, or pass a text export."
            ),
            details={"mime_type": info.mime_type, "extension": info.extension},
        )
