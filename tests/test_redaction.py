"""Log redaction tests (SPEC §13): the API key travels in the query string."""

from __future__ import annotations

import io
import logging

from bird_recognizer.logging_setup import REDACTED, setup_logging


def _log(message: str, *args: object, secrets: tuple[str, ...] = ()) -> str:
    stream = io.StringIO()
    setup_logging("INFO", stream=stream, secrets=secrets)
    logging.getLogger("test.redaction").info(message, *args)
    return stream.getvalue()


def test_key_in_url_is_masked() -> None:
    output = _log(
        'GET https://xeno-canto.org/api/3/recordings?query=sp:%22Parus+major%22&key=SECRET123'
    )
    assert "SECRET123" not in output
    assert f"key={REDACTED}" in output


def test_key_in_formatted_arguments_is_masked() -> None:
    output = _log("pobieram %s", "https://xeno-canto.org/api/3/recordings?key=SECRET123&page=2")
    assert "SECRET123" not in output
    assert "page=2" in output


def test_authorization_header_is_masked() -> None:
    output = _log("wysyłam Authorization: Bearer abc.def.ghi")
    assert "abc.def.ghi" not in output


def test_registered_secret_is_masked_anywhere() -> None:
    output = _log("klucz to XYZSECRET w środku zdania", secrets=("XYZSECRET",))
    assert "XYZSECRET" not in output


def test_plain_message_is_unchanged() -> None:
    output = _log("zwykły komunikat bez sekretów")
    assert "zwykły komunikat bez sekretów" in output
