"""Client tests: queries, pagination, retries and error mapping (SPEC §4, §9)."""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest

from bird_recognizer.errors import ApiError, AuthError, MissingApiKeyError, RateLimitError
from bird_recognizer.xc.client import (
    XCClient,
    build_query,
    build_url,
    clamp_per_page,
    parse_error_payload,
    parse_retry_after,
)
from support import FakeTransport, file_response, json_response, load_fixture, page, recording


def _client(config, responses, *, key: str | None = "test-key"):
    transport = FakeTransport(responses)
    sleeps: list[float] = []
    client = XCClient(config, api_key=key, transport=transport, sleep=sleeps.append)
    return client, transport, sleeps


# --------------------------------------------------------------------------- #
# Query building
# --------------------------------------------------------------------------- #
def test_query_uses_tags_and_quotes(config) -> None:
    query = build_query(config.species[0], config.api)
    assert 'sp:"Passer domesticus"' in query
    assert "cnt:poland" in query
    assert 'q:">C"' in query
    assert "status:" not in query  # tag wyłączony — API go odrzuca (api-notes §3.1)


def test_status_tag_returned_when_configured(config) -> None:
    api = replace(config.api, status="identified")
    assert "status:identified" in build_query(config.species[0], api)


def test_per_page_is_clamped_to_api_range(config) -> None:
    assert clamp_per_page(10) == 50
    assert clamp_per_page(1000) == 500
    assert clamp_per_page(None) == 100


def test_url_carries_key_and_page(config) -> None:
    url = build_url(config.api, 'sp:"Parus major"', key="secret", page=2, per_page=50)
    assert "page=2" in url
    assert "per_page=50" in url
    assert "key=secret" in url
    assert "sp:" in url


# --------------------------------------------------------------------------- #
# Pagination
# --------------------------------------------------------------------------- #
def test_search_returns_parsed_page(config) -> None:
    client, transport, _ = _client(config, [json_response(load_fixture("parus_page1.json"))])
    result = client.search('sp:"Parus major"')
    assert result.page == 1
    assert result.num_pages == 3
    assert result.num_recordings == 3
    assert len(result.recordings) == 2
    assert len(transport.calls) == 1


def test_iter_recordings_walks_all_pages(config) -> None:
    responses = [
        json_response(load_fixture("parus_page1.json")),
        json_response(load_fixture("parus_page2.json")),
        json_response(load_fixture("parus_page3.json")),
    ]
    client, transport, _ = _client(config, responses)
    ids = [item["id"] for item in client.iter_recordings('sp:"Parus major"')]
    assert ids == ["XC1001", "XC1002", "XC1003", "XC1004"]
    assert len(transport.calls) == 3
    assert "page=2" in transport.calls[1]
    assert "page=3" in transport.calls[2]


def test_iter_recordings_respects_max_pages(config) -> None:
    responses = [
        json_response(load_fixture("parus_page1.json")),
        json_response(load_fixture("parus_page2.json")),
    ]
    client, transport, _ = _client(config, responses)
    ids = [item["id"] for item in client.iter_recordings('sp:"x"', max_pages=1)]
    assert len(ids) == 2
    assert len(transport.calls) == 1


def test_throttle_sleeps_between_requests(config) -> None:
    responses = [json_response(page([])), json_response(page([]))]
    client, _, sleeps = _client(config, responses)
    client.search("q:A")
    client.search("q:A")
    assert sleeps and sleeps[0] >= 0.9


# --------------------------------------------------------------------------- #
# Error mapping (SPEC §9)
# --------------------------------------------------------------------------- #
def test_missing_key_raises_before_any_request(config, monkeypatch) -> None:
    monkeypatch.delenv("BIRD_XC_API_KEY", raising=False)
    monkeypatch.delenv("xc_api_key", raising=False)
    client = XCClient(config, api_key=None, transport=FakeTransport([]))
    with pytest.raises(MissingApiKeyError) as excinfo:
        client.search("q:A")
    assert excinfo.value.exit_code == 2


def test_400_maps_to_api_error_without_retry(config) -> None:
    client, transport, _ = _client(
        config, [json_response(load_fixture("error_400.json"), status=400)]
    )
    with pytest.raises(ApiError) as excinfo:
        client.search('sp:"Parus major"')
    assert excinfo.value.status == 400
    assert excinfo.value.code == "client_error"
    assert len(transport.calls) == 1


def test_403_maps_to_auth_error(config) -> None:
    client, transport, _ = _client(
        config, [json_response(load_fixture("error_403.json"), status=403)]
    )
    with pytest.raises(AuthError) as excinfo:
        client.search("q:A")
    assert excinfo.value.status == 403
    assert len(transport.calls) == 1


def test_429_honours_retry_after_then_succeeds(config) -> None:
    responses = [
        json_response(
            load_fixture("error_429.json"), status=429, headers={"Retry-After": "7"}
        ),
        json_response(page([recording("XC1")])),
    ]
    client, transport, sleeps = _client(config, responses)
    result = client.search("q:A")
    assert len(result.recordings) == 1
    assert 7.0 in sleeps
    assert len(transport.calls) == 2


def test_429_without_retry_after_backs_off_from_5s(config) -> None:
    responses = [
        json_response(load_fixture("error_429.json"), status=429),
        json_response(page([])),
    ]
    client, _, sleeps = _client(config, responses)
    client.search("q:A")
    assert sleeps[0] >= 5.0


def test_429_exhausts_attempts_and_raises_rate_limit(config) -> None:
    attempts = max(1, config.api.max_retries)
    responses = [json_response(load_fixture("error_429.json"), status=429) for _ in range(attempts)]
    client, transport, _ = _client(config, responses)
    with pytest.raises(RateLimitError) as excinfo:
        client.search("q:A")
    assert excinfo.value.status == 429
    assert len(transport.calls) == attempts


def test_5xx_retries_then_fails(config) -> None:
    responses = [
        json_response({"error": "server_error", "message": "boom"}, status=500)
        for _ in range(4)
    ]
    client, transport, sleeps = _client(config, responses)
    with pytest.raises(ApiError):
        client.search("q:A")
    assert len(transport.calls) == 4
    assert sleeps  # backoff between attempts


def test_network_error_is_retried(config) -> None:
    responses = [ApiError("connection reset"), json_response(page([recording("XC9")]))]
    client, transport, _ = _client(config, responses)
    result = client.search("q:A")
    assert len(result.recordings) == 1
    assert len(transport.calls) == 2


def test_network_error_gives_up_after_three_attempts(config) -> None:
    responses = [ApiError("timeout") for _ in range(3)]
    client, transport, _ = _client(config, responses)
    with pytest.raises(ApiError):
        client.search("q:A")
    assert len(transport.calls) == 3


def test_invalid_json_response_raises_api_error(config) -> None:
    from support import HttpResponse

    client, _, _ = _client(config, [HttpResponse(status=200, body=b"not json")])
    with pytest.raises(ApiError, match="JSON"):
        client.search("q:A")


def test_bare_error_shape_is_accepted(config) -> None:
    client, _, _ = _client(
        config, [json_response({"error": "client_error", "message": "zły tag"}, status=400)]
    )
    with pytest.raises(ApiError) as excinfo:
        client.search("q:A")
    assert excinfo.value.code == "client_error"
    assert "zły tag" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def test_error_payload_shapes() -> None:
    flat = b'{"error": "client_error", "message": "m"}'
    nested = b'{"error": {"code": "missing_parameter", "message": "n"}}'
    assert parse_error_payload(flat) == ("client_error", "m")
    assert parse_error_payload(nested) == ("missing_parameter", "n")
    assert parse_error_payload(b"<html>") == (None, None)


def test_retry_after_parsing() -> None:
    assert parse_retry_after("5") == 5.0
    assert parse_retry_after("0") == 0.0
    assert parse_retry_after("nie-liczba") is None
    assert parse_retry_after(None) is None


def test_key_never_appears_in_logs(config, caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="bird_recognizer.xc.client")
    client, _, _ = _client(config, [json_response(page([recording("XC1")]))], key="TOPSECRET")
    client.search("q:A")
    assert "TOPSECRET" not in caplog.text
    assert "key=***" in caplog.text


def test_fetch_bytes_returns_payload_and_headers(config) -> None:
    client, _, _ = _client(config, [file_response(b"ID3-audio", content_type="audio/wav")])
    body, headers = client.fetch_bytes("https://xeno-canto.org/1/download")
    assert body == b"ID3-audio"
    assert headers["Content-Type"] == "audio/wav"


def test_fetch_bytes_rejects_empty_body(config) -> None:
    client, _, _ = _client(config, [file_response(b"")])
    with pytest.raises(ApiError, match="pusty"):
        client.fetch_bytes("https://xeno-canto.org/1/download")
