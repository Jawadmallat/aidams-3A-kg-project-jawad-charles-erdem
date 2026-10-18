"""Health of the running Compose stack, checked from outside like a client.

Run after `make docker-run`, against the published Nginx port:

    STACK_URL=http://localhost:8080 uv run pytest tests/test_container_health.py -v

Skipped unless STACK_URL is set, so CI and `make test` never need Docker.
The checks follow the request path of the architecture: Nginx (the only
published port) -> API -> Neo4j, plus the healthchecks Docker itself runs.
"""

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request

import pytest

STACK_URL = os.environ.get("STACK_URL", "").rstrip("/")
SERVICES = {"neo4j", "api", "nginx"}

pytestmark = pytest.mark.skipif(not STACK_URL, reason="STACK_URL not set (stack not running)")


def get(path: str) -> tuple[int, dict | str]:
    """GET through the proxy: (status, JSON body or text)."""
    request = urllib.request.Request(STACK_URL + path)
    return _send(request)


def post(path: str, body: dict) -> tuple[int, dict | str]:
    """POST JSON through the proxy: (status, JSON body or text)."""
    request = urllib.request.Request(
        STACK_URL + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    return _send(request)


def _send(request: urllib.request.Request) -> tuple[int, dict | str]:
    """Send a request; HTTP errors are answers too."""
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status, text = response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        status, text = error.code, error.read().decode()
    try:
        return status, json.loads(text)
    except json.JSONDecodeError:
        return status, text


def compose_services() -> list[dict]:
    """`docker compose ps` as dicts, or skip when Docker is not available."""
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not available")
    output = subprocess.run(
        ["docker", "compose", "ps", "--format", "json"],
        capture_output=True, text=True, check=True,
    ).stdout
    # One JSON object per line (Compose v2.21+), or a single JSON array before.
    stripped = output.strip()
    if stripped.startswith("["):
        return json.loads(stripped)
    return [json.loads(line) for line in stripped.splitlines() if line]


def test_api_health_answers_through_nginx():
    """Nginx -> API: the liveness probe answers on the published port."""
    assert get("/api/health") == (200, {"status": "ok"})


def test_api_reaches_neo4j():
    """API -> Neo4j: a data endpoint answers 200, not the 503 of a dead database."""
    status, body = get("/api/movies/trending?limit=1")
    assert status == 200, body
    assert "results" in body


def test_swagger_and_openapi_are_served():
    """Swagger UI and the OpenAPI document (with the LLM endpoints) go through Nginx."""
    status, _ = get("/docs")
    assert status == 200
    status, spec = get("/openapi.json")
    assert status == 200
    assert "/api/ask" in spec["paths"]
    assert "/api/users/{user_id}/recommendations/{movie_id}/why" in spec["paths"]


def test_ask_answers_with_a_handled_status():
    """/api/ask answers 200/400 with an LLM, or a clear 503 without one; never a 500."""
    status, body = post("/api/ask", {"question": "How many movies are in the database?"})
    assert status in {200, 400, 503}, body
    if status == 503:
        assert "LLM_API_KEY" in body["detail"] or body["detail"] == "database unavailable"


def test_unrouted_paths_stop_at_the_proxy():
    """Nginx exposes an allowlist: anything else 404s before reaching the API."""
    assert get("/admin")[0] == 404


def test_every_container_reports_healthy():
    """Docker's own healthchecks (cypher-shell, /api/health, wget) all pass."""
    services = {service["Service"]: service for service in compose_services()}
    assert SERVICES <= set(services), f"missing: {SERVICES - set(services)}"
    for name in SERVICES:
        assert services[name]["Health"] == "healthy", f"{name}: {services[name]['Status']}"


def test_only_nginx_publishes_a_port():
    """Access control by topology: Neo4j and the API are internal only."""
    published = {
        service["Service"]
        for service in compose_services()
        for publisher in service.get("Publishers") or []
        if publisher.get("PublishedPort")
    }
    assert published == {"nginx"}
