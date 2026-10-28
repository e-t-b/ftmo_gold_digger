"""ChatGPT-plan OAuth for the local, open-source Gold Digger client.

This implements OpenAI's public-client PKCE flow. Credentials are kept outside
the repository and are never printed. The module deliberately has no trading
or MetaTrader capability.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import secrets
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


AUTHORIZE_URL = "https://auth.openai.com/api/accounts/authorize"
TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
OIDC_CONFIG_URL = "https://auth.openai.com/.well-known/openid-configuration"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
AGENT_NAME = "Gold Digger"
REQUIRED_PLAN_SCOPE = "chatgpt.tokens.use.direct"


def default_config_dir() -> Path:
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "ftmo-gold"
    if os.environ.get("XDG_CONFIG_HOME"):
        return Path(os.environ["XDG_CONFIG_HOME"]) / "ftmo-gold"
    return Path.home() / ".config" / "ftmo-gold"


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def make_pkce() -> tuple[str, str]:
    verifier = _b64url(secrets.token_bytes(48))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def _post_form(url: str, fields: dict, timeout: int = 30) -> dict:
    request = urllib.request.Request(
        url,
        data=urllib.parse.urlencode(fields).encode("ascii"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _read_json(url: str, token: str | None = None, timeout: int = 30) -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
        return json.load(response)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class _CallbackHandler(BaseHTTPRequestHandler):
    result: dict | None = None
    event = threading.Event()

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/auth/callback":
            self.send_error(404)
            return
        type(self).result = {key: values[-1] for key, values in urllib.parse.parse_qs(parsed.query).items()}
        body = b"Gold Digger is connected. You may close this window."
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        type(self).event.set()

    def log_message(self, *_):
        return


def _validate_id_token(id_token: str, client_id: str, nonce: str) -> dict:
    try:
        import jwt
    except ImportError as exc:
        raise RuntimeError('Install the authentication extra: pip install -e ".[openai]"') from exc
    configuration = _read_json(OIDC_CONFIG_URL)
    issuer = configuration["issuer"]
    signing_key = jwt.PyJWKClient(configuration["jwks_uri"]).get_signing_key_from_jwt(id_token)
    claims = jwt.decode(
        id_token,
        signing_key.key,
        algorithms=["RS256"],
        audience=client_id,
        issuer=issuer,
    )
    if not secrets.compare_digest(str(claims.get("nonce", "")), nonce):
        raise RuntimeError("OpenAI ID-token nonce mismatch")
    return claims


class ChatGPTPlanCredentials:
    """One local ChatGPT registration for Gold Digger."""

    def __init__(self, config_dir: str | Path | None = None):
        self.config_dir = Path(config_dir) if config_dir else default_config_dir()
        self.host_path = self.config_dir / "host.json"
        self.profile_path = self.config_dir / "chatgpt-profile.json"

    def _host_id(self) -> str:
        if self.host_path.exists():
            value = json.loads(self.host_path.read_text(encoding="utf-8"))["ext_agent_host_id"]
            if not value.startswith("urn:uuid:"):
                raise RuntimeError("Invalid stored Gold Digger host identifier")
            return value
        value = "urn:uuid:" + str(uuid.uuid4())
        _atomic_json(self.host_path, {"ext_agent_host_id": value})
        return value

    def load(self) -> dict:
        if not self.profile_path.exists():
            raise RuntimeError("ChatGPT is not connected; run: python -m ftmo_gold.chatgpt_auth login")
        profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
        if REQUIRED_PLAN_SCOPE not in profile.get("scopes", []):
            raise RuntimeError("ChatGPT plan usage was not authorized")
        return profile

    def login(self, timeout: int = 180) -> dict:
        prior = json.loads(self.profile_path.read_text(encoding="utf-8")) if self.profile_path.exists() else None
        client_id = prior["client_id"] if prior else "dynamic_agent_client"
        verifier, challenge = make_pkce()
        state, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        _CallbackHandler.result = None
        _CallbackHandler.event = threading.Event()
        server = ThreadingHTTPServer(("127.0.0.1", 0), _CallbackHandler)
        port = server.server_address[1]
        redirect_uri = f"http://127.0.0.1:{port}/auth/callback"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        parameters = {
            "client_id": client_id,
            "ext_agent_host_id": self._host_id(),
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": SCOPES,
            "resource": RESOURCE,
            "state": state,
            "nonce": nonce,
            "code_challenge_method": "S256",
            "code_challenge": challenge,
        }
        if prior:
            if prior.get("id_token"):
                parameters["id_token_hint"] = prior["id_token"]
            if prior.get("email"):
                parameters["login_hint"] = prior["email"]
        else:
            parameters["agent_name_hint"] = AGENT_NAME
        url = AUTHORIZE_URL + "?" + urllib.parse.urlencode(parameters)
        try:
            if not webbrowser.open(url):
                raise RuntimeError("Could not open the system browser")
            if not _CallbackHandler.event.wait(timeout):
                raise TimeoutError("ChatGPT authorization timed out")
        finally:
            server.shutdown()
            server.server_close()
        callback = _CallbackHandler.result or {}
        if not secrets.compare_digest(callback.get("state", ""), state):
            raise RuntimeError("OAuth state mismatch")
        if callback.get("error"):
            raise RuntimeError("ChatGPT authorization was declined")
        issued_client_id = callback.get("client_id", client_id)
        if client_id == "dynamic_agent_client" and issued_client_id == client_id:
            raise RuntimeError("OpenAI did not issue a client identifier")
        if prior and issued_client_id != client_id:
            raise RuntimeError("OpenAI returned a different registered client")
        token = _post_form(TOKEN_URL, {
            "grant_type": "authorization_code",
            "client_id": issued_client_id,
            "code": callback["code"],
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
            "resource": RESOURCE,
        })
        claims = _validate_id_token(token["id_token"], issued_client_id, nonce)
        scopes = token.get("scope", "").split()
        if REQUIRED_PLAN_SCOPE not in scopes:
            raise RuntimeError("ChatGPT plan usage permission was not granted")
        now = time.time()
        profile = {
            "email": claims.get("email"),
            "issuer": claims.get("iss"),
            "subject": claims["sub"],
            "client_id": issued_client_id,
            "ext_agent_host_id": self._host_id(),
            "id_token": token["id_token"],
            "access_token": token["access_token"],
            "refresh_token": token["refresh_token"],
            "token_type": token.get("token_type", "Bearer"),
            "expires_in": int(token["expires_in"]),
            "expires_at": now + int(token["expires_in"]),
            "scopes": scopes,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(self.profile_path, profile)
        return profile

    def refresh(self) -> dict:
        profile = self.load()
        token = _post_form(TOKEN_URL, {
            "grant_type": "refresh_token",
            "client_id": profile["client_id"],
            "refresh_token": profile["refresh_token"],
            "resource": RESOURCE,
        })
        now = time.time()
        updated = dict(profile)
        for key in ("access_token", "refresh_token", "id_token", "token_type"):
            if token.get(key):
                updated[key] = token[key]
        updated["expires_in"] = int(token["expires_in"])
        updated["expires_at"] = now + int(token["expires_in"])
        updated["scopes"] = token.get("scope", " ".join(profile["scopes"])).split()
        updated["saved_at"] = datetime.now(timezone.utc).isoformat()
        if REQUIRED_PLAN_SCOPE not in updated["scopes"]:
            raise RuntimeError("Refreshed token no longer permits ChatGPT plan usage")
        _atomic_json(self.profile_path, updated)
        return updated

    def access_token(self) -> str:
        environment_token = os.environ.get("OPENAI_CHATGPT_ACCESS_TOKEN")
        if environment_token:
            return environment_token
        profile = self.load()
        if time.time() >= float(profile.get("expires_at", 0)) - 120:
            profile = self.refresh()
        return profile["access_token"]

    def status(self) -> dict:
        try:
            profile = self.load()
        except RuntimeError:
            return {"connected": False, "plan_usage": False}
        return {
            "connected": True,
            "plan_usage": REQUIRED_PLAN_SCOPE in profile.get("scopes", []),
            "email": profile.get("email"),
            "expires_at": profile.get("expires_at"),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Connect Gold Digger to a ChatGPT plan")
    parser.add_argument("command", choices=("login", "status"))
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args()
    credentials = ChatGPTPlanCredentials(args.config_dir)
    if args.command == "login":
        profile = credentials.login()
        print(json.dumps({"connected": True, "email": profile.get("email"), "plan_usage": True}))
    else:
        print(json.dumps(credentials.status()))


if __name__ == "__main__":
    main()
