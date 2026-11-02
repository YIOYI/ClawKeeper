"""HTTP client with SSRF policy enforced at the socket connection boundary.

Each new TCP connection resolves once, rejects the entire answer if any address
is non-public, and connects a numeric sockaddr directly. The URL/Host/TLS SNI
remain the original hostname. Redirects run through the same transport. Proxy
environment variables are deliberately disabled: a proxy would resolve elsewhere.
"""

from __future__ import annotations

import ipaddress
import socket
import ssl
import time

import httpcore
import httpx
from httpcore._backends.sync import SyncStream

from clawkeeper_core.guards.url_safety import _ip_is_blocked, url_safety_guard


class UnsafeDestination(httpx.TransportError):
    """The request cannot satisfy the public-destination policy."""


def _public_address(value: str):
    try:
        addr = ipaddress.ip_address(value)
    except ValueError as exc:
        raise UnsafeDestination('Resolver returned an invalid address') from exc
    if '%' in value:
        raise UnsafeDestination('Scoped IP destinations are unsupported')
    blocked, _ = _ip_is_blocked(value)
    if blocked or not addr.is_global or addr.is_multicast:
        raise UnsafeDestination('Destination is not a public unicast address')
    return addr


_IPV6_VERSION = 6


class PublicNetworkBackend(httpcore.NetworkBackend):
    """No hostname reaches connect(); failed DNS or mixed answers fail closed."""

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):  # noqa: PLR0912 -- validate the entire answer before any dial
        if local_address is not None:
            raise UnsafeDestination('Custom local addresses are unsupported')
        started = time.monotonic()
        try:
            # Numeric literals bypass DNS entirely, including mapped IPv6 forms.
            literal = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        try:
            answers = ([(socket.AF_INET6 if literal.version == _IPV6_VERSION else socket.AF_INET,
                         socket.SOCK_STREAM, socket.IPPROTO_TCP, '',
                         (str(literal), port, 0, 0) if literal.version == _IPV6_VERSION else (str(literal), port))]
                       if literal is not None else socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
        except OSError as exc:
            raise UnsafeDestination('DNS resolution failed') from exc
        if not answers:
            raise UnsafeDestination('DNS returned no addresses')
        targets = []
        for family, socktype, _proto, _, address in answers:
            ip = _public_address(address[0])
            expected = socket.AF_INET6 if ip.version == _IPV6_VERSION else socket.AF_INET
            if family != expected or socktype != socket.SOCK_STREAM or address[1] != port:
                raise UnsafeDestination('Resolver returned an inconsistent address')
            # Rebuild from the validated numeric IP. Do not let a resolver's
            # IPv6 scope identifier alter the destination.
            target = (str(ip), port, 0, 0) if ip.version == _IPV6_VERSION else (str(ip), port)
            if (family, target) not in targets:
                targets.append((family, target))
        last_error = None
        for family, address in targets:
            sock = socket.socket(family, socket.SOCK_STREAM, socket.IPPROTO_TCP)
            try:
                remaining = None if timeout is None else max(0.0, timeout - (time.monotonic() - started))
                if remaining == 0:
                    raise TimeoutError('Connection deadline exceeded')
                sock.settimeout(remaining)
                for option in socket_options or ():
                    sock.setsockopt(*option)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.connect(address)
                peer = sock.getpeername()
                _public_address(peer[0])
                if ipaddress.ip_address(peer[0]) != ipaddress.ip_address(address[0]) or peer[1] != port:
                    raise UnsafeDestination('Connected peer differs from validated destination')
                return SyncStream(sock)
            except UnsafeDestination:
                sock.close()
                raise
            except OSError as exc:
                sock.close()
                last_error = exc
        if isinstance(last_error, TimeoutError):
            raise httpcore.ConnectTimeout('Connection timed out') from last_error
        raise httpcore.ConnectError('All validated destinations failed') from last_error

    def connect_unix_socket(self, *args, **kwargs):
        raise UnsafeDestination('Unix sockets are unsupported')


class PublicHTTPTransport(httpx.HTTPTransport):
    """HTTPX transport using a direct, public-only httpcore connection pool."""

    def __init__(self):
        # HTTPTransport's request/response adapter is reused; its pool is built
        # explicitly so httpcore owns TLS for the original URL host and SNI.
        self._pool = httpcore.ConnectionPool(
            ssl_context=ssl.create_default_context(), network_backend=PublicNetworkBackend(),
            max_connections=20, max_keepalive_connections=10, retries=0,
        )

    def handle_request(self, request):
        if request.url.scheme not in {'http', 'https'} or not request.url.host:
            raise UnsafeDestination('Only absolute HTTP(S) URLs are supported', request=request)
        if request.url.username or request.url.password:
            raise UnsafeDestination('URL userinfo is unsupported', request=request)
        # A custom SNI extension could change the TLS authority independently
        # of the URL. The client always uses the original URL authority.
        if 'sni_hostname' in request.extensions:
            raise UnsafeDestination('Custom TLS SNI is unsupported', request=request)
        decision = url_safety_guard({'params': {'url': str(request.url)}})
        if decision and decision.get('block'):
            raise UnsafeDestination(decision['reason'], request=request)
        return super().handle_request(request)


def public_http_client(*, timeout: float = 10.0, max_redirects: int = 10) -> httpx.Client:
    """Use this client inside network tool handlers; close it after use.

    This protects requests made through this client. Pre-tool URL text checks
    cannot provide connection isolation for a host's arbitrary curl/browser/tool.
    """
    return httpx.Client(transport=PublicHTTPTransport(), trust_env=False,
                        follow_redirects=True, max_redirects=max_redirects, timeout=timeout)
