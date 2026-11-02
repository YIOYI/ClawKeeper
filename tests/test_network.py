"""Fake DNS and sockets exercise the real HTTP stack without network access."""

# Concrete HTTP status codes, address families and connection counts are fixtures.
# ruff: noqa: PLR2004

import socket
from types import SimpleNamespace

import httpcore
import httpx
import pytest

from clawkeeper_core import network

PUBLIC = '93.184.216.34'
PUBLIC_V6 = '2606:4700:4700::1111'


def answer(ip, port=80):
    family = socket.AF_INET6 if ':' in ip else socket.AF_INET
    return family, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', (ip, port)


@pytest.fixture
def fake_network(monkeypatch):
    state = SimpleNamespace(dns=[], connections=[], sent=[], closed=[],
                            answers=[answer(PUBLIC)], peer=None, replies=[], tls=[])

    def resolve(host, port, **kwargs):
        state.dns.append((host, port))
        if isinstance(state.answers, Exception):
            raise state.answers
        return state.answers(host, port) if callable(state.answers) else state.answers

    class Socket:
        def __init__(self, *args):
            self.address = None
            self.response = b''

        def settimeout(self, timeout):
            pass

        def setsockopt(self, *args):
            pass

        def connect(self, address):
            self.address = address
            state.connections.append(address)
            self.response = state.replies.pop(0) if state.replies else b''

        def getpeername(self):
            return state.peer or self.address

        def send(self, data):
            state.sent.append(bytes(data))
            return len(data)

        def recv(self, size):
            part, self.response = self.response[:size], self.response[size:]
            return part

        def close(self):
            state.closed.append(self.address)

    monkeypatch.setattr(network, 'socket', SimpleNamespace(
        AF_INET=socket.AF_INET, AF_INET6=socket.AF_INET6, SOCK_STREAM=socket.SOCK_STREAM,
        IPPROTO_TCP=socket.IPPROTO_TCP, TCP_NODELAY=socket.TCP_NODELAY,
        timeout=socket.timeout, socket=Socket, getaddrinfo=resolve))
    return state


@pytest.mark.parametrize('ip', [
    '127.0.0.1', '10.0.0.1', '169.254.169.254', '100.64.0.1', '0.0.0.0',
    '192.168.1.1', '::1', 'fe80::1', 'fc00::1', '::ffff:127.0.0.1', 'ff02::1',
])
def test_domain_resolving_to_private_is_blocked(fake_network, ip):
    fake_network.answers = [answer(ip)]
    with network.public_http_client() as client, pytest.raises(network.UnsafeDestination):
        client.get('http://private.test/')
    assert fake_network.connections == []


def test_all_answers_validated_before_any_connection(fake_network):
    fake_network.answers = [answer(PUBLIC), answer('::1')]
    with network.public_http_client() as client, pytest.raises(network.UnsafeDestination):
        client.get('http://mixed.test/')
    assert fake_network.connections == []


@pytest.mark.parametrize('answers', [[], socket.gaierror('fake DNS failure')])
def test_dns_failures_are_closed(fake_network, answers):
    fake_network.answers = answers
    with network.public_http_client() as client, pytest.raises(network.UnsafeDestination):
        client.get('http://unresolved.test/')
    assert fake_network.connections == []


def reply(status='200 OK', headers=b'', body=b'ok'):
    return (b'HTTP/1.1 ' + status.encode() + b'\r\nConnection: close\r\n' + headers
            + b'Content-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body)


@pytest.mark.parametrize('ip', [PUBLIC, PUBLIC_V6])
def test_public_request_uses_numeric_address_and_original_host(fake_network, ip, monkeypatch):
    monkeypatch.setenv('HTTP_PROXY', 'http://127.0.0.1:8888')
    fake_network.answers = [answer(ip)]
    fake_network.replies = [reply()]
    with network.public_http_client() as client:
        assert client.get('http://public.test/path').text == 'ok'
    assert fake_network.dns == [('public.test', 80)]
    assert fake_network.connections[0][:2] == (ip, 80)
    assert b'Host: public.test\r\n' in b''.join(fake_network.sent)


@pytest.mark.parametrize('location', ['http://127.0.0.1/secret', 'http://private.test/secret', '//private.test/secret'])
def test_public_to_private_redirect_is_blocked(fake_network, location):
    fake_network.answers = lambda host, port: [answer(PUBLIC if host == 'public.test' else '10.0.0.1', port)]
    fake_network.replies = [reply('302 Found', b'Location: ' + location.encode() + b'\r\n')]
    with network.public_http_client() as client, pytest.raises(network.UnsafeDestination):
        client.get('http://public.test/')
    assert fake_network.connections == [(PUBLIC, 80)]


def test_every_redirect_and_new_connection_resolves_again(fake_network):
    def resolve(host, port):
        return [answer(PUBLIC if len(fake_network.dns) == 1 else '127.0.0.1', port)]
    fake_network.answers = resolve
    fake_network.replies = [reply('302 Found', b'Location: /next\r\n')]
    with network.public_http_client() as client, pytest.raises(network.UnsafeDestination):
        client.get('http://rebinding.test/')
    assert len(fake_network.dns) == 2
    assert fake_network.connections == [(PUBLIC, 80)]


def test_no_second_name_resolution_between_check_and_connect(fake_network):
    fake_network.answers = lambda host, port: [answer(PUBLIC if len(fake_network.dns) == 1 else '127.0.0.1', port)]
    fake_network.replies = [reply()]
    with network.public_http_client() as client:
        assert client.get('http://rebinding.test/').status_code == 200
    assert len(fake_network.dns) == 1
    assert fake_network.connections == [(PUBLIC, 80)]


def test_connected_peer_change_is_rejected_and_closed(fake_network):
    fake_network.peer = ('127.0.0.1', 80)
    with pytest.raises(network.UnsafeDestination):
        network.PublicNetworkBackend().connect_tcp('peer.test', 80)
    assert fake_network.closed == [(PUBLIC, 80)]
    assert fake_network.sent == []


def test_https_retains_original_sni_and_verified_context(fake_network, monkeypatch):
    class TLSContext:
        check_hostname = True
        verify_mode = 2
        def set_alpn_protocols(self, protocols):
            pass
        def wrap_socket(self, sock, server_hostname=None):
            fake_network.tls.append(server_hostname)
            return sock
    monkeypatch.setattr(network.ssl, 'create_default_context', TLSContext)
    fake_network.answers = [answer(PUBLIC, 443)]
    fake_network.replies = [reply()]
    with network.public_http_client() as client:
        assert client.get('https://secure.test/').text == 'ok'
    assert fake_network.connections == [(PUBLIC, 443)]
    assert fake_network.tls == ['secure.test']
    assert b'Host: secure.test\r\n' in b''.join(fake_network.sent)


def test_relative_public_redirect_succeeds(fake_network):
    fake_network.replies = [reply('302 Found', b'Location: /next\r\n'), reply(body=b'done')]
    with network.public_http_client() as client:
        result = client.get('http://public.test/')
    assert result.text == 'done' and len(result.history) == 1
    assert fake_network.connections == [(PUBLIC, 80), (PUBLIC, 80)]


def test_redirect_limit(fake_network):
    fake_network.replies = [reply('302 Found', b'Location: /loop\r\n')] * 3
    with network.public_http_client(max_redirects=1) as client, pytest.raises(httpx.TooManyRedirects):
        client.get('http://public.test/')
    assert len(fake_network.connections) == 2


@pytest.mark.parametrize('url', ['http://user:pass@public.test/', 'http://metadata.google.internal./',
                               'http://[::ffff:127.0.0.1]/'])
def test_rejected_url_never_connects(fake_network, url):
    with network.public_http_client() as client, pytest.raises(network.UnsafeDestination):
        client.get(url)
    assert fake_network.connections == []


def test_custom_sni_and_unix_connections_are_rejected(fake_network):
    with network.public_http_client() as client, pytest.raises(network.UnsafeDestination):
        client.get('https://public.test/', extensions={'sni_hostname': 'different.test'})
    with pytest.raises(network.UnsafeDestination):
        network.PublicNetworkBackend().connect_unix_socket('/tmp/fake.sock')
    assert fake_network.connections == []


def test_connect_failure_is_httpcore_error(fake_network, monkeypatch):
    def fail(self, address):
        raise OSError('fake refusal')
    monkeypatch.setattr(network.socket.socket, 'connect', fail)
    with pytest.raises(httpcore.ConnectError):
        network.PublicNetworkBackend().connect_tcp('public.test', 80)
    assert fake_network.closed == [None]
