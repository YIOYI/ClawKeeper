# Path, script and HTTP destination checks

The local guard chain shares bounded Bash/Python facts with learned-rule
validation and enforcement. Candidate commands are parsed, never executed.

## Path operands and script associations (#18, #6)

- `echo '~/.aws/credentials'`, Python `print(...)`, quoted documentation
  heredocs and writing that text into an ordinary document do not access the
  named credential file. They are excluded from the path subject.
- `cat ~/.aws/credentials`, file redirections, supported interpreter script
  operands, explicit file-tool fields and Python file access operands retain
  access origins. Nested shell calls, command substitutions and visible Python
  stdin use the shared execution parser.
- Python associations cover literal assignments, imported aliases, string/path
  joins, Path.home/expanduser, lists and visible loop bindings. The access operand
  carries the sensitive path regardless of intervening comments or whitespace.
  An unrelated `.aws` string does not contaminate a read of `/tmp/guide.txt`.
- Unknown access operands, dynamic code and unsupported contexts return `ask`.
  Unknown programs receiving path-shaped operands are uncertain. This does not
  infer arbitrary program behavior or follow arbitrary Python function calls.
- Relative paths use the caller's supplied working directory, falling back to
  the guard process directory. Relative access in a source with directory
  changes requires review. The guard resolves filesystem paths at assessment
  time; preventing concurrent symlink changes requires a host filesystem sandbox.

Explicit fields include `path`, `paths`, `file`, `file_path`, `filePath`,
`filename`, `file_name`, `directory`, `dir_path`, `source`, `destination`, `target`
and `location`, including nested envelopes. Content, descriptions, environment
metadata and headers are excluded. Hosts with another schema should map it to
these fields before evaluation.

Learning uses the same access candidates, cwd normalization and uncertainty
handling. Validator 6 rejects older snapshots. Old normalized path samples
cannot prove whether they came from output or file access, so they become
inactive and must be relearned from a tool event. Existing records with original
tool arguments are reanalyzed; command-rule projections remain compatible.
Inactive records retain their reason and deletion identity.

## HTTP connections (#5)

`clawkeeper_core.network.public_http_client()` returns a synchronous HTTPX client
for use inside a network tool's actual fetch implementation. The transport:

1. Checks the URL policy for every request, including every redirect.
2. Resolves a hostname at every new TCP connection. Empty/failed DNS and any
   non-public address in a mixed answer reject the connection before dialing.
3. Validates IPv4, IPv6 and IPv4-mapped IPv6, then passes only a numeric sockaddr
   directly to the socket. There is no second hostname lookup between check and
   connect. Connection fallback uses only addresses from the validated answer.
4. Checks the connected peer before sending HTTP data or starting TLS. Keeps the
   original URL host for Host, TLS SNI and certificate verification.
5. Repeats these steps for redirected destinations and reconnects. A reused
   connection remains tied to its previously checked peer. Limits redirects and
   disables environment proxies, Unix sockets and custom SNI overrides.

The client raises `UnsafeDestination` (an HTTPX transport error) when policy or
resolution fails. Connection/HTTP/TLS errors retain HTTPX error behavior. Tool
handlers must treat these errors as failed requests, without an unguarded retry.
Close the client via its context manager. Async hosts can use a worker thread.

The existing `url_safety_guard` intentionally performs no DNS or HTTP requests.
It checks URL text in commands and returned content; it cannot constrain a
separate host browser, shell command, proxy or HTTP client. Hosts must route
fetches through the protected client or enforce their own network boundary.
Arbitrary network execution still requires host egress isolation.

## Verification

`tests/test_path_context.py` checks output/access pairs in the default chain,
padding variants, unrelated-text counterexamples, relative and nested inputs,
learning/reload/deletion and retirement of samples without provenance.
`tests/test_network.py` uses fake DNS and sockets through the real HTTPX/httpcore
stack to exercise private/mixed IPv4/IPv6 answers, public and relative redirects,
private redirects, rebinding/reconnects, numeric dialing, peer changes, Host/SNI,
proxy environment isolation and DNS/connection failures. These tests make no
external requests and never access real credentials or cloud metadata.
