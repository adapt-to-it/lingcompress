"""Forza IPv4 per le connessioni Python: su questa rete IPv6 verso CloudFront (Hugging Face) viene resettato."""
import socket

_orig = socket.getaddrinfo


def _solo_v4(host, port, family=0, *a, **k):
    return _orig(host, port, socket.AF_INET, *a, **k)


socket.getaddrinfo = _solo_v4
