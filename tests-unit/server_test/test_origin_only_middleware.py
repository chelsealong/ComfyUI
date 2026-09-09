"""Tests for the origin_only_middleware (server.py)"""

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from server import create_origin_only_middleware

pytestmark = pytest.mark.asyncio  # Apply asyncio mark to all tests


async def ok_handler(request):
    return web.Response(status=200, text="ok")


@pytest.fixture
def middleware():
    return create_origin_only_middleware()


async def test_cross_site_to_loopback_host_is_blocked(middleware):
    """A cross-site request aimed at a loopback host is the threat this guards against -> 403."""
    request = make_mocked_request(
        "POST",
        "/prompt",
        headers={"Host": "127.0.0.1:8188", "Sec-Fetch-Site": "cross-site"},
    )
    response = await middleware(request, ok_handler)
    assert response.status == 403


async def test_cross_site_to_public_host_is_allowed(middleware):
    """Regression for #16203: a cross-site navigation to a public (non-loopback) host must not be blocked.

    This is the post-login redirect from an auth proxy (Cloudflare Access, oauth2-proxy, Authelia, ...)
    on a separate hostname. Before the fix the middleware returned a bare 403 for any cross-site request.
    """
    request = make_mocked_request(
        "GET",
        "/",
        headers={"Host": "93.184.216.34", "Sec-Fetch-Site": "cross-site"},
    )
    response = await middleware(request, ok_handler)
    assert response.status == 200


async def test_same_origin_to_loopback_is_allowed(middleware):
    """Normal same-origin use of a local deployment is unaffected."""
    request = make_mocked_request(
        "GET",
        "/",
        headers={"Host": "127.0.0.1:8188", "Sec-Fetch-Site": "same-origin"},
    )
    response = await middleware(request, ok_handler)
    assert response.status == 200
