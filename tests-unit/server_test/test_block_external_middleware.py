"""Tests for the --disable-api-nodes CSP middleware"""

import torch
from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import pytest  # noqa: E402
from aiohttp import web  # noqa: E402
from aiohttp.test_utils import make_mocked_request  # noqa: E402

from server import create_block_external_middleware  # noqa: E402

pytestmark = pytest.mark.asyncio


async def test_csp_allows_blob_media_src():
    middleware = create_block_external_middleware()

    async def handler(request):
        return web.Response()

    request = make_mocked_request("GET", "/")
    response = await middleware(request, handler)

    csp = response.headers["Content-Security-Policy"]
    directives = {d.strip().split(" ", 1)[0]: d.strip() for d in csp.split(";") if d.strip()}
    assert "media-src" in directives
    assert "blob:" in directives["media-src"]
