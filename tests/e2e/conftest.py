"""Start a real toolkit API process for each user journey."""

import pytest
from api_process import ApiProcess


@pytest.fixture
def api_service(tmp_path, request):
    service = ApiProcess(tmp_path)
    service.env.update(getattr(request, "param", {}))
    try:
        service.start()
        yield service
    finally:
        service.stop()
