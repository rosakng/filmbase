import json

import pytest
from app import app
from chalice.config import Config
from chalice.local import LocalGateway


@pytest.fixture
def gateway_factory():
    def create_gateway(config=None):
        if config is None:
            config = Config()
        return LocalGateway(app, config)

    return create_gateway


def test_post_user_input_request(gateway_factory):
    gateway = gateway_factory()

    request = {
        "1": {
            "title": "Breakfast Club, The",
            "rating": 5
        },
        "2": {
            "title": "Toy Story",
            "rating": 3.5
        },
        "3": {
            "title": "Jumanji",
            "rating": 2
        },
        "4": {
            "title": "Pulp Fiction",
            "rating": 5
        },
        "5": {
            "title": "Akira",
            "rating": 4.5
        }
    }
    response = gateway.handle_request(
        method="POST",
        path=f"/v1/filmbase/results/reccs",
        headers={
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST,PATCH,GET,DELETE,OPTIONS",
            "Access-Control-Max-Age": "600",
        },
        body=json.dumps(request),
    )
    print(response)
    assert False
