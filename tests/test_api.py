import requests

# We can use `TestClient`, but the relative import fails when the test is in the root/tests directory. I prefer not to place the test inside the api/ folder.
# To avoid dealing with the import issue, we need to spin up docker-compose to test this.
LOCAL_API_URL = "http://localhost:4510"


def test_health():
    response = requests.get(LOCAL_API_URL)
    assert response.status_code == 200
    assert "message" in response.json()


def test_text_to_graph_strat(api_auth_header, pipeline="Location to Stratigraphy"):
    response = requests.post(
        f"{LOCAL_API_URL}/text_to_graph",
        headers=api_auth_header,
        json={
            "text": "Smithville formation is in the United States.",
            "model": "mixtral",
            "extraction_pipeline": pipeline,
        },
    )
    assert response.status_code == 200
    assert "triplets" in response.json()


def test_text_to_graph_mineral(api_auth_header, pipeline="Location to Mineral"):
    response = requests.post(
        f"{LOCAL_API_URL}/text_to_graph",
        headers=api_auth_header,
        json={
            "text": "Smithville formation is in the United States.",
            "model": "mixtral",
            "extraction_pipeline": pipeline,
        },
    )
    assert response.status_code == 200
    assert "triplets" in response.json()


def test_search_to_graph_slow_strat(
    api_auth_header, pipeline="Location to Stratigraphy"
):
    response = requests.post(
        f"{LOCAL_API_URL}/search_to_graph_slow",
        headers=api_auth_header,
        json={
            "query": "Smithville formation",
            "top_k": 1,
            "ttl": True,
            "hydrate": False,
            "extraction_pipeline": pipeline,
        },
    )
    assert response.status_code == 200
    assert "@prefix" in response.json()[0]


def test_search_to_graph_slow_mineral(api_auth_header, pipeline="Location to Mineral"):
    response = requests.post(
        f"{LOCAL_API_URL}/search_to_graph_slow",
        headers=api_auth_header,
        json={
            "query": "Critical mineral in the USA.",
            "top_k": 1,
            "ttl": True,
            "hydrate": False,
            "extraction_pipeline": pipeline,
        },
    )
    assert response.status_code == 200
    print(response.json()[0])
    assert "@prefix" in response.json()[0]


def test_search_to_graph_fast(api_auth_header, pipeline="Location to Stratigraphy"):
    response = requests.post(
        f"{LOCAL_API_URL}/search_to_graph_fast",
        headers=api_auth_header,
        json={
            "query": "Smithville formation",
            "top_k": 1,
            "ttl": True,
            "hydrate": False,
            "extraction_pipeline": pipeline,
        },
    )
    assert response.status_code == 200
    assert "@prefix" in response.json()[0]
