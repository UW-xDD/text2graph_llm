import requests

# We can use `TestClient`, but the relative import fails when the test is in the root/tests directory. I prefer not to place the test inside the api/ folder.
# To avoid dealing with the import issue, we need to spin up docker-compose to test this.
LOCAL_API_URL = "http://localhost:4510"


def test_health():
    response = requests.get(LOCAL_API_URL)
    assert response.status_code == 200
    assert "message" in response.json()


def test_text_to_graph(api_auth_header):
    response = requests.post(
        f"{LOCAL_API_URL}/text_to_graph",
        headers=api_auth_header,
        json={
            "text": "Smithville formation is in the United States.",
            "model": "mixtral",
        },
    )
    assert response.status_code == 200
    assert "triplets" in response.json()


def test_slow_llm_graph(api_auth_header):
    response = requests.post(
        f"{LOCAL_API_URL}/slow_llm_graph",
        headers=api_auth_header,
        json={
            "query": "Smithville formation",
            "top_k": 1,
            "ttl": True,
            "hydrate": False,
        },
    )
    assert response.status_code == 200
    assert "@prefix" in response.json()


def test_llm_graph(api_auth_header):
    response = requests.post(
        f"{LOCAL_API_URL}/llm_graph",
        headers=api_auth_header,
        json={
            "query": "Smithville formation",
            "top_k": 1,
            "ttl": True,
            "hydrate": False,
        },
    )
    assert response.status_code == 200
    assert "@prefix" in response.json()
