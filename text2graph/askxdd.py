import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()


class Retriever:
    """Retrieve text from ask-xdd endpoint."""

    def query(self, query: str, top_k: int) -> dict:
        """Query the AskXDD API and return the response."""

        ASK_XDD_APIKEY = os.getenv("ASK_XDD_APIKEY")
        ASK_XDD_URL = os.getenv("ASK_XDD_URL")
        headers = {"Content-Type": "application/json", "Api-Key": ASK_XDD_APIKEY}
        data = {
            "topic": "criticalmaas",
            "question": query,
            "top_k": top_k,
        }

        response = requests.post(f"{ASK_XDD_URL}/hybrid", headers=headers, json=data)
        response.raise_for_status()
        paragraphs = response.json()
        for paragraph in paragraphs:
            paragraph["url"] = self.get_url(paragraph["paper_id"])
        return paragraphs

    def get_url(self, paper_id: str) -> str | None:
        """Get the URL for a paper in the XDD database."""

        XDD_ARTICLE_ENDPOINT = os.getenv("XDD_ARTICLE_ENDPOINT")
        response = requests.get(f"{XDD_ARTICLE_ENDPOINT}?docid={paper_id}")
        response.raise_for_status()

        try:
            data = response.json()["success"]["data"]
            # Return the first publisher link
            for d in data:
                links = d["link"]
                for link in links:
                    if link["type"] == "publisher":
                        return link["url"]
            return links
        except Exception as e:
            logging.error(f"Error getting URL for paper {paper_id}: {e}")
