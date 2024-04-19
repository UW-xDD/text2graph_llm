import logging
import os

import requests
from dotenv import load_dotenv

from text2graph.schema import Paragraph, Provenance

load_dotenv()


class Retriever:
    """Retrieve text from ask-xdd endpoint."""

    ASK_XDD_URL = os.getenv("ASK_XDD_URL")

    def __init__(self, topic: str = "criticalmaas", version: str = "0.0.3"):
        self.topic = topic
        self.version = version

    def query(self, query: str, top_k: int) -> list[Paragraph]:
        """Query the AskXDD API and return the response."""

        headers = {
            "Content-Type": "application/json",
            "Api-Key": os.getenv("ASK_XDD_APIKEY"),
        }
        data = {
            "topic": self.topic,
            "question": query,
            "top_k": top_k,
        }

        response = requests.post(
            f"{self.ASK_XDD_URL}/hybrid", headers=headers, json=data
        )
        response.raise_for_status()
        paragraphs = response.json()
        for paragraph in paragraphs:
            paragraph["url"] = self.get_url(paragraph["paper_id"])
            paragraph["provenance"] = self.get_provenance(paragraph)
        return [Paragraph(**p) for p in paragraphs]

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

    def get_provenance(self, paragraph: dict) -> Provenance:
        # Ask-xDD versioning, not implemented yet. TODO: Implement versioning in upstream ask-xDD API.
        source_version = None
        try:
            source_version = paragraph["version"]
        except KeyError:
            source_version = self.version  # Hardcoded current version as of Apr 2024

        selected_keys_dict = {}
        selected_keys = [
            "paper_id",
            "url",
            "preprocessor_id",
            "doc_type",
            "topic_list",
            "hashed_text",
            "cosmos_object_id",
            "distance",
        ]
        for x in selected_keys:
            try:
                selected_keys_dict[x] = paragraph[x]
            except KeyError:
                pass

        return Provenance(
            source_name="Ask_xDD_hybrid_API",
            source_url=self.ASK_XDD_URL,
            source_version=source_version,
            additional_values=selected_keys_dict,
        )
