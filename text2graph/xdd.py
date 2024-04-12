import os
import logging

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, AnyUrl
from text2graph.schema import Provenance


load_dotenv()


class Paragraph(BaseModel):
    """class for Retriever results"""

    paper_id: str
    preprocessor_id: str
    doc_type: str
    topic_list: list[str]
    text_content: str
    hashed_text: str
    cosmos_object_id: str | None
    distance: float
    url: AnyUrl
    provenance: Provenance


class USGSRetriever:
    """This is a mockup for the USGS specific retriever."""

    def query_ask_xdd(self, query: str) -> Paragraph:
        """Query the AskXDD API and return the response."""

        ASK_XDD_APIKEY = os.getenv("ASK_XDD_APIKEY")
        ASK_XDD_URL = os.getenv("ASK_XDD_URL")
        headers = {"Content-Type": "application/json", "Api-Key": ASK_XDD_APIKEY}
        data = {
            "topic": "criticalmaas",
            "question": query,
            "top_k": 1,
        }

        response = requests.post(ASK_XDD_URL + "/hybrid", headers=headers, json=data)
        response.raise_for_status()
        paragraph = response.json()[0]
        paragraph["url"] = self.get_url(paragraph["paper_id"])
        source_version = None
        try:
            source_version = response.json()[0]["version"]
        except KeyError:
            pass
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
        paragraph["provenance"] = Provenance(
            source_name="Ask_xDD_hybrid_API",
            source_url=ASK_XDD_URL,
            source_version=source_version,
            additional_values=selected_keys_dict,
        )
        return Paragraph(**paragraph)

    @staticmethod
    def get_url(paper_id: str) -> str | None:
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
