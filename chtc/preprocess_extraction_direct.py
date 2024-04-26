import argparse
import asyncio
import logging
import pickle

import tenacity
import weaviate
from db import Triplets, get_all_processed_ids, push
from dotenv import load_dotenv
from tqdm import tqdm

from text2graph.askxdd import get_weaviate_client
from text2graph.llm import ask_llm

logging.basicConfig(level=logging.ERROR)

load_dotenv()


@tenacity.retry(wait=tenacity.wait_fixed(5), stop=tenacity.stop_after_attempt(3))
async def extract(text: str, doc_id: str) -> str:
    """Extract json GraphOutput from text."""

    graph = await ask_llm(text, model="mixtral", hydrate=False, doc_ids=[doc_id])
    return graph.model_dump_json(exclude_unset=True)  # type: ignore


def process_paragraph(
    id: str, weaviate_client: weaviate.Client
) -> dict[str, str] | None:
    """Process extraction pipeline for a paragraph."""

    paragraph = weaviate_client.data_object.get_by_id(id)

    # Unpack useful fields
    text_content = paragraph["properties"]["text_content"]
    paper_id = paragraph["properties"]["paper_id"]
    hashed_text = paragraph["properties"]["hashed_text"]

    try:
        triplets = asyncio.run(extract(text_content, paper_id))
        output = {
            "id": id,
            "hashed_text": hashed_text,
            "paper_id": paper_id,
            "triplets": triplets,
        }
    except tenacity.RetryError:
        logging.error(f"Failed to extract {id}.")
        return

    logging.info(f"Extracted paragraph {id}: {output}")
    return output


def main(job_index: int = 0, batch_size: int = 2000):
    """Main function to process paragraphs."""

    weaviate_client = get_weaviate_client()

    # Get ids to process
    batch_start_idx = job_index * batch_size

    all_ids_pickle = "/staging/clo36/text2graph/preprocess/geoarchive_paragraph_ids.pkl"
    with open(all_ids_pickle, "rb") as f:
        all_ids = pickle.load(f)
    batch_ids = all_ids[batch_start_idx : batch_start_idx + batch_size]

    ## Remove processed
    processed = get_all_processed_ids()
    batch_ids = [id for id in batch_ids if id not in processed]

    for id in tqdm(batch_ids):
        try:
            out = process_paragraph(id, weaviate_client)
            if out is None:
                continue
            push([Triplets(**out)])

        except Exception as e:
            logging.error(f"Failed to process paragraph {id}: {e}")
            continue

    logging.info(f"Finished processing batch {job_index=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_index", type=int)
    main(**vars(parser.parse_args()))
