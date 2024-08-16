import json
import httpx
import logging
from pathlib import Path
from rich.progress import Progress
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
XDD_V1_BASE_URL = "https://xdd.wisc.edu/api/v1/"


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_api_response(client, url):
    r = client.get(url)
    r.raise_for_status()
    return r.json()


def all_strat_names_in_all_docids_for_dataset(dataset: str) -> dict[str, list[str]]:
    """
    Get all informal strat names in a docid.
    """
    route = "/articles"
    params = {
        "dataset": dataset,
        "full_results": "true",
        "known_terms": "stratigraphic_names",
        "known_entities": "stratname_candidates",
    }
    param_str = "&".join([f"{k}={v}" for k, v in params.items()])
    url = f"{XDD_V1_BASE_URL}{route}?{param_str}"
    strat_names = {}
    transport = httpx.HTTPTransport(retries=1)
    timeout_seconds = 30
    with httpx.Client(timeout=timeout_seconds, transport=transport) as client:
        with Progress() as progress:
            response_json = get_api_response(client, url)
            expected_article_count = response_json["success"]["hits"]
            get_paginated_stratnames = progress.add_task(
                "get_paginated_stratnames", total=expected_article_count
            )
            counter = 1
            while url:
                for doc in response_json["success"]["data"]:
                    counter += 1
                    progress.update(get_paginated_stratnames, advance=1)
                    docid = doc["_gddid"]
                    strat_names[docid] = {}
                    try:
                        strat_names[docid]["stratnames"] = doc["known_terms"][0][
                            "stratigraphic_names"
                        ]
                    except (KeyError, IndexError):
                        strat_names[docid]["stratnames"] = []

                    try:
                        strat_names[docid]["stratname_candidates"] = doc[
                            "known_entities"
                        ]["stratname_candidates"][0]["candidates"]
                    except (KeyError, IndexError):
                        strat_names[docid]["stratname_candidates"] = []
                try:
                    url = response_json["success"]["next_page"]
                    if url:
                        response_json = get_api_response(client, url)
                    else:
                        url = None
                except (KeyError, httpx.UnsupportedProtocol, RetryError) as e:
                    if not isinstance(
                        e, KeyError
                    ):  # expect next_page missing and KeyError at last page of API results
                        logger.warning(
                            f"Error fetching xdd article strat_names {url=}\n{e}"
                        )
                    url = None

    if counter < expected_article_count:
        logger.warning(
            f"Expected {expected_article_count} articles, but only got {counter}"
        )
    return strat_names


def get_cached_strat_names_for_dataset(
    dataset: str, datafolder: Path | None = None, clobber: bool = False
) -> dict[str, list[str]]:
    """
    Get all informal strat names in a docid.
    """
    if not datafolder:
        datafolder = Path("./data")

    datafolder.mkdir(exist_ok=True, parents=True)

    datapath = datafolder / f"{dataset}_strat_names.json"
    if datapath.exists() and not clobber:
        with open(datapath, "r") as f:
            return json.load(f)

    strat_names = all_strat_names_in_all_docids_for_dataset(dataset=dataset)
    total_docids = len(strat_names)
    total_extracted_stratnames_and_candidates = sum(
        [
            len(v["stratnames"]) + len(v["stratname_candidates"])
            for v in strat_names.values()
        ]
    )
    print(
        f"extracted {total_extracted_stratnames_and_candidates} stratnames and candidates from {total_docids} docids"
    )
    with open(datapath, "w") as f:
        json.dump(strat_names, f)

    return strat_names


def merge_strat_names_and_candidates(
    d: dict[str, dict[str, list[str]]],
) -> dict[str, list[str]]:
    merged = {}
    for docid, stratname_dict in d.items():
        stratnames = stratname_dict.get("stratnames", [])
        candidates = stratname_dict.get("stratname_candidates", [])
        merged[docid] = stratnames + candidates
    return merged


def informal_strat_names(stratname: str) -> list[str]:
    """
    Extract informal strat names from a string.
    :param stratname: A stratigraphic name.
    :return: A list of informal strat names.
    """
    strat_words = stratname.split()[:-1]
    strat_words_count = len(strat_words)
    return [
        " ".join(strat_words[strat_words_count - x :])
        for x in range(1, strat_words_count + 1)
    ]


def add_informal_strat_names(doc_id_strat_names_dct: dict[str, list[str]]) -> list[str]:
    """
    Get all informal strat names in a docid.
    """
    lol_informal_strat_names_dct = {
        docid: [informal_strat_names(stratname) for stratname in strat_names]
        for docid, strat_names in doc_id_strat_names_dct.items()
    }
    return {
        docid: doc_id_strat_names_dct[docid] + [y for x in lol for y in x]
        for docid, lol in lol_informal_strat_names_dct.items()
    }
