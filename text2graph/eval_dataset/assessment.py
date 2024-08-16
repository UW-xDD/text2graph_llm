import os
import requests
import dotenv

from text2graph.eval_dataset.schema import SQuADStratPipelineQuestion


dotenv.load_dotenv()
API_ENDPOINT = "http://cosmos0002.chtc.wisc.edu:4510/text_to_graph"
API_KEY = os.environ["API_KEY"]


def query_text_to_graph_api(text) -> dict[str, any]:
    """
    Perform triplet extaction with API call for given text
    :param text: text to extract triplets from
    :return: extracted triplets
    """
    headers = {"Content-Type": "application/json", "Api-Key": API_KEY}
    data = {
        "text": text,
        "model": "mixtral",
        "extraction_pipeline": "LOCATION_STRATNAME",
    }
    response = requests.post(API_ENDPOINT, headers=headers, json=data)
    response.raise_for_status()
    return response.json()


def response_subject_predicate_objects(
    response: dict[str, any],
) -> list[dict[str, str]]:
    """
    Extract subject, predicate and object from response
    :param response: response from text to graph API
    :return: list of string representation of subject, predicate and object
    """
    subject_predicate_object = []
    for t in response["triplets"]:
        subject = t["subject"]
        predicate = t["predicate"]
        object = t["object"]
        subject_predicate_object.append(
            {
                "subject": subject.get("name"),
                "predicate": predicate,
                "object": object.get("strat_name"),
            }
        )

    return subject_predicate_object


def question_best_exact_match(
    squad_question: SQuADStratPipelineQuestion,
    subject_predicate_object_strs: list[dict[str, str]],
) -> list[int]:
    """
    Return 1 if exact match score between squad questions and extracted triplets else 0
    returns 1 for impossible questions if no triplets are extracted.
    :param squad_questions: list of squad questions
    :param subject_predicate_object_strs: list of extracted triplets
    :return: 1 or 0, exact match found or not.
    """
    sq_dict = squad_question.model_dump()
    triplet_keys = ["subject", "predicate", "object"]

    if squad_question.impossible and not subject_predicate_object_strs:
        return 1

    for triplet in subject_predicate_object_strs:
        matches = []
        for triplet_key in triplet_keys:
            matches.append(triplet[triplet_key] == sq_dict[triplet_key])
        if all(matches):
            return 1
        else:
            pass
    return 0


def f1(s1: str, s2: str) -> float:
    """
    Compute F1 score between two strings
    :param s1: first string
    :param s2: second string
    :return: F1 score
    """
    s1_tokens = s1.split()
    s2_tokens = s2.split()
    common_tokens = set(s1_tokens).intersection(s2_tokens)
    try:
        precision = len(common_tokens) / len(s1_tokens)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = len(common_tokens) / len(s2_tokens)
    except ZeroDivisionError:
        recall = 0
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def question_best_mean_f1(
    squad_question: SQuADStratPipelineQuestion,
    subject_predicate_object_strs: list[dict[str, str]],
) -> dict[str, float]:
    """
    Compute best macro F1 score between squad question and API response extracted triplets
    :param squad_questions: squad question
    :param subject_predicate_object_strs: list of extracted triplets
    :return: F1 score
    """
    if squad_question.impossible:
        if not subject_predicate_object_strs:
            return 1
        else:
            return 0

    if not subject_predicate_object_strs:
        return 0

    comparisons = []
    sq_dict = squad_question.model_dump()
    triplet_keys = ["subject", "predicate", "object"]
    for spo in subject_predicate_object_strs:
        this_comparison = {}
        for triplet_key in triplet_keys:
            this_comparison[triplet_key] = f1(
                sq_dict.get(triplet_key, ""), spo.get(triplet_key, "")
            )
        comparisons.append(this_comparison)

    comparisons_means = [sum(x.values()) / len(x.values()) for x in comparisons]
    return max(comparisons_means)
