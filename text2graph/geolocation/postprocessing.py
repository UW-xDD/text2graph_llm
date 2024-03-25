from text2graph.geolocation.macrostrat import StratNameGPSLookup
from text2graph.geolocation.serpapi import serpapi_location_result


def geolocate_model_graph_extraction(
    extracted_graphs: list[dict[str, str | list[str]]],
) -> list[dict[str, str | list[str] | dict[str, list[tuple[float, float]]]]]:
    """
    use external APIs to geolocate extracted entities
    :param extracted_graphs:
    :return:
    """
    sns = StratNameGPSLookup()
    for loc_dct in extracted_graphs:
        loc_dct["entity_coords"] = {}
        # geolocation lookup API
        name = loc_dct["name"]
        p_name = serpapi_location_result(q=name)
        if p_name:
            loc_dct["entity_coords"][name] = (p_name.lat, p_name.lon)

        # macrostrat stratname location lookup
        for i, strat_name in enumerate(loc_dct["stratigraphic_units"]):
            p = sns(strat_name)
            if p:
                loc_dct["entity_coords"][strat_name] = (p.lat, p.lon)

    return extracted_graphs
