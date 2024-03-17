from text2graph.schema import RelationshipTriples

TripletJson = dict[str, str | dict[str, str]]
GKMProperties = list[dict[str, str | dict[str, str]]]
MACROSTRAT_IGNORE_KEYS = [
    "strat_name_long", "strat_name", "t_units",
]


def triplet_to_gkm(triplet: RelationshipTriples | TripletJson) -> GKMProperties:
    """
    Convert a triplet object to a list of GKM properties
    """
    try:
        triplet_json = triplet.dict()
    except (AttributeError, TypeError):
        triplet_json = triplet

    subject_dict = triplet_json['subject']
    predicate = triplet_json['predicate']
    object_dict = triplet_json['object']

    gkm_properties = []
    subject_name = subject_dict['strat_name_long'].replace(' ', '')
    gkm_properties.append(
        {"name": f"xdd:{subject_name}"}
    )

    for k, v in subject_dict.items():
        if k[-3:] != "_id" and k not in MACROSTRAT_IGNORE_KEYS and v:
            gkm_properties.append(
                {"gsoc:hasQuality": {"xdd:" + k: {"hasValue:": v}}}
            )

    if predicate == "is_in":
        gkm_properties.append(
            {f"xdd:isIn": {"gsoc:SpatialLocation": {"gsoc:hasValue": object_dict['name']}}}
        )

    gkm_properties.append(
        {"gsoc:hasQuality": {"gsoc:SpatialLocation": {"gsoc:hasValue": object_dict['name']}}}
    )
    gkm_properties.append(
        {"gsoc:hasQuality": {"gsoc:SpatialLocation": {
            "gsoc:hasValue": {"gsocSpatialValue": f"(POINT {object_dict['lat']} {object_dict['lon']}"}}}}
    )

    return gkm_properties


def nested_dict_to_str(d) -> str:
    """
    Concat a nested dict's keys and values to a string
    """
    result_str = ""
    try:
        for k, v in d.items():
            result_str = k + " " + nested_dict_to_str(v)
    except (AttributeError, TypeError):
        result_str = d
    return result_str


def gkm_to_string(gkm_properties: GKMProperties, indent: int = 2) -> str:
    """
    Convert a list of GKM properties into a formatted string
    """
    end_of_line = " ;\n"
    result_str = ""
    for gkm_entry in gkm_properties:
        print(gkm_entry)
        try:
            result_str += gkm_entry["name"] + end_of_line
            continue
        except KeyError:
            pass
        try:
            result_str += nested_dict_to_str(gkm_entry) + end_of_line
        except TypeError:
            print(f"Unnesting error: {gkm_entry}")

    return result_str
