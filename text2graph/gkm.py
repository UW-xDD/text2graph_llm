import re
import numpy as np
from text2graph.schema import RelationshipTriples


TripletJson = dict[str, str | dict[str, str]]
GKMProperties = list[dict[str, str | dict[str, str]]]
MACROSTRAT_IGNORE_KEYS = [
    "strat_name_long", "strat_name", "t_units",
]
RANK_LOOKUP = {
    "Mbr": "Member",
    "Fm": "Formation",
    "Gp": "Group",
    "SGp": "Supergroup"
}


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
        if k == 'rank':
            try:
                type_value = RANK_LOOKUP[v]
            except KeyError:
                type_value = v
            gkm_properties.append(
                {
                    "rdf:type": "gsgu:" + type_value
                }
            )
            continue
        elif k[-3:] != "_id" and k not in MACROSTRAT_IGNORE_KEYS and v:
            gkm_properties.append(
                {
                    "gsoc:hasQuality": {
                        "xdd:" + k: {
                            "gsoc:hasValue": format_gkm_value(v)
                        }
                    }
                }
            )

    if predicate == "is_in":
        gkm_properties.append(
            {
                f"xdd:isIn": {
                    "gsoc:SpatialLocation": {
                        "gsoc:hasValue": format_gkm_value(object_dict['name'])
                    }
                }
            }
        )

    gkm_properties.append(
        {
            "gsoc:hasQuality": {
                "gsoc:SpatialLocation": {
                    "gsoc:hasValue": format_gkm_value(object_dict['name'])
                }
            }
        }
    )

    spatial_value_str = f"(POINT {format_gkm_value(object_dict['lat'])} {format_gkm_value(object_dict['lon'])})"
    gkm_properties.append(
        {
            "gsoc:hasValue": {
                "gsoc:SpatialLocation": {
                    "gsoc:hasValue": {
                        "gsoc:SpatialValue": spatial_value_str
                    }
                }
            }
        }
    )
    return gkm_properties


def format_gkm_value(v: int | float | str) -> str:
    """
    return str value with quotes for strs, and not for numbers
    :param v: object_dict value
    :return: str value with quotes for strs, and not for numbers
    """

    if isinstance(v, float) or isinstance(v, int):
        return str(v)
    if isinstance(v, str) and v.isnumeric():
        return str(v)
    else:
        return '"' + str(v) + '"'


def gkm_to_string(gkm_properties: GKMProperties, indent: int = 2) -> str:
    """
    Convert a list of GKM properties into a formatted string
    """
    end_of_line = "\n"
    result_str = ""
    indent_spacer = " " * indent
    for gkm_entry in gkm_properties:
        try:
            result_str += str(gkm_entry["name"]) + end_of_line
            continue
        except KeyError:
            pass

        try:
            property_str = indent_spacer + (nested_dict_to_str(gkm_entry)) + end_of_line
            result_str += property_str
        except TypeError as e:
            print(f"Unnesting error: {gkm_entry}, {e}")

    result_str = square_bracket_indentation(result_str, indent_spacer=indent_spacer)
    result_str = result_str.replace("]", "] ;")
    result_str += "\n."

    return result_str


def nested_dict_to_str(d) -> str:
    """
    Concat a nested dict's keys and values to a string
    """
    result_str = ""
    try:
        for k, v in d.items():
            if k in ["gsoc:hasQuality", "gsoc:hasValue", "gsoc:SpatialLocation", "gsoc:SpatialValue"]:
                result_str = k + " [ " + nested_dict_to_str(v) + " ] "
            else:
                result_str = k + " " + nested_dict_to_str(v)

    except (AttributeError, TypeError):
        result_str = d + " ;"
    return result_str


def square_bracket_indentation(property_str: str, indent_spacer: str) -> str:
    indented_str = ""
    indent_level = 1
    while len(property_str) > 0:
        open_bracket_hit = re.search(r"\[", property_str)
        close_bracket_hit = re.search(r"]", property_str)
        open_bracket_span = open_bracket_hit.span() if open_bracket_hit else (np.inf, np.inf)
        close_bracket_span = close_bracket_hit.span() if close_bracket_hit else (np.inf, np.inf)
        if open_bracket_span == close_bracket_span == (np.inf, np.inf):
            break
        if open_bracket_span[0] < close_bracket_span[0]:
            indented_str += property_str[:open_bracket_span[1]] + "\n"
            indent_level += 1
            indent = indent_spacer * indent_level
            property_str = indent + property_str[open_bracket_span[1]:]
        else:
            indented_str += property_str[:close_bracket_span[0]] + "\n"
            indent_level -= 1
            indent_level = max(0, indent_level)
            indent = indent_spacer * indent_level
            indented_str += indent + property_str[close_bracket_span[0]:close_bracket_span[1]]
            try:
                property_str = indent + property_str[close_bracket_span[1] + 1:]
            except IndexError:
                property_str = indent + property_str[close_bracket_span[1]:]

    return indented_str
