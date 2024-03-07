from text2graph.dict2ttl.entities import (
    GKMxDDEntity,
    RDFTYPE_LOOKUP,
    ttl_prefixes,
)


def gkm_xdd_entity_from_macrostrat_unit_dict(macrostrat_unit_dict: dict[str, str]) -> GKMxDDEntity:
    """
    process macrostrat unit into GKM
    :param macrostrat_unit_dict:
    """
    gxe = GKMxDDEntity(name=macrostrat_unit_dict["unit_name"])
    for k, v in RDFTYPE_LOOKUP.items():
        if macrostrat_unit_dict[k]:
            new_name = f"xdd:{macrostrat_unit_dict[k]}{k}"
            gxe.add_property(type="gsoc:isPartOf", item=new_name)

    if macrostrat_unit_dict["lith"]:
        lith_name = macrostrat_unit_dict['lith'][0]['name']
        gxe.add_lithology(lith_name)

    if macrostrat_unit_dict["clat"] and macrostrat_unit_dict["clng"]:
        gxe.add_location_coords(
            lat=float(macrostrat_unit_dict["clat"]),
            lon=float(macrostrat_unit_dict["clng"]),
            reference_system_name="WGS 84"
        )

    return gxe


def gkm_xdd_entity_from_geolocation_dict(geolocation_dict) -> GKMxDDEntity:
    for location, strat_names in geolocation_dict.items():
        pass

def final_gkm_file(gkm_xdd_entities: list[GKMxDDEntity]) -> str:
    final = ttl_prefixes()
    for gkm_xdd_entity in gkm_xdd_entities:
        final += gkm_xdd_entity.full_gkm_str()

    return final
