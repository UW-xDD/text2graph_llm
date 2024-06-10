def test_stratname_alignment(stratname_alignment_handler):
    expected_n = 45646
    assert stratname_alignment_handler.known_entity_embeddings.shape[0] == expected_n
    assert len(stratname_alignment_handler.known_entity_names) == expected_n


def test_get_closest_known_stratname(stratname_alignment_handler):
    stratname = "Abbey head bed."
    closest_stratname = stratname_alignment_handler.get_closest_known_entity(stratname)
    assert closest_stratname == "Abbey Head Bed"


def test_mineral_alignment(mineral_alignment_handler):
    expected_n = 6387
    assert mineral_alignment_handler.known_entity_embeddings.shape[0] == expected_n
    assert len(mineral_alignment_handler.known_entity_names) == expected_n


def test_get_closest_known_mineral(mineral_alignment_handler):
    mineral = "Gold"
    closest_mineral = mineral_alignment_handler.get_closest_known_entity(mineral)
    assert closest_mineral == "gold"
