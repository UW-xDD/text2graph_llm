from functools import cache
from importlib.resources import files
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pairwise_cos_sim

from text2graph.macrostrat import EntityType, get_all_mineral_names, get_all_strat_names
from text2graph.usgs import CRITICAL_MINERALS


class AlignmentHandler:
    def __init__(
        self,
        entity_type: EntityType,
        known_entity_names: list[str],
        known_entity_embeddings: np.ndarray | None = None,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self.entity_type = entity_type
        self.known_entity_names = known_entity_names
        self.known_entity_embeddings = known_entity_embeddings
        self.model_name = model_name

        self.model = SentenceTransformer(model_name, device=device)

        # Instantiate from scratch
        if self.known_entity_embeddings is None:
            self.known_entity_embeddings = self.model.encode(self.known_entity_names)

        assert len(self.known_entity_names) == len(self.known_entity_embeddings)

    @property
    def default_save_path(self) -> Path:
        return Path(
            f"text2graph/binaries/known_entity_embeddings/{self.entity_type.value}/{self.model_name}"
        )

    def save(self, path: str | Path | None = None) -> None:
        """Save handler to disk."""

        if path is None:
            path = self.default_save_path

        if isinstance(path, str):
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)

        # Purge existing files
        for file in path.glob("*"):
            file.unlink()

        # Save model name
        with open(path / "model.txt", "w") as f:
            f.write(self.model_name + "\n")

        # Save known entities
        assert self.known_entity_names is not None
        with open(path / "known_entity_names.txt", "w") as f:
            for name in self.known_entity_names:
                f.write(name + "\n")

        # Save known entity embeddings
        assert self.known_entity_embeddings is not None
        np.savez(
            path / "known_entity_embeddings.npz",
            embeddings=self.known_entity_embeddings,
        )

    @classmethod
    def load(
        cls,
        entity_type: EntityType,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> "AlignmentHandler":
        """Load handler from disk."""

        path = (
            files("text2graph.binaries.known_entity_embeddings")
            / entity_type.value
            / model_name
        )
        model_name_file = path / "model.txt"
        known_entity_names_file = path / "known_entity_names.txt"
        known_entity_embeddings_file = str(path / "known_entity_embeddings.npz")

        # Load model name
        with open(model_name_file, "r") as f:  # type: ignore
            model_name = f.readline().strip()

        # Load known entities
        with open(known_entity_names_file, "r") as f:  # type: ignore
            known_entity_names = [line.strip() for line in f.readlines()]

        # Load known entity embeddings
        known_entity_embeddings = np.load(known_entity_embeddings_file)["embeddings"]

        return cls(
            entity_type=entity_type,
            known_entity_names=known_entity_names,
            known_entity_embeddings=known_entity_embeddings,
            model_name=model_name,
            device=device,
        )

    def get_closest_known_entity(self, name: str, threshold: float = 0.95) -> str:
        """Get the closest known entity to a given name or return itself if not found."""

        x = self.model.encode([name])
        similarity = pairwise_cos_sim(x, self.known_entity_embeddings)  # type: ignore
        idx_closest = np.argmax(similarity)

        if similarity[idx_closest] < threshold:
            return name
        return self.known_entity_names[idx_closest]

    @property
    def version(self) -> str:
        return "v1"


def _generate_known_entity_embeddings() -> None:
    """Generate all known entity embeddings for alignment."""

    # Stratigraphic names
    AlignmentHandler(
        entity_type=EntityType.STRAT_NAME, known_entity_names=get_all_strat_names()
    ).save()

    # Minerals
    macrostrat_minerals = get_all_mineral_names()
    usgs_minerals = CRITICAL_MINERALS
    all_minerals = list(set(macrostrat_minerals + usgs_minerals))
    AlignmentHandler(
        entity_type=EntityType.MINERAL, known_entity_names=all_minerals
    ).save()


@cache
def get_alignment_handler(entity_type: EntityType | str) -> AlignmentHandler:
    """Get alignment handler for a given entity type."""

    if isinstance(entity_type, str):
        entity_type = EntityType(entity_type)
    return AlignmentHandler.load(entity_type=entity_type)
