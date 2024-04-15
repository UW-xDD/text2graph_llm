from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pairwise_cos_sim


class AlignmentHandler:
    def __init__(
        self,
        known_entity_names: list[str],
        known_entity_embeddings: np.ndarray | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.model_name = model_name
        self.known_entity_names = known_entity_names
        self.known_entity_embeddings = known_entity_embeddings
        self.model = SentenceTransformer(model_name, device="cpu")

        # Instantiate from scratch
        if self.known_entity_embeddings is None:
            self.known_entity_embeddings = self.model.encode(self.known_entity_names)

        assert len(self.known_entity_names) == len(self.known_entity_embeddings)

    def get_closest_known_entity(self, name: str, threshold: float = 0.95) -> str:
        """Get the closest known entity to a given name or return itself if not found."""

        x = self.model.encode([name])
        similarity = pairwise_cos_sim(x, self.known_entity_embeddings)
        idx_closest = np.argmax(similarity)

        if similarity[idx_closest] < threshold:
            return name
        return self.known_entity_names[idx_closest]

    def save(self, path: str | Path | None = None) -> None:
        """Save handler to disk."""

        if path is None:
            path = Path(f"data/known_entity_embeddings/{self.model_name}")

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
    def load(cls, path: str | Path) -> "AlignmentHandler":
        """Load handler from disk."""

        if isinstance(path, str):
            path = Path(path)

        # Load model name
        with open(path / "model.txt", "r") as f:
            model_name = f.readline().strip()

        # Load known entities
        with open(path / "known_entity_names.txt", "r") as f:
            known_entity_names = [line.strip() for line in f.readlines()]

        # Load known entity embeddings
        known_entity_embeddings = np.load(path / "known_entity_embeddings.npz")[
            "embeddings"
        ]

        return cls(
            model_name=model_name,
            known_entity_names=known_entity_names,
            known_entity_embeddings=known_entity_embeddings,
        )
