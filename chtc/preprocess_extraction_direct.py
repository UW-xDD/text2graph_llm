import argparse
import asyncio
import logging
import pickle
import re

import db
import vllm
from tqdm import tqdm

from text2graph.alignment import AlignmentHandler
from text2graph.askxdd import get_weaviate_client
from text2graph.llm import post_process
from text2graph.prompt import PromptHandlerV3
from text2graph.schema import Provenance


def get_paragraph_ids(job_index: int, batch_size: int, ids_pickle: str) -> list[str]:
    """Get a list of paragraph ids from Weaviate."""

    with open(ids_pickle, "rb") as f:
        all_ids = pickle.load(f)
    batch_start_idx = job_index * batch_size
    batch_ids = all_ids[batch_start_idx : batch_start_idx + batch_size]

    processed = db.get_all_processed_ids(job_index=job_index, max_size=batch_size)
    return [id for id in batch_ids if id not in processed]


class BatchInferenceRunner:
    """Batch inference runner for extracting triplets from paragraphs using vllm."""

    def __init__(
        self,
        id_pickle: str,
        batch_size: int = 2000,
    ):
        self.id_pickle = id_pickle
        # Do not change across runs, it will mess up indexing
        self.batch_size = batch_size

        # Infrastructure
        self.weaviate_client = get_weaviate_client()
        self.prompt_handler = PromptHandlerV3()
        self.alignment_handler = AlignmentHandler.load()
        self.llm = vllm.LLM(
            model="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
            dtype="float16",
            tensor_parallel_size=1,
            enforce_eager=True,
        )
        self.sampling_params = vllm.SamplingParams(
            temperature=0, max_tokens=2048, stop=["[/INST]", "[INST]"]
        )
        self.mixtral_prompt_template = "<s> [INST] {system} {user} [/INST] Model answer</s> [INST] Reply the output json only, do not provide any explanation or notes. [/INST]"

    def run(self, job_index: int, mini_batch_size: int = 200) -> None:
        """Run the job in mini-batches."""

        batch_ids = get_paragraph_ids(
            job_index, self.batch_size, ids_pickle=self.id_pickle
        )

        # Mini-batching
        db_objects = []
        while len(batch_ids) > 0:
            n_in_batch = min(mini_batch_size, len(batch_ids))
            mini_batch_ids = [batch_ids.pop() for _ in range(n_in_batch)]

            # Intermediate outputs contain raw llm outputs
            intermediate_outputs = self.process_mini_batch(mini_batch_ids)

            # Outputs contain post-processed triplets (with provenance)
            outputs = self.post_process_with_prov(**intermediate_outputs)

            # Create database ORM objects.
            db_objects.extend(
                [db.Triplets(**output, job_id=job_index) for output in outputs]
            )

            # Push to database
            db.push(db_objects)

    def process_mini_batch(self, ids: list[str]) -> dict:
        """Process a mini-batch to produce raw output with meta-data."""

        # Get prompts and metadata
        hashed_texts, paper_ids, prompts = [], [], []

        for id in tqdm(ids, desc="Created prompts"):
            paragraph = self.weaviate_client.data_object.get_by_id(
                id, class_name="Paragraph"
            )
            if not paragraph:
                logging.error(f"Paragraph {id} not found.")
                continue

            text = paragraph["properties"]["text_content"]
            messages = self.prompt_handler.get_gpt_messages(text)
            prompt = self.mixtral_prompt_template.format(
                system=messages[0]["content"], user=messages[1]["content"]
            )
            prompts.append(prompt)
            hashed_texts.append(paragraph["properties"]["hashed_text"])
            paper_ids.append(paragraph["properties"]["paper_id"])

        # Generate LLM outputs
        llm_outputs = self.llm.generate(prompts, self.sampling_params)
        raw_outputs = [output.outputs[0].text.strip() for output in llm_outputs]
        return {
            "ids": ids,
            "paper_ids": paper_ids,
            "hashed_texts": hashed_texts,
            "raw_outputs": raw_outputs,
        }

    def post_process_with_prov(
        self,
        ids: list[str],
        paper_ids: list[str],
        hashed_texts: list[str],
        raw_outputs: list[str],
    ) -> list[dict]:
        """Post processing with provenance."""

        outputs = []
        for id, paper_id, hashed_text, raw_output in zip(
            ids, paper_ids, hashed_texts, raw_outputs
        ):
            vllm_prov = Provenance(
                source_name="vllm",
                source_version=self.llm.llm_engine.model_config.__dict__["model"],
                additional_values={
                    "temperature": self.sampling_params.temperature,
                    "paragraph_id": id,
                    "doc_ids": [paper_id],
                },
            )

            # vllm-specific clean up and json conversion
            raw_output = raw_output.replace("\n", "").replace("\\", "")
            raw_output = re.sub(r"\}[^}]*$", "}", raw_output)

            try:
                triplets = asyncio.run(
                    post_process(
                        raw_llm_output=raw_output,
                        prompt_handler=self.prompt_handler,
                        alignment_handler=self.alignment_handler,
                        hydrate=False,
                        provenance=vllm_prov,
                    )
                )

                # Convert to plain json
                triplets = triplets.model_dump_json(exclude_unset=True)

                output = {
                    "id": id,
                    "hashed_text": hashed_text,
                    "paper_id": paper_id,
                    "triplets": triplets,
                }
                outputs.append(output)

            except Exception as e:
                logging.error(
                    f"Error post-processing paragraph {id}: {e}, {raw_output}"
                )
                pass

        return outputs


def main(
    id_pickle: str, batch_size: int, job_index: int, mini_batch_size: int, debug: bool
):
    logging_level = logging.DEBUG if debug else logging.ERROR
    logging.basicConfig(level=logging_level)
    runner = BatchInferenceRunner(id_pickle=id_pickle, batch_size=batch_size)
    return runner.run(job_index=job_index, mini_batch_size=mini_batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id_pickle", type=str, required=True)
    parser.add_argument("--job_index", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--mini_batch_size", type=int, default=200)
    parser.add_argument("--debug", action="store_true")

    outputs = main(**vars(parser.parse_args()))
