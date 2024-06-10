import argparse
import asyncio
import logging
import pickle
import re
import time

import db
import vllm
from tqdm import tqdm

from text2graph.alignment import AlignmentHandler
from text2graph.askxdd import get_weaviate_client
from text2graph.llm import post_process
from text2graph.prompt import StratPromptHandlerV3
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
        self.infrastructure_loaded = False

    def load_infrastructure(self):
        """Load the infrastructure for the runner."""

        # Delay loading of the infrastructure to allow quick fail (e.g., batch already processed)
        self.weaviate_client = get_weaviate_client()
        self.prompt_handler = StratPromptHandlerV3()
        self.alignment_handler = AlignmentHandler.load(
            name="all-MiniLM-L6-v2", device="cuda"
        )
        self.llm = vllm.LLM(
            model="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
            dtype="float16",
            tensor_parallel_size=1,
            enforce_eager=True,
            disable_custom_all_reduce=True,
        )
        self.sampling_params = vllm.SamplingParams(
            temperature=0, max_tokens=2048, stop=["[/INST]", "[INST]"]
        )
        self.mixtral_prompt_template = "<s> [INST] {system} {user} [/INST] Model answer</s> [INST] Reply the output json only, do not provide any explanation or notes. [/INST]"
        self.infrastructure_loaded = True

    def run(self, job_index: int, mini_batch_size: int = 200) -> None:
        """Run the job in mini-batches."""

        batch_ids = get_paragraph_ids(
            job_index, self.batch_size, ids_pickle=self.id_pickle
        )

        # Faster exit condition
        if not batch_ids:
            logging.info(f"Batch {job_index} already processed.")
            return

        if not self.infrastructure_loaded:
            self.load_infrastructure()

        # Mini-batching
        while len(batch_ids) > 0:
            logging.info(
                f"Remaining {len(batch_ids)} paragraphs to process in this batch."
            )
            n_in_batch = min(mini_batch_size, len(batch_ids))
            mini_batch_ids = [batch_ids.pop() for _ in range(n_in_batch)]

            # Intermediate outputs contain raw llm outputs
            intermediate_outputs = self.process_mini_batch(mini_batch_ids)

            # Outputs contain post-processed triplets (with provenance)
            outputs = self.post_process_with_prov(**intermediate_outputs)

            # Create unprocessed (failed) ids in DB and push
            processed_ids = [output["id"] for output in outputs]
            unprocessed_ids = [id for id in mini_batch_ids if id not in processed_ids]
            for id in unprocessed_ids:
                outputs.append(
                    dict(
                        id=id,
                        hashed_text="NA",
                        paper_id="NA",
                        triplets="NA",
                    )
                )

            # Push to database
            db.push(outputs, job_id=job_index)

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
        for id, paper_id, hashed_text, raw_output in tqdm(
            zip(ids, paper_ids, hashed_texts, raw_outputs),
            desc="Post-processing",
        ):
            t0 = time.perf_counter()
            vllm_prov = Provenance(
                source_name="vllm",
                source_version=self.llm.llm_engine.model_config.__dict__["model"],
                additional_values={
                    "temperature": self.sampling_params.temperature,
                    "paragraph_id": id,
                    "doc_ids": [paper_id],
                },
            )
            t1 = time.perf_counter()
            # vllm-specific clean up and json conversion
            raw_output = raw_output.replace("\n", "").replace("\\", "")
            raw_output = re.sub(r"\}[^}]*$", "}", raw_output)

            t2 = time.perf_counter()

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
            t3 = time.perf_counter()

            logging.debug(
                f"Time taken: {t3 - t0:.2f}s (prov: {t1-t0:.2f}s, regex cleanup: {t2-t1:.2f}s, alignment: {t3-t2:.2f}s)"
            )

        return outputs


def main(
    id_pickle: str,
    batch_size: int,
    job_index_start: int,
    job_index_end: int,
    mini_batch_size: int,
    debug: bool,
):
    logging_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=logging_level)
    runner = BatchInferenceRunner(id_pickle=id_pickle, batch_size=batch_size)

    for job_index in range(job_index_start, job_index_end):
        runner.run(job_index=job_index, mini_batch_size=mini_batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id_pickle", type=str, required=True)
    parser.add_argument("--job_index_start", type=int, required=True)
    parser.add_argument("--job_index_end", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--mini_batch_size", type=int, default=100)
    parser.add_argument("--debug", action="store_true")

    outputs = main(**vars(parser.parse_args()))
