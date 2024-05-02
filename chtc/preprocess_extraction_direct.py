import argparse
import pickle
import logging
import asyncio
from tqdm import tqdm
from text2graph.prompt import PromptHandlerV3, PromptHandler
from text2graph.alignment import AlignmentHandler
from text2graph.askxdd import get_weaviate_client
from text2graph.llm import post_process
from text2graph.schema import Provenance
import vllm
import db
import re
logging.basicConfig(level=logging.ERROR)



def post_process_with_prov(
    ids: list[str],
    paper_ids: list[str],
    hashed_texts: list[str],
    raw_outputs: list[str],
    llm: vllm.LLM,
    prompt_handler: PromptHandler,
    alignment_handler: AlignmentHandler,
    sampling_params: vllm.SamplingParams,
) -> list[dict]:
    """Post processing with provenance."""

    outputs = []
    for id, paper_id, hashed_text, raw_output in zip(
        ids, paper_ids, hashed_texts, raw_outputs
    ):
        vllm_prov = Provenance(
            source_name="vllm",
            source_version=llm.llm_engine.model_config.__dict__["model"],
            additional_values={
                "temperature": sampling_params.temperature,
                "paragraph_id": id,
                "doc_ids": [paper_id],
            },
        )

        # vllm-specific clean up and json conversion
        raw_output = raw_output.replace("\n", "").replace("\\", "")
        raw_output = re.sub(r'\}[^}]*$', '}', raw_output)
                                        
        try:
            triplets = asyncio.run(
                post_process(
                    raw_output,
                    prompt_handler,
                    alignment_handler,
                    hydrate=False,
                    provenance=vllm_prov,
                )
            )
            this_output = {
                "id": id,
                "hashed_text": hashed_text,
                "paper_id": paper_id,
                "triplets": triplets,
            }
            outputs.append(this_output)

        except Exception as e:
            logging.error(f"Error post-processing paragraph {id}: {e}, {raw_output}")
    return outputs




def main(
    job_index: int, batch_size: int = 2000, mini_batch_size: int = 200
) -> list[str]:
    """Get a batch of prompts for benchmarking."""

    # Infrastructure
    weaviate_client = get_weaviate_client()
    prompt_handler = PromptHandlerV3()
    llm = vllm.LLM(
        model="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
        dtype="float16",
        tensor_parallel_size=1,
        enforce_eager=True,
    )
    alignment_handler = AlignmentHandler.load()

    # Mixtral settings
    MIXTRAL_TEMPLATE = "<s> [INST] {system} {user} [/INST] Model answer</s> [INST] Reply the output json only, do not provide any explanation or notes. [/INST]"
    sampling_params = vllm.SamplingParams(
        temperature=0, max_tokens=2048, stop=["[/INST]", "[INST]"]
    )

    # Get batch_ids
    all_ids_pickle = "/root/geoarchive_paragraph_ids.pkl"
    with open(all_ids_pickle, "rb") as f:
        all_ids = pickle.load(f)
    batch_start_idx = job_index * batch_size
    batch_ids = all_ids[batch_start_idx : batch_start_idx + batch_size]

    # Get processed ids
    # processed = db.get_all_processed_ids(job_index=job_index, max_size=batch_size)
    # batch_ids = [id for id in batch_ids if id not in processed]

    # Process a mini batch
    def process_mini_batch(ids: list[str]) -> list[dict]:
        """Process a batch of ids."""

        # Get prompts and metadata
        hashed_texts, paper_ids, prompts = [], [], []

        for id in tqdm(ids, desc="Created prompts"):
            paragraph = weaviate_client.data_object.get_by_id(
                id, class_name="Paragraph"
            )
            text = paragraph["properties"]["text_content"]
            messages = prompt_handler.get_gpt_messages(text)
            prompt = MIXTRAL_TEMPLATE.format(
                system=messages[0]["content"], user=messages[1]["content"]
            )
            prompts.append(prompt)
            hashed_texts.append(paragraph["properties"]["hashed_text"])
            paper_ids.append(paragraph["properties"]["paper_id"])

        # Generate LLM outputs
        # This will use vllm's offline inference for speed
        # Benchmark: A100: around 500 tokens/s at batch size = 200 (Perhaps can go higher with larger batch sizes)
        # Benchmark: `tensor_parallel_size` do not affect speed/GPU too much
        # Benchmark: `enforce_eager` also do not affect speed too much
        llm_outputs = llm.generate(prompts, sampling_params)
        raw_outputs = [output.outputs[0].text.strip() for output in llm_outputs]

        # Post-process
        outputs = post_process_with_prov(
            ids, paper_ids, hashed_texts, raw_outputs, llm, prompt_handler, alignment_handler, sampling_params
        )

        return outputs

    # Mini-batching
    db_objects = []
    while len(batch_ids) > 0:
        n_in_batch = min(mini_batch_size, len(batch_ids))
        mini_batch_ids = [batch_ids.pop() for _ in range(n_in_batch)]
        outputs = process_mini_batch(mini_batch_ids)
        db_objects.extend([db.Triplets(**output, job_id=job_index) for output in outputs])

    return db_objects


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_index", type=int)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--mini_batch_size", type=int, default=50)
    outputs = main(**vars(parser.parse_args()))

    count_success = 0
    for output in outputs:
        if output.triplets:
            count_success += 1
        else:
            logging.error(f"Failed: {output.triplets}")

    print(f"Success: {count_success}")
