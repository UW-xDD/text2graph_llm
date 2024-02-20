import json
import gradio as gr
from concurrent.futures import ThreadPoolExecutor
from text2graph.llm import llm_graph

EXAMPLE_TEXT = """
The top of the Sauk megasequence in Minnesota is at the unconformable contact of the Shakopee Formation with the St. Peter Sandstone. Younger rocks are present beneath the St. Peter Sandstone on the southern and east- ern flanks of the Ozark dome, where the upper Sauk succession includes the Roubidoux, Jefferson City, Cotter, Powell – Smithville – Black Rock, and Everton units in that stratigraphic order (Ethington et al., 2012; Palmer et al., 2012). The Shakopee Formation is equivalent to some lower part of this succession, but sparse inverte- brate faunas and long-ranging conodonts in these units preclude correlation with high resolution. The Jasper Member of the Everton Formation of northern Arkansas contains conodonts of the Histiodella holodentata Biozone, which demonstrates the latest early Whiterockian age for the top of the rocks of the GACB in that region. No faunal evidence is available there for the age of the base of the St. Peter Sandstone. The boundary between the Sauk and Tippecanoe megasequences may be a cor- relative conformity in the Reelfoot rift of southeastern Missouri and northeastern Missouri, but this has not been demonstrated.
"""


# Submit to both OpenAI and OSS models
def pipeline(text: str, model: str) -> str:
    """Models pipeline with formatting."""

    def _try_to_format(text: str) -> str:
        try:
            x = json.loads(text)
            return json.dumps(x, indent=4, sort_keys=True)
        except Exception:
            return text

    raw_output = llm_graph(text, model)
    formatted_output = _try_to_format(raw_output)
    return formatted_output


def run_models(text: str) -> tuple[str, str]:
    """Run both OpenAI model and OSS model for comparison."""
    with ThreadPoolExecutor(max_workers=3) as executor:
        m1 = executor.submit(pipeline, text, "gpt-4-turbo-preview")
        m2 = executor.submit(pipeline, text, "mixtral")
    return [m1.result(), m2.result()]


# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    title = gr.Markdown("# Ask-XDD: Location extraction demo")
    input_text = gr.Textbox(label="Input Text")
    run_button = gr.Button(value="Run Models")

    # Output side by side
    with gr.Row() as row:
        out1 = gr.JSON(label="GPT-4 Turbo Preview")
        out2 = gr.JSON(label="Locally hosted open source model")

    run_button.click(run_models, inputs=[input_text], outputs=[out1, out2])

    gr.Markdown("## Example usage")
    gr.Examples(
        [EXAMPLE_TEXT],
        [input_text],
        [out1, out2],
        run_models,
        cache_examples=True,
    )

demo.launch(server_name="0.0.0.0")
