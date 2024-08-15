import numpy as np
import base64
import io
import pydantic
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
import pyproj
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

from text2graph.eval_dataset.relationships import SpatialRelationship
from text2graph.prompt import PromptHandler, StratPromptHandlerV3
from text2graph.macrostrat import find_all_occurrences


BURGUNDY = "#8D021F"
BLUE = "#1F77B4"
MINT = "#00CC99"
RASPBERRY = "#DD1155"
ORANGE = "#FFA500"
LIGHT_GREY = "#DDDDDD"
DARK_GREY = "#888888"


def sanitize_text_for_html(text: str) -> str:
    return (
        text.replace('"', "&quot;")
        .replace("'", "&apos;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("&", "&amp;")
        .replace(r"\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
        .replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
        .replace(r"\n", "<br>")
        .replace("\n", "<br>")
    )


class SQuADStratPipelineQuestion(pydantic.BaseModel):
    context: str
    object: str | None
    object_start: int | None
    predicate: str | None
    predicate_start: int | None
    subject: str | None
    subject_start: int | None
    context_docid: str
    context_hashed_text: str
    question: str | None = None
    distance_in_miles: float | None = None
    object_lat: float | None = None
    object_lon: float | None = None
    subject_lat: float | None = None
    subject_lon: float | None = None
    augmented_prompt: str | None = None

    def create_augmented_prompt(
        self, prompt_handler: PromptHandler | None = None, clobber=False
    ) -> str | None:
        if not clobber and self.augmented_prompt:
            return self.augmented_prompt

        if not prompt_handler:
            prompt_handler = StratPromptHandlerV3()

        try:
            self.augmented_prompt = (
                prompt_handler.get_system_prompt(text=self.context)
                + "\n"
                + prompt_handler.get_user_prompt(text=self.context)
            )
        except Exception as e:
            print(f"Failed to get prompt for {self.context} with error: {e}")
            self.augmented_prompt = None

        return self.augmented_prompt

    def midpoint(self, geod: pyproj.Geod | None = None) -> tuple[float, float]:
        """return the midpoint lat, lon between subject and object. default geod is WGS84."""
        if not geod:
            geod = pyproj.Geod(ellps="WGS84")
        return geod.npts(
            self.subject_lon, self.subject_lat, self.object_lon, self.object_lat, 1
        )[0]

    def plot_extent(self, pad_degrees: float = 10) -> tuple[float, float, float, float]:
        lons = [self.subject_lon, self.object_lon]
        lats = [self.subject_lat, self.object_lat]
        return (
            min(lons) - pad_degrees,
            max(lons) + pad_degrees,
            min(lats) - pad_degrees,
            max(lats) + pad_degrees,
        )

    def plot_subject_object_relationship(
        self,
        ax: plt.Axes | None = None,
        extent: tuple[float, float, float, float] | None = None,
    ) -> plt.Axes | None:
        if (
            self.object_lat is None
            or self.object_lon is None
            or self.subject_lat is None
            or self.subject_lon is None
        ):
            return None

        if not extent:
            extent = self.plot_extent()

        fig = plt.figure(figsize=[4, 4])
        geo_ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        geo_ax.set_extent(extent, ccrs.PlateCarree())
        geo_ax.add_feature(cfeature.LAND)
        geo_ax.add_feature(cfeature.OCEAN)
        geo_ax.add_feature(cfeature.COASTLINE)
        geo_ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray")
        geo_ax.add_feature(cfeature.STATES)
        transform = ccrs.PlateCarree()._as_mpl_transform(geo_ax)
        geo_ax.plot(
            [self.subject_lon, self.object_lon],
            [self.subject_lat, self.object_lat],
            color=MINT,
            linewidth=1,
            transform=ccrs.Geodetic(),
        )
        geo_ax.plot(
            [self.subject_lon],
            [self.subject_lat],
            color=BLUE,
            linewidth=1,
            marker="o",
            transform=ccrs.Geodetic(),
        )
        geo_ax.plot(
            [self.object_lon],
            [self.object_lat],
            color=BURGUNDY,
            linewidth=1,
            marker="o",
            transform=ccrs.Geodetic(),
        )

        # Define the xticks for longitude
        coordinate_tick_count = 3
        lon_spacing = (extent[1] - extent[0]) / coordinate_tick_count
        geo_ax.set_xticks(
            np.arange(extent[0], extent[1], lon_spacing), crs=ccrs.PlateCarree()
        )
        lon_formatter = cticker.LongitudeFormatter()
        geo_ax.xaxis.set_major_formatter(lon_formatter)

        # Define the yticks for latitude
        lat_spacing = (extent[3] - extent[2]) / coordinate_tick_count
        geo_ax.set_yticks(
            np.arange(extent[2], extent[3], lat_spacing), crs=ccrs.PlateCarree()
        )
        lat_formatter = cticker.LatitudeFormatter()
        geo_ax.yaxis.set_major_formatter(lat_formatter)
        geo_ax.annotate(
            self.object,
            xy=(self.object_lon, self.object_lat),
            xycoords=transform,
            color=BURGUNDY,
        )
        geo_ax.annotate(
            self.subject,
            xy=(self.subject_lon, self.subject_lat),
            xycoords=transform,
            color=BLUE,
        )
        geo_ax.annotate(
            round(self.distance_in_miles, 2),
            xy=self.midpoint(),
            xycoords=transform,
            color=MINT,
        )
        geo_ax.gridlines()
        return geo_ax

    def plot_subject_object_relationship_to_buffer(
        self,
        extent: tuple[float, float, float, float] | None = None,
        color: str = "red",
    ) -> io.BytesIO | None:
        ax = self.plot_subject_object_relationship(extent=extent)
        if ax:
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            return buf
        plt.close()
        return None

    def html_format_text(self) -> str:
        if not self.object:
            actual_object_name = None
        elif self.object in self.context:
            actual_object_name = self.object
        else:
            try:
                actual_object_name = " ".join(self.object.split()[:-1])
                if not actual_object_name:
                    actual_object_name = self.object
            except Exception:
                actual_object_name = self.object
        render_miles = (
            round(self.distance_in_miles, 2) if self.distance_in_miles else "Unknown"
        )
        object_for_html = [
            v for v in [self.object, actual_object_name, "Undefined"] if v
        ][0]
        predicate_for_html = [
            v for v in [self.predicate, actual_object_name, "Undefined"] if v
        ][0]
        subject_for_html = [v for v in [self.subject, "Undefined"] if v][0]
        html_key = f'<p><font color="{BURGUNDY}">{object_for_html}</font> - <b>{predicate_for_html}</b> - <font color="{BLUE}">{subject_for_html}</font>: {render_miles} mi</p>'
        html_text = sanitize_text_for_html(self.context)
        html_str = f'<p style="font-size:12px;">{html_text}</p>'
        if actual_object_name:
            html_str = html_str.replace(
                actual_object_name,
                f'<font color="{BURGUNDY}">{actual_object_name}</font>',
            )
        if self.subject:
            html_str = html_str.replace(
                self.subject, f'<font color="{BLUE}">{self.subject}</font>'
            )
        if self.predicate:
            html_str = html_str.replace(self.predicate, f'<b>"{self.predicate}</b>')
        return html_key + html_str

    def html_format(self) -> str:
        buf = self.plot_subject_object_relationship_to_buffer()
        if buf:
            buf_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            image_html = f"<img src='data:image/png;base64,{buf_base64}'>"
        else:
            image_html = "<p>Failed to render map image.</p>"
        text_html = self.html_format_text()
        css = """.container {
        display: grid;
        align-items: center;
        grid-template-columns: 1fr 1fr;
        column-gap: 5px;
        }

        img {
        max-width: 100%;
        max-height:100%;
        }

        .text {
        font-size: 12px;
        background-color: {DARK_GREY};
        text-color: {LIGHT_GREY};
        }"""
        return f"""<!DOCTYPE html>
        <html>
        <style>{css}</style>
        <head>
            <title>SQuAD Question HTML visualization</title>
        </head>
        <body>
            <div class="container">
            <div class="image">
                {image_html}
            </div>
            <div class="text">
                {text_html}
            </div>
            </div>
        </body>
        </html>
        """


def questions_from_spatial_relationships(
    spatial_relationships: list[SpatialRelationship],
) -> list[SQuADStratPipelineQuestion]:
    questions = []
    geod = pyproj.Geod(ellps="WGS84")
    for sr in spatial_relationships:
        if sr.original_paragraph:
            substrings_dict = {
                "object": sr.object_name,
                "predicate": sr.predicate,
                "subject": sr.subject_name,
            }
            indexes_dict = {}
            for k, substring in substrings_dict.items():
                try:
                    indexes_dict[k] = sr.original_paragraph.index(substring)
                except ValueError:
                    indexes_dict[k] = None
            try:
                distance_in_miles = sr.distance(geod=geod)
            except Exception as e:
                print(
                    f"Failed distance for {sr.object_name} - {sr.predicate} - {sr.subject_name} with error: {e}"
                )
                distance_in_miles = None

            for object_to_find in [
                sr.object_name,
                " ".join(sr.object_name.split()[:-1]),
            ]:
                try:
                    object_start = find_all_occurrences(
                        sr.original_paragraph, [object_to_find]
                    )[0]["start"]
                    break
                except IndexError:
                    object_start = None

            try:
                predicate_start = find_all_occurrences(
                    sr.original_paragraph, [sr.predicate]
                )[0]["start"]
            except IndexError:
                predicate_start = None

            try:
                subject_start = find_all_occurrences(
                    sr.original_paragraph, [sr.subject_name]
                )[0]["start"]
            except IndexError:
                subject_start = None

            questions.append(
                SQuADStratPipelineQuestion(
                    context=sr.original_paragraph,
                    object=sr.object_name,
                    object_start=object_start,
                    predicate=sr.predicate,
                    predicate_start=predicate_start,
                    subject=sr.subject_name,
                    subject_start=subject_start,
                    context_docid=sr.doc_id,
                    context_hashed_text=sr.hashed_text,
                    distance_in_miles=distance_in_miles,
                    object_lat=sr.object_lat,
                    object_lon=sr.object_lon,
                    subject_lat=sr.subject_lat,
                    subject_lon=sr.subject_lon,
                )
            )
    return questions


def render_question_to_html_eval_format(question: SQuADStratPipelineQuestion) -> str:
    return f"""
    <div style="border: 1px solid black; padding: 10px; margin: 10px;">
        <h3>Context</h3>
        <p>{question.html_format()}</p>
        <h3>augmented_prompt</h3>
        <p><b>{question.create_augmented_prompt()}</b></p>
    </div>
    """


def new_question_from_old(
    question: SQuADStratPipelineQuestion, redefine_start_idx: bool = True, **kwargs
) -> SQuADStratPipelineQuestion:
    """
    create a new instance of a question from an existing question with updated parameters. Provide the new/replacement parameters as kwargs.
    :param question: The existing SQuADStraPipelineQuestion instance.
    :param redefine_start_idx: Default True, If True, will redefine the start index of the object, predicate, and subject if a new value is supplied.
    :return: A new instance of SQuADStratPipelineQuestion.
    """
    existing_question_data = question.model_dump()
    update_params = {
        k: v for k, v in kwargs.items() if k in list(existing_question_data.keys())
    }

    # if subject, predicate or object are updated, redefine the relevant start index
    update_start_idx_params = [
        k for k in update_params if k in ["object", "predicate", "subject"]
    ]
    if update_start_idx_params and redefine_start_idx:
        for start_idx_param in update_start_idx_params:
            start_idx_key = f"{start_idx_param}_start"
            start_idx_value = update_params[start_idx_param]
            # if pass None to update kwargs for object predicate subject, also pass None to the start index, and for object and subject None lat/lon too
            if not start_idx_value:
                update_params[start_idx_key] = None
                if start_idx_param in ["object", "subject"]:
                    update_params[start_idx_param + "_lat"] = None
                    update_params[start_idx_param + "_lon"] = None
                    update_params["distance_in_miles"] = None
                continue
            # get on with trying to update the start index for new object, predicate, or subject
            if (
                start_idx_param == "object"
            ):  # for stratnames allow dropping the last word (Formation, Group etc.)
                search_values = [
                    start_idx_value,
                    " ".join(start_idx_value.split()[:-1]),
                ]
            else:
                search_values = [start_idx_value]
            for search_value in search_values:
                try:
                    update_params[start_idx_key] = find_all_occurrences(
                        question.context, [search_value]
                    )[0]["start"]
                    break
                except IndexError:
                    continue

    update_kwargs = existing_question_data | update_params
    return SQuADStratPipelineQuestion(**update_kwargs)


def serialize_questions_to_json_by_index(
    questions: list[SQuADStratPipelineQuestion], output_path: Path, indexes: list[int]
):
    """
    Serialize a list of SQuADStratPipelineQuestion to a JSON file by index.
    """
    json_strings = []
    for idx in indexes:
        json_strings.append(questions[idx].model_dump_json())
    with open(output_path, "w") as f:
        f.write("\n".join(json_strings))
