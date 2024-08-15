import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj
import folium

from text2graph.eval_dataset.relationships import SpatialRelationship


BURGUNDY = "#8D021F"
BLUE = "#1F77B4"
CONUS_EXTENT = [-125, -66.5, 20, 50]
CONUS_PLUS_EXTENT = [-145, -46.5, 10, 70]
WORLD_EXTENT = [-175, 180, -75, 75]


def extent_from_spatial_relationships(
    spatial_relationships: list[SpatialRelationship], pad: float = 0
) -> tuple[float, float, float, float]:
    lons = [
        sr.subject_lon for sr in spatial_relationships if sr.subject_lon is not None
    ]
    lons += [sr.object_lon for sr in spatial_relationships if sr.object_lon is not None]
    lats = [
        sr.subject_lat for sr in spatial_relationships if sr.subject_lat is not None
    ]
    lats += [sr.object_lat for sr in spatial_relationships if sr.object_lat is not None]
    return min(lons), max(lons), min(lats), max(lats)


def plot_spatial_relationships(
    spatial_relationships: list[SpatialRelationship],
    cutoff_miles: float = 500,
    labels: bool = False,
    plot_extent: tuple[float, float, float, float] | None = None,
    geod: pyproj.Geod = pyproj.Geod(ellps="WGS84"),
) -> plt.Axes:
    if not plot_extent:
        plot_extent = extent_from_spatial_relationships(spatial_relationships)

    fig = plt.figure(figsize=[24, 12])
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax1.set_extent(plot_extent, ccrs.PlateCarree())
    ax1.add_feature(cfeature.LAND)
    ax1.add_feature(cfeature.OCEAN)
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray")
    ax1.add_feature(cfeature.STATES)

    transform = ccrs.PlateCarree()._as_mpl_transform(ax1)
    under_cutoff_counter = 0
    over_cutoff_counter = 0
    no_coords_counter = 0
    within_extent_counter = 0
    for sr in spatial_relationships:
        if (
            sr.object_lat is None
            or sr.object_lon is None
            or sr.subject_lat is None
            or sr.subject_lon is None
        ):
            no_coords_counter += 1
            continue  # Skip relationships where we couldn't get both gps locations
        sr_distance = round(sr.distance(geod=geod), 2)
        if sr_distance <= cutoff_miles:
            under_cutoff_counter += 1
            if sr.within_extent(extent=plot_extent):
                within_extent_counter += 1
                ax1.plot(
                    [sr.subject_lon, sr.object_lon],
                    [sr.subject_lat, sr.object_lat],
                    color="red",
                    linewidth=1,
                    marker="o",
                    transform=ccrs.Geodetic(),
                )
                if labels:
                    ax1.annotate(
                        sr.object_name,
                        xy=(sr.object_lon, sr.object_lat),
                        xycoords=transform,
                    )
                    ax1.annotate(
                        sr.subject_name,
                        xy=(sr.subject_lon, sr.subject_lat),
                        xycoords=transform,
                    )
                    ax1.annotate(
                        sr_distance, xy=sr.midpoint(geod=geod), xycoords=transform
                    )
        else:
            over_cutoff_counter += 1

    # print(f"within extent: {within_extent_counter}")
    # print(f"Under {cutoff_miles=}: {under_cutoff_counter}")
    # print(f"Over {cutoff_miles=}: {over_cutoff_counter}")
    # print(f"No coords: {no_coords_counter}")

    ax1.gridlines()
    return ax1


def html_format_context(sr: SpatialRelationship) -> str:
    try:
        actual_object_name = " ".join(sr.object_name.split()[:-1])
        if not actual_object_name:
            actual_object_name = sr.object_name
    except Exception:
        actual_object_name = sr.object_name
    html_key = f'<p><font color="{BURGUNDY}">{sr.object_name}</font> - <b>{sr.predicate}</b> - <font color="{BLUE}">{sr.subject_name}</font></p>'
    html_str = f'<p style="font-size:12px;">{sr.original_paragraph}</p>'
    html_str = html_str.replace(
        actual_object_name, f'<font color="{BURGUNDY}">{actual_object_name}</font>'
    )
    html_str = html_str.replace(
        sr.subject_name, f'<font color="{BLUE}">{sr.subject_name}</font>'
    )
    html_str = html_str.replace(sr.predicate, f'<b>"{sr.predicate}</b>')
    return html_key + html_str


def original_paragraph_formatted_html_popup(
    sr: SpatialRelationship,
) -> folium.Popup | None:
    if sr.original_paragraph:
        return folium.Popup(html_format_context(sr), min_width=100, max_width=400)
    return None


def bin_spatial_relationships_by_distance(
    spatial_relationships: list[SpatialRelationship], bins: list[int], geod: pyproj.Geod
) -> dict[int, list[SpatialRelationship]]:
    binned_relationships = {b: [] for b in bins}
    for sr in spatial_relationships:
        if sr.has_all_coords():
            distance = sr.distance(geod=geod)
            for b in bins:
                if distance <= b:
                    binned_relationships[b].append(sr)
                    break
    return binned_relationships


def group_from_spatial_relationships(
    spatial_relationships: list[SpatialRelationship],
    group_name: str,
    geod: pyproj.Geod = pyproj.Geod(ellps="WGS84"),
) -> folium.FeatureGroup:
    group = folium.FeatureGroup(name=group_name)
    for sr in spatial_relationships:
        folium.Marker(
            location=[sr.subject_lat, sr.subject_lon],
            tooltip=sr.subject_name,
            popup=sr.subject_name,
            icon=folium.Icon(icon="location-crosshairs", prefix="fa"),
        ).add_to(group)
        folium.Marker(
            location=[sr.object_lat, sr.object_lon],
            tooltip=sr.object_name,
            popup=original_paragraph_formatted_html_popup(sr),
            icon=folium.Icon(icon="gem", prefix="fa"),
        ).add_to(group)
        folium.PolyLine(
            [(sr.subject_lat, sr.subject_lon), (sr.object_lat, sr.object_lon)],
            tooltip=f"{sr.object_name} - {sr.predicate} - {sr.subject_name}: {round(sr.distance(geod=geod), 2)} miles",
        ).add_to(group)
    return group
