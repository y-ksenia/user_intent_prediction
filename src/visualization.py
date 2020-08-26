from bokeh.plotting import figure, gmap, output_file, save, show
from bokeh.models import ColumnDataSource, GMapOptions, ColorBar, LinearColorMapper
from bokeh.palettes import Viridis256, Magma256, Category20_19
from bokeh.io import output_notebook, export_png
from bokeh.layouts import column, row, gridplot
output_notebook()


def plot_map(dataframe, column, min_value=None, max_value=None, save_png=False):
    """
    Visualization of the location (using Google Maps).
    :param dataframe: (pd.DataFrame) a data frame that stores location data for visualization (must have "longitude"
    and "latitude" columns)
    :param column: (str) the column whose values will be colored
    :param min_value: (float) min value for color bar
    :param max_value: (float) max value for color bar
    :param save_png: (bool) save or not the picture to png-format
    :return: None
    """
    map_options = GMapOptions(lat=dataframe[dataframe['latitude'] > 0].mean().latitude,
                              lng=dataframe[dataframe['longitude'] > 0].mean().longitude,
                              map_type="roadmap",
                              zoom=10)
    p = gmap('GoogleMaps-API', map_options,
             title=f"Spatio-temporal clusters")

    source = ColumnDataSource(
        data=dict(lat=dataframe.latitude.to_list(),
                  lon=dataframe.longitude.to_list(),
                  color=dataframe[column].tolist())
    )
    if min_value is None:
        min_value = min(dataframe[column].tolist())
    if max_value is None:
        max_value = max(dataframe[column].tolist())
    color_mapper = LinearColorMapper(palette=Viridis256, low=min_value, high=max_value)

    p.circle(
        x="lon",
        y="lat",
        size=10,
        fill_color={'field': 'color', 'transform': color_mapper},
        line_color={'field': 'color', 'transform': color_mapper},
        fill_alpha=0.3,
        line_alpha=0.3,
        source=source
    )

    # add color bar
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')

    if save_png:
        export_png(p, filename=f"../output/time_cluster.png")
        # output_file(f'{column}_map.html')
        # save(p)
    else:
        show(p)


def plot_time(dataframe, which, case, color_bar=False):
    """
    Plot spatial clusters as cos/sin plot.
    :param dataframe: (pd.DataFrame) a data frame that stores location data for visualization (must have "longitude"
    and "latitude" columns)
    :param which: (str) specify for which poriod we draw a plot: Day/Week/Month (dataframe should have columns
    time{which}_cos and time{which}_sin)
    :param case: (str) test or train
    :param color_bar: (bool) add or not a color bar
    :return: (bokeh.Figure)
    """
    source = ColumnDataSource(dataframe)
    color_mapper = LinearColorMapper(palette=Viridis256, low=dataframe['time_cluster'].min(),
                                     high=dataframe['time_cluster'].max())

    if color_bar:
        s = figure(width=500, height=450,
                   title=f'Time of the {which.lower()} for {case}',
                   toolbar_location=None, tools="")
        color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, border_line_color=None, location=(0, 0))
        s.add_layout(color_bar, 'right')
    else:
        s = figure(width=450, height=450,
                   title=f'Time of the {which.lower()} for {case}',
                   toolbar_location=None, tools="")

    s.circle(
        x=f"time{which}_cos",
        y=f"time{which}_sin",
        size=10,
        fill_color={'field': 'time_cluster', 'transform': color_mapper},
        line_color=None,
        fill_alpha=0.3,
        line_alpha=0.3,
        source=source
    )

    # configure text properties
    s.xaxis.axis_label = 'cos'
    s.yaxis.axis_label = 'sin'
    s.xaxis.axis_label_text_font_style = 'normal'
    s.yaxis.axis_label_text_font_style = 'normal'
    s.xaxis.axis_label_text_font = "times"
    s.yaxis.axis_label_text_font = "times"
    s.xaxis.axis_label_text_font_size = "14pt"
    s.xaxis.major_label_text_font_size = "11pt"
    s.yaxis.axis_label_text_font_size = "14pt"
    s.yaxis.major_label_text_font_size = "12pt"
    s.xaxis.major_label_text_font = "times"
    s.yaxis.major_label_text_font = "times"
    s.title.text_font_size = '16pt'
    s.title.text_font = "times"
    s.title.text_font_style = 'normal'

    return s


def plot_temporal_clusters(dataframe, case, save_png=False):
    """
    Plot combination of spatial clusters for different periods as cos/sin plot.
    :param dataframe: (pd.DataFrame) a data frame that stores location data for visualization (must have "longitude"
    and "latitude" columns)
    :param save_png: (bool) save or not the picture to png-format
    :return: None
    """
    s1 = plot_time(dataframe, 'Day', case)
    s2 = plot_time(dataframe, 'Week', case, color_bar=True)

    p = gridplot([[s1, s2]])

    if save_png:
        export_png(p, filename=f"../output/time_clusters.png")
    else:
        show(p)
