import pathlib
from datetime import datetime
import logging
from logging.config import dictConfig

import yaml
import json
import pandas as pd

import plotly.express as px
import dash
from dash import dash_table, dcc, html


# ======================================================================================================================
# VARIABLES AND SETUP HERE - MAIN METHOD AT BOTTOM
# ======================================================================================================================


def setup():
    """Pull in config.yaml from config folder"""

    with open(proj_dir / "config" / "config.yaml", "r") as file:
        conf = yaml.safe_load(file)

        # Pull in the logger config
        dictConfig(conf["logging"])
        log = logging.getLogger(__name__)

    return conf, log


# Variables
proj_dir = pathlib.Path.cwd()
config, logger = setup()


# ======================================================================================================================
# IMPORT FUNCTIONS
# ======================================================================================================================
def import_raw():
    """ Imports the raw data from file, separates the two different radios, smushes it into either a dict or Pandas
    DataFrame. Returns the below dict containing all the data."""

    data = {
        "latency": {
            "subsix": {},
            "mmwave": {}
        },
        "throughput": {
            "subsix": {},
            "mmwave": {}
        },
    }

    # Path to where all the data is stored (likely somewhere on the SharePoint now)
    oneDrivePath = pathlib.Path(rf"C:\Users\Joseph Clancy\OneDrive - National University of Ireland, Galway\Thesis "
                                rf"Stuff\P4")

    # Sub-6GHz Data
    # ===============================================================
    sub_six_dir = oneDrivePath / "Cork_Experiments_17_02_23\Data"

    # Import latency data
    latency_dir = sub_six_dir / "ping_tests"
    for data_file_path in latency_dir.iterdir():
        with open(data_file_path, "r") as data_file:
            data["latency"]["subsix"][data_file_path.stem] = data_file.readlines()

    # Import throughput tests
    throughput_dir = sub_six_dir / "main_tests"
    for data_file_path in throughput_dir.iterdir():
        with open(data_file_path, "r") as data_file:
            try:
                data["throughput"]["subsix"][data_file_path.stem] = json.load(data_file)
            except json.JSONDecodeError:
                data["throughput"]["subsix"][data_file_path.stem] = [json.loads(line) for line in data_file]

    # mmWave Data
    # ===============================================================
    mmwave_dir = oneDrivePath / "Bray_Experiments\output\csv_data"

    # Import latency data
    latency_dir = mmwave_dir / "latency"
    for data_file_path in latency_dir.iterdir():
        data["latency"]["mmwave"][data_file_path.stem] = pd.read_csv(data_file_path)

    # Import throughput tests
    throughput_dir = mmwave_dir / "throughput"
    for data_file_path in throughput_dir.iterdir():
        data["throughput"]["mmwave"][data_file_path.stem] = pd.read_csv(data_file_path)

    return data


# ======================================================================================================================
# PROCESSING FUNCTIONS
# ======================================================================================================================
def process_latency(raw_lat_data):
    """Given the raw latency data in dictionary format, this function extracts the latency and jitter information
    from raw ping logs for the Sub-6GHz radio and simply handles the DataFrame from the mmWave radio.

    At the end, a big dirty awful dict is returned with all the stats and dataframes for both radios."""

    # Sub-6GHz Latency Data
    # =================================================
    latencies = []
    for file_name, file_data in raw_lat_data["subsix"].items():
        for line in file_data:
            if "time=" in line:
                line = line.split(" ")
                latencies.append(float(line[6].replace("time=", "")) / 2)

    subsix_lat_df = pd.DataFrame(latencies, columns=["Latency"])

    df = pd.read_csv("output/csvs/new_df.csv")
    subsix_lat_df = pd.concat([subsix_lat_df, df])

    subsix_jit_list = []
    for index, value in subsix_lat_df["Latency"].items():
        if index >= 1:
            prev = subsix_lat_df["Latency"].iloc[index - 1]
            subsix_jit_list.append(abs(value - prev))

    subsix_jit_df = pd.DataFrame(subsix_jit_list, columns=["Jitter"])

    # subsix_less_than_20 = subsix_jit_df[subsix_jit_df["Jitter"] < 20.0]
    # print(f"Sub-6GHz Jitter <20ms: {(len(subsix_less_than_20.index) / len(subsix_jit_df.index))}")

    subsix_lat_stats = subsix_lat_df.describe().to_dict()
    subsix_lat_stats["Latency"]["skew"] = subsix_lat_df["Latency"].skew()
    subsix_lat_stats["Latency"]["kurt"] = subsix_lat_df["Latency"].kurtosis()

    subsix_lat_stats_df = pd.DataFrame.from_dict(subsix_lat_stats, orient="index")

    subsix_jit_stats = subsix_jit_df.describe().to_dict()
    subsix_jit_stats["Jitter"]["skew"] = subsix_jit_df["Jitter"].skew()
    subsix_jit_stats["Jitter"]["kurt"] = subsix_jit_df["Jitter"].kurtosis()

    subsix_jit_stats_df = pd.DataFrame.from_dict(subsix_jit_stats, orient="index")

    subsix_proc_lat_data = {
        "Statistics": subsix_lat_stats_df,
        "DataFrame": subsix_lat_df
    }

    subsix_proc_jit_data = {
        "Statistics": subsix_jit_stats_df,
        "DataFrame": subsix_jit_df
    }

    # mmWave Latency Data
    # =================================================
    mmwave_df_list = raw_lat_data["mmwave"].values()

    mmwave_lat_df = pd.concat(mmwave_df_list)
    mmwave_lat_df.rename(columns={"latency": "Latency"}, inplace=True)

    mmwave_lat_df.to_csv("mmwave_lat_df.csv")

    mmwave_jit_list = []
    for index, value in mmwave_lat_df["Latency"].items():
        if index >= 1:
            prev = mmwave_lat_df["Latency"].iloc[index - 1]
            mmwave_jit_list.append(abs(value - prev))

    mmwave_jit_df = pd.DataFrame(mmwave_jit_list, columns=["Jitter"])
    mmwave_jit_df["Jitter"].replace(0.0, 0.000489, inplace=True)

    # mmwave_less_than_20 = mmwave_jit_df[mmwave_jit_df["Jitter"] < 20.0]
    # print(f"mmWave Jitter <20ms: {(len(mmwave_less_than_20.index) / len(mmwave_jit_df.index))}")

    mmwave_lat_stats = mmwave_lat_df.describe().to_dict()
    mmwave_lat_stats["Latency"]["skew"] = mmwave_lat_df["Latency"].skew()
    mmwave_lat_stats["Latency"]["kurt"] = mmwave_lat_df["Latency"].kurtosis()

    mmwave_lat_stats_df = pd.DataFrame.from_dict(mmwave_lat_stats, orient="index")

    mmwave_jit_stats = mmwave_jit_df.describe().to_dict()
    mmwave_jit_stats["Jitter"]["skew"] = mmwave_jit_df["Jitter"].skew()
    mmwave_jit_stats["Jitter"]["kurt"] = mmwave_jit_df["Jitter"].kurtosis()

    mmwave_jit_stats_df = pd.DataFrame.from_dict(mmwave_jit_stats, orient="index")

    mmwave_proc_lat_data = {
        "Statistics": mmwave_lat_stats_df,
        "DataFrame": mmwave_lat_df
    }

    mmwave_proc_jit_data = {
        "Statistics": mmwave_jit_stats_df,
        "DataFrame": mmwave_jit_df
    }

    subsix_lat_df["radio"] = subsix_lat_df.apply(lambda row: "Sub-6GHz", axis=1)
    mmwave_lat_df["radio"] = mmwave_lat_df.apply(lambda row: "mmWave", axis=1)
    both_proc_lat_data = pd.concat([subsix_lat_df, mmwave_lat_df])

    subsix_jit_df["radio"] = subsix_jit_df.apply(lambda row: "Sub-6GHz", axis=1)
    mmwave_jit_df["radio"] = mmwave_jit_df.apply(lambda row: "mmWave", axis=1)
    both_proc_jit_data = pd.concat([subsix_jit_df, mmwave_jit_df])

    proc_lat_data = {
        "subsix": subsix_proc_lat_data,
        "mmwave": mmwave_proc_lat_data,
        "both": both_proc_lat_data,
    }

    proc_jit_data = {
        "subsix": subsix_proc_jit_data,
        "mmwave": mmwave_proc_jit_data,
        "both": both_proc_jit_data,
    }

    return proc_lat_data, proc_jit_data


def process_throughput(raw_thru_data):
    """Given the raw latency data in dictionary format, this function extracts the latency and jitter information
    from the iPerf logs for the Sub-6GHz radio and simply handles the DataFrame from the mmWave radio.

    At the end, a big dirty awful dict is returned with all the stats and dataframes for both radios."""

    # Sub-6GHz Throughput Data
    # =================================================
    subsix_stats_dict = {}
    subsix_df_list = []
    for rate in ["12.5kb", "25kb", "50kb", "8Mb", "50Mb", "150Mb", "800Mb"]:
        useful_data = []
        for file_name, file_data in raw_thru_data["subsix"].items():
            if file_name in [rate + "-1", rate + "-2", rate + "-3"]:

                interval_iterator = file_data["intervals"]
                if "jitter_ms" not in file_data["intervals"][0]["sum"].keys():
                    interval_iterator = file_data["server_output_json"]["intervals"]

                for interval in interval_iterator:
                    temp_dict = {}
                    for param in ["bytes", "bits_per_second", "jitter_ms", "lost_packets", "packets", "lost_percent"]:
                        temp_dict[param] = interval["sum"][param]
                    useful_data.append(temp_dict)

        def fix_rows(row, drate):
            if drate == "12.5kb":
                if row["packets"] == 2:
                    row["bytes"] = row["bytes"] / 2
                    row["bits_per_second"] = row["bits_per_second"] / 2
            elif drate == "25kb":
                if row["packets"] == 3:
                    row["bytes"] = row["bytes"] * (2 / 3)
                    row["bits_per_second"] = row["bits_per_second"] * (2 / 3)
            elif drate == "50kb":
                if row["packets"] == 6:
                    row["bytes"] = row["bytes"] * (2 / 3)
                    row["bits_per_second"] = row["bits_per_second"] * (2 / 3)
                elif row["packets"] == 3:
                    row["bytes"] = row["bytes"] * 1.4
                    row["bits_per_second"] = row["bits_per_second"] * 1.4
            return row

        iperf_df = pd.DataFrame.from_records(useful_data)
        iperf_df = iperf_df.apply(lambda row: fix_rows(row, rate), axis=1)
        iperf_df["kilobits_per_second"] = iperf_df.apply(lambda row: row["bits_per_second"] / 1000, axis=1)
        iperf_df["megabits_per_second"] = iperf_df.apply(lambda row: row["bits_per_second"] / 1000000, axis=1)
        # iperf_df = iperf_df[iperf_df["kilobits_per_second"] < 13.00]

        stats = iperf_df["megabits_per_second"].describe().to_dict()
        stats["skew"] = iperf_df["megabits_per_second"].skew()
        stats["kurtosis"] = iperf_df["megabits_per_second"].kurtosis()

        subsix_stats_dict[rate] = stats
        iperf_df["data_rate"] = iperf_df.apply(lambda row: rate, axis=1)
        subsix_df_list.append(iperf_df)

    subsix_stats_df = pd.DataFrame.from_dict(subsix_stats_dict, orient="index")
    subsix_stats_df = subsix_stats_df.reset_index(level=0)

    subsix_thru_df = pd.concat(subsix_df_list)

    subsix_proc_thru_data = {
        "Statistics": subsix_stats_df,
        "DataFrame": subsix_thru_df
    }

    # mmWave Throughput Data
    # =================================================
    mmwave_df_list = []
    for file_name, file_data in raw_thru_data["mmwave"].items():
        file_name = file_name.split("_")
        file_data["data_rate"] = file_data.apply(lambda row: file_name[2].replace("m", "M"), axis=1)
        mmwave_df_list.append(file_data)

    mmwave_thru_df = pd.concat(mmwave_df_list)
    mmwave_thru_df.rename(columns={"throughput": "Throughput"}, inplace=True)
    mmwave_thru_df["Throughput"] = mmwave_thru_df.apply(lambda row: row["Throughput"] / 1000000, axis=1)
    mmwave_thru_df["data_rate"] = pd.Categorical(mmwave_thru_df["data_rate"],
                                                 ["50Mb", "100Mb", "150Mb", "500Mb", "800Mb"])
    mmwave_thru_df.sort_values("data_rate", inplace=True)
    mmwave_thru_df.reset_index(inplace=True)
    mmwave_thru_df.to_csv("mmwave_thru_df.csv")

    mmwave_stats_dict = {}
    for data_rate in mmwave_thru_df["data_rate"].unique():
        stats = mmwave_thru_df["Throughput"][mmwave_thru_df["data_rate"] == data_rate].describe().to_dict()
        stats["skew"] = mmwave_thru_df["Throughput"][mmwave_thru_df["data_rate"] == data_rate].skew()
        stats["kurtosis"] = mmwave_thru_df["Throughput"][mmwave_thru_df["data_rate"] == data_rate].kurtosis()
        mmwave_stats_dict[data_rate] = stats

    mmwave_stats_df = pd.DataFrame.from_dict(mmwave_stats_dict, orient="index")
    mmwave_stats_df.to_csv("mmwave_stats_df.csv")
    mmwave_stats_df = mmwave_stats_df.reset_index(level=0)

    mmwave_proc_thru_data = {
        "Statistics": mmwave_stats_df,
        "DataFrame": mmwave_thru_df
    }

    proc_thru_data = {
        "subsix": subsix_proc_thru_data,
        "mmwave": mmwave_proc_thru_data,
    }

    return proc_thru_data


def process_raw(data):
    """This is just a wrapper function to split off the processing tasks to other functions."""

    proc_lat_data, proc_jit_data = process_latency(data["latency"])

    proc_data = {
        "latency": proc_lat_data,
        "jitter": proc_jit_data,
        "throughput": process_throughput(data["throughput"]),
    }

    return proc_data


# ======================================================================================================================
# MAIN BODY OF CODE
# ======================================================================================================================

# Start the clock!
start = datetime.now()
logger.info("Starting post-processing @ " + start.strftime('%Y/%m/%d %H:%M:%S') + "...")

# Import raw data from the raw folder
logger.info("Importing raw data...")
raw_data = import_raw()

# Process raw data
logger.info("Processing raw data...")
processed_data = process_raw(raw_data)

# Finish and collect runtime
finish = datetime.now()
logger.info("Finished post-processing @ " + finish.strftime('%Y/%m/%d %H:%M:%S') + "...")
exec_time = finish - start
logger.info("Execution time: " + str(exec_time))

# ======================================================================================================================
# PLOTLY CODE TO GENERATE GRAPHS
# ======================================================================================================================

# LATENCY ECDF PLOT FOR BOTH RADIOS
lat_ecdf = px.ecdf(
    processed_data["latency"]["both"],
    x="Latency",
    color="radio",
    labels={
        "Latency": "End-to-End Delay (ms)",
        "percent": "Percent (%)"
    },
    ecdfnorm='percent',
    marginal="box",
    log_x=True
)
lat_ecdf.update_layout(
    font_family="Times New Roman", font_size=20,
    font_color="black",
    margin_l=5, margin_t=5, margin_b=5, margin_r=5
)
lat_ecdf.update_layout(legend_title_text='')
lat_ecdf.update_yaxes(title_text="Percent (%)", row=1, col=1)

lat_ecdf.write_image("output/images/lat_ecdf.pdf", width=3.5 * 300, height=2 * 300, scale=1)

# JITTER ECDF PLOT FOR BOTH RADIOS
jit_ecdf = px.ecdf(
    processed_data["jitter"]["both"],
    x="Jitter",
    color="radio",
    labels={
        "Jitter": "End-to-End Jitter (ms)",
        "percent": "Percent (%)"
    },
    ecdfnorm='percent',
    marginal="box",
    log_x=True
)
jit_ecdf.update_layout(
    font_family="Times New Roman", font_size=20,
    font_color="black",
    margin_l=5, margin_t=5, margin_b=5, margin_r=5
)
jit_ecdf.update_layout(legend_title_text='')
jit_ecdf.update_yaxes(title_text="Percent (%)", row=1, col=1)

jit_ecdf.write_image("output/images/jit_ecdf.pdf", width=3.5 * 300, height=2 * 300, scale=1)

# OBJECT DATA THROUGHPUT FOR SUB-6GHZ RADIO
thru_df = processed_data["throughput"]["subsix"]["DataFrame"]
thru_color_map = {
    "12.5kb": '#636EFA',
    "25kb": '#EF553B',
    "50kb": '#00CC96',
    "8Mb": '#AB63FA',
    "50Mb": '#FFA15A',
    "150Mb": '#19D3F3',
    "800Mb": '#FF6692'}
obj_ecdf = px.ecdf(
    thru_df[thru_df["data_rate"].isin(["12.5kb", "25kb", "50kb"])],
    x="kilobits_per_second",
    color="data_rate",
    color_discrete_map=thru_color_map,
    labels={
        "kilobits_per_second": "Throughput (kbps)",
        "megabits_per_second": "Throughput (Mbps)",
        "percent": "Percent (%)"
    },
    ecdfnorm='percent',
    marginal="box",
    facet_col="data_rate",
    facet_col_spacing=0.05,
    log_x=True
)
obj_ecdf.update_layout(
    font_family="Times New Roman", font_size=20,
    font_color="black",
    margin_l=5, margin_t=30, margin_b=5, margin_r=5
)
obj_ecdf.update_xaxes(matches=None)
obj_ecdf.update_layout(showlegend=False)
obj_ecdf.update_yaxes(title_text="Percent (%)", row=1, col=1)

obj_ecdf.write_image(f"output/images/obj_ecdf.pdf", width=3.5 * 400, height=2 * 300, scale=1)

# SENSOR DATA THROUGHPUT FOR SUB-6GHZ RADIO
sens_ecdf = px.ecdf(
    thru_df[thru_df["data_rate"].isin(["8Mb", "50Mb", "150Mb", "800Mb"])],
    x="megabits_per_second",
    color="data_rate",
    color_discrete_map=thru_color_map,
    labels={
        "kilobits_per_second": "Throughput (kbps)",
        "megabits_per_second": "Throughput (Mbps)",
        "percent": "Percent (%)"
    },
    ecdfnorm='percent',
    marginal="box",
    facet_col="data_rate",
    facet_col_spacing=0.025,
    facet_col_wrap=2,
    log_x=True
)
sens_ecdf.update_layout(
    font_family="Times New Roman", font_size=20,
    font_color="black",
    margin_l=5, margin_t=30, margin_b=5, margin_r=5
)
sens_ecdf.update_xaxes(matches=None)
sens_ecdf.update_layout(showlegend=False)
sens_ecdf.update_yaxes(title_text="Percent (%)", row=1, col=1)

sens_ecdf.write_image(f"output/images/sens_ecdf.pdf", width=3.5 * 400, height=2 * 300, scale=1)

# SENSOR DATA THROUGHPUT FOR MMWAVE RADIO
mmwave_df = processed_data["throughput"]["mmwave"]["DataFrame"]
mmwave_ecdf = px.ecdf(
    mmwave_df[mmwave_df["data_rate"].isin(["8Mb", "50Mb", "150Mb", "800Mb"])],
    x="Throughput",
    color="data_rate",
    color_discrete_map=thru_color_map,
    labels={
        "Throughput": "Throughput (Mbps)",
        "percent": "Percent (%)"
    },
    ecdfnorm='percent',
    marginal="box",
    facet_col="data_rate",
    facet_col_spacing=0.025,
    log_x=True
)
mmwave_ecdf.update_layout(
    font_family="Times New Roman", font_size=20,
    font_color="black",
    margin_l=5, margin_t=30, margin_b=5, margin_r=5
)
mmwave_ecdf.update_xaxes(matches=None)
mmwave_ecdf.update_layout(showlegend=False)
mmwave_ecdf.update_yaxes(title_text="Percent (%)", row=1, col=1)

mmwave_ecdf.write_image(f"output/images/mmwave_ecdf.pdf", width=3.5 * 400, height=2 * 300, scale=1)

# ======================================================================================================================
# DASH/PLOTLY WEB-APP SETUP
# ======================================================================================================================

# App instantiation/configuration
app = dash.Dash(__name__, external_stylesheets=config["external_stylesheets"], suppress_callback_exceptions=True)
app.title = "NUI Galway Car Group - Cellular Network Measurements"

# HTML layout definition
app.layout = html.Div([
    # Header/Title definition
    html.Div(
        children=[
            html.H1(
                children="Cellular Network Measurements", className="header-title"
            ),
            html.P(children="Joseph Clancy", className="header-author"),
        ],
        className="header",
    ),
    # Wrapper for the main body of the page containing graphs
    html.Div(id='view-container', children=[

        # Combined Data Visuals
        # =====================================================
        # LATENCY
        html.Div(children=f"End-to-End Delay Statistics (ms)", className="menu-title"),
        dash_table.DataTable(
            id=f"both_lat_stats_table",
            columns=[{"name": i, "id": i, "type": "numeric", "format": {"specifier": ',.2f'}} for i in
                     processed_data["latency"]["subsix"]["Statistics"].columns],
            data=[processed_data["latency"]["subsix"]["Statistics"].to_dict(orient="index")["Latency"],
                  processed_data["latency"]["mmwave"]["Statistics"].to_dict(orient="index")["Latency"]],
            style_cell=dict(textAlign='right'),
            style_header=dict(backgroundColor="lightGrey"),
            style_as_list_view=True,
            style_cell_conditional=[
                {
                    'if': {'column_id': 'View'},
                    'textAlign': 'left'
                }
            ]
        ),
        html.Div(children=f"End-to-End Delay eCDF", className="menu-title"),
        dcc.Graph(id="both_lat_cdf", figure=lat_ecdf),
        html.Div(children=f"End-to-End Delay Histogram", className="menu-title"),
        dcc.Graph(id="both_lat_hist", figure=px.histogram(
            processed_data["latency"]["both"],
            x="Latency",
            color="radio",
            labels={
                "Latency": "End-to-End Delay (ms)",
            },
            nbins=250,
            marginal="box",
            log_x=True
        )),

        # JITTER
        html.Div(children=f"End-to-End Jitter Statistics (ms)", className="menu-title"),
        dash_table.DataTable(
            id=f"both_jit_stats_table",
            columns=[{"name": i, "id": i, "type": "numeric", "format": {"specifier": ',.2f'}} for i in
                     processed_data["latency"]["subsix"]["Statistics"].columns],
            data=[processed_data["jitter"]["subsix"]["Statistics"].to_dict(orient="index")["Jitter"],
                  processed_data["jitter"]["mmwave"]["Statistics"].to_dict(orient="index")["Jitter"]],
            style_cell=dict(textAlign='right'),
            style_header=dict(backgroundColor="lightGrey"),
            style_as_list_view=True,
            style_cell_conditional=[
                {
                    'if': {'column_id': 'View'},
                    'textAlign': 'left'
                }
            ]
        ),
        html.Div(children=f"End-to-End Jitter eCDF", className="menu-title"),
        dcc.Graph(id="both_jit_cdf", figure=jit_ecdf),

        # OBJECT DATA
        html.Div(children=f"Object Data Throughput Statistics (kbps)", className="menu-title"),
        dash_table.DataTable(
            id=f"subsix_thru_stats_table",
            columns=[{"name": i, "id": i, "type": "numeric", "format": {"specifier": ',.5f'}} for i in
                     processed_data["throughput"]["subsix"]["Statistics"].columns],
            data=processed_data["throughput"]["subsix"]["Statistics"].to_dict('records'),
            style_cell=dict(textAlign='right'),
            style_header=dict(backgroundColor="lightGrey"),
            style_as_list_view=True,
            style_cell_conditional=[
                {
                    'if': {'column_id': 'View'},
                    'textAlign': 'left'
                }
            ]
        ),
        html.Div(children=f"Object Data Throughput eCDF", className="menu-title"),
        dcc.Graph(id="obj_cdf", figure=obj_ecdf),
        html.Div(children=f"Sensor Data Throughput eCDF", className="menu-title"),
        dcc.Graph(id="sens_cdf", figure=sens_ecdf),
        html.Div(children=f"mmWave Data Throughput eCDF", className="menu-title"),
        dcc.Graph(id="mmWave_cdf", figure=mmwave_ecdf),
        # html.Div(children=f"Object Data Throughput eCDF", className="menu-title"),
        #     dcc.Graph(id="obj_cdf", figure=thru_ecdfs[0]),
        # html.Div(children=f"Object Data Throughput eCDF", className="menu-title"),
        #     dcc.Graph(id="obj_cdf", figure=thru_ecdfs[1]),
        # html.Div(children=f"Object Data Throughput eCDF", className="menu-title"),
        #     dcc.Graph(id="obj_cdf", figure=thru_ecdfs[2]),
        # html.Div(children=f"Object Data Throughput eCDF", className="menu-title"),
        #     dcc.Graph(id="obj_cdf", figure=thru_ecdfs[3]),
        # html.Div(children=f"Object Data Throughput eCDF", className="menu-title"),
        #     dcc.Graph(id="obj_cdf", figure=thru_ecdfs[4]),
        # html.Div(children=f"Object Data Throughput eCDF", className="menu-title"),
        #     dcc.Graph(id="obj_cdf", figure=thru_ecdfs[5]),
        # html.Div(children=f"Object Data Throughput eCDF", className="menu-title"),
        #     dcc.Graph(id="obj_cdf", figure=thru_ecdfs[6]),

        # Sub-6GHz Data Visuals
        # =====================================================
        html.Div(children=f"Sub-6GHz End-to-End Delay Statistics (ms)", className="menu-title"),
        dash_table.DataTable(
            id=f"subsix_lat_stats_table",
            columns=[{"name": i, "id": i, "type": "numeric", "format": {"specifier": ',.2f'}} for i in
                     processed_data["latency"]["subsix"]["Statistics"].columns],
            data=[processed_data["latency"]["subsix"]["Statistics"].to_dict(orient="index")["Latency"]],
            style_cell=dict(textAlign='right'),
            style_header=dict(backgroundColor="lightGrey"),
            style_as_list_view=True,
            style_cell_conditional=[
                {
                    'if': {'column_id': 'View'},
                    'textAlign': 'left'
                }
            ]
        ),
        html.Div(children=f"Sub-6GHz End-to-End Delay eCDF", className="menu-title"),
        dcc.Graph(id="subsix_lat_cdf", figure=px.ecdf(
            processed_data["latency"]["subsix"]["DataFrame"],
            x="Latency",
            labels={
                "Latency": "End-to-End Delay (ms)",
            },
            ecdfnorm='percent',
            marginal="box",
            log_x=True
        )),
        html.Div(children=f"Sub-6GHz End-to-End Delay Histogram", className="menu-title"),
        dcc.Graph(id="subsix_lat_hist", figure=px.histogram(
            processed_data["latency"]["subsix"]["DataFrame"],
            x="Latency",
            labels={
                "Latency": "End-to-End Delay (ms)",
            },
            nbins=250,
            marginal="box"
        )),

        html.Div(children=f"Sub-6GHz Throughput Statistics (Mbps)", className="menu-title"),
        dash_table.DataTable(
            id=f"subsix_thru_stats_table",
            columns=[{"name": i, "id": i, "type": "numeric", "format": {"specifier": ',.5f'}} for i in
                     processed_data["throughput"]["subsix"]["Statistics"].columns],
            data=processed_data["throughput"]["subsix"]["Statistics"].to_dict('records'),
            style_cell=dict(textAlign='right'),
            style_header=dict(backgroundColor="lightGrey"),
            style_as_list_view=True,
            style_cell_conditional=[
                {
                    'if': {'column_id': 'View'},
                    'textAlign': 'left'
                }
            ]
        ),

        html.Div(children=f"Sub-6GHz Throughput Line Chart", className="menu-title"),
        dcc.Graph(id="subsix_thru_line", figure=px.line(
            processed_data["throughput"]["subsix"]["DataFrame"],
            y="megabits_per_second",
            color='data_rate'
        )),

        html.Div(children=f"Sub-6GHz Throughput Distribution Plot", className="menu-title"),
        dcc.Graph(id="subsix_thru_line", figure=px.histogram(
            processed_data["throughput"]["subsix"]["DataFrame"],
            x="megabits_per_second",
            color="data_rate",
            log_y=True,
            barmode="overlay",
            nbins=100,
        )),

        # mmWave Data Visuals
        # =====================================================
        html.Div(children=f"mmwave End-to-End Delay Statistics (ms)", className="menu-title"),
        dash_table.DataTable(
            id=f"mmwave_lat_stats_table",
            columns=[{"name": i, "id": i, "type": "numeric", "format": {"specifier": ',.2f'}} for i in
                     processed_data["latency"]["mmwave"]["Statistics"].columns],
            data=[processed_data["latency"]["mmwave"]["Statistics"].to_dict(orient="index")["Latency"]],
            style_cell=dict(textAlign='right'),
            style_header=dict(backgroundColor="lightGrey"),
            style_as_list_view=True,
            style_cell_conditional=[
                {
                    'if': {'column_id': 'View'},
                    'textAlign': 'left'
                }
            ]
        ),
        html.Div(children=f"mmwave End-to-End Delay eCDF", className="menu-title"),
        dcc.Graph(id="mmwave_lat_cdf", figure=px.ecdf(
            processed_data["latency"]["mmwave"]["DataFrame"],
            x="Latency",
            labels={
                "Latency": "End-to-End Delay (ms)",
            },
            ecdfnorm='percent',
            marginal="box",
            log_x=True
        )),
        html.Div(children=f"mmwave End-to-End Delay Histogram", className="menu-title"),
        dcc.Graph(id="mmwave_lat_hist", figure=px.histogram(
            processed_data["latency"]["mmwave"]["DataFrame"],
            x="Latency",
            labels={
                "Latency": "End-to-End Delay (ms)",
            },
            nbins=250,
            marginal="box",
            log_x=True
        )),

        html.Div(children=f"mmwave Throughput Statistics (Mbps)", className="menu-title"),
        dash_table.DataTable(
            id=f"mmwave_thru_stats_table",
            columns=[{"name": i, "id": i, "type": "numeric", "format": {"specifier": ',.5f'}} for i in
                     processed_data["throughput"]["mmwave"]["Statistics"].columns],
            data=processed_data["throughput"]["mmwave"]["Statistics"].to_dict('records'),
            style_cell=dict(textAlign='right'),
            style_header=dict(backgroundColor="lightGrey"),
            style_as_list_view=True,
            style_cell_conditional=[
                {
                    'if': {'column_id': 'View'},
                    'textAlign': 'left'
                }
            ]
        ),

        html.Div(children=f"mmwave Throughput Line Chart", className="menu-title"),
        dcc.Graph(id="mmwave_thru_line", figure=px.line(
            processed_data["throughput"]["mmwave"]["DataFrame"],
            y="Throughput",
            color='data_rate'
        )),

        html.Div(children=f"mmwave Throughput Distribution Plot", className="menu-title"),
        dcc.Graph(id="mmwave_thru_line", figure=px.histogram(
            processed_data["throughput"]["mmwave"]["DataFrame"],
            x="Throughput",
            color="data_rate",
            log_y=True,
            barmode="overlay",
            nbins=100,
        )),

    ], className="wrapper"),

], style={'backgroundColor': '#EDEDED'})

if __name__ == "__main__":
    app.run_server(debug=True)
