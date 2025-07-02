// Plotly.js version of GDP per Capita Chart with Rounded Corner Labels
import json

trace_data_js_code = f"const traceData = {json.dumps(trace_data_js, indent=2)};"

# Save the JavaScript code to a .js file
js_file_path = "/mnt/data/chart.js"
with open(js_file_path, "w") as js_file:
    js_file.write(trace_data_js_code)

js_file_path

var traceData = [
  // Placeholder - Replace with actual country data traces as needed
];

var annotations = [
  {
    x: 2023,
    y: 8800,
    text: "COSTA RICA",
    showarrow: false,
    font: {
      color: "rgb(0,102,204)",
      size: 13,
      family: "Arial Black"
    },
    align: "left",
    xanchor: "left",
    bgcolor: "rgba(0,102,204,0.12)",
    bordercolor: "rgb(0,102,204)",
    borderwidth: 1,
    borderpad: 6,
    opacity: 0.95
  }
];

var layout = {
  title: {
    text: "GDP per Capita (constant 2015 US$) in Central America (1975–2023)",
    font: {
      size: 20,
      color: "#003893",
      family: "Arial Black"
    }
  },
  xaxis: {
    title: "Year",
    showgrid: false,
    linecolor: "black",
    tickfont: { size: 13 },
    titlefont: { size: 14, color: "black" }
  },
  yaxis: {
    title: "GDP per Capita (constant 2015 US$)",
    gridcolor: "lightgray",
    gridwidth: 1,
    tickfont: { size: 13 },
    titlefont: { size: 14, color: "black" },
    zeroline: false
  },
  plot_bgcolor: "#f0f8ff",
  paper_bgcolor: "white",
  font: { color: "black", size: 14 },
  showlegend: false,
  annotations: annotations
};

Plotly.newPlot("myDiv", traceData, layout);
