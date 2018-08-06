# Leafage

Leafage is a library to extract explanations for the predictions of black-box machine learning methods.
LEAFaGE stands for Local Example And Feature-based model-Agnostic Explanation.

An example of the usage of the library is shown in Usage.ipynb.

The project is written in `Python 2.7`

[Plotly](https://plot.ly/python/getting-started/) is used for visualization.
To extract visualizations, you need a free plotly account and set the credentials as follows:
```python 
import plotly
plotly.tools.set_credentials_file(username='DemoAccount', api_key='lr1c37zw81')
```