import plotly.express as px

color_scale = [('train', 'orange'), ('val','blue'), ('test', 'green')]

def plot(data):
    fig = px.scatter_mapbox(
        data, 
        lat="lat", 
        lon="lon", 
        color_continuous_scale=color_scale,
        color='dataset',
        zoom=0, 
        height=1080,
        width=1920
    )

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()