import numpy as np
import plotly.graph_objects as go

"""
Create current figure from given trace
"""
def get_current_fig(x, y):

    # Create scatter
    t1 = go.Scatter(x=x, y=y, mode='lines', marker_color='#1D33FF')

    # Define laylout of figure
    layout = go.Layout(
        paper_bgcolor='#fff',  # everything around the plot
        plot_bgcolor='#fff',  # the plot itself
        xaxis_title='time steps',
        yaxis_title='current',
        margin=dict(t=0),
        height=350,
        showlegend=False
    )
    fig = go.Figure(layout=layout)

    # Add traces
    fig.add_traces(t1)

    # Update axis and grid layout
    fig.update_xaxes(showline=True, linewidth=1, linecolor='#000', gridcolor='#eee')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='#000', gridcolor='#eee')

    return fig

"""
Create voltage figure from given trace
"""
def get_voltage_fig(x, y, treshold):

    # Create scatter
    t1 = go.Scatter(x=x, y=y, mode='lines', marker_color='#1D33FF')
    t2 = go.Scatter(x=x, y=np.repeat(treshold*2**6, len(x)), mode='lines', marker_color='#8E00FF')

    # Define laylout of figure
    layout = go.Layout(
        paper_bgcolor='#fff',  # everything around the plot
        plot_bgcolor='#fff',  # the plot itself
        xaxis_title='time steps',
        yaxis_title='voltage',
        margin=dict(t=0),
        height=350,
        showlegend=False
    )
    fig = go.Figure(layout=layout)

    # Add traces
    fig.add_traces(t1)
    fig.add_traces(t2)

    # Update axis and grid layout
    fig.update_xaxes(showline=True, linewidth=1, linecolor='#000', gridcolor='#eee')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='#000', gridcolor='#eee')

    return fig

"""
Create figure that compares Loihi fit with original model, based on fitter
"""
def get_fit_fig(trace_generator, loihi_fitter):
    # Define x values
    x_axis = np.arange(trace_generator.get_simulation_steps())

    # Normalize trace where threshold is fixed at 1, [trace]/threshold
    b_trace = trace_generator.get_trace()
    l_trace = loihi_fitter.get_trace()
    thresh  = np.array([1]*int(trace_generator.get_simulation_steps()))

    # Create scatter
    t0 = go.Scatter(x=x_axis, y=b_trace, mode='lines', name='Model', marker_color='#FFD679', line=dict(width=6))
    t1 = go.Scatter(x=x_axis, y=l_trace, mode='lines', name='Loihi fit', marker_color='#1D33FF')
    t2 = go.Scatter(x=x_axis, y=np.repeat(thresh, len(x_axis)), name='Threshold', mode='lines', marker_color='#8E00FF')

    # Define laylout of figure
    layout = go.Layout(
        paper_bgcolor='#fff',  # everything around the plot
        plot_bgcolor='#fff',  # the plot itself
        xaxis_title='time steps',
        yaxis_title='voltage (% of threshold)',
        margin=dict(t=20),
        height=300,
        legend_title="Legend"
    )
    fig = go.Figure(layout=layout)

    # Add traces
    fig.add_traces(t0)
    fig.add_traces(t1)
    fig.add_traces(t2)

    # Update axis and grid layout
    fig.update_xaxes(showline=True, linewidth=1, linecolor='#000', gridcolor='#eee')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='#000', gridcolor='#eee')

    return fig
