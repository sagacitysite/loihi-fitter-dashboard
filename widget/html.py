import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
#from dash import html
#from dash import dcc

content_parse = html.Div([
    html.H3('Enter equation:'),
    dcc.Textarea(
        id='equation',
        #value='dv/dt  = (ge-(v-El))/(10*ms) : volt (unless refractory) \ndge/dt = -ge/(11*ms) : volt',
        value='dv/dt  = (ge-(v-El))/taum : volt (unless refractory) \ndge/dt = -ge/taum : volt',
        #value='dv/dt  = (ge-(v-El))/taum : volt (unless refractory) \ndge/dt = -ge/taue : volt',
        #value='dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory) \n dge/dt = -ge/taue : volt \n dgi/dt = -gi/taui : volt',
        title='enter equation here'
    ),
    html.H3('Assign variables:'),
    dbc.Table([
        html.Thead([
            html.Tr([
                html.Th('Variable type'),
                html.Th('Variable name'),
                html.Th('Unit')
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td([
                    html.Span('Membrane potential'),
                    html.Br(),
                    html.Small('e.g. voltage')]),
                html.Td(dcc.Input(id='output-variable', type='text', value='v', placeholder='e.g. v')),
                html.Td(dcc.Input(id='output-variable-unit', type='text', value='mV', placeholder='e.g. mV'))
            ]),
            html.Tr([
                html.Td([
                    html.Span('Synaptic input'),
                    html.Br(),
                    html.Small('e.g. current')
                ]),
                html.Td(dcc.Input(id='input-variable', type='text',  value='ge', placeholder='e.g. I')),
                html.Td(dcc.Input(id='input-variable-unit', type='text', value='mV', placeholder='e.g. nA'))
            ]),
            html.Tr(
                html.Td(colSpan=3, children=[
                    dbc.Checklist(
                        options=[{"label": "Model contains a resting potential parameter", "value": 1}],
                        value=[1],
                        id='has-resting-variable',
                        switch=True,
                    )
                ])
            ),
            html.Tr(id='resting-variable-row', children=[
                html.Td([
                    html.Span('Resting potential'),
                    html.Br(),
                    html.Small('e.g. voltage')
                ]),
                html.Td(dcc.Input(id='resting-variable', type='text',  value='El', placeholder='e.g. v_r')),
                html.Td(dcc.Input(id='resting-variable-unit', type='text', value='mV', placeholder='e.g. mV'))
            ])
        ])
    ]),
    dbc.Button('Parse', color='success', id='submit-parse', n_clicks=0)
])

content_parameters = html.Div([
    html.H3('Neuron:'),
    dbc.Table([
        html.Thead([
            html.Tr([
                html.Th('Parameter'),
                html.Th('Value'),
                html.Th('Unit')
            ])
        ]),
        html.Tbody(id='neuron-model-parameters', children=[
            html.Tr([
                html.Td('Treshold'),
                html.Td(dcc.Input(id='threshold-value', type='number', value='32')),
                html.Td(dcc.Input(id='threshold-unit', type='text', value='mV', placeholder='e.g. mV'))
            ])
        ])
    ]),
    html.H3('Synapse:'),
    dbc.Table([
        html.Thead([
            html.Tr([
                html.Th('Parameter'),
                html.Th('Value'),
                html.Th('Unit')
            ])
        ]),
        html.Tbody(id='synapse-model-parameters', children=[
            html.Tr([
                html.Td('Weight'),
                html.Td(dcc.Input(id='weight-value', type='number', value='64')),
                html.Td(dcc.Input(id='weight-unit', type='text', value='mV', placeholder='e.g. mV'))
            ])
        ])
    ]),
    #html.H3('Fitting (optional):'),
    #dbc.Table([
    #    html.Thead([
    #        html.Tr([
    #            html.Th('Parameter'),
    #            html.Th('Value'),
    #            html.Th('Unit')
    #        ])
    #    ]),
    #    html.Tbody(id='synapse-model-parameters', children=[
    #        html.Tr([
    #            html.Td('dt'),
    #            html.Td(dcc.Input(id='dt-value', type='text')),
    #            html.Td(dcc.Input(id='dt-unit', type='text'))
    #        ]),
    #        html.Tr([
    #            html.Td('Runtime'),
    #            html.Td(dcc.Input(id='runtime-value', type='text')),
    #            html.Td(dcc.Input(id='runtime-unit', type='text'))
    #        ])
    #    ])
    #]),
    dbc.Button('Fit', color='success', id='submit-fit', n_clicks=0),
    dbc.Button('Cancel', color='secondary', id='cancel-fit', n_clicks=0)
])

content_result = html.Div([
    html.Div(id='results', children=[
        dbc.Spinner(color="primary")
    ]),
    dcc.Graph(id='plot-fit'),
    dbc.Button('Apply to dashboard', color='success', id='apply-parameters', n_clicks=0),
    dbc.Button('New model', color='secondary', id='restart-fit', n_clicks=0)
])

fitter = dbc.Card(id="fitter", children=[
    dbc.CardHeader(html.Strong('1. Parse equation')),
    dbc.Collapse(
        dbc.CardBody(content_parse),
        id='collapse-parse',
        is_open=True,
    ),
    dbc.CardHeader(html.Strong('2. Define parameters')),
    dbc.Collapse(
        dbc.CardBody(content_parameters),
        id='collapse-parameters',
        is_open=False,
    ),
    dbc.CardHeader(html.Strong('3. Result')),
    dbc.Collapse(
        dbc.CardBody(content_result),
        id='collapse-result',
        is_open=False,
    ),
])

dashboard = dbc.Card([
    dbc.CardHeader(html.Strong('Dashboard')),
    dbc.CardBody([
        html.Div(id='figures', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id='plot-itrace'), width=6),
                dbc.Col(dcc.Graph(id='plot-vtrace'), width=6)
            ])
        ]),
        html.Div(id='sliders', children=[
            dbc.Row([
                dbc.Col(html.Strong('Simulation runtime'), className='slider-label', width=3),
                dbc.Col(dcc.Slider(
                    id='slider-runtime', min=100, max=250, step=1, value=100,
                    marks={100: '100', 150: '150', 200: '200', 250: '250'}
                ), width=7),
                dbc.Col(html.Strong(id='value-runtime', children=[]), className='slider-value', width=2)
            ]),
            dbc.Row([
                dbc.Col(html.Strong('Voltage decay'), className='slider-label', width=3),
                dbc.Col(dcc.Slider(
                    id='slider-vdecay', min=0, max=4096, step=1, value=2048,
                    marks={0: '0', 1024: '1024', 2048: '2048', 3072: '3072', 4096: '4096'}
                ), width=7),
                dbc.Col(html.Strong(id='value-vdecay', children=[]), className='slider-value', width=2)
            ]),
            dbc.Row([
                dbc.Col(html.Strong('Current decay'), className='slider-label', width=3),
                dbc.Col(dcc.Slider(
                    id='slider-idecay', min=0, max=4096, step=1, value=2048,
                    marks={0: '0', 1024: '1024', 2048: '2048', 3072: '3072', 4096: '4096'}
                ), width=7),
                dbc.Col(html.Strong(id='value-idecay', children=[]), className='slider-value', width=2)
            ]),
            dbc.Row([
                dbc.Col(html.Strong('Weight mantissa'), className='slider-label', width=3),
                dbc.Col(dcc.Slider(
                    id='slider-wmant', min=0, max=255, step=1, value=128,
                    marks={0: '0', 64: '64', 128: '128', 192: '192', 255: '255'}
                ), width=7),
                dbc.Col(html.Strong(id='value-wmant', children=[]), className='slider-value', width=2)
            ]),
            dbc.Row([
                dbc.Col(html.Strong('Weight exponent'), className='slider-label', width=3),
                dbc.Col(dcc.Slider(
                    id='slider-wexp', min=-8, max=7, step=1, value=0,
                    marks={-8: '-8', -2: '-2', -1: '-1', 0: '0', 1: '1', 2: '2', 7: '7'}
                ), width=7),
                dbc.Col(html.Strong(id='value-wexp', children=[]), className='slider-value', width=2)
            ]),
            dbc.Row([
                dbc.Col(html.Strong('Weight bits'), className='slider-label', width=3),
                dbc.Col(dcc.Slider(
                    id='slider-wbits', min=1, max=8, step=1, value=8,
                    marks={1: '1', 6: '6', 7: '7', 8: '8'}
                ), width=7),
                dbc.Col(html.Strong(id='value-wbits', children=[]), className='slider-value', width=2)
            ]),
            dbc.Row([
                dbc.Col(html.Strong('Threshold'), className='slider-label', width=3),
                dbc.Col(dcc.Slider(
                    id='slider-thresh', min=0, max=131071, step=1, value=512,
                    marks={0: '0', 16384: '16384', 32768: '32768', 49152: '49152', 65536: '65536', 81920: '81920', 98304: '98304', 114688: '114688', 131071: '131071'}
                ), width=7),
                dbc.Col(html.Strong(id='value-thresh', children=[]), className='slider-value', width=2)
            ]),
            dbc.Row([
                dbc.Col(html.Strong('Refractory period'), className='slider-label', width=3),
                dbc.Col(dcc.Slider(
                    id='slider-ref', min=1, max=10, step=1, value=1,
                    marks={1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 10: '10'}
                ), width=7),
                dbc.Col(html.Strong(id='value-ref', children=[]), className='slider-value', width=2)
            ])
        ])
    ])
])

container = dbc.Container(fluid=True, children=[
    html.H1('Tune your Loihi neuron model and fit Brian2 differential equation to Loihi parameters', id='main-title'),
    dbc.Row([
        dbc.Col(fitter, width=4),
        dbc.Col(dashboard, width=8)
    ]),
    dbc.Modal(
        [
            dbc.ModalHeader('Error'),
            dbc.ModalBody('Make sure that all fields are filled.'),
            dbc.ModalFooter(dbc.Button("Close", id="close-modal-parse", color='danger', n_clicks=0)),
        ],
        id='modal-parse',
        is_open=False,
    ),
    dbc.Modal(
        [
            dbc.ModalHeader('Error'),
            dbc.ModalBody('Make sure that all fields are filled.'),
            dbc.ModalFooter(dbc.Button("Close", id="close-modal-fit", color='danger', n_clicks=0)),
        ],
        id='modal-fit',
        is_open=False,
    )
])
