import dash

from brian2 import * 
from brian2.units.allunits import * # get everything

from widget.html import *
from widget.callbacks import *

# Initialize app storage and add brian2 units
app_storage = {}
app_storage['brian_units'] = { unit: eval(unit) for unit in [*units.allunits.__all__, *units.__all__] }

# Create app
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# Add root html element
app.layout = html.Div(container)

# Add callbacks to app
init_callbacks(app, app_storage)

# Run server
app.run_server(port=8899, debug=True)
