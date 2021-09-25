import dash_bootstrap_components as dbc
import dash_html_components as html

def is_identifier(child):
    if(('className' not in child['props']) or (child['props']['className'] != 'identifier')):
       return True
    else:
       return False

def remove_identifiers(children):
    return list(filter(is_identifier, children))

def get_html_from_fit(fit):
    return dbc.Table([
        html.Thead([
            html.Tr([
                html.Th('Loihi variable'),
                html.Th('Value')
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td([
                    html.Span('Current decay '),
                    html.Small('(compartmentCurrentDecay)')
                ]),
                html.Td(fit['I_decay'])
            ]),
            html.Tr([
                html.Td([
                    html.Span('Voltage decay '),
                    html.Small('(compartmentVoltageDecay)')
                ]),
                html.Td(fit['v_decay'])
            ]),
            html.Tr([
                html.Td([
                    html.Span('Threshold mantissa '),
                    html.Small('(vThMant)')
                ]),
                html.Td(fit['threshold_mant'])
            ]),
            html.Tr([
                html.Td([
                    html.Span('Weight mantissa '),
                    html.Small('(weight)')
                ]),
                html.Td(fit['weight_mant'])
            ])
        ])
    ])
