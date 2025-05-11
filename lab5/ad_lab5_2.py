import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go

# ---------------------------
# Функції
# ---------------------------
def harmonic_with_noise(t, amplitude, frequency, phase, noise_mean, noise_covariance, noise=None):
    y_clean = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    if noise is None:
        noise = np.random.normal(noise_mean, np.sqrt(noise_covariance), size=len(t))
    y_noisy = y_clean + noise
    return y_clean, y_noisy, noise

def moving_average(signal, window_size=21):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

# ---------------------------
# Початкові параметри
# ---------------------------
init_amplitude = 1.0
init_frequency = 1.0
init_phase = 0.0
init_noise_mean = 0.0
init_noise_covariance = 0.1

# Масив часу
t = np.linspace(0, 10, 1000)
current_noise = np.random.normal(init_noise_mean, np.sqrt(init_noise_covariance), size=len(t))

# ---------------------------
# Dash App
# ---------------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Інтерактивна гармоніка з шумом і фільтрацією"),

    html.Div([
        html.Label('Amplitude:'),
        dcc.Slider(0.1, 5.0, 0.1, value=init_amplitude, id='amplitude-slider'),

        html.Label('Frequency:'),
        dcc.Slider(0.1, 5.0, 0.1, value=init_frequency, id='frequency-slider'),

        html.Label('Phase:'),
        dcc.Slider(0.0, 2*np.pi, 0.1, value=init_phase, id='phase-slider'),

        html.Label('Noise Mean:'),
        dcc.Slider(-1.0, 1.0, 0.05, value=init_noise_mean, id='mean-slider'),

        html.Label('Noise Covariance:'),
        dcc.Slider(0.001, 1.0, 0.01, value=init_noise_covariance, id='covariance-slider'),

        html.Label('Вибір фільтра:'),
        dcc.Dropdown([
            {'label': 'Без фільтрації', 'value': 'none'},
            {'label': 'Ковзне середнє', 'value': 'ma'}
        ], value='ma', id='filter-select'),

        dcc.Checklist(
            options=[
                {'label': 'Показати шум', 'value': 'show_noise'},
                {'label': 'Показати фільтрований сигнал', 'value': 'show_filtered'}
            ],
            value=['show_noise', 'show_filtered'],
            id='display-options'
        )
    ], style={'width': '50%', 'padding': '20px'}),

    dcc.Graph(id='harmonic-graph'),
])

@app.callback(
    Output('harmonic-graph', 'figure'),
    Input('amplitude-slider', 'value'),
    Input('frequency-slider', 'value'),
    Input('phase-slider', 'value'),
    Input('mean-slider', 'value'),
    Input('covariance-slider', 'value'),
    Input('display-options', 'value'),
    Input('filter-select', 'value'),
)
def update_graph(amplitude, frequency, phase, noise_mean, noise_covariance, display_options, filter_type):
    global current_noise

    noise_changed = noise_mean != update_graph.prev_noise_mean or noise_covariance != update_graph.prev_noise_covariance

    if noise_changed:
        current_noise = np.random.normal(noise_mean, np.sqrt(noise_covariance), size=len(t))
        update_graph.prev_noise_mean = noise_mean
        update_graph.prev_noise_covariance = noise_covariance

    y_clean, y_noisy, _ = harmonic_with_noise(t, amplitude, frequency, phase, noise_mean, noise_covariance, noise=current_noise)

    traces = [
        go.Scatter(x=t, y=y_clean, mode='lines', name='Clean Signal')
    ]

    if 'show_noise' in display_options:
        traces.append(go.Scatter(x=t, y=y_noisy, mode='lines', name='Noisy Signal'))

    if 'show_filtered' in display_options:
        if filter_type == 'ma':
            y_filtered = moving_average(y_noisy)
            traces.append(go.Scatter(x=t, y=y_filtered, mode='lines', name='Filtered (MA)'))

    return {
        'data': traces,
        'layout': go.Layout(
            title='Гармоніка з шумом та фільтрацією',
            xaxis={'title': 'Time'},
            yaxis={'title': 'Amplitude'},
            height=600
        )
    }

# Ініціалізація попередніх значень шуму
update_graph.prev_noise_mean = init_noise_mean
update_graph.prev_noise_covariance = init_noise_covariance

if __name__ == '__main__':
    app.run(debug=True)

