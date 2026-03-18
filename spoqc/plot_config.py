import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go

def set_pub_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12
    })

    font_template = go.layout.Template(
        layout=dict(
            font=dict(size=14),
            title=dict(font=dict(size=18)),
            xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)),
            yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)),
            legend=dict(font=dict(size=14))
        )
    )

    pio.templates["publication_fonts"] = font_template
    pio.templates.default = "plotly_white+publication_fonts"