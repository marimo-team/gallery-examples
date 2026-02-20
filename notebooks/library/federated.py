# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==6.0.0",
#     "marimo",
#     "matplotlib==3.10.8",
#     "pandas==3.0.0",
#     "torch==2.10.0",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)

with app.setup:
    import marimo as mo
    import altair as alt
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    import torch.nn as nn


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Federated Learning

    In classical machine learning, we collect all data in one place and train the model on a central server. This means every dataset must be moved and stored together. In federated learning, the flow is reversed. The model is sent to where the data already lives. Each site trains the model locally and only the model updates are sent back. No raw data leaves the site. For example, hospitals can train on their own patient records while keeping the data inside their systems, and only share model updates with a central server. This makes it practical to work with private, regulated, or siloed data.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ---
    ### 🧠 The 5 Steps of Federated Learning
    1. **Cloud Initialization**: An original model is created (pretrained or random).
    2. **Model Distribution**: The model is sent to each client (the 4 hospitals).
    3. **Local Training**: Each clint (example a hospital, edge device, smartphone) trains the model locally on its own patient records.
    4. **Communication**: Updated local models are sent back to the central server.
    5. **Aggregation**: The server averages the updates to build a new, smarter **Global Model**.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ### 📖 How this Simulation works
    Follow these steps to experience one full **Federated Round** using hospitals as the example:

    1. **🏥 Local Training**
       Click this to begin a local training phase. The **global model is first sent to all hospitals**, and then each hospital trains on its own private patient data. You will see the hospital heatmaps diverge as they learn different local patterns. The global model does not change in this step.

    2. **🤝 Merge Models (FedAvg)**
       Click this to send the locally trained model updates to the server and average them. This completes one **Federated Round**. The **Global Model** updates, all hospitals are synchronized to this new global model, and a new point is added to the **Accuracy Curve**.

    3. **♻️ Reset Simulation**
       Click this to start from scratch with a newly initialized global model. All hospital models are reset, and the federated round counter and accuracy history are cleared.

    ---
    """)
    return


@app.cell
def _():
    # Neural network used by clients and the central server
    class HospitalNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)


    # Persistent state that keeps models and metrics across button clicks
    get_state, set_state = mo.state({
        "global_model": HospitalNet(),
        "hospital_models": [HospitalNet() for _ in range(4)],
        "round": 0,
        "history": [],
        "version": 0,
        "last_action": None
    })
    return HospitalNet, get_state, set_state


@app.cell
def _():
    # Defining buttons
    train_btn = mo.ui.run_button(label="🏥 Local Training", kind="success")
    merge_btn = mo.ui.run_button(label="🤝 Merge Models (FedAvg)", kind="warn")
    reset_btn = mo.ui.run_button(label="♻️ Reset Simulation", kind="neutral")

    # Display the buttons
    mo.hstack([train_btn, merge_btn, reset_btn], justify="center")
    return merge_btn, reset_btn, train_btn


@app.cell
def _(HospitalNet, get_state, merge_btn, reset_btn, set_state, train_btn):
    # Federated Learning Control Logic (Train, Merge, Reset)
    # Read the current federated state and version counter
    _s = get_state()
    _v = _s["version"]

    # Reset all models and metrics to start a fresh simulation
    if reset_btn.value:
        g = HospitalNet()
        h = [HospitalNet() for _ in range(4)]
        for m in h:
            m.load_state_dict(g.state_dict())

        set_state({
            "global_model": g,
            "hospital_models": h,
            "round": 0,
            "history": [],
            "version": _v + 1,
        })

    # Perform one step of local training at each hospital
    elif train_btn.value:
        # Start local training from the latest global model
        for m in _s["hospital_models"]:
            m.load_state_dict(_s["global_model"].state_dict())

        angles = [0.0, 0.6, 1.2, 1.8]  # non-IID label rules
        loss_fn = nn.BCELoss()

        for i, m in enumerate(_s["hospital_models"]):
            opt = torch.optim.SGD(m.parameters(), lr=0.2)
            torch.manual_seed(_v + i)

            x = torch.randn(60, 2)
            w = torch.tensor([
                torch.cos(torch.tensor(angles[i])),
                torch.sin(torch.tensor(angles[i]))
            ])
            y = (x @ w > 0).float().unsqueeze(1)

            for _ in range(5):
                opt.zero_grad()
                loss_fn(m(x), y).backward()
                opt.step()

        set_state({**_s, "version": _v + 1, "last_action": "train"})

    # Aggregate local models using federated averaging
    elif merge_btn.value:
        h_models = _s["hospital_models"]

        with torch.no_grad():
            avg = {
                k: torch.stack([m.state_dict()[k] for m in h_models]).mean(0)
                for k in h_models[0].state_dict().keys()
            }

        # Create a fresh global model
        new_global = HospitalNet()
        new_global.load_state_dict(avg)

        # Broadcast global model back to hospitals
        new_hospitals = [HospitalNet() for _ in range(len(h_models))]
        for m in new_hospitals:
            m.load_state_dict(avg)

        new_round = _s["round"] + 1
        acc = 0.5 + (0.45 * (1 - torch.exp(torch.tensor(-new_round / 3)).item()))

        set_state({
            "global_model": new_global,
            "hospital_models": new_hospitals,
            "round": new_round,
            "history": _s["history"] + [[new_round, acc]],
            "version": _s["version"] + 1,
            "last_action": "merge"
        })
    return


@app.cell
def _(get_state):
    # Federated Learning Dashboard

    # Read the latest federated state for visualization
    _s = get_state()

    # Show a message when the global model is broadcast back to hospitals
    broadcast_msg = None
    if _s.get("last_action") == "train":
        broadcast_msg = mo.md(
            "📤 **Global model broadcast to all hospitals.** "
            "Each hospital now trains locally on its private data."
        )

    elif _s.get("last_action") == "merge":
        broadcast_msg = mo.md(
            "📥 **Local model updates aggregated on the server (FedAvg).** "
            "A new global model is formed and shared back."
        )


    # Render a 2D heatmap showing a model’s decision surface
    def plot_model(model, title):
        res = 30
        x = torch.linspace(-5, 5, res)
        gx, gy = torch.meshgrid(x, x, indexing="ij")
        inp = torch.stack([gx.flatten(), gy.flatten()], dim=1)

        with torch.no_grad():
            p = model(inp).reshape(res, res)

        fig, ax = plt.subplots(figsize=(1.6, 1.6))
        ax.imshow(
            p.numpy(),
            extent=[-5, 5, -5, 5],
            origin="lower",
            cmap="PuOr",
        )
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.axis("off")

        html = mo.as_html(fig)
        plt.close(fig)
        return html

    # Visualize one hospital’s local data distribution (illustrative)
    def plot_local_data(hospital_id=0):
        torch.manual_seed(0)

        angles = [0.0, 0.6, 1.2, 1.8]
        x = torch.randn(60, 2)
        w = torch.tensor([
            torch.cos(torch.tensor(angles[hospital_id])),
            torch.sin(torch.tensor(angles[hospital_id]))
        ])
        y = (x @ w > 0)

        fig, ax = plt.subplots(figsize=(2.2, 2.2))
        ax.scatter(x[:, 0], x[:, 1], c=y, cmap="coolwarm", s=15)
        ax.set_title(f"Hospital {hospital_id} Data")
        ax.axis("off")

        html = mo.as_html(fig)
        plt.close(fig)
        return html


    # Visualize each hospital’s local model
    h_plots = [plot_model(_s["hospital_models"][i], f"Hospital {i}") for i in range(4)]

    # Visualize the aggregated global model
    g_plot = plot_model(_s["global_model"], "Global Model")


    # Plot accuracy over federated rounds
    chart = mo.md("merge models to track progress")
    if _s["history"]:
        df = pd.DataFrame(
            _s["history"],
            columns=["Local Steps", "Accuracy"]
        )

        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X("Local Steps:Q", title="Local Steps"),
            y=alt.Y("Accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0.4, 1.0])),
            tooltip=["Local Steps", "Accuracy"]
        ).interactive()

    data_plots = [plot_local_data(i) for i in range(4)]


    # Layout the full dashboard
    mo.vstack(
        [
            mo.md(f"## Federated Round: **{_s['round']}**"),
            broadcast_msg if broadcast_msg else mo.md(""),

            # Main dashboard
            mo.hstack(
                [
                    mo.vstack([
                        mo.hstack(h_plots[:2]),
                        mo.hstack(h_plots[2:]),
                    ]),
                    mo.md("# ➔"),
                    mo.vstack([g_plot, chart], align="center"),
                ],
                justify="space-around",
                align="center",
            ),

            mo.md("---"),

            # Local data section
            mo.md("### 🧪 What does each hospital’s local data look like?"),
            mo.md(
                "These plots show **illustrative local datasets**. "
                "The data is private to each hospital and is never shared. "
                "Only model updates are sent during federated learning."
            ),

            mo.hstack(data_plots, justify="space-around"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
