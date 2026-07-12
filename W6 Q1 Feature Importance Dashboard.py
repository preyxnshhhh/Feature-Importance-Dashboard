import pandas as pd
from flask import Flask, render_template_string
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

@app.route("/")
def dashboard():
    df = pd.read_csv("student_data.csv")

    # Preprocessing
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    target = df.columns[-1]
    X = df.drop(target, axis=1)
    y = df[target]

    # Model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importances = rf.feature_importances_

    data = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
    features = [x[0] for x in data]
    importance = [round(x[1], 4) for x in data]

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Feature Importance Dashboard</title>

        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels"></script>

        <style>
            body {
                margin: 0;
                font-family: 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
                color: white;
            }

            .container {
                width: 90%;
                margin: auto;
                text-align: center;
            }

            h1 {
                margin-top: 20px;
                font-size: 32px;
            }

            .card {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                padding: 20px;
                border-radius: 15px;
                margin-top: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            }

            canvas {
                max-width: 100%;
                height: 500px !important;
            }

            select {
                padding: 10px;
                border-radius: 10px;
                border: none;
                margin-top: 10px;
                font-size: 16px;
            }

            .info {
                margin-top: 20px;
                font-size: 14px;
                opacity: 0.8;
            }
        </style>
    </head>

    <body>

    <div class="container">
        <h1>📊 AI Feature Importance Dashboard</h1>

        <div class="card">
            <label>Select Top Features:</label>
            <br>
            <select id="topN">
                <option value="5">Top 5</option>
                <option value="10" selected>Top 10</option>
                <option value="15">Top 15</option>
                <option value="all">All</option>
            </select>

            <canvas id="chart"></canvas>

            <div class="info">
                Hover over bars to see exact importance values
            </div>
        </div>
    </div>

    <script>
        const allFeatures = {{ features | tojson }};
        const allImportance = {{ importance | tojson }};

        let chart;

        function renderChart(n) {
            let features = allFeatures;
            let importance = allImportance;

            if (n !== 'all') {
                features = allFeatures.slice(0, n);
                importance = allImportance.slice(0, n);
            }

            const ctx = document.getElementById('chart').getContext('2d');

            if (chart) chart.destroy();

            chart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: features,
                    datasets: [{
                        label: 'Importance',
                        data: importance,
                        backgroundColor: 'rgba(0, 200, 255, 0.6)',
                        borderRadius: 8
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false },
                        datalabels: {
                            color: 'white',
                            anchor: 'end',
                            align: 'top',
                            formatter: (val) => val
                        }
                    },
                    scales: {
                        x: {
                            ticks: {
                                color: "white",
                                maxRotation: 45,
                                minRotation: 45
                            }
                        },
                        y: {
                            ticks: {
                                color: "white"
                            }
                        }
                    }
                },
                plugins: [ChartDataLabels]
            });
        }

        document.getElementById('topN').addEventListener('change', function() {
            renderChart(this.value);
        });

        renderChart(10);
    </script>

    </body>
    </html>
    """

    return render_template_string(html, features=features, importance=importance)


if __name__ == "__main__":
    app.run(debug=True)