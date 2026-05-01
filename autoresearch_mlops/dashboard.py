# import streamlit as st
# import mlflow
# import pandas as pd

# st.set_page_config(page_title="AutoResearch Dashboard", layout="wide")

# st.title("🧠 AutoResearch MLOps Dashboard")

# # ---------------------------
# # 1. MLflow runs
# # ---------------------------
# st.header("📊 Experiment Runs (MLflow)")

# runs = mlflow.search_runs()

# if len(runs) > 0:
#     st.dataframe(runs[[
#         "run_id",
#         "metrics.rmse",
#         "params.model_type",
#         "params.lr",
#         "params.epochs"
#     ]].sort_values("metrics.rmse"))

#     st.subheader("RMSE Trend")
#     st.line_chart(runs["metrics.rmse"])

#     st.subheader("Best Run")
#     best = runs.sort_values("metrics.rmse").iloc[0]
#     st.json(best.to_dict())
# else:
#     st.warning("No runs found yet. Train model first.")

# # ---------------------------
# # 2. Program.md Viewer
# # ---------------------------
# st.header("📄 program.md (Control Brain)")

# try:
#     with open("program.md", "r") as f:
#         program = f.read()
#     st.text_area("Program", program, height=500)
# except:
#     st.error("program.md not found")

# # ---------------------------
# # 3. Live Metrics Insight
# # ---------------------------
# st.header("📈 Insights")

# if len(runs) > 5:
#     st.write("Recent performance trend:")
#     st.line_chart(runs.tail(20)["metrics.rmse"])