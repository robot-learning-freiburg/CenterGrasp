import wandb
import pandas as pd

pd.set_option("display.precision", 2)

# Grasping
print("Grasping Results")
api = wandb.Api()
runs = api.runs("robot-learning-lab/[CenterGrasp] SimEvalYCB")
data_list = []
for run in runs:
    if run.state in ["running", "failed"] or "robot_type" not in run.config:
        continue
    data = {
        "env": run.config["env"],
        "method": run.config["method"],
        "robot_type": run.config["robot_type"],
        "success_rate": run.summary["success_rate"],
        "declutter_rate": run.summary["declutter_rate"],
    }
    data_list.append(data)
grasp_data_frame = pd.DataFrame.from_records(data_list)
grasp_comparison_frame = grasp_data_frame.groupby(["robot_type", "env", "method"]).mean()
print(grasp_comparison_frame)

# Shape reconstruction
print("Shape Reconstruction Results")
api = wandb.Api()
runs = api.runs("robot-learning-lab/[CenterGrasp] SimEvalShape")
data_list = []
for run in runs:
    if run.state in ["running", "failed"] or "env" not in run.config:
        continue
    data = [
        {
            "env": run.config["env"],
            "method": "centergrasp",
            "avg_bi": run.summary["avg_bi_centergrasp"],
            "avg_iou": run.summary["avg_iou_centergrasp"],
        },
        {
            "env": run.config["env"],
            "method": "centergrasp_noicp",
            "avg_bi": run.summary["avg_bi_centergrasp_noicp"],
            "avg_iou": run.summary["avg_iou_centergrasp_noicp"],
        },
        {
            "env": run.config["env"],
            "method": "giga",
            "avg_bi": run.summary["avg_bi_giga"],
            "avg_iou": run.summary["avg_iou_giga"],
        },
    ]
    data_list.extend(data)
shape_data_frame = pd.DataFrame.from_records(data_list)
shape_comparison_frame = shape_data_frame.groupby(["env", "method"]).mean()
print(shape_comparison_frame)

# Print overall improvement
avg_centergrasp_success_rate = (
    grasp_comparison_frame["success_rate"].loc["franka", :, "centergrasp"].mean()
)
avg_giga_success_rate = grasp_comparison_frame["success_rate"].loc["franka", :, "giga"].mean()
improvement_grasp = avg_centergrasp_success_rate - avg_giga_success_rate
print(f"{improvement_grasp=:.2f}")

avg_centergrasp_cd = shape_comparison_frame["avg_bi"].loc[:, "centergrasp"].mean()
avg_giga_cd = shape_comparison_frame["avg_bi"].loc[:, "giga"].mean()
improvement_shape = avg_giga_cd - avg_centergrasp_cd
print(f"{improvement_shape=:.2f}")

print("done")
