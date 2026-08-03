import glob
import json
import os
import pandas as pd


def compile_evaluations(results_dir="data/05_eval_results"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = (
        os.path.normpath(os.path.join(script_dir, "../../", results_dir))
        if not os.path.isabs(results_dir)
        else results_dir
    )

    json_files = glob.glob(os.path.join(target_dir, "*_final_metrics_api.json"))

    if not json_files:
        print(f"No JSON evaluation files found in {target_dir}")
        return

    all_models_data = []

    for file_path in json_files:
        model_name = os.path.basename(file_path).replace("_final_metrics_api.json", "")

        with open(file_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        model_row = {"Model": model_name}

        metric_totals = {}
        metric_counts = {}

        for topic, details in eval_data.items():
            script_id = details.get("id", "unknown_script")
            metrics = details.get("averaged_metrics", {})

            for metric, score in metrics.items():
                col_name = f"{script_id}_{metric}"
                model_row[col_name] = score
                metric_totals[metric] = metric_totals.get(metric, 0.0) + score
                metric_counts[metric] = metric_counts.get(metric, 0) + 1

        for metric, total in metric_totals.items():
            final_col_name = f"final_{metric}"
            model_row[final_col_name] = round(total / metric_counts[metric], 2)

        all_models_data.append(model_row)

    df = pd.DataFrame(all_models_data)

    sort_col = "final_overall_score" if "final_overall_score" in df.columns else "final_overall"

    if sort_col in df.columns:
        df = df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    else:
        print("Warning: Could not find an overall score column to sort by.")

    cols = df.columns.tolist()
    cols.remove("Model")
    final_cols = sorted([c for c in cols if c.startswith("final_")])
    script_cols = sorted([c for c in cols if not c.startswith("final_")])
    ordered_cols = ["Model"] + final_cols + script_cols

    df = df[ordered_cols]

    output_csv = os.path.join(target_dir, "final_evaluation_results.csv")
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"\nSuccessfully saved compiled results to:\n{output_csv}\n")
    print("MODEL RANKING (Highest to Lowest Overall Score)")
    print("-" * 55)

    for index, row in df.iterrows():
        score = row.get(sort_col, "N/A")
        print(f"{index + 1}. {row['Model'].ljust(30)} | Avg Score: {score}")
    print("-" * 55 + "\n")


if __name__ == "__main__":
    compile_evaluations()
