"""Export scored GMNPS results into manuscript-ready assets."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _escape_latex(value: object, float_format: str) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        text = float_format % value
    else:
        text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def latex_table_from_frame(
    frame: pd.DataFrame,
    caption: str,
    label: str,
    max_rows: int | None = None,
    float_format: str = "%.3f",
) -> str:
    """Render a compact LaTeX table for Nature Portfolio draft assets."""

    table = frame.head(max_rows) if max_rows is not None else frame
    colspec = "l" * len(table.columns)
    header = " & ".join(_escape_latex(c, float_format) for c in table.columns)
    rows = [
        " & ".join(_escape_latex(value, float_format) for value in row)
        for row in table.itertuples(index=False, name=None)
    ]
    body = "\\\\\n".join(rows)
    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        f"\\caption{{{_escape_latex(caption, float_format)}}}\n"
        f"\\label{{{_escape_latex(label, float_format)}}}\n"
        f"\\begin{{tabular}}{{{colspec}}}\n"
        "\\hline\n"
        f"{header}\\\\\n"
        "\\hline\n"
        f"{body}\\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def manuscript_table_summaries(
    food_summary: pd.DataFrame,
    preservation_metrics: dict[str, object] | None = None,
    benchmark: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build small source tables for the main text and supplements."""

    tables: dict[str, pd.DataFrame] = {}
    cols = [
        "food_id",
        "food_name",
        "food_group",
        "FCS2",
        "GMNPS_mean",
        "GMNPS_sd",
        "GMNPS_p05",
        "GMNPS_p95",
        "dominant_channel",
    ]
    available = [c for c in cols if c in food_summary.columns]
    tables["top_food_summary"] = food_summary[available].sort_values(
        ["food_group", "GMNPS_mean"], ascending=[True, False]
    )
    if preservation_metrics is not None:
        tables["nps_preservation_metrics"] = pd.DataFrame([preservation_metrics])
    if benchmark is not None:
        tables["synthetic_benchmark"] = benchmark.copy()
    return tables


def export_article_bundle(
    output_dir: str | Path,
    individual_food: pd.DataFrame,
    food_summary: pd.DataFrame,
    manifest: dict[str, object],
    preservation_metrics: dict[str, object] | None = None,
    heterogeneity: pd.DataFrame | None = None,
    benchmark: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Write reproducible score tables, figure source data and LaTeX snippets."""

    out = Path(output_dir)
    tables_dir = out / "tables"
    latex_dir = out / "latex"
    figure_data_dir = out / "figure_source_data"
    for path in (tables_dir, latex_dir, figure_data_dir):
        path.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    paths["individual_food_csv"] = str(tables_dir / "individual_food_scores.csv")
    individual_food.to_csv(paths["individual_food_csv"], index=False)

    paths["food_summary_csv"] = str(tables_dir / "food_summary.csv")
    food_summary.to_csv(paths["food_summary_csv"], index=False)

    if preservation_metrics is not None:
        paths["nps_preservation_json"] = str(tables_dir / "nps_preservation_metrics.json")
        Path(paths["nps_preservation_json"]).write_text(
            json.dumps(preservation_metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if heterogeneity is not None:
        paths["heterogeneity_csv"] = str(figure_data_dir / "food_group_heterogeneity.csv")
        heterogeneity.to_csv(paths["heterogeneity_csv"], index=False)
    if benchmark is not None:
        paths["synthetic_benchmark_csv"] = str(figure_data_dir / "synthetic_benchmark.csv")
        benchmark.to_csv(paths["synthetic_benchmark_csv"], index=False)

    full_manifest = dict(manifest)
    full_manifest["exported_paths"] = paths
    paths["manifest_json"] = str(out / "article_export_manifest.json")
    Path(paths["manifest_json"]).write_text(
        json.dumps(full_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for name, table in manuscript_table_summaries(food_summary, preservation_metrics, benchmark).items():
        tex = latex_table_from_frame(
            table,
            caption=name.replace("_", " ").title(),
            label=f"tab:{name}",
            max_rows=25 if name == "top_food_summary" else None,
        )
        paths[f"{name}_tex"] = str(latex_dir / f"{name}.tex")
        Path(paths[f"{name}_tex"]).write_text(tex, encoding="utf-8")

    return paths
