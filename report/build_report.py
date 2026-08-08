#!/usr/bin/env python3
"""Build the post-course portfolio technical report.

This script validates the preserved aggregate result files before generating the PDF.
It does not read raw prompts, responses, or per-example judge outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report" / "llm-rl-final-project-portfolio-report.pdf"

NAVY = HexColor("#17324D")
BLUE = HexColor("#2F6BFF")
TEAL = HexColor("#138A7E")
AMBER = HexColor("#D78126")
INK = HexColor("#1D2733")
MUTED = HexColor("#5E6A78")
LINE = HexColor("#D8E0E8")
PALE_BLUE = HexColor("#EDF3FF")
PALE_TEAL = HexColor("#EAF7F4")
PALE_AMBER = HexColor("#FFF4E5")
PALE_GRAY = HexColor("#F4F6F8")


def load_and_validate_results() -> dict[str, object]:
    """Load aggregate records and stop if the preserved facts no longer match."""
    method_a_path = (
        ROOT
        / "experiments"
        / "methodA_traincalib"
        / "methodA_repeated_gpt_summary.json"
    )
    method_b_path = (
        ROOT
        / "experiments"
        / "methodB_rank_traincalib_seeds12"
        / "methodB_summary.json"
    )
    reward_path = ROOT / "experiments" / "reward_model_summary.json"

    method_a = json.loads(method_a_path.read_text(encoding="utf-8"))
    method_b = json.loads(method_b_path.read_text(encoding="utf-8"))
    reward = json.loads(reward_path.read_text(encoding="utf-8"))

    a_values = [round(float(row["offline"]), 4) for row in method_a["rows"]]
    if a_values != [0.8636, 0.8621, 0.8571]:
        raise ValueError(f"Unexpected Method A values: {a_values}")
    if round(float(method_a["mean_offline"]), 4) != 0.8609:
        raise ValueError("Unexpected Method A mean")

    best_b = next(
        row
        for row in method_b["summary_by_seed_step"]
        if row["seed"] == 2 and row["step"] == 100
    )
    b_values = [round(float(value), 4) for value in best_b["values"]]
    if b_values != [0.7258, 0.7636, 0.7705]:
        raise ValueError(f"Unexpected Method B values: {b_values}")
    if round(float(best_b["mean_online"]), 4) != 0.7533:
        raise ValueError("Unexpected Method B mean")
    if sum(value > 0.75 for value in b_values) != 2:
        raise ValueError("Unexpected Method B threshold count")

    reward_accuracy = float(reward["value"])
    if reward_accuracy != 0.84375:
        raise ValueError("Unexpected reward-model pair accuracy")

    return {
        "method_a_values": a_values,
        "method_a_mean": 0.8609,
        "method_b_values": b_values,
        "method_b_mean": 0.7533,
        "reward_accuracy": reward_accuracy,
    }


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=32,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=15,
            leading=20,
            textColor=MUTED,
            spaceAfter=18,
        ),
        "kicker": ParagraphStyle(
            "Kicker",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=11,
            tracking=1.2,
            textColor=BLUE,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "Heading1Custom",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=25,
            textColor=NAVY,
            spaceBefore=0,
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "Heading2Custom",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=INK,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=INK,
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "SmallCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=MUTED,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "CaptionCustom",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7.8,
            leading=10,
            textColor=MUTED,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "callout": ParagraphStyle(
            "CalloutCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=INK,
        ),
        "table_header": ParagraphStyle(
            "TableHeaderCustom",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=colors.white,
        ),
        "number": ParagraphStyle(
            "NumberCustom",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=21,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
        "number_label": ParagraphStyle(
            "NumberLabelCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "mono": ParagraphStyle(
            "MonoCustom",
            parent=base["BodyText"],
            fontName="Courier",
            fontSize=8.2,
            leading=11,
            textColor=INK,
        ),
    }


def p(text: str, styles: dict[str, ParagraphStyle], name: str = "body") -> Paragraph:
    return Paragraph(text, styles[name])


def info_box(text: str, styles: dict[str, ParagraphStyle], background=PALE_BLUE) -> Table:
    table = Table([[p(text, styles, "callout")]], colWidths=[6.55 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.8, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return table


def result_cards(data: list[tuple[str, str]], styles: dict[str, ParagraphStyle]) -> Table:
    cells = []
    for value, label in data:
        cells.append(
            [
                p(value, styles, "number"),
                p(label, styles, "number_label"),
            ]
        )
    row = [Table([[cell[0]], [cell[1]]], colWidths=[1.95 * inch]) for cell in cells]
    table = Table([row], colWidths=[2.12 * inch] * len(row))
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_GRAY),
                ("BOX", (0, 0), (-1, -1), 0.8, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.8, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return table


def workflow_drawing() -> Drawing:
    drawing = Drawing(470, 92)
    labels = [
        ("5,000 preference pairs", PALE_BLUE, BLUE),
        ("Offline + online methods", PALE_TEAL, TEAL),
        ("Local head-to-head eval", PALE_AMBER, AMBER),
    ]
    x_positions = [0, 165, 330]
    for index, ((label, fill, stroke), x) in enumerate(zip(labels, x_positions)):
        drawing.add(Rect(x, 25, 140, 48, rx=7, ry=7, fillColor=fill, strokeColor=stroke))
        drawing.add(
            String(
                x + 70,
                50,
                label,
                fontName="Helvetica-Bold",
                fontSize=8.3,
                fillColor=INK,
                textAnchor="middle",
            )
        )
        if index < 2:
            drawing.add(Rect(x + 143, 47, 17, 3, fillColor=LINE, strokeColor=LINE))
            drawing.add(String(x + 153, 43, ">", fontName="Helvetica-Bold", fontSize=11, fillColor=MUTED))
    return drawing


def horizontal_bars(
    labels: list[str], values: list[float], bar_colors: list[colors.Color]
) -> Drawing:
    drawing = Drawing(470, 38 + 34 * len(values))
    label_x = 0
    bar_x = 105
    track_width = 295
    score_x = 414
    height = drawing.height
    for index, (label, value, color) in enumerate(zip(labels, values, bar_colors)):
        y = height - 28 - index * 34
        drawing.add(
            String(label_x, y + 2, label, fontName="Helvetica", fontSize=8.5, fillColor=INK)
        )
        drawing.add(Rect(bar_x, y, track_width, 11, fillColor=PALE_GRAY, strokeColor=LINE))
        drawing.add(
            Rect(
                bar_x,
                y,
                track_width * value,
                11,
                fillColor=color,
                strokeColor=color,
            )
        )
        drawing.add(
            String(
                score_x,
                y + 1,
                f"{value:.4f}",
                fontName="Helvetica-Bold",
                fontSize=8.5,
                fillColor=INK,
            )
        )
    return drawing


def rank_drawing() -> Drawing:
    drawing = Drawing(470, 88)
    values = ["-1", "-1/3", "1/3", "1"]
    fills = [HexColor("#FDECEC"), PALE_AMBER, PALE_TEAL, PALE_BLUE]
    strokes = [HexColor("#C75656"), AMBER, TEAL, BLUE]
    for index, (value, fill, stroke) in enumerate(zip(values, fills, strokes)):
        x = 25 + index * 110
        drawing.add(Rect(x, 25, 86, 44, rx=7, ry=7, fillColor=fill, strokeColor=stroke))
        drawing.add(
            String(
                x + 43,
                48,
                value,
                fontName="Helvetica-Bold",
                fontSize=13,
                fillColor=INK,
                textAnchor="middle",
            )
        )
        drawing.add(
            String(
                x + 43,
                11,
                f"rank {index + 1}",
                fontName="Helvetica",
                fontSize=7.5,
                fillColor=MUTED,
                textAnchor="middle",
            )
        )
    return drawing


def data_table(
    rows: list[list[object]],
    widths: list[float],
    styles: dict[str, ParagraphStyle],
) -> Table:
    rendered = []
    for row_index, row in enumerate(rows):
        rendered.append(
            [
                cell
                if isinstance(cell, Paragraph)
                else p(str(cell), styles, "small" if row_index else "table_header")
                for cell in row
            ]
        )
    table = Table(rendered, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.6, LINE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_GRAY]),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def first_page(canvas, doc) -> None:
    canvas.saveState()
    width, height = letter
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 0.28 * inch, width, 0.28 * inch, fill=1, stroke=0)
    canvas.setFillColor(BLUE)
    canvas.rect(0, 0, 0.18 * inch, height, fill=1, stroke=0)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(0.65 * inch, 0.38 * inch, "Yifan Xu | Portfolio Technical Report")
    canvas.drawRightString(width - 0.55 * inch, 0.38 * inch, "Page 1")
    canvas.restoreState()


def later_pages(canvas, doc) -> None:
    canvas.saveState()
    width, height = letter
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.6)
    canvas.line(0.72 * inch, height - 0.48 * inch, width - 0.72 * inch, height - 0.48 * inch)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(0.72 * inch, height - 0.36 * inch, "Calibrated Reward Ensembles for LLM RLHF")
    canvas.drawRightString(width - 0.72 * inch, 0.38 * inch, f"Page {doc.page}")
    canvas.drawString(0.72 * inch, 0.38 * inch, "Post-course portfolio edition")
    canvas.restoreState()


def build_report() -> None:
    results = load_and_validate_results()
    styles = make_styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="Calibrated Reward Ensembles for LLM RLHF - Portfolio Technical Report",
        author="Yifan Xu",
        subject="Post-course CS 185/285 final-project portfolio report",
        creator="ReportLab",
    )

    story = []

    # Page 1: cover and executive summary.
    story.extend(
        [
            Spacer(1, 0.34 * inch),
            p("PORTFOLIO TECHNICAL REPORT", styles, "kicker"),
            p("Calibrated Reward Ensembles for LLM RLHF", styles, "title"),
            p(
                "Offline preference optimization, reward modeling, and online RL for open-ended instruction following",
                styles,
                "subtitle",
            ),
            result_cards(
                [
                    ("0.84375", "Reward-model pair accuracy"),
                    ("0.8609", "Method A mean local win rate"),
                    ("0.7533", "Method B best mean"),
                ],
                styles,
            ),
            Spacer(1, 0.23 * inch),
            data_table(
                [
                    ["Project field", "Preserved value"],
                    ["Author", "Yifan Xu"],
                    ["Course context", "UC Berkeley CS 185/285"],
                    ["Base model", "Qwen/Qwen2.5-1.5B-Instruct"],
                    ["Benchmark scale", "5,000 preference pairs"],
                    ["Document status", "Post-course portfolio edition, August 2026"],
                ],
                [1.55 * inch, 5.0 * inch],
                styles,
            ),
            Spacer(1, 0.18 * inch),
            info_box(
                "<b>Provenance.</b> Prepared after course completion from Yifan Xu's original code, experiments, and submitted materials. This is not the report submitted for grading and is not an official course artifact.",
                styles,
                PALE_AMBER,
            ),
            p("Executive summary", styles, "h2"),
            p(
                "The project evaluates a reward-model pipeline and two project investigations: calibrated reward-ensemble reranking (Method A) and rank-advantage GRPO (Method B). The preserved records show a reward-model pair accuracy of 0.84375, a Method A mean local GPT-5.4 win rate of 0.8609 across three evaluations, and a best Method B three-evaluation mean of 0.7533.",
                styles,
            ),
            p(
                "Raw prompt/response corpora and per-example generations are excluded from the portfolio snapshot. This report summarizes only aggregate values checked against preserved project records.",
                styles,
                "small",
            ),
        ]
    )

    # Page 2: framing and implementation.
    story.extend(
        [
            PageBreak(),
            p("01 / PROJECT FRAMING", styles, "kicker"),
            p("From preference data to evaluated policies", styles, "h1"),
            p(
                "The implementation covers the main components of an LLM RLHF workflow: offline preference objectives, reward-model training and evaluation, online group-relative policy optimization, deterministic submission generation, and local head-to-head evaluation.",
                styles,
            ),
            Spacer(1, 0.08 * inch),
            workflow_drawing(),
            p(
                "Figure 1. High-level workflow represented by the cleaned portfolio snapshot.",
                styles,
                "caption",
            ),
            p("Preserved method surface", styles, "h2"),
            data_table(
                [
                    ["Component", "Implementation represented in the codebase"],
                    ["Offline preference optimization", "DPO, IPO, and AOT objectives"],
                    ["Reward modeling", "Pairwise reward-model training, scoring, calibration, and ensemble evaluation"],
                    ["Online policy optimization", "GRPO, DrGRPO, GSPO, and rank-advantage GRPO pathways"],
                    ["Experiment orchestration", "Modal entry points, configuration records, and build/evaluation utilities"],
                ],
                [1.75 * inch, 4.8 * inch],
                styles,
            ),
            p("Artifact boundary", styles, "h2"),
            info_box(
                "<b>Included:</b> method code, orchestration scripts, aggregate summaries, configuration metadata, manifests, and reproduction notes.<br/><b>Excluded:</b> raw open-domain prompts and responses, generated candidate JSONLs, detailed judge outputs, checkpoints, caches, build products, and submission bundles.",
                styles,
                PALE_TEAL,
            ),
            p("Why the boundary matters", styles, "h2"),
            p(
                "The source corpus and generation files contain third-party or model-generated text. Excluding those files reduces privacy risk while leaving the implementation and aggregate evidence available for technical review. The cleaned repository is therefore a presentation artifact, not a self-contained training distribution.",
                styles,
            ),
        ]
    )

    # Page 3: Method A.
    a_values = results["method_a_values"]
    story.extend(
        [
            PageBreak(),
            p("02 / METHOD A", styles, "kicker"),
            p("Calibrated reward-ensemble reranking", styles, "h1"),
            p(
                "Method A uses three reward models. For each model, score statistics from the training preference data define a center and scale. Each candidate score is standardized before the three standardized scores are averaged.",
                styles,
            ),
            info_box(
                "<b>Calibration rule</b><br/><font name='Courier'>z_i(x, y) = (r_i(x, y) - c_i) / s_i</font><br/>The ensemble reward is the mean of the calibrated scores. For each prompt, Method A selects the fixed candidate with the highest ensemble score.",
                styles,
            ),
            p("Fixed-candidate protocol", styles, "h2"),
            data_table(
                [
                    ["Step", "Preserved protocol"],
                    ["1", "Train three reward models using different seeds."],
                    ["2", "Select one checkpoint per seed using held-out preference diagnostics."],
                    ["3", "Calibrate scores with training preference statistics."],
                    ["4", "Score fixed DPO, IPO, and AOT candidates and select the maximum per prompt."],
                ],
                [0.65 * inch, 5.9 * inch],
                styles,
            ),
            p("Repeated local evaluations", styles, "h2"),
            horizontal_bars(
                ["Evaluation 1", "Evaluation 2", "Evaluation 3"],
                a_values,
                [BLUE, TEAL, AMBER],
            ),
            data_table(
                [
                    ["Evaluation", "Local GPT-5.4 win rate"],
                    ["1", "0.8636"],
                    ["2", "0.8621"],
                    ["3", "0.8571"],
                    ["Mean", "0.8609"],
                ],
                [2.3 * inch, 4.25 * inch],
                styles,
            ),
            p(
                "The three recorded values span 0.0065. This narrow range is a descriptive observation about these three local evaluations; it is not a statistical-significance claim.",
                styles,
                "caption",
            ),
        ]
    )

    # Page 4: Method B.
    b_values = results["method_b_values"]
    story.extend(
        [
            PageBreak(),
            p("03 / METHOD B", styles, "kicker"),
            p("Rank-advantage GRPO", styles, "h1"),
            p(
                "Method B tests whether the ordering induced by the calibrated reward ensemble can replace reward magnitude in the group-relative advantage. For a four-response group, the responses are ordered by calibrated ensemble score and assigned fixed rank advantages.",
                styles,
            ),
            rank_drawing(),
            p(
                "Figure 2. Rank-only advantages used for a four-response group.",
                styles,
                "caption",
            ),
            info_box(
                "<b>Design question.</b> Rank-only advantages remove dependence on the absolute reward scale, but they also discard differences in reward magnitude. The preserved Method B records evaluate this tradeoff as an online ablation.",
                styles,
                PALE_TEAL,
            ),
            p("Best preserved configuration", styles, "h2"),
            p(
                "The best preserved configuration is policy seed 2 at step 100. Its three repeated local evaluations are shown below.",
                styles,
            ),
            horizontal_bars(
                ["Evaluation 1", "Evaluation 2", "Evaluation 3"],
                b_values,
                [AMBER, TEAL, BLUE],
            ),
            data_table(
                [
                    ["Evaluation", "Local win rate", "Above 0.75"],
                    ["1", "0.7258", "No"],
                    ["2", "0.7636", "Yes"],
                    ["3", "0.7705", "Yes"],
                    ["Mean", "0.7533", "2 of 3"],
                ],
                [1.6 * inch, 2.5 * inch, 2.45 * inch],
                styles,
            ),
            p(
                "The phrase '2 of 3' refers to repeated evaluations of one configuration, not to training seeds.",
                styles,
                "caption",
            ),
        ]
    )

    # Page 5: findings and limitations.
    story.extend(
        [
            PageBreak(),
            p("04 / FINDINGS", styles, "kicker"),
            p("What the preserved records support", styles, "h1"),
            result_cards(
                [
                    ("0.84375", "Reward-model pair accuracy"),
                    ("0.8609", "Method A mean"),
                    ("0.7533", "Method B best mean"),
                ],
                styles,
            ),
            p(
                "These three values represent different evaluation components and should not be treated as directly interchangeable measurements. They are presented together only as a compact index of the preserved project outcomes.",
                styles,
                "caption",
            ),
            p("Supported observations", styles, "h2"),
            data_table(
                [
                    ["Observation", "Evidence boundary"],
                    ["Reward model", "The preserved pair accuracy is 0.84375."],
                    ["Method A", "Three local win rates average 0.8609 and range from 0.8571 to 0.8636."],
                    ["Method B", "The best preserved three-evaluation mean is 0.7533; two values exceed 0.75."],
                ],
                [1.3 * inch, 5.25 * inch],
                styles,
            ),
            p("Limitations", styles, "h2"),
            info_box(
                "1. The GPT-5.4 numbers are preserved local head-to-head evaluation outcomes, not an official course grade.<br/>2. Three repeated evaluations are insufficient for claims of statistical significance.<br/>3. Raw prompt/response records are excluded, so the public snapshot cannot reproduce the reported evaluations by itself.<br/>4. No result is claimed for unrecorded configurations or evaluation sets.",
                styles,
                PALE_AMBER,
            ),
            p("Interpretation", styles, "h2"),
            p(
                "Within the preserved records, calibrated fixed-candidate reranking produced the stronger and more consistent local result. Rank-advantage GRPO remained a meaningful ablation: its best recorded mean was above 0.75, but one of the three repeated evaluations fell below that value. This comparison is descriptive and limited to the recorded settings.",
                styles,
            ),
        ]
    )

    # Page 6: reproducibility, record sources, and disclosure.
    story.extend(
        [
            PageBreak(),
            p("05 / REPRODUCIBILITY AND PROVENANCE", styles, "kicker"),
            p("A reviewable, privacy-aware portfolio snapshot", styles, "h1"),
            p("Repository map", styles, "h2"),
            data_table(
                [
                    ["Path", "Purpose"],
                    ["llm_rl_final_proj/", "Training, reward-model, rollout, and RL implementation"],
                    ["scripts/modal_train.py", "Remote training and evaluation entry points"],
                    ["experiments/methodA_traincalib/", "Method A protocol, calibration metadata, and aggregate summaries"],
                    ["experiments/methodB_rank_traincalib_seeds12/", "Method B configuration and aggregate summaries"],
                    ["README_for_experiments.md", "Historical reproduction commands; restricted assets required"],
                ],
                [2.4 * inch, 4.15 * inch],
                styles,
            ),
            p("Aggregate record sources", styles, "h2"),
            p(
                "<font name='Courier'>experiments/reward_model_summary.json</font><br/><font name='Courier'>experiments/methodA_traincalib/methodA_repeated_gpt_summary.json</font><br/><font name='Courier'>experiments/methodB_rank_traincalib_seeds12/methodB_summary.json</font><br/><font name='Courier'>dataset/wildchat_min4_judged_5k_v1/manifest.json</font>",
                styles,
                "mono",
            ),
            p("Portfolio status and AI disclosure", styles, "h2"),
            info_box(
                "This Portfolio Technical Report was prepared after course completion from Yifan Xu's original code, experiments, and submitted materials. It was not submitted for grading and is not an official course document. Editorial organization and typesetting were assisted by OpenAI Codex. No new experimental result was generated for this report; numerical claims were constrained to the preserved aggregate records listed above.",
                styles,
                PALE_BLUE,
            ),
            p("Attribution", styles, "h2"),
            p(
                "The project was completed in the context of UC Berkeley CS 185/285. The repository contains portions of course starter code and Yifan Xu's project implementation and experiments. The retained MIT license and NOTICE.md document this distinction and do not imply endorsement by UC Berkeley or course staff.",
                styles,
            ),
            p("Conclusion", styles, "h2"),
            p(
                "The cleaned snapshot preserves the technical core of the project while separating code and aggregate evidence from potentially sensitive text corpora and generation artifacts. Method A records a 0.8609 mean across three local evaluations; Method B records a best three-evaluation mean of 0.7533; and the reward model records 0.84375 pair accuracy.",
                styles,
            ),
        ]
    )

    doc.build(story, onFirstPage=first_page, onLaterPages=later_pages)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build_report()
