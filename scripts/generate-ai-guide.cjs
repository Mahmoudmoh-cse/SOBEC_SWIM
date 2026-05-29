const {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  Header,
  HeadingLevel,
  LevelFormat,
  Packer,
  PageBreak,
  PageNumber,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableLayoutType,
  TableRow,
  TextRun,
  VerticalAlign,
  WidthType,
} = require("docx");
const fs = require("fs");
const path = require("path");

const C = {
  navy: "0D2B45",
  blue: "1A6B9E",
  blueLight: "E6F1FB",
  teal: "0F6E56",
  tealLight: "E1F5EE",
  purple: "3C3489",
  purpleLight: "EEEDFE",
  amber: "854F0B",
  amberLight: "FAEEDA",
  coral: "8B2500",
  coralLight: "FAECE7",
  green: "1A6B3A",
  greenLight: "E8F5ED",
  gray: "475467",
  lightGray: "F4F6F8",
  midGray: "D0D5DD",
  white: "FFFFFF",
  black: "101828",
  codeGray: "1E1E2E",
  codeText: "CDD6F4",
};

const W = 9360;

const bdr = (color = C.midGray, size = 4) => ({ style: BorderStyle.SINGLE, size, color });
const noBdr = () => ({ style: BorderStyle.NONE, size: 0, color: C.white });
const allB = (color = C.midGray, size = 4) => ({
  top: bdr(color, size),
  bottom: bdr(color, size),
  left: bdr(color, size),
  right: bdr(color, size),
});
const noB = () => ({ top: noBdr(), bottom: noBdr(), left: noBdr(), right: noBdr() });

const tx = (text, opts = {}) =>
  new TextRun({
    text,
    font: "Arial",
    size: opts.size || 22,
    bold: opts.bold || false,
    color: opts.color || C.black,
    italics: opts.italic || false,
  });

const txCode = (text) => new TextRun({ text, font: "Courier New", size: 16, color: C.codeText });

const p = (children, opts = {}) =>
  new Paragraph({
    children: Array.isArray(children) ? children : [children],
    alignment: opts.align || AlignmentType.LEFT,
    spacing: { before: opts.before || 0, after: opts.after === undefined ? 100 : opts.after },
    keepLines: opts.keepLines || false,
    keepNext: opts.keepNext || false,
    ...(opts.extra || {}),
  });

const h1 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, font: "Arial", size: 36, bold: true, color: C.navy })],
    spacing: { before: 420, after: 160 },
    keepNext: true,
  });

const h2 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, font: "Arial", size: 28, bold: true, color: C.blue })],
    spacing: { before: 300, after: 120 },
    keepNext: true,
  });

const h3 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text, font: "Arial", size: 24, bold: true, color: C.teal })],
    spacing: { before: 220, after: 90 },
    keepNext: true,
  });

const body = (text, color = C.gray) => p([tx(text, { size: 22, color })], { before: 0, after: 120 });

const bl = (text, prefix = "") =>
  new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    children: prefix
      ? [tx(prefix, { bold: true, size: 21, color: C.navy }), tx(text, { size: 21, color: C.gray })]
      : [tx(text, { size: 21, color: C.gray })],
    spacing: { before: 20, after: 60 },
  });

const nb = (text, prefix = "") =>
  new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    children: prefix
      ? [tx(prefix, { bold: true, size: 21, color: C.navy }), tx(text, { size: 21, color: C.gray })]
      : [tx(text, { size: 21, color: C.gray })],
    spacing: { before: 20, after: 60 },
  });

const sp = (before = 140) => p([tx("")], { before, after: 0 });
const pg = () => new Paragraph({ children: [new PageBreak()] });

const hr = (color = C.blue) =>
  new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color } },
    spacing: { before: 220, after: 220 },
    children: [],
  });

const table = (rows, columnWidths) =>
  new Table({
    width: { size: columnWidths.reduce((sum, width) => sum + width, 0), type: WidthType.DXA },
    columnWidths,
    layout: TableLayoutType.FIXED,
    rows,
  });

const tCell = (content, opts = {}) =>
  new TableCell({
    borders: opts.borders || allB(opts.bColor || C.midGray),
    width: { size: opts.w || W, type: WidthType.DXA },
    shading: { fill: opts.fill || C.white, type: ShadingType.CLEAR },
    verticalAlign: opts.verticalAlign || VerticalAlign.CENTER,
    margins: { top: 100, bottom: 100, left: 140, right: 140 },
    children: Array.isArray(content)
      ? content
      : [p([tx(content, { size: opts.ts || 20, bold: opts.bold || false, color: opts.tc || C.gray })], { before: 0, after: 0 })],
  });

const tbl2 = (rows, w1 = 2800, w2 = 6560, hFill = C.navy) =>
  table(
    rows.map(
      (row, i) =>
        new TableRow({
          children: [
            tCell(row[0], {
              w: w1,
              fill: i === 0 ? hFill : i % 2 === 0 ? C.lightGray : C.white,
              bold: true,
              tc: i === 0 ? C.white : C.navy,
              ts: i === 0 ? 21 : 20,
            }),
            tCell(
              Array.isArray(row[1])
                ? row[1].map((line) => p([tx(line, { size: 19, color: i === 0 ? C.white : C.gray })], { before: 0, after: 44 }))
                : [p([tx(row[1], { size: i === 0 ? 21 : 20, color: i === 0 ? C.white : C.gray })], { before: 0, after: 0 })],
              { w: w2, fill: i === 0 ? hFill : i % 2 === 0 ? C.lightGray : C.white },
            ),
          ],
        }),
    ),
    [w1, w2],
  );

const tbl3 = (rows, widths = [3120, 3120, 3120], hFill = C.navy) =>
  table(
    rows.map(
      (row, i) =>
        new TableRow({
          children: row.map((content, j) =>
            tCell(
              Array.isArray(content)
                ? content.map((line) => p([tx(line, { size: 18, color: i === 0 ? C.white : C.gray })], { before: 0, after: 44 }))
                : [p([tx(content, { size: i === 0 ? 20 : 18, bold: i === 0, color: i === 0 ? C.white : C.gray })], { before: 0, after: 0 })],
              { w: widths[j], fill: i === 0 ? hFill : i % 2 === 0 ? C.lightGray : C.white },
            ),
          ),
        }),
    ),
    widths,
  );

const infoBox = (title, lines, fill = C.blueLight, accentColor = C.blue, textColor = C.gray) =>
  new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: [W],
    layout: TableLayoutType.FIXED,
    rows: [
      new TableRow({
        children: [
          new TableCell({
            borders: { top: bdr(accentColor, 10), bottom: bdr(C.midGray), left: bdr(C.midGray), right: bdr(C.midGray) },
            width: { size: W, type: WidthType.DXA },
            shading: { fill, type: ShadingType.CLEAR },
            margins: { top: 140, bottom: 160, left: 220, right: 220 },
            children: [
              p([tx(title, { size: 22, bold: true, color: accentColor })], { before: 0, after: 80 }),
              ...lines.map((line) =>
                line === ""
                  ? p([tx("")], { before: 0, after: 40 })
                  : p([tx(line, { size: 20, color: textColor })], { before: 0, after: 60 }),
              ),
            ],
          }),
        ],
      }),
    ],
  });

const banner = (num, title, subtitle, fill = C.blue) =>
  new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: [W],
    layout: TableLayoutType.FIXED,
    rows: [
      new TableRow({
        children: [
          new TableCell({
            borders: noB(),
            width: { size: W, type: WidthType.DXA },
            shading: { fill, type: ShadingType.CLEAR },
            margins: { top: 220, bottom: 220, left: 300, right: 300 },
            children: [
              p([tx(`${num} `, { size: 20, color: C.white }), tx(title, { size: 34, bold: true, color: C.white })], {
                before: 0,
                after: 60,
                keepNext: true,
              }),
              p([tx(subtitle, { size: 20, color: C.white, italic: true })], { before: 0, after: 0 }),
            ],
          }),
        ],
      }),
    ],
  });

const codeBlock = (lines) =>
  new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: [W],
    layout: TableLayoutType.FIXED,
    rows: [
      new TableRow({
        children: [
          new TableCell({
            borders: allB(C.codeGray, 8),
            width: { size: W, type: WidthType.DXA },
            shading: { fill: C.codeGray, type: ShadingType.CLEAR },
            margins: { top: 160, bottom: 160, left: 220, right: 220 },
            children: lines.map((line) => p([txCode(line)], { before: 0, after: 24 })),
          }),
        ],
      }),
    ],
  });

const coverStats = () =>
  table(
    [
      new TableRow({
        children: [
          statCell("4", "AI engines", C.blue, C.blueLight),
          statCell("12+", "AI models used", C.teal, C.tealLight),
          statCell("30+", "Prompt templates", C.purple, C.purpleLight),
          statCell("100%", "Swimmer-specific", C.coral, C.coralLight),
        ],
      }),
    ],
    [2340, 2340, 2340, 2340],
  );

const statCell = (value, label, color, fill) =>
  tCell(
    [
      p([tx(value, { size: 28, bold: true, color })], { before: 0, after: 20, align: AlignmentType.CENTER }),
      p([tx(label, { size: 18, color: C.gray })], { before: 0, after: 0, align: AlignmentType.CENTER }),
    ],
    { w: 2340, fill, borders: allB(color) },
  );

const header = new Header({
  children: [
    table(
      [
        new TableRow({
          children: [
            tCell([p([tx("AquaIQ - AI Engineering Guide", { size: 18, bold: true, color: C.blue })], { before: 0, after: 0 })], {
              w: W / 2,
              fill: C.white,
              borders: noB(),
            }),
            tCell(
              [p([tx("Confidential - Internal Build Reference", { size: 18, color: C.gray })], { before: 0, after: 0, align: AlignmentType.RIGHT })],
              { w: W / 2, fill: C.white, borders: noB() },
            ),
          ],
        }),
      ],
      [W / 2, W / 2],
    ),
    new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: C.blue } }, spacing: { before: 80, after: 0 }, children: [] }),
  ],
});

const footer = new Footer({
  children: [
    new Paragraph({ border: { top: { style: BorderStyle.SINGLE, size: 4, color: C.midGray } }, spacing: { before: 80, after: 0 }, children: [] }),
    p(
      [
        tx("AquaIQ AI Guide  |  ", { size: 18, color: C.gray }),
        new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 18, color: C.gray }),
      ],
      { before: 60, after: 0, align: AlignmentType.CENTER },
    ),
  ],
});

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: C.navy },
        paragraph: { spacing: { before: 420, after: 160 }, outlineLevel: 0 },
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: C.blue },
        paragraph: { spacing: { before: 300, after: 120 }, outlineLevel: 1 },
      },
      {
        id: "Heading3",
        name: "Heading 3",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: C.teal },
        paragraph: { spacing: { before: 220, after: 90 }, outlineLevel: 2 },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "\u2022",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
      {
        reference: "numbers",
        levels: [
          {
            level: 0,
            format: LevelFormat.DECIMAL,
            text: "%1.",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1080, right: 1080, bottom: 1080, left: 1080, header: 708, footer: 708 },
        },
      },
      headers: { default: header },
      footers: { default: footer },
      children: [
        sp(1500),
        new Table({
          width: { size: W, type: WidthType.DXA },
          columnWidths: [W],
          layout: TableLayoutType.FIXED,
          rows: [
            new TableRow({
              children: [
                new TableCell({
                  borders: noB(),
                  width: { size: W, type: WidthType.DXA },
                  shading: { fill: C.navy, type: ShadingType.CLEAR },
                  margins: { top: 500, bottom: 500, left: 500, right: 500 },
                  children: [
                    p([tx("AquaIQ", { size: 80, bold: true, color: C.white })], { before: 0, after: 100, align: AlignmentType.CENTER }),
                    p([tx("AI Engineering Guide", { size: 32, color: "7EC8E3", italic: true })], { before: 0, after: 180, align: AlignmentType.CENTER }),
                    p([tx("The Complete AI Blueprint", { size: 28, bold: true, color: C.white })], {
                      before: 0,
                      after: 80,
                      align: AlignmentType.CENTER,
                    }),
                    p([tx("Models | Prompts | Pipelines | Integrations | Real Code", { size: 22, color: "7EC8E3" })], {
                      before: 0,
                      after: 0,
                      align: AlignmentType.CENTER,
                    }),
                  ],
                }),
              ],
            }),
          ],
        }),
        sp(300),
        coverStats(),
        pg(),

        banner("01", "AI System Overview", "What AI does in AquaIQ, which models are used, and how they connect", C.navy),
        sp(160),
        h2("1.1 The role of AI in AquaIQ"),
        body("AquaIQ uses AI in four distinct ways. Each way solves a different coaching problem and uses a different type of model. The result is a system that feels intelligent without hiding the coach's judgment."),
        tbl2(
          [
            ["AI role in AquaIQ", "What it means in practice"],
            ["Natural language coaching", "Claude reads swimmer data and writes coaching advice in plain language: specific, motivating, and immediately usable."],
            ["Computer vision", "MediaPipe Pose reads swimmer video frame by frame, extracts joint coordinates, and detects technique faults without human labeling."],
            ["Predictive modeling", "Classical ML models predict race times, fatigue levels, optimal pacing, and readiness from swimmer-specific history."],
            ["Adaptive rule engine", "Transparent rules read check-in, HRV, RPE, and session-load signals to adjust training while keeping outputs auditable."],
          ],
          2600,
          6760,
        ),
        sp(160),
        h2("1.2 Full AI model stack"),
        tbl3(
          [
            ["Model / framework", "Role", "Priority"],
            ["Claude Sonnet", "Main coaching intelligence for plans, reports, and deep analysis", "Critical"],
            ["Claude Haiku", "Fast summaries, alerts, check-in reads, drill lookups", "High"],
            ["MediaPipe Pose", "33-landmark pose estimation for uploaded swim video", "Critical"],
            ["scikit-learn", "Fatigue curves, split prediction, performance regression", "High"],
            ["statsmodels / SciPy", "Mental correlation analysis, HRV trends, signal processing", "Medium"],
            ["OpenCV", "Frame extraction, preprocessing, stabilization before pose analysis", "Medium"],
          ],
          [3000, 4560, 1800],
        ),
        sp(160),
        infoBox(
          "Complete AI request lifecycle",
          [
            "1. Data collection: swimmer logs a check-in, session, race result, or uploads video.",
            "2. Validation: FastAPI validates and persists the record.",
            "3. Context assembly: the backend reads swimmer profile, recent sessions, active faults, phase, and mental scores.",
            "4. Prompt construction: Jinja-style templates convert that context into a structured system and user prompt.",
            "5. Provider call: Claude or a local mock provider generates a structured output.",
            "6. Parsing and storage: JSON is parsed, metadata is stored, and the dashboard receives the result.",
            "7. Feedback loop: coach overrides and ratings become future context.",
          ],
          C.blueLight,
          C.blue,
        ),
        pg(),

        banner("02", "Claude API - The Language Brain", "How to use Claude for coaching intelligence, plan generation, and feedback", C.blue),
        sp(160),
        h2("2.1 Why Claude is the first LLM adapter"),
        tbl2(
          [
            ["Decision factor", "Why it matters for AquaIQ"],
            ["Instruction following", "Coaching outputs need valid JSON, strict sections, and parseable fields."],
            ["Long context", "A six-month swimmer history can be compressed and sent as one prompt when needed."],
            ["Reasoning quality", "Plan and race outputs must explain why a recommendation fits the swimmer."],
            ["API consistency", "Stable model configuration and versioning make production behavior easier to control."],
          ],
          2600,
          6760,
        ),
        sp(160),
        h2("2.2 API setup and configuration"),
        codeBlock([
          "# requirements.txt",
          "anthropic==0.25.0",
          "python-dotenv==1.0.0",
          "",
          "# .env",
          "ANTHROPIC_API_KEY=sk-ant-...",
          "CLAUDE_MODEL_HEAVY=claude-sonnet-4-20250514",
          "CLAUDE_MODEL_FAST=claude-haiku-4-5-20251001",
          "",
          "# ai/client.py",
          "import anthropic",
          "from functools import lru_cache",
          "",
          "@lru_cache(maxsize=1)",
          "def get_claude_client() -> anthropic.Anthropic:",
          "    return anthropic.Anthropic()",
        ]),
        sp(160),
        h2("2.3 Prompt architecture"),
        body("Every Claude call uses a stable system prompt plus a dynamic user prompt. The system prompt defines coaching behavior; the user prompt supplies swimmer context and the exact output contract."),
        h3("System prompt - training plan generator"),
        codeBlock([
          "You are AquaIQ Coach - an elite AI swimming coach with deep knowledge of:",
          "- Swimming periodization science: base, build, peak, taper",
          "- Stroke biomechanics and event-specific demands",
          "- Energy systems: aerobic, threshold, VO2 max, anaerobic speed",
          "- Adaptive training based on recovery and compliance",
          "",
          "All outputs must be personalized, scientifically grounded, actionable,",
          "and coach-readable. Return valid JSON unless explicitly asked for prose.",
          "Never invent missing swimmer data. Flag gaps explicitly.",
        ]),
        h3("User prompt - training plan generator"),
        codeBlock([
          "Generate a {{ weeks }}-week training plan for this swimmer.",
          "",
          "SWIMMER PROFILE",
          "Name: {{ swimmer.name }}",
          "Level: {{ swimmer.level }}",
          "Primary event: {{ swimmer.primary_event }}",
          "Personal best: {{ swimmer.pb_seconds }} seconds",
          "Sessions per week available: {{ swimmer.sessions_per_week }}",
          "",
          "CURRENT STATUS",
          "Current phase: {{ plan.phase }}",
          "Weeks to race: {{ plan.weeks_to_race }}",
          "Active technique faults: {{ faults | join(', ') }}",
          "",
          "OUTPUT REQUIRED",
          "{",
          '  "phase_rationale": "string",',
          '  "weekly_structure": [{ "week": 1, "phase": "base", "volume_m": 24000, "sessions": [] }],',
          '  "adaptation_flags": ["string"]',
          "}",
        ]),
        sp(160),
        h2("2.4 Response parsing and error handling"),
        codeBlock([
          "# ai/parser.py",
          "import json, re",
          "from dataclasses import dataclass",
          "",
          "@dataclass",
          "class AIResponse:",
          "    success: bool",
          "    data: object",
          "    raw_text: str",
          "    tokens_used: int",
          "    model: str",
          "",
          "def parse_claude_response(response) -> AIResponse:",
          "    raw = response.content[0].text",
          "    tokens = response.usage.input_tokens + response.usage.output_tokens",
          "    model = response.model",
          "    try:",
          "        return AIResponse(True, json.loads(raw), raw, tokens, model)",
          "    except json.JSONDecodeError:",
          "        pass",
          '    match = re.search(r"```(?:json)?\\s*([\\s\\S]+?)```", raw)',
          "    if match:",
          "        return AIResponse(True, json.loads(match.group(1).strip()), raw, tokens, model)",
          '    return AIResponse(True, {"text": raw}, raw, tokens, model)',
        ]),
        pg(),

        banner("03", "MediaPipe - Technique Vision", "How AquaIQ sees and analyzes swimmer movement using computer vision", C.teal),
        sp(160),
        h2("3.1 How MediaPipe works in AquaIQ"),
        body("MediaPipe Pose detects 33 human landmarks in a video frame. AquaIQ runs it offline on uploaded swimmer videos to extract biomechanical measurements a coach cannot reliably measure by eye."),
        infoBox(
          "What computer vision adds",
          [
            "Frame-by-frame precision: 30-60 frames per second instead of a few observed key positions.",
            "Exact angles: elbow, shoulder, hip, knee, and ankle angles can be measured consistently.",
            "Symmetry detection: left-right timing asymmetry can be seen in milliseconds.",
            "Trend tracking: the same metric can be compared across weeks of training.",
          ],
          C.tealLight,
          C.teal,
        ),
        sp(160),
        h2("3.2 Landmarks AquaIQ uses"),
        tbl3(
          [
            ["Landmark ID", "Body part", "Used for"],
            ["11, 12", "Shoulders", "Entry angle, body rotation, recovery path"],
            ["13, 14", "Elbows", "Early vertical forearm and catch mechanics"],
            ["15, 16", "Wrists", "Hand entry, catch position, push-through path"],
            ["23, 24", "Hips", "Hip rotation and body-line stability"],
            ["25-28", "Knees / ankles", "Kick amplitude, breaststroke alignment, ankle flexibility"],
            ["0", "Nose", "Head position and breathing height"],
          ],
          [1600, 2200, 5560],
          C.teal,
        ),
        sp(160),
        h2("3.3 Video analysis pipeline"),
        codeBlock([
          "# ai/technique/video_pipeline.py",
          "import cv2, mediapipe as mp, numpy as np",
          "",
          "mp_pose = mp.solutions.pose",
          "",
          "class SwimVideoAnalyzer:",
          "    def __init__(self):",
          "        self.pose = mp_pose.Pose(",
          "            static_image_mode=False,",
          "            model_complexity=2,",
          "            min_detection_confidence=0.6,",
          "            min_tracking_confidence=0.5,",
          "        )",
          "",
          "    def analyze_video(self, video_path: str) -> dict:",
          "        cap = cv2.VideoCapture(video_path)",
          "        fps = cap.get(cv2.CAP_PROP_FPS) or 30",
          "        all_landmarks = []",
          "        frame_index = 0",
          "        while cap.isOpened():",
          "            ok, frame = cap.read()",
          "            if not ok:",
          "                break",
          "            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)",
          "            result = self.pose.process(rgb)",
          "            if result.pose_landmarks:",
          "                lm = result.pose_landmarks.landmark",
          "                all_landmarks.append({",
          '                    "frame": frame_index,',
          '                    "timestamp_s": frame_index / fps,',
          '                    "landmarks": {i: {"x": lm[i].x, "y": lm[i].y, "z": lm[i].z}',
          "                                  for i in [0,11,12,13,14,15,16,23,24,25,26,27,28]}",
          "                })",
          "            frame_index += 1",
          "        cap.release()",
          "        angles = self._calculate_angles(all_landmarks)",
          "        faults = self._detect_faults(angles)",
          "        return {\"angles\": angles, \"faults\": faults, \"frame_count\": frame_index}",
        ]),
        sp(160),
        h2("3.4 Fault detection thresholds"),
        tbl3(
          [
            ["Fault", "Detection logic", "Severity"],
            ["Dropped elbow", "elbow_angle_min > 155 degrees during catch phase", "Critical above 160"],
            ["Wide kick", "ankle_y_range > 0.45 normalized per kick cycle", "Warning above 0.40"],
            ["Under-rotation", "hip_rotation_max < 38 degrees for freestyle", "Warning below 40"],
            ["Head-high breathing", "nose_y rises above configured surface proxy", "Critical above threshold"],
            ["Asymmetric kick", "left-right knee angle difference > 8 degrees", "Critical above 12"],
          ],
          [2300, 4560, 2500],
          C.teal,
        ),
        pg(),

        banner("04", "ML Models - Predictive Analytics", "scikit-learn models for fatigue, race prediction, and performance trends", C.purple),
        sp(160),
        h2("4.1 The five ML models in AquaIQ"),
        tbl2(
          [
            ["Model", "What it predicts"],
            ["Race time predictor", "Linear regression from load, technique score, HRV average, and taper state to predicted race time."],
            ["Fatigue curve model", "Exponential decay fitted to split deterioration patterns for the swimmer."],
            ["HRV threshold detector", "Rolling z-score model that flags suppressed recovery versus a personal baseline."],
            ["Training load optimizer", "Finds the weekly volume that maximizes performance gain without crossing fatigue thresholds."],
            ["Mental-performance correlator", "Identifies which mental dimensions predict better race execution for this swimmer."],
          ],
          2800,
          6560,
          C.purple,
        ),
        sp(160),
        h2("4.2 Fatigue curve model"),
        codeBlock([
          "# ai/ml/fatigue_model.py",
          "import numpy as np",
          "from scipy.optimize import curve_fit",
          "from sklearn.metrics import r2_score",
          "from dataclasses import dataclass",
          "",
          "@dataclass",
          "class FatigueCurve:",
          "    alpha: float",
          "    beta: float",
          "    offset: float",
          "    r2: float",
          "    n_races: int",
          "",
          "    def predict_split(self, length_number: int) -> float:",
          "        return self.alpha * np.exp(self.beta * length_number) + self.offset",
          "",
          "def fit_fatigue_curve(race_splits_list: list[list[float]]) -> FatigueCurve:",
          "    if len(race_splits_list) < 3:",
          "        return FatigueCurve(12.0, 0.03, 10.5, 0.0, 0)",
          "    xs, ys = [], []",
          "    for splits in race_splits_list:",
          "        for i, split in enumerate(splits):",
          "            xs.append(i)",
          "            ys.append(split)",
          "    xs, ys = np.array(xs, dtype=float), np.array(ys)",
          "    def model(x, alpha, beta, offset):",
          "        return alpha * np.exp(beta * x) + offset",
          "    popt, _ = curve_fit(model, xs, ys, p0=[ys[xs == 0].mean(), 0.05, ys[-1]], maxfev=5000)",
          "    return FatigueCurve(*popt, r2=r2_score(ys, model(xs, *popt)), n_races=len(race_splits_list))",
        ]),
        sp(160),
        h2("4.3 HRV-based load adaptation"),
        codeBlock([
          "# ai/ml/hrv_monitor.py",
          "import numpy as np",
          "from enum import Enum",
          "",
          "class LoadDecision(Enum):",
          '    FULL = "full"',
          '    REDUCE = "reduce"',
          '    RECOVER = "recover"',
          '    REST = "rest"',
          "",
          "def analyze_hrv(today_hrv: float, history: list[float]) -> tuple[str, float, str]:",
          "    if len(history) < 5:",
          '        return LoadDecision.FULL.value, 1.0, "Insufficient HRV history."',
          "    baseline = np.mean(history[-14:])",
          "    z_score = (today_hrv - baseline) / (np.std(history[-14:]) + 1e-6)",
          "    if z_score >= -0.5:",
          '        return LoadDecision.FULL.value, 1.0, "HRV normal. Proceed with plan."',
          "    if z_score >= -1.5:",
          '        return LoadDecision.REDUCE.value, 0.7, "Reduce main-set volume by 30%."',
          "    if z_score >= -2.5:",
          '        return LoadDecision.RECOVER.value, 0.4, "Technique and light aerobic only."',
          '    return LoadDecision.REST.value, 0.0, "Full rest day and coach review."',
        ]),
        pg(),

        banner("05", "Race Strategy AI", "How AquaIQ builds optimal split plans and models race performance", C.coral),
        sp(160),
        h2("5.1 Race strategy pipeline"),
        nb("Swimmer or coach requests strategy for an upcoming event."),
        nb("System loads previous race splits for the swimmer and event."),
        nb("Fatigue curve is fitted to the swimmer's split history."),
        nb("Four candidate strategies are generated: negative, even, positive, and custom."),
        nb("Each strategy is scored and the best one becomes the race brief."),
        nb("Claude translates the selected plan into coach-readable instructions."),
        sp(160),
        infoBox(
          "Strategy Score = (A x 40) + (B x 30) + (C x 20) + (D x 10)",
          [
            "A = Historical correlation: similarity to split patterns that produced PBs.",
            "B = Physiological appropriateness: fit for the event distance and energy system.",
            "C = Fatigue curve alignment: similarity to the swimmer's personal decay model.",
            "D = Readiness adjustment: taper stage, recovery, and recent RPE.",
          ],
          C.coralLight,
          C.coral,
        ),
        sp(160),
        h2("5.2 Split plan generation"),
        codeBlock([
          "# ai/race/strategy_engine.py",
          "from dataclasses import dataclass",
          "",
          "@dataclass",
          "class SplitPlan:",
          "    strategy_type: str",
          "    splits_s: list[float]",
          "    total_time_s: float",
          "    strategy_score: float",
          "    vs_pb_delta_s: float",
          "",
          "def normalize_to_total(splits: list[float], target: float) -> list[float]:",
          "    factor = target / sum(splits)",
          "    return [s * factor for s in splits]",
          "",
          "def generate_split_plans(event_distance_m: int, pb_time_s: float, fatigue_curve) -> list[SplitPlan]:",
          "    segment = 50 if event_distance_m >= 200 else 25",
          "    n_segments = event_distance_m // segment",
          "    target_time = pb_time_s * 0.995",
          "    even = target_time / n_segments",
          "    negative = normalize_to_total([even * (1 + (n_segments / 2 - i) * 0.008) for i in range(n_segments)], target_time)",
          "    even_plan = [even] * n_segments",
          "    custom = normalize_to_total(fatigue_curve.predict_full_race(n_segments), target_time)",
          "    return [",
          '        SplitPlan("negative", negative, sum(negative), 0, sum(negative) - pb_time_s),',
          '        SplitPlan("even", even_plan, sum(even_plan), 0, sum(even_plan) - pb_time_s),',
          '        SplitPlan("custom", custom, sum(custom), 0, sum(custom) - pb_time_s),',
          "    ]",
        ]),
        pg(),

        banner("06", "Mental Coach AI", "Pattern detection, routine generation, and visualization script delivery", C.purple),
        sp(160),
        h2("6.1 Mental AI architecture"),
        tbl2(
          [
            ["Component", "Technology and role"],
            ["Mental state collector", "FastAPI endpoint stores six-dimensional check-ins."],
            ["Correlation analyzer", "Pearson correlation links pre-race dimensions with race performance deltas."],
            ["Threshold detector", "Rules compare today's scores against personal thresholds."],
            ["Routine generator", "LLM adapter builds a timed pre-race protocol."],
            ["Pattern report", "Monthly coach-facing narrative summary of mental-performance signals."],
          ],
          2800,
          6560,
          C.purple,
        ),
        sp(160),
        h2("6.2 Mental pattern detection"),
        codeBlock([
          "# ai/mental/pattern_analyzer.py",
          "import numpy as np",
          "from scipy.stats import pearsonr",
          "from dataclasses import dataclass",
          "",
          'DIMENSIONS = ["focus", "confidence", "energy", "calm", "recovery", "motivation"]',
          "",
          "@dataclass",
          "class MentalThreshold:",
          "    dimension: str",
          "    threshold_value: float",
          "    correlation_r: float",
          "    performance_impact_s: float",
          "    confidence: str",
          "",
          "def build_mental_profile(checkins: list[dict], race_results: list[dict]) -> list[MentalThreshold]:",
          "    matched = match_checkins_to_races(checkins, race_results)",
          "    if len(matched) < 5:",
          "        return []",
          "    thresholds = []",
          "    for dim in DIMENSIONS:",
          '        scores = np.array([m["checkin"][dim] for m in matched])',
          '        performance = np.array([m["race"]["pb_pct_diff"] for m in matched])',
          "        if np.std(scores) < 0.5:",
          "            continue",
          "        r, pvalue = pearsonr(scores, performance)",
          "        if abs(r) > 0.3 and pvalue < 0.15:",
          "            thresholds.append(MentalThreshold(dim, find_threshold(scores, performance), r, 0.0, 'emerging'))",
          "    return thresholds",
        ]),
        sp(160),
        h2("6.3 Pre-race routine generation prompt"),
        codeBlock([
          "You are AquaIQ Mental Coach.",
          "Generate a specific, timed pre-race routine for the swimmer.",
          "Address the flagged mental dimensions and return JSON only.",
          "",
          "Swimmer: {{ swimmer.name }}",
          "Event: {{ swimmer.primary_event }}",
          "Minutes to race: {{ minutes_to_race }}",
          "Flagged dimensions: {{ flags | join(', ') }}",
          "",
          "{",
          '  "total_minutes": 30,',
          '  "routine_rationale": "string",',
          '  "steps": [{ "order": 1, "title": "string", "duration_min": 3, "instruction": "string" }]',
          "}",
        ]),
        pg(),

        banner("07", "AI Context Builder", "The system that assembles swimmer data into AI-ready context", C.amber),
        sp(160),
        h2("7.1 Why context matters"),
        body("The model has no memory between calls. Every AI response must be supplied with the right swimmer history, compressed enough to stay efficient and structured enough to reason over."),
        infoBox(
          "The Context Builder is the most important non-AI code in AquaIQ",
          [
            "Bad context produces generic advice that could apply to anyone.",
            "Good context produces specific recommendations that feel like the system knows the swimmer.",
            "The quality difference is mostly in how well the backend describes the swimmer to the model.",
          ],
          C.amberLight,
          C.amber,
        ),
        sp(160),
        h2("7.2 Context Builder implementation sketch"),
        codeBlock([
          "# ai/context_builder.py",
          "from dataclasses import dataclass, asdict",
          "from datetime import date, timedelta",
          "import json",
          "",
          "@dataclass",
          "class SwimmerContext:",
          "    name: str",
          "    level: str",
          "    primary_event: str",
          "    personal_bests: dict",
          "    current_phase: str",
          "    weeks_to_race: int",
          "    current_week_volume_m: int",
          "    today_hrv: float",
          "    hrv_baseline_14d: float",
          "    active_faults: list",
          "    recent_races: list",
          "    mental_thresholds: list",
          "",
          "def build_swimmer_context(swimmer_id: str, db, today: date | None = None) -> SwimmerContext:",
          "    today = today or date.today()",
          "    swimmer = db.query(Swimmer).get(swimmer_id)",
          "    plan = db.query(TrainingPlan).filter_by(swimmer_id=swimmer_id, is_active=True).first()",
          "    hrv_history = (",
          "        db.query(WearableData)",
          "        .filter(WearableData.swimmer_id == swimmer_id)",
          "        .filter(WearableData.timestamp >= today - timedelta(days=14))",
          "        .order_by(WearableData.timestamp)",
          "        .all()",
          "    )",
          "    races = (",
          "        db.query(RaceAnalysis)",
          "        .filter_by(swimmer_id=swimmer_id)",
          "        .order_by(RaceAnalysis.race_date.desc())",
          "        .limit(5)",
          "        .all()",
          "    )",
          "    return SwimmerContext(",
          "        name=swimmer.name,",
          "        level=swimmer.level,",
          "        primary_event=swimmer.primary_event,",
          "        personal_bests=swimmer.personal_bests or {},",
          "        current_phase=plan.current_phase if plan else 'base',",
          "        weeks_to_race=plan.weeks_total if plan else 12,",
          "        current_week_volume_m=current_week_volume(swimmer_id, db),",
          "        today_hrv=today_hrv(hrv_history),",
          "        hrv_baseline_14d=hrv_baseline(hrv_history),",
          "        active_faults=active_faults(swimmer_id, db),",
          "        recent_races=[race_to_dict(r) for r in races],",
          "        mental_thresholds=swimmer.mental_profile.get('thresholds', []),",
          "    )",
          "",
          "def context_to_prompt_string(ctx: SwimmerContext) -> str:",
          "    return json.dumps(asdict(ctx), indent=2, default=str)",
        ]),
        pg(),

        banner("08", "Testing and Quality Control", "How to make sure AI outputs work and keep working", C.navy),
        sp(160),
        h2("8.1 The three types of AI tests"),
        tbl2(
          [
            ["Test type", "What it checks"],
            ["Structural tests", "Does the output parse as JSON and contain required keys?"],
            ["Semantic tests", "Do low-HRV swimmers receive lower volume? Do taper weeks reduce load?"],
            ["Regression tests", "Do known profiles continue to produce similar plan quality after prompt changes?"],
          ],
          2600,
          6760,
        ),
        sp(160),
        h2("8.2 Structural test example"),
        codeBlock([
          "# tests/test_ai_plan.py",
          "def test_plan_structure():",
          "    swimmer = make_test_swimmer(level='age_group', weeks_to_race=12)",
          "    plan = generate_training_plan(swimmer)",
          "    assert plan['phase_rationale']",
          "    assert len(plan['weekly_structure']) == 12",
          "    for week in plan['weekly_structure']:",
          "        assert week['phase'] in ['base', 'build', 'peak', 'taper']",
          "        assert 1000 <= week['volume_m'] <= 60000",
          "",
          "def test_taper_week_volume():",
          "    plan = generate_training_plan(make_test_swimmer(weeks_to_race=8))",
          "    peak = [w['volume_m'] for w in plan['weekly_structure'] if w['phase'] == 'peak']",
          "    taper = [w['volume_m'] for w in plan['weekly_structure'] if w['phase'] == 'taper']",
          "    assert sum(taper) / len(taper) < (sum(peak) / len(peak)) * 0.75",
        ]),
        sp(160),
        h2("8.3 Prompt versioning"),
        codeBlock([
          "# ai/prompts/__init__.py",
          "PROMPT_REGISTRY = {",
          '    "training_plan_system": {"v1": "prompts/v1/training_plan_system.txt", "active": "v1"},',
          '    "technique_analysis_system": {"v1": "prompts/v1/technique_system.txt", "active": "v1"},',
          '    "race_strategy_system": {"v1": "prompts/v1/race_system.txt", "active": "v1"},',
          '    "mental_checkin_system": {"v1": "prompts/v1/mental_system.txt", "active": "v1"},',
          "}",
          "",
          "def get_prompt(name: str, version: str | None = None) -> str:",
          "    entry = PROMPT_REGISTRY[name]",
          "    selected = version or entry['active']",
          "    with open(entry[selected], encoding='utf-8') as prompt:",
          "        return prompt.read()",
        ]),
        pg(),

        banner("09", "AI Costs and Optimization", "How to keep Claude API costs low while delivering quality", C.teal),
        sp(160),
        h2("9.1 Token cost estimate per swimmer per month"),
        tbl3(
          [
            ["AI call type", "Frequency", "Estimated cost"],
            ["Weekly plan generation", "4x", "About $0.10 per month"],
            ["Daily check-in summary", "20x", "About $0.01 per month with fast model"],
            ["Technique analysis language", "4x", "About $0.04 per month"],
            ["Race strategy plan", "2x", "About $0.04 per month"],
            ["Mental pattern report", "1x", "About $0.03 per month"],
            ["Total per swimmer", "Monthly", "About $0.23"],
          ],
          [3000, 2000, 4360],
          C.teal,
        ),
        sp(160),
        h2("9.2 Cost reduction strategies"),
        tbl2(
          [
            ["Strategy", "Implementation"],
            ["Cache plans", "Only regenerate when race date, target, phase, or recovery status changes."],
            ["Use fast model for simple tasks", "Check-ins, alerts, and drill lookups do not need deep analysis."],
            ["Compress context", "Send derived summaries instead of raw session history unless detail is required."],
            ["Batch summaries", "Generate weekly mental summaries instead of one call after every check-in."],
            ["Async processing", "Let heavier reports run in the background with retry metadata."],
          ],
          2600,
          6760,
          C.teal,
        ),
        pg(),

        banner("10", "Wearable Data Integrations", "How to connect Garmin, Apple Watch, Polar, Whoop, and swim-specific devices", C.blue),
        sp(160),
        h2("10.1 Priority order"),
        tbl2(
          [
            ["Priority", "Device and integration detail"],
            ["1 - Garmin", "Most common in competitive swimming. Pulls SWOLF, stroke rate, strokes per length, lap times, and HR."],
            ["2 - Apple HealthKit", "Useful for iOS users, active calories, workouts, and accessible health data sync."],
            ["3 - Polar H10", "High-quality chest-strap HRV for serious athletes who value precision."],
            ["4 - Whoop", "Recovery, strain, and sleep-stage data for athletes already using the ecosystem."],
            ["5 - FORM goggles", "Real-time swim-specific split data for Phase 3 race-execution workflows."],
          ],
          1800,
          7560,
          C.blue,
        ),
        sp(160),
        h2("10.2 Garmin connector sketch"),
        codeBlock([
          "# integrations/garmin.py",
          "import requests",
          "from datetime import date, timedelta",
          "",
          'GARMIN_API_BASE = "https://apis.garmin.com/wellness-api/rest"',
          "",
          "class GarminConnector:",
          "    def __init__(self, access_token: str):",
          '        self.headers = {"Authorization": f"Bearer {access_token}"}',
          "",
          "    def get_swim_activities(self, start: date, end: date) -> list[dict]:",
          '        response = requests.get(f"{GARMIN_API_BASE}/activities", headers=self.headers)',
          '        activities = response.json().get("activities", [])',
          '        return [a for a in activities if a.get("activityType") == "POOL_SWIMMING"]',
          "",
          "    def sync_swimmer(self, swimmer_id: str, db) -> dict:",
          "        end = date.today()",
          "        start = end - timedelta(days=7)",
          "        activities = self.get_swim_activities(start, end)",
          '        result = {"sessions_synced": 0, "hrv_readings": 0}',
          "        for activity in activities:",
          "            db.merge(garmin_activity_to_session(swimmer_id, activity))",
          '            result["sessions_synced"] += 1',
          "        db.commit()",
          "        return result",
        ]),
        pg(),

        banner("11", "AI Launch Sequence", "The exact order to build and ship the AI components", C.amber),
        sp(160),
        h2("11.1 Minimum viable AI build order"),
        tbl2(
          [
            ["Build order", "Deliverable and reason"],
            ["Week 1", "Claude or local mock provider returns one coaching recommendation from a hardcoded swimmer profile."],
            ["Week 2", "Daily check-in form stores six dimensions and returns a two-sentence coaching read."],
            ["Week 3", "Training session generator outputs one full session with sets, rest, and targets."],
            ["Week 4", "Week plan generator creates five sessions that a coach can edit and save."],
            ["Weeks 5-6", "HRV and RPE adaptation modifies the next session safely."],
            ["Weeks 7-8", "Race strategy creates candidate split plans and a race brief."],
            ["Weeks 9-10", "Technique video analysis flags a small set of deterministic faults."],
            ["Weeks 11-12", "Mental routine generation starts from collected check-in data."],
          ],
          1600,
          7760,
          C.amber,
        ),
        sp(160),
        h2("11.2 Three questions before every AI feature"),
        infoBox(
          "Answer these before writing AI code",
          [
            "1. What data do I have? Design the context before the prompt.",
            "2. What is the exact output format? Define the JSON contract before testing prompts.",
            "3. How will I test it? Write structural and semantic checks before trusting the result.",
          ],
          C.amberLight,
          C.amber,
        ),
        hr(C.blue),
        p([tx("You have the vision. You have the swimmer knowledge. You have the coach experience.", { size: 22, bold: true, color: C.navy })], {
          before: 180,
          after: 80,
          align: AlignmentType.CENTER,
        }),
        p([tx("Now you have the AI blueprint. Build the first 50 lines this week.", { size: 22, color: C.blue, italic: true })], {
          before: 0,
          after: 200,
          align: AlignmentType.CENTER,
        }),
      ],
    },
  ],
});

const outputDir = path.join(__dirname, "..", "docs", "output");
const outputPath = path.join(outputDir, "AquaIQ_AI_Guide.docx");

fs.mkdirSync(outputDir, { recursive: true });

Packer.toBuffer(doc)
  .then((buffer) => {
    fs.writeFileSync(outputPath, buffer);
    console.log(`SUCCESS: generated ${outputPath}`);
  })
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
