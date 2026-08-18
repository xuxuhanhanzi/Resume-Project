from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUTPUT = Path(__file__).with_name("个人简历_内部审查预测版_v0.1.docx")

# Design authority: compact_reference_guide.
# Named override "technical_resume_density": one-column ATS resume, Microsoft YaHei
# for CJK coverage, 9.4 pt body, compact heading scale, and no decorative tables.
TOKENS = {
    "page": {
        "width": 8.5,
        "height": 11.0,
        "margin_vertical": 0.68,
        "margin_horizontal": 0.72,
        "header": 0.30,
        "footer": 0.34,
    },
    "font": "Microsoft YaHei",
    "latin_font": "Aptos",
    "body_size": 9.2,
    "body_after": 2.0,
    "body_line": 1.04,
    "title_size": 23,
    "subtitle_size": 10.5,
    "h1_size": 12.5,
    "h1_before": 5,
    "h1_after": 3,
    "h2_size": 10.2,
    "h2_before": 3,
    "h2_after": 1,
    "bullet_left": 0.27,
    "bullet_hanging": 0.16,
    "bullet_after": 0.8,
    "navy": "17365D",
    "blue": "2E75B6",
    "ink": "20242A",
    "muted": "666666",
    "line": "B9C7D8",
    "prediction_fill": "FFF2CC",
    "placeholder_fill": "FFE699",
    "review_fill": "EAF2F8",
    "risk": "9C0006",
}


def set_cell_margins(*args, **kwargs):
    raise RuntimeError("Tables are intentionally not used in this ATS-first resume.")


def set_run_font(run, size=None, bold=None, italic=None, color=None, font=None):
    font_name = font or TOKENS["font"]
    run.font.name = font_name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), TOKENS["latin_font"])
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), TOKENS["latin_font"])
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), font_name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def shade_paragraph(paragraph, fill):
    p_pr = paragraph._p.get_or_add_pPr()
    shd = p_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        p_pr.append(shd)
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)


def add_bottom_border(paragraph, color=None, size="10", space="2"):
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = p_pr.find(qn("w:pBdr"))
    if p_bdr is None:
        p_bdr = OxmlElement("w:pBdr")
        p_pr.append(p_bdr)
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), size)
    bottom.set(qn("w:space"), space)
    bottom.set(qn("w:color"), color or TOKENS["blue"])
    p_bdr.append(bottom)


def keep_with_next(paragraph):
    paragraph.paragraph_format.keep_with_next = True


def set_repeatable_numbering(doc: Document) -> int:
    numbering = doc.part.numbering_part.element
    abstract_ids = [
        int(el.get(qn("w:abstractNumId")))
        for el in numbering.findall(qn("w:abstractNum"))
    ]
    num_ids = [int(el.get(qn("w:numId"))) for el in numbering.findall(qn("w:num"))]
    abstract_id = max(abstract_ids or [0]) + 1
    num_id = max(num_ids or [0]) + 1

    abstract = OxmlElement("w:abstractNum")
    abstract.set(qn("w:abstractNumId"), str(abstract_id))
    multi = OxmlElement("w:multiLevelType")
    multi.set(qn("w:val"), "singleLevel")
    abstract.append(multi)

    lvl = OxmlElement("w:lvl")
    lvl.set(qn("w:ilvl"), "0")
    start = OxmlElement("w:start")
    start.set(qn("w:val"), "1")
    num_fmt = OxmlElement("w:numFmt")
    num_fmt.set(qn("w:val"), "bullet")
    lvl_text = OxmlElement("w:lvlText")
    lvl_text.set(qn("w:val"), "•")
    lvl_jc = OxmlElement("w:lvlJc")
    lvl_jc.set(qn("w:val"), "left")
    lvl.extend([start, num_fmt, lvl_text, lvl_jc])

    p_pr = OxmlElement("w:pPr")
    tabs = OxmlElement("w:tabs")
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "num")
    tab.set(qn("w:pos"), str(int(TOKENS["bullet_left"] * 1440)))
    tabs.append(tab)
    ind = OxmlElement("w:ind")
    ind.set(qn("w:left"), str(int(TOKENS["bullet_left"] * 1440)))
    ind.set(qn("w:hanging"), str(int(TOKENS["bullet_hanging"] * 1440)))
    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:after"), str(int(TOKENS["bullet_after"] * 20)))
    spacing.set(qn("w:line"), "259")
    spacing.set(qn("w:lineRule"), "auto")
    p_pr.extend([tabs, ind, spacing])
    lvl.append(p_pr)

    r_pr = OxmlElement("w:rPr")
    r_fonts = OxmlElement("w:rFonts")
    r_fonts.set(qn("w:ascii"), TOKENS["latin_font"])
    r_fonts.set(qn("w:hAnsi"), TOKENS["latin_font"])
    r_pr.append(r_fonts)
    lvl.append(r_pr)
    abstract.append(lvl)
    numbering.append(abstract)

    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(num_id))
    abstract_ref = OxmlElement("w:abstractNumId")
    abstract_ref.set(qn("w:val"), str(abstract_id))
    num.append(abstract_ref)
    numbering.append(num)
    return num_id


def apply_numbering(paragraph, num_id):
    p_pr = paragraph._p.get_or_add_pPr()
    num_pr = p_pr.find(qn("w:numPr"))
    if num_pr is None:
        num_pr = OxmlElement("w:numPr")
        p_pr.append(num_pr)
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), "0")
    num_id_el = OxmlElement("w:numId")
    num_id_el.set(qn("w:val"), str(num_id))
    num_pr.extend([ilvl, num_id_el])


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("内部审查版")
    set_run_font(run, size=8, color=TOKENS["muted"])


def configure_styles(doc):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = TOKENS["font"]
    normal._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), TOKENS["latin_font"])
    normal._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), TOKENS["latin_font"])
    normal._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), TOKENS["font"])
    normal.font.size = Pt(TOKENS["body_size"])
    normal.font.color.rgb = RGBColor.from_string(TOKENS["ink"])
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(TOKENS["body_after"])
    normal.paragraph_format.line_spacing = TOKENS["body_line"]

    for name, size, before, after in [
        ("Heading 1", TOKENS["h1_size"], TOKENS["h1_before"], TOKENS["h1_after"]),
        ("Heading 2", TOKENS["h2_size"], TOKENS["h2_before"], TOKENS["h2_after"]),
        ("Heading 3", 9.6, 2, 1),
    ]:
        style = styles[name]
        style.font.name = TOKENS["font"]
        style._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), TOKENS["latin_font"])
        style._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), TOKENS["latin_font"])
        style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), TOKENS["font"])
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(TOKENS["navy"])
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    meta = styles.add_style("Resume Meta", WD_STYLE_TYPE.PARAGRAPH)
    meta.font.name = TOKENS["font"]
    meta.font.size = Pt(8.7)
    meta.font.color.rgb = RGBColor.from_string(TOKENS["muted"])
    meta.paragraph_format.space_before = Pt(0)
    meta.paragraph_format.space_after = Pt(1)
    meta.paragraph_format.line_spacing = 1.0

    audit = styles.add_style("Audit Note", WD_STYLE_TYPE.PARAGRAPH)
    audit.font.name = TOKENS["font"]
    audit.font.size = Pt(9.2)
    audit.font.color.rgb = RGBColor.from_string(TOKENS["ink"])
    audit.paragraph_format.space_before = Pt(0)
    audit.paragraph_format.space_after = Pt(4)
    audit.paragraph_format.line_spacing = 1.12


def configure_sections(doc):
    doc.settings.odd_and_even_pages_header_footer = False
    for section in doc.sections:
        section.page_width = Inches(TOKENS["page"]["width"])
        section.page_height = Inches(TOKENS["page"]["height"])
        margin_v = Inches(TOKENS["page"]["margin_vertical"])
        margin_h = Inches(TOKENS["page"]["margin_horizontal"])
        section.top_margin = margin_v
        section.bottom_margin = margin_v
        section.left_margin = margin_h
        section.right_margin = margin_h
        section.header_distance = Inches(TOKENS["page"]["header"])
        section.footer_distance = Inches(TOKENS["page"]["footer"])
        section.different_first_page_header_footer = False
        section.footer.is_linked_to_previous = False
        footer_p = section.footer.paragraphs[0]
        footer_p.text = ""
        add_page_number(footer_p)


def add_section_heading(doc, text):
    p = doc.add_paragraph(text, style="Heading 1")
    add_bottom_border(p, color=TOKENS["line"], size="6", space="1")
    return p


def add_project_heading(doc, title, role, dates, status=None):
    p = doc.add_paragraph(style="Heading 2")
    keep_with_next(p)
    r = p.add_run(title)
    set_run_font(r, size=TOKENS["h2_size"], bold=True, color=TOKENS["navy"])
    r = p.add_run(f"  |  {role}")
    set_run_font(r, size=9.5, bold=True, color=TOKENS["ink"])
    r = p.add_run(f"  |  {dates}")
    set_run_font(r, size=8.7, color=TOKENS["muted"])
    if status:
        r = p.add_run(f"  [{status}]")
        set_run_font(r, size=8.7, bold=True, color=TOKENS["blue"])
    return p


def add_bullet(doc, num_id, text, label=None, prediction=False, placeholder=False):
    p = doc.add_paragraph()
    apply_numbering(p, num_id)
    p.paragraph_format.space_after = Pt(TOKENS["bullet_after"])
    p.paragraph_format.line_spacing = TOKENS["body_line"]
    if label:
        r = p.add_run(label)
        set_run_font(r, size=TOKENS["body_size"], bold=True, color=TOKENS["navy"])
    r = p.add_run(text)
    set_run_font(r, size=TOKENS["body_size"], color=TOKENS["ink"])
    if prediction:
        shade_paragraph(p, TOKENS["prediction_fill"])
    if placeholder:
        shade_paragraph(p, TOKENS["placeholder_fill"])
    return p


def add_review_item(doc, num_id, title, body, severity="P0"):
    p = doc.add_paragraph(style="Audit Note")
    apply_numbering(p, num_id)
    r = p.add_run(f"[{severity}] {title}：")
    set_run_font(r, size=9.2, bold=True, color=TOKENS["risk"] if severity == "P0" else TOKENS["navy"])
    r = p.add_run(body)
    set_run_font(r, size=9.2, color=TOKENS["ink"])
    return p


def build():
    doc = Document()
    configure_styles(doc)
    configure_sections(doc)
    num_id = set_repeatable_numbering(doc)

    # Page 1: ATS-first resume opening and two most relevant algorithm projects.
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title.paragraph_format.space_after = Pt(1)
    r = title.add_run("[姓名待补]")
    set_run_font(r, size=TOKENS["title_size"], bold=True, color=TOKENS["navy"])
    shade_paragraph(title, TOKENS["placeholder_fill"])

    subtitle = doc.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(2)
    r = subtitle.add_run("人工智能算法工程师  |  多模态 / Agent Runtime / VLA")
    set_run_font(r, size=TOKENS["subtitle_size"], bold=True, color=TOKENS["blue"])

    contact = doc.add_paragraph(style="Resume Meta")
    contact.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = contact.add_run("[电话待补]  |  [邮箱待补]  |  [GitHub/作品集链接待补]  |  [所在城市待补]")
    set_run_font(r, size=8.7, color=TOKENS["muted"])
    shade_paragraph(contact, TOKENS["placeholder_fill"])
    add_bottom_border(contact, color=TOKENS["blue"], size="10", space="4")

    banner = doc.add_paragraph()
    banner.paragraph_format.space_before = Pt(4)
    banner.paragraph_format.space_after = Pt(4)
    banner.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = banner.add_run("内部审查版 v0.1  |  ◇ = 预测/目标区间，尚未验证，正式投递前必须替换")
    set_run_font(r, size=8.4, bold=True, color=TOKENS["risk"])
    shade_paragraph(banner, TOKENS["prediction_fill"])

    add_section_heading(doc, "个人概述")
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(
        "围绕多模态大模型后训练、可治理 Agent Runtime 与自动驾驶 VLA 推理构建作品集，"
        "强调身份冻结、可恢复实验、逐样本证据和严格主张边界；具备 Python/PyTorch 算法实现、"
        "Docker 隔离评测、API 服务化与 Java 后端工程能力。"
    )
    set_run_font(r, size=TOKENS["body_size"], color=TOKENS["ink"])

    add_section_heading(doc, "核心技能")
    add_bullet(
        doc,
        num_id,
        "Python、PyTorch、Qwen2.5-VL、ms-swift、PEFT/TRL、QLoRA/GRPO、Reward/Verifier、统计检验。",
        label="模型与算法：",
    )
    add_bullet(
        doc,
        num_id,
        "Agent Runtime、ReAct、检索/重排、工具协议、checkpoint/journal/trace、HTTP/SSE、并发与恢复。",
        label="Agent 与系统：",
    )
    add_bullet(
        doc,
        num_id,
        "Docker/CI、pytest、Ruff、strict mypy、Spring Boot、MySQL/Redis、Keycloak、Prometheus/Grafana、k6。",
        label="工程与质量：",
    )

    add_section_heading(doc, "项目经历")
    add_project_heading(
        doc,
        "ForgeMM",
        "图表问答可信推理的多模态后训练",
        "2026.08–至今",
        "正式训练进行中",
    )
    add_bullet(
        doc,
        num_id,
        "基于 Qwen2.5-VL-3B 与 ms-swift 设计 Answer-only/Structured QLoRA、标准 GRPO、固定多奖励 GRPO 与动态约束 Chart-FGRPO 的 E0–E5/A1–A2 对照矩阵。",
    )
    add_bullet(
        doc,
        num_id,
        "构建 1,763 条严格 EvidenceStore，生成 3,526 条双模板 SFT 序列与 1,763 个 GRPO prompt；3,526/3,526 gold completion 通过答案/证据/运算三通道 reward 回放。",
    )
    add_bullet(
        doc,
        num_id,
        "实现安全 Parser/Executor、三通道 reward、group advantage、constraint mask、dual update/clip/checkpoint 恢复，以及 FCR、paired bootstrap 95% CI、exact McNemar 评测；57 项测试、Ruff、strict mypy 通过。",
    )
    add_bullet(
        doc,
        num_id,
        "在 RTX 4090 完成真实 GPU smoke：底座推理 0.899 samples/s，50-step Structured QLoRA 最终 token accuracy 0.9889；GRPO 三路 reward 非零并保存完整 checkpoint。该结果仅证明训练链路可执行。",
    )
    add_bullet(
        doc,
        num_id,
        "◇ 预测/目标：若正式三种子实验达到预注册门槛，预计 E5 相对同起点 E3 的 FCR 提升 5–8 pp、推理不一致率降低 30%–40%，ChartQA relaxed accuracy 控制在 -1 至 +1 pp；以冻结 test/ChartQAPro 实测为准。",
        prediction=True,
    )

    add_project_heading(
        doc,
        "DriveVLA-Guard",
        "自动驾驶 VLA 风险感知推理与评测",
        "2026.08–至今",
        "官方 NAVSIM 待跑",
    )
    add_bullet(
        doc,
        num_id,
        "面向 AutoVLA/Qwen2.5-VL-3B 实现多候选 Action Token 生成、轨迹解码、可解释风险分解与 risk-triggered fast/slow routing；顺序候选生成用于控制 KV-cache 峰值。",
    )
    add_bullet(
        doc,
        num_id,
        "建立 manifest/config/code hash 驱动的可恢复评测，逐场景保存 token、轨迹、风险分量、路由原因和延迟；NAVSIM agent 仅使用历史末帧 annotations 与 constant-velocity 外推，隔离 future GT。",
    )
    add_bullet(
        doc,
        num_id,
        "完成 12 个冻结合成场景 B0/B1/E1–E4 的 72/72 运行及 1,200 场景、20,400 scene-runs 压力诊断（0 执行失败）；K=4 为最小零碰撞代理配置，slow path 25%。上述结果不是 NAVSIM 性能。",
    )
    add_bullet(
        doc,
        num_id,
        "◇ 预测/目标：官方 B0/E2 若通过资产与复现门，预计碰撞/TTC 类安全失败相对下降 10%–20%，PDMS 变化 -0.5 至 +1.0 pp，端到端延迟增加 20%–35%；若安全门未满足则停止 E3/E4。",
        prediction=True,
    )

    # Page 2: system project, engineering project, education and internship placeholders.
    doc.add_page_break()
    add_section_heading(doc, "项目经历（续）")
    add_project_heading(
        doc,
        "RepoPilot",
        "本地优先、反馈驱动的可治理 Agent Runtime Lab",
        "2026.07–2026.08",
        "本地实验闭环",
    )
    add_bullet(
        doc,
        num_id,
        "以约 5K 行 Python 实现 provider-neutral ReAct runtime，覆盖硬预算、检索/重排、工具调用、确定性 verifier、checkpoint/journal/trace、记忆、multi-agent 状态图与漂移检测。",
    )
    add_bullet(
        doc,
        num_id,
        "实现 Bearer 鉴权 HTTP JSON API 与可续传 SSE，覆盖限流、请求上限、路径 allowlist、断连恢复和硬并发排队；Docker sandbox 固化 non-root、read-only rootfs、cap-drop、no-network 与资源限制。",
    )
    add_bullet(
        doc,
        num_id,
        "冻结三领域评测：FRAMES 60 题 18.3%（11/60，三次一致）、DABench 35 题 × 3 为 73.3% 均值、SWE-bench-Live 5 个 fresh task 为 0/5；负结果触发停止规则，避免继续污染验证集。",
    )
    add_bullet(
        doc,
        num_id,
        "真实 Qwen2.5-7B Q4_K_M 服务在 1/4/8 并发下完成 72/72 非空响应，吞吐 5.52/16.80/18.62 req/s；90 项测试、strict mypy 与现场 Docker 安全测试通过。",
    )

    add_project_heading(
        doc,
        "Hospital Workforce Platform",
        "医院人力与排班平台的独立现代化重构",
        "2026.08",
        "工程实验闭环",
    )
    add_bullet(
        doc,
        num_id,
        "使用 Java 21/Spring Boot 模块化单体重构组织人事、排班、薪资、招聘、培训、审计与 RBAC；通过员工行悲观锁、重叠校验、数据库约束、乐观版本与幂等键保障排班并发安全。",
    )
    add_bullet(
        doc,
        num_id,
        "搭建 MySQL/Redis/Keycloak/前后端/Prometheus/Grafana 七服务 Compose；7/7 后端测试通过。在 50 名员工、1,000 条排班的已认证查询负载下完成 16,746 请求、0% 错误、55.70 req/s、p95 5.83 ms。",
    )
    add_bullet(
        doc,
        num_id,
        "真实性边界：该仓库是对 2021 年实习项目的后续个人重构，不应把 2026 年架构、容器化、压测和指标写成实习期间成果。",
    )

    add_section_heading(doc, "教育背景")
    add_bullet(
        doc,
        num_id,
        "[学校全称]  |  [专业]  |  [学历]  |  [入学年月–毕业年月]  |  [GPA/排名/核心课程按需补充]",
        placeholder=True,
    )

    add_section_heading(doc, "实习经历")
    p = doc.add_paragraph(style="Heading 2")
    r = p.add_run("[公司全称]  |  [岗位名称]  |  2021.[月]–2021.[月]")
    set_run_font(r, size=9.8, bold=True, color=TOKENS["navy"])
    shade_paragraph(p, TOKENS["placeholder_fill"])
    add_bullet(
        doc,
        num_id,
        "参与医院人事管理系统相关开发；负责模块、技术栈、需求规模、协作方式和可核验成果均待依据原始材料补齐。",
        placeholder=True,
    )

    add_section_heading(doc, "补充信息")
    add_bullet(
        doc,
        num_id,
        "语言：[英语水平待补]；奖项/论文/证书：[待补或删除本栏]；作品集：四个项目均保留 README、实验记录、冻结结果与复现入口，公开链接待补。",
        placeholder=True,
    )

    # Page 3: internal review appendix, intentionally excluded from application copy.
    doc.add_page_break()
    title = doc.add_paragraph()
    title.paragraph_format.space_after = Pt(3)
    r = title.add_run("审查附录｜问题定位与修正优先级")
    set_run_font(r, size=20, bold=True, color=TOKENS["navy"])
    add_bottom_border(title, color=TOKENS["blue"], size="12", space="4")

    note = doc.add_paragraph(style="Audit Note")
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = note.add_run("本页仅用于内部审查，正式投递时整页删除。预测区间不是实验结论，也不是承诺值。")
    set_run_font(r, size=9.4, bold=True, color=TOKENS["risk"])
    shade_paragraph(note, TOKENS["prediction_fill"])

    add_section_heading(doc, "必须先修（P0）")
    add_review_item(doc, num_id, "个人信息缺失", "姓名、电话、邮箱、城市、GitHub/作品集链接均为空；这是当前版本无法投递的直接原因。")
    add_review_item(doc, num_id, "教育背景缺失", "学校、专业、学历与时间决定岗位匹配和 ATS 筛选，必须提供真实信息；不建议由项目材料推断。")
    add_review_item(doc, num_id, "实习事实不完整", "需补公司全称、岗位、起止年月、真实职责与成果；必须把 2021 年工作和 2026 年个人重构严格拆开。")
    add_review_item(doc, num_id, "预测值不可直接投递", "ForgeMM 与 DriveVLA-Guard 的黄色条目只能作为审查目标。正式结果出来后，按相同实验协议替换；未达门槛则改写为工程闭环或负结果。")

    add_section_heading(doc, "内容优化（P1）")
    add_review_item(doc, num_id, "目标岗位尚未单一化", "当前同时覆盖多模态后训练、Agent 与 VLA。投递时建议至少派生两个版本：多模态/VLA 算法版、Agent/AI 系统版。", severity="P1")
    add_review_item(doc, num_id, "项目过多", "正文保留四个项目便于当前审查；正式两页版可按岗位删除最低相关项目，并将最相关项目扩展为 4 条高密度要点。", severity="P1")
    add_review_item(doc, num_id, "链接与演示缺失", "为每个保留项目补 GitHub、README、结果报告或 2–3 分钟演示链接；公开前做许可证、密钥与大文件检查。", severity="P1")
    add_review_item(doc, num_id, "数字需带口径", "保留样本量、数据集、重复次数和测试环境；不要把 smoke、合成代理、查询压测或模型服务吞吐泛化成生产能力。", severity="P1")

    add_section_heading(doc, "预测结果回填规则")
    add_bullet(doc, num_id, "ForgeMM：只在 E3/E5 同起点、同数据、三种子、冻结 test 与统计检验完成后填写 FCR、Evidence F1、Operation Consistency 和 accuracy；优先填真实差值与置信区间。")
    add_bullet(doc, num_id, "DriveVLA-Guard：先完成官方 AutoVLA/NAVSIM B0 对齐，再在同一 manifest 上比较 E2；只有安全子指标改善且 EP/Comfort 可接受时才继续 E3/E4。")
    add_bullet(doc, num_id, "若预测失败：删除预测数字，保留可验证工程成果、失败原因、停止规则和下一步修正，不用调参后的最好单次结果替代预注册矩阵。")

    add_section_heading(doc, "建议的下一轮输入")
    add_bullet(doc, num_id, "提供一份目标 JD，以及姓名/联系方式/城市、教育背景、2021 实习原始经历、英语/奖项/论文信息；据此生成真正可投递的定向版本。")
    add_bullet(doc, num_id, "确认公开仓库与作品集 URL；如果暂未公开，正文先删除链接字段，避免留下明显占位符。")

    # Core properties and compatibility flags.
    doc.core_properties.title = "个人简历｜内部审查预测版 v0.1"
    doc.core_properties.subject = "多模态、Agent Runtime、VLA 项目简历审查稿"
    doc.core_properties.author = "Candidate"
    settings = doc.settings.element
    update_fields = settings.find(qn("w:updateFields"))
    if update_fields is None:
        update_fields = OxmlElement("w:updateFields")
        settings.append(update_fields)
    update_fields.set(qn("w:val"), "true")

    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
