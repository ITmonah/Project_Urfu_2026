"""Generate documentation diagrams as BPMN 2.0, SVG and PNG.

Run from the repository root: python docs/diagrams/generate_diagrams.py
Only Pillow is required; no model weights or inference are loaded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parent
NS = {
    "bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
    "bpmndi": "http://www.omg.org/spec/BPMN/20100524/DI",
    "dc": "http://www.omg.org/spec/DD/20100524/DC",
    "di": "http://www.omg.org/spec/DD/20100524/DI",
}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def element(parent, prefix, tag, **attrs):
    return ET.SubElement(parent, f"{{{NS[prefix]}}}{tag}", {k: str(v) for k, v in attrs.items()})


@dataclass
class Node:
    id: str
    kind: str
    text: str
    lane: int
    x: int
    y: int
    w: int = 180
    h: int = 82
    proposed: bool = False
    event: str = ""
    label_xy: tuple[int, int] | None = None


@dataclass
class Flow:
    source: str
    target: str
    points: list[tuple[int, int]]
    label: str = ""
    label_xy: tuple[int, int] | None = None
    condition: str = ""


class Canvas:
    """Draw the same vector primitives into SVG and a high-resolution PNG."""

    def __init__(self, width, height, title):
        self.scale = 2
        self.svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", width=str(width),
                              height=str(height), viewBox=f"0 0 {width} {height}", role="img")
        ET.SubElement(self.svg, "title").text = title
        self.image = Image.new("RGB", (width * self.scale, height * self.scale), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.rect(0, 0, width, height, "white", None)

    def rect(self, x, y, w, h, fill, stroke="#334155", sw=1.5, radius=0):
        attrs = dict(x=str(x), y=str(y), width=str(w), height=str(h), fill=fill or "none",
                     stroke=stroke or "none", **{"stroke-width": str(sw), "rx": str(radius)})
        ET.SubElement(self.svg, "rect", attrs)
        bounds = tuple(int(v * self.scale) for v in (x, y, x + w, y + h))
        self.draw.rounded_rectangle(bounds, radius=int(radius * self.scale), fill=fill,
                                    outline=stroke, width=max(1, round(sw * self.scale)))

    def circle(self, x, y, r, fill="white", stroke="#334155", sw=2):
        ET.SubElement(self.svg, "circle", cx=str(x), cy=str(y), r=str(r), fill=fill or "none",
                      stroke=stroke, **{"stroke-width": str(sw)})
        self.draw.ellipse(tuple(round(v * self.scale) for v in (x-r, y-r, x+r, y+r)),
                          fill=fill, outline=stroke, width=round(sw * self.scale))

    def line(self, points, color="#475569", sw=2):
        ET.SubElement(self.svg, "polyline", points=" ".join(f"{x},{y}" for x, y in points),
                      fill="none", stroke=color, **{"stroke-width": str(sw), "stroke-linejoin": "round"})
        self.draw.line([(round(x*self.scale), round(y*self.scale)) for x, y in points],
                       fill=color, width=round(sw*self.scale), joint="curve")

    def polygon(self, points, fill="white", stroke="#334155", sw=2):
        ET.SubElement(self.svg, "polygon", points=" ".join(f"{x},{y}" for x, y in points),
                      fill=fill, stroke=stroke or "none", **{"stroke-width": str(sw)})
        coords = [(round(x*self.scale), round(y*self.scale)) for x, y in points]
        self.draw.polygon(coords, fill=fill)
        if stroke:
            self.draw.line(coords + [coords[0]], fill=stroke, width=round(sw*self.scale), joint="curve")

    def text(self, x, y, text, size=16, color="#172b4d", anchor="middle", bold=False):
        font_path = Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
        try:
            font = ImageFont.truetype(str(font_path), size * self.scale)
        except OSError:
            font = ImageFont.truetype("DejaVuSans.ttf", size * self.scale)
        for i, line in enumerate(text.split("\n")):
            yy = y + i * (size + 5)
            attrs = {"x": str(x), "y": str(yy), "fill": color, "font-family": "Arial, DejaVu Sans, sans-serif",
                     "font-size": str(size), "text-anchor": anchor,
                     "font-weight": "bold" if bold else "normal"}
            ET.SubElement(self.svg, "text", attrs).text = line
            self.draw.text((x*self.scale, yy*self.scale), line, font=font, fill=color,
                           anchor="ms" if anchor == "middle" else "ls")

    def arrow(self, points):
        self.line(points)
        x, y = points[-1]
        px, py = points[-2]
        angle = math.atan2(y-py, x-px)
        tip = [(x, y), (x-10*math.cos(angle)+4*math.sin(angle), y-10*math.sin(angle)-4*math.cos(angle)),
               (x-10*math.cos(angle)-4*math.sin(angle), y-10*math.sin(angle)+4*math.cos(angle))]
        self.polygon(tip, "#475569", None)

    def save(self, stem):
        ET.indent(self.svg)
        ET.ElementTree(self.svg).write(OUT / f"{stem}.svg", encoding="utf-8", xml_declaration=True)
        self.image.save(OUT / f"{stem}.png", optimize=True)


def draw_node(c, n):
    fill = "#fff4dd" if n.proposed else "#eaf4ff"
    x, y, w, h = n.x, n.y, n.w, n.h
    if n.kind in {"startEvent", "endEvent"}:
        c.circle(x+w/2, y+h/2, w/2, sw=4 if n.kind == "endEvent" else 2)
        if n.event == "timer":
            c.circle(x+w/2, y+h/2, 11, sw=1)
            c.line([(x+w/2, y+8), (x+w/2, y+h/2), (x+w/2+6, y+h/2+4)], sw=1.5)
        elif n.event == "conditional":
            c.rect(x+11, y+9, 16, 20, "white", sw=1)
            for dy in (14, 19, 24):
                c.line([(x+14, y+dy), (x+24, y+dy)], sw=1)
        c.text(*(n.label_xy or (x+w/2, y+h+25)), n.text, size=15)
    elif n.kind == "exclusiveGateway":
        c.polygon([(x+w/2, y), (x+w, y+h/2), (x+w/2, y+h), (x, y+h/2)], "#f0fdf4")
        c.line([(x+18, y+18), (x+w-18, y+h-18)], sw=2)
        c.line([(x+w-18, y+18), (x+18, y+h-18)], sw=2)
        if n.text:
            c.text(*(n.label_xy or (x+w/2, y+h+24)), n.text, size=15)
    else:
        c.rect(x, y, w, h, fill, radius=10)
        if n.kind == "userTask":
            c.circle(x+17, y+13, 4, fill=None, sw=1.4)
            c.line([(x+10, y+26), (x+10, y+22), (x+13, y+19), (x+21, y+19), (x+24, y+22), (x+24, y+26)], sw=1.4)
        elif n.kind == "serviceTask":
            pts = []
            for i in range(32):
                angle = i*math.pi/16
                r = 9 if i % 4 in (0, 1) else 6.5
                pts.append((x+17+r*math.cos(angle), y+18+r*math.sin(angle)))
            c.polygon(pts, "#475569", None)
            c.circle(x+17, y+18, 3, fill=fill, sw=1)
        lines = len(n.text.split("\n"))
        c.text(x+w/2, y+h/2-(lines-1)*10+11, n.text, size=16)
        if n.kind == "subProcess":
            c.rect(x+w/2-7, y+h-19, 14, 14, "white", sw=1)
            c.line([(x+w/2-4, y+h-12), (x+w/2+4, y+h-12)], sw=1)
            c.line([(x+w/2, y+h-16), (x+w/2, y+h-8)], sw=1)


def export(stem, title, subtitle, lanes, nodes, flows, width, height):
    c = Canvas(width, height, title)
    c.text(35, 43, title, size=28, bold=True, anchor="start", color="#075ca8")
    c.text(35, 73, subtitle, size=16, anchor="start", color="#526078")
    pool_x, pool_y, pool_w = 35, lanes[0][1], width-70
    pool_h = sum(lane[2] for lane in lanes)
    c.rect(pool_x, pool_y, pool_w, pool_h, "white")
    c.rect(pool_x, pool_y, 32, pool_h, "#e2e8f0")
    for i, (name, top, lane_h) in enumerate(lanes):
        c.rect(67, top, width-102, lane_h, "#f8fafc" if i % 2 == 0 else "white")
        c.rect(67, top, 110, lane_h, "#eef2f6")
        c.text(122, top+lane_h/2-10, name, size=13, bold=True)
    for f in flows:
        c.arrow(f.points)
        if f.label:
            c.text(*(f.label_xy or f.points[len(f.points)//2]), f.label, size=14, color="#475569")
    for n in nodes:
        draw_node(c, n)
    c.rect(35, height-57, 22, 17, "#eaf4ff")
    c.text(66, height-43, "Этап сервиса / действие сотрудника", size=14, anchor="start")
    if any(n.proposed for n in nodes):
        c.rect(390, height-57, 22, 17, "#fff4dd")
        c.text(421, height-43, "Предлагаемый организационный этап", size=14, anchor="start")
    c.text(35, height-17, "BPMN 2.0 · Один пул, роли в дорожках · Сплошные стрелки — sequence flow · X — исключающее ветвление", size=13, anchor="start")
    c.save(stem)

    root = ET.Element(f"{{{NS['bpmn']}}}definitions", id=f"Definitions_{stem.replace('-', '_')}",
                      targetNamespace="https://example.org/kgo/documentation", exporter="KGO documentation")
    process = element(root, "bpmn", "process", id="Process_KGO", name=title, isExecutable="false")
    element(process, "bpmn", "documentation").text = subtitle
    lane_set = element(process, "bpmn", "laneSet", id="LaneSet_KGO")
    for i, (name, _, _) in enumerate(lanes):
        lane = element(lane_set, "bpmn", "lane", id=f"Lane_{i}", name=name.replace("\n", " "))
        for n in nodes:
            if n.lane == i:
                element(lane, "bpmn", "flowNodeRef").text = n.id
    for n in nodes:
        el = element(process, "bpmn", n.kind, id=n.id, name=n.text.replace("\n", " "))
        if n.proposed:
            element(el, "bpmn", "documentation").text = "Предлагаемый этап; автоматизация в репозитории не реализована."
        for i, f in enumerate(flows):
            if f.target == n.id:
                element(el, "bpmn", "incoming").text = f"Flow_{i}"
            if f.source == n.id:
                element(el, "bpmn", "outgoing").text = f"Flow_{i}"
        if n.event == "timer":
            element(el, "bpmn", "timerEventDefinition", id=f"EventDefinition_{n.id}")
        elif n.event == "conditional":
            event = element(el, "bpmn", "conditionalEventDefinition", id=f"EventDefinition_{n.id}")
            element(event, "bpmn", "condition").text = "Наступило условие пересмотра модели по мониторингу качества"
    for i, f in enumerate(flows):
        el = element(process, "bpmn", "sequenceFlow", id=f"Flow_{i}", sourceRef=f.source,
                     targetRef=f.target, name=f.label)
        if f.condition:
            element(el, "bpmn", "conditionExpression").text = f.condition
    collaboration = element(root, "bpmn", "collaboration", id="Collaboration_KGO")
    element(collaboration, "bpmn", "participant", id="Participant_KGO", name="Процесс КГО", processRef="Process_KGO")
    diagram = element(root, "bpmndi", "BPMNDiagram", id="Diagram_KGO")
    plane = element(diagram, "bpmndi", "BPMNPlane", id="Plane_KGO", bpmnElement="Collaboration_KGO")

    def shape(id_, bounds, **attrs):
        s = element(plane, "bpmndi", "BPMNShape", id=f"Shape_{id_}", bpmnElement=id_, **attrs)
        element(s, "dc", "Bounds", **dict(zip(("x", "y", "width", "height"), bounds)))
        return s

    shape("Participant_KGO", (pool_x, pool_y, pool_w, pool_h), isHorizontal="true")
    for i, (_, top, lane_h) in enumerate(lanes):
        shape(f"Lane_{i}", (67, top, width-102, lane_h), isHorizontal="true")
    for n in nodes:
        attrs = {"isExpanded": "false"} if n.kind == "subProcess" else {}
        s = shape(n.id, (n.x, n.y, n.w, n.h), **attrs)
        if n.kind in {"startEvent", "endEvent", "exclusiveGateway"} and n.text:
            x, y = n.label_xy or (n.x+n.w/2, n.y+n.h+25)
            label = element(s, "bpmndi", "BPMNLabel")
            element(label, "dc", "Bounds", x=x-85, y=y-15, width=170, height=42)
    for i, f in enumerate(flows):
        edge = element(plane, "bpmndi", "BPMNEdge", id=f"Edge_Flow_{i}", bpmnElement=f"Flow_{i}")
        for x, y in f.points:
            element(edge, "di", "waypoint", x=x, y=y)
        if f.label:
            x, y = f.label_xy or f.points[len(f.points)//2]
            label = element(edge, "bpmndi", "BPMNLabel")
            element(label, "dc", "Bounds", x=x-40, y=y-15, width=80, height=24)
    ET.indent(root)
    ET.ElementTree(root).write(OUT / f"{stem}.bpmn", encoding="utf-8", xml_declaration=True)


def model_use():
    nodes = [
        Node("Start", "startEvent", "Нужна оценка\nплощадки", 0, 205, 178, 38, 38),
        Node("Photo", "userTask", "Сфотографировать\nплощадку", 0, 290, 155),
        Node("Upload", "userTask", "Отправить фото", 0, 525, 155),
        Node("Correct", "userTask", "Исправить запрос\nили заменить фото", 0, 800, 155),
        Node("Input", "serviceTask", "Получить веса\nи декодировать фото", 1, 525, 360),
        Node("ErrorGate", "exclusiveGateway", "Обработка\nвыполнилась?", 1, 800, 375, 54, 54, label_xy=(827, 343)),
        Node("Error", "serviceTask", "Вернуть ошибку\nобработки", 1, 930, 360),
        Node("Detect", "serviceTask", "YOLO: найти\nkgo_platform", 1, 525, 500),
        Node("Zone", "exclusiveGateway", "Площадка\nнайдена?", 1, 800, 515, 54, 54, label_xy=(827, 485)),
        Node("Score", "subProcess", "Оценить выбранным\nML-пайплайном", 1, 960, 485, 205, 100),
        Node("Reply", "serviceTask", "Вернуть метку /\nотсутствие результата\nи визуализацию", 1, 1320, 485, 220, 100),
        Node("End", "endEvent", "Ответ API\nвозвращён", 1, 1585, 516, 38, 38),
    ]
    flows = [
        Flow("Start", "Photo", [(243,197),(290,197)]),
        Flow("Photo", "Upload", [(470,197),(525,197)]),
        Flow("Upload", "Input", [(615,237),(615,360)]),
        Flow("Input", "ErrorGate", [(705,402),(800,402)]),
        Flow("ErrorGate", "Error", [(854,402),(930,402)], "Нет", (894,391), "обработка не выполнена"),
        Flow("Error", "Correct", [(1020,360),(1020,196),(980,196)]),
        Flow("Correct", "Upload", [(800,196),(705,196)]),
        Flow("ErrorGate", "Detect", [(827,429),(827,469),(615,469),(615,500)], "Да", (759,458), "обработка выполнена"),
        Flow("Detect", "Zone", [(705,542),(800,542)]),
        Flow("Zone", "Score", [(854,542),(960,542)], "Да", (898,529), "найдена площадка"),
        Flow("Zone", "Reply", [(827,569),(827,620),(1375,620),(1375,585)], "Нет", (1010,609), "площадка не найдена"),
        Flow("Score", "Reply", [(1165,535),(1320,535)]),
        Flow("Reply", "End", [(1540,535),(1585,535)]),
    ]
    export("model-use", "Использование модели: обработка фото через API",
           "Граница процесса: получение фотографии, ML-обработка и возврат ответа API.",
           [("Сотрудник",110,190),("ML-сервис /\nAPI",300,360)],
           nodes, flows, 1700, 740)


def model_retraining():
    nodes = [
        Node("Schedule", "startEvent", "По расписанию", 0, 230, 160, 38, 38, True, "timer"),
        Node("Signal", "startEvent", "Сигнал\nмониторинга", 0, 230, 260, 38, 38, True, "conditional"),
        Node("Merge", "exclusiveGateway", "", 0, 300, 220, 54, 54, True),
        Node("Load", "serviceTask", "Загрузить новые\nфото и обратную связь", 0, 405, 205, 205, 82, True),
        Node("Prepare", "serviceTask", "Версионировать\nданные и разделить\ntrain / val / test", 0, 830, 205, 205, 90, True),
        Node("Train", "serviceTask", "Обучить нужные\nкомпоненты", 0, 1090, 205, 190, 90, True),
        Node("Evaluate", "serviceTask", "Оценить кандидата\nи весь пайплайн", 0, 1350, 205, 190, 90, True),
        Node("Label", "userTask", "Проверить фото\nи разметить данные", 1, 405, 420, 205, 82, True),
        Node("Validate", "userTask", "Проверить метрики,\nошибки и срезы", 1, 1350, 420, 190, 82, True),
        Node("Quality", "exclusiveGateway", "Критерии качества\nвыполнены?", 1, 1130, 432, 54, 54, True, label_xy=(1157, 402)),
        Node("Register", "serviceTask", "Выпустить веса\nи обновить model card", 2, 1100, 610, 215, 82, True),
        Node("Deploy", "serviceTask", "Подключить версию\nи перезапустить сервис", 2, 1370, 610, 215, 82, True),
        Node("Smoke", "serviceTask", "Проверить сервис\nпосле выпуска", 2, 1370, 760, 215, 82, True),
        Node("Health", "exclusiveGateway", "Проверка\nуспешна?", 2, 1140, 774, 54, 54, True, label_xy=(1167, 745)),
        Node("Rollback", "serviceTask", "Откатить\nна прежнюю версию", 2, 850, 760, 205, 82, True),
        Node("Log", "serviceTask", "Сохранить причину,\nуведомить ML-команду", 2, 570, 610, 215, 82, True),
        Node("Monitor", "serviceTask", "Продолжить мониторинг\nдействующей версии", 2, 300, 760, 215, 82, True),
        Node("End", "endEvent", "Цикл завершён", 2, 227, 781, 38, 38, True),
    ]
    flows = [
        Flow("Schedule", "Merge", [(268,179),(285,179),(285,247),(300,247)]),
        Flow("Signal", "Merge", [(268,279),(285,279),(285,247),(300,247)]),
        Flow("Merge", "Load", [(354,247),(405,247)]),
        Flow("Load", "Label", [(507,287),(507,420)]),
        Flow("Label", "Prepare", [(610,461),(740,461),(740,250),(830,250)]),
        Flow("Prepare", "Train", [(1035,250),(1090,250)]),
        Flow("Train", "Evaluate", [(1280,250),(1350,250)]),
        Flow("Evaluate", "Validate", [(1445,295),(1445,420)]),
        Flow("Validate", "Quality", [(1350,461),(1184,461)]),
        Flow("Quality", "Register", [(1157,486),(1157,558),(1207,558),(1207,610)], "Да", (1235,549), "критерии качества выполнены"),
        Flow("Quality", "Log", [(1130,459),(1080,459),(1080,520),(677,520),(677,610)], "Нет", (977,509), "критерии качества не выполнены"),
        Flow("Register", "Deploy", [(1315,651),(1370,651)]),
        Flow("Deploy", "Smoke", [(1477,692),(1477,760)]),
        Flow("Smoke", "Health", [(1370,801),(1194,801)]),
        Flow("Health", "Rollback", [(1140,801),(1055,801)], "Нет", (1098,790), "проверка сервиса не пройдена"),
        Flow("Rollback", "Log", [(952,760),(952,720),(677,720),(677,692)]),
        Flow("Health", "Monitor", [(1167,828),(1167,885),(407,885),(407,842)], "Да", (1100,873), "проверка сервиса пройдена"),
        Flow("Log", "Monitor", [(570,651),(407,651),(407,760)]),
        Flow("Monitor", "End", [(300,801),(265,801)]),
    ]
    export("model-retraining", "Переобучение: новые данные, контроль качества, выпуск",
           "Предлагаемый процесс. Пороги согласуются заранее; при отказе сохраняется или восстанавливается прежняя версия.",
           [("ML / MLOps",110,240),("Эксперт /\nразметчик",350,200),("Эксплуатация",550,365)],
           nodes, flows, 1700, 995)


if __name__ == "__main__":
    model_use()
    model_retraining()
    print("Generated two diagrams: BPMN 2.0 + SVG + PNG")
