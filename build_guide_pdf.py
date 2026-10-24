"""Render the editable Markdown guide with Vietnamese fonts and vector figures."""
from pathlib import Path
import re,html
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,KeepTogether,Flowable
from reportlab.lib.styles import ParagraphStyle,getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY,TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'docs'/'Cam_Nang_Gemma4_T4.pdf'
FONTS=Path('/usr/share/fonts/truetype/dejavu')
for name,path in [('Book','DejaVuSerif.ttf'),('Book-Bold','DejaVuSerif-Bold.ttf'),
                  ('Sans','DejaVuSans.ttf'),('Sans-Bold','DejaVuSans-Bold.ttf'),('Mono','DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(FONTS/path)))
pdfmetrics.registerFontFamily('Book',normal='Book',bold='Book-Bold',italic='Book',boldItalic='Book-Bold')
pdfmetrics.registerFontFamily('Sans',normal='Sans',bold='Sans-Bold',italic='Sans',boldItalic='Sans-Bold')
INK=colors.HexColor('#182638');GRAY=colors.HexColor('#52606a');LINE=colors.HexColor('#c6cbd2')
styles={
    'body':ParagraphStyle('body',fontName='Book',fontSize=10.4,leading=15.3,spaceAfter=8,alignment=TA_JUSTIFY,textColor=INK),
    'h1':ParagraphStyle('h1',fontName='Book-Bold',fontSize=17.8,leading=23.4,spaceAfter=15,textColor=INK),
    'h2':ParagraphStyle('h2',fontName='Book-Bold',fontSize=13.8,leading=19,spaceBefore=5,spaceAfter=12,textColor=INK),
    'cell':ParagraphStyle('cell',fontName='Sans',fontSize=8.8,leading=12.4,textColor=INK,splitLongWords=True),
    'th':ParagraphStyle('th',fontName='Sans-Bold',fontSize=8.8,leading=12.4,textColor=colors.white),
    'bullet':ParagraphStyle('bullet',fontName='Book',fontSize=10.2,leading=15,spaceAfter=6,leftIndent=11,firstLineIndent=-9,textColor=INK),
    'caption':ParagraphStyle('caption',fontName='Sans',fontSize=8.6,leading=12,textColor=GRAY,spaceAfter=8),
}

def inline(s):
    s=s.replace('—','-').replace('–','-').replace('‑','-')
    s=html.escape(s)
    s=re.sub(r'`([^`]+)`',r'<font name="Mono" size="8.7">\1</font>',s)
    s=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'(https://[^\s<>]+)',lambda m:f'<link href="{m.group(1)}" color="#275a86">{m.group(1)}</link>',s)
    return s

class Diagram(Flowable):
    def __init__(self,kind):super().__init__();self.kind=kind;self.width=467;self.height=135 if kind=='architecture' else 116
    def box(self,x,y,w,h,title,sub=''):
        c=self.canv;c.setFillColor(colors.HexColor('#f1f3f5'));c.setStrokeColor(LINE);c.roundRect(x,y,w,h,5,fill=1)
        c.setFillColor(INK);c.setFont('Sans-Bold',9);c.drawCentredString(x+w/2,y+h-17,title)
        if sub:c.setFont('Sans',8);c.drawCentredString(x+w/2,y+12,sub)
    def arrow(self,x1,y1,x2,y2):
        c=self.canv;c.setStrokeColor(GRAY);c.setFillColor(GRAY);c.line(x1,y1,x2,y2)
        import math
        a=math.atan2(y2-y1,x2-x1);p=c.beginPath();p.moveTo(x2,y2)
        p.lineTo(x2-5*math.cos(a-.45),y2-5*math.sin(a-.45));p.lineTo(x2-5*math.cos(a+.45),y2-5*math.sin(a+.45));p.close();c.drawPath(p,fill=1)
    def draw(self):
        if self.kind=='architecture':
            self.box(0,76,138,49,'01 · Baseline','Model + số đo ban đầu')
            self.box(164,76,138,49,'02 · Train / chọn','Ứng viên + predictor')
            self.box(328,76,138,49,'03 · Test / demo','Đối chứng cùng T4')
            self.arrow(139,100,162,100);self.arrow(303,100,326,100)
            self.box(0,0,220,49,'Cùng mã benchmark','Prompt · thời gian · bộ nhớ · lỗi')
            self.box(247,0,220,49,'Cùng Gemma + Ollama','Digest và phiên bản được khóa')
            self.arrow(232,76,110,51);self.arrow(234,76,357,51)
        elif self.kind=='decode':
            self.box(0,64,115,42,'Prompt','Token đầu vào')
            self.box(143,64,127,42,'Prefill','Xử lý + tạo cache')
            self.box(301,64,165,42,'Decode','Token mới + cập nhật cache')
            self.arrow(116,85,141,85);self.arrow(271,85,299,85)
            self.box(301,1,165,40,'Dừng?','EOS hoặc chạm giới hạn')
            self.arrow(384,64,384,43)
            self.canv.setFont('Sans',8);self.canv.setFillColor(GRAY);self.canv.drawString(8,18,'Chưa dừng: lặp lại decode với token vừa sinh.')
        elif self.kind=='train':
            self.box(0,65,138,45,'Train','Học quan hệ độ trễ')
            self.box(164,65,138,45,'Validation','Chọn và lọc cấu hình')
            self.box(328,65,138,45,'Test','Chỉ đánh giá cuối')
            self.arrow(139,87,162,87);self.arrow(303,87,326,87)
            self.box(100,0,267,42,'Khóa lựa chọn trước khi dùng test','Không cập nhật Gemma trong pipeline này')
            self.arrow(233,65,233,44)

def parse_chunk(chunk,index):
    lines=chunk.strip().splitlines();flow=[];i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('# '):
            flow.append(Paragraph(inline(line[2:]),styles['h1']));i+=1;continue
        if line.startswith('## '):
            flow.append(Paragraph(inline(line[3:]),styles['h2']));i+=1;continue
        if line.startswith('|'):
            data=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',c) for c in cells):data.append(cells)
                i+=1
            n=len(data[0]);widths={2:[155,312],3:[136,149,182],4:[99,125,125,118],5:[117,87,87,88,88],6:[90,75,75,75,77,75]}.get(n,[467/n]*n)
            cells=[[Paragraph(inline(s),styles['th'] if j==0 else styles['cell']) for s in row] for j,row in enumerate(data)]
            t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),INK),('VALIGN',(0,0),(-1,-1),'TOP'),
                ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
                ('LINEBELOW',(0,0),(-1,-1),.35,LINE),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f7f8')])]))
            flow.extend([t,Spacer(1,11)]);continue
        if line.startswith('- ') or re.match(r'^\d+\. ',line):
            marker='• '+line[2:] if line.startswith('- ') else line
            marker=marker.replace('[ ]','□')
            flow.append(Paragraph(inline(marker),styles['bullet']));i+=1;continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#','|','- ')):
            paragraph.append(lines[i].strip());i+=1
        flow.append(Paragraph(inline(' '.join(paragraph)),styles['body']))
    if index==0:
        flow.insert(3,Diagram('architecture'));flow.insert(4,Spacer(1,12))
    return flow

def footer(canvas,doc):
    canvas.saveState();w,h=doc.pagesize
    canvas.setStrokeColor(LINE);canvas.line(64,h-38,w-64,h-38)
    canvas.setFillColor(GRAY);canvas.setFont('Sans',8)
    canvas.drawString(64,h-29,'GEMMA 4 12B / T4  |  CẨM NANG THỰC HÀNH')
    canvas.drawString(64,29,'13.10.2026  ·  Ví dụ số học được ghi rõ; chưa có số đo T4 kèm gói')
    canvas.drawRightString(w-64,29,str(doc.page));canvas.restoreState()

text=(ROOT/'docs'/'CAM_NANG.md').read_text()
chunks=text.split('<!-- PAGEBREAK -->')
story=[]
for i,chunk in enumerate(chunks):
    if i:story.append(PageBreak())
    story.extend(parse_chunk(chunk,i))
doc=SimpleDocTemplate(str(OUT),pagesize=(595.276,841.89),leftMargin=64,rightMargin=64,
    topMargin=57,bottomMargin=51,title='Gemma 4 12B trên Colab T4 - Cẩm nang thực hành',author='Tài liệu đồ án')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
