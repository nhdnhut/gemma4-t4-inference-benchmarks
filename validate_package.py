"""Check notebooks and embedded shared code without executing install/model cells."""
import ast,base64,hashlib,json,zlib
from pathlib import Path
R=Path(__file__).resolve().parent
for p in sorted((R/'notebooks').glob('*.ipynb')):
    n=json.loads(p.read_text());assert n['nbformat']==4
    assert n['metadata']['colab']['gpuType']=='T4'
    ids=[]
    for i,c in enumerate(n['cells']):
        assert c['cell_type'] in ['code','markdown'];ids.append(c['id'])
        if c['cell_type']=='code':
            assert c['outputs']==[] and c['execution_count'] is None
            ast.parse(c['source'],filename=f'{p.name}:cell{i}')
    assert len(ids)==len(set(ids))
    cell=next(c['source'] for c in n['cells'] if c['cell_type']=='code' and '_compressed = ' in c['source'])
    tree=ast.parse(cell)
    assignment=next(t for t in tree.body if isinstance(t,ast.Assign) and isinstance(t.targets[0],ast.Name) and t.targets[0].id=='_compressed')
    packed=base64.b64decode(ast.literal_eval(assignment.value.args[0]))
    files=json.loads(zlib.decompress(packed))
    assert files=={p.name:p.read_text() for p in (R/'src').glob('*.py')}
    print(p.name,': syntax, cell IDs, clean outputs, exact embedded source OK')
for p in (R/'src').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
print('Shared source syntax OK')
