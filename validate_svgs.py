import xml.etree.ElementTree as ET
import os

base = r'c:\Users\Shobhit Raj\Downloads\shobhit-jarvis-exact-readme\New folder\lucifer9973'

svgs = [
    'assets/hero-dark.svg',
    'assets/hero-light.svg',
    'assets/jarvis-dark.svg',
    'assets/jarvis-light.svg',
    'dark.svg',
    'light.svg',
    'assets/dark.svg',
    'assets/light.svg',
]

for svg in svgs:
    path = os.path.join(base, svg)
    if not os.path.exists(path):
        print(f'❌ MISSING: {svg}')
        continue
    size = os.path.getsize(path)
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        w = root.get('width')
        h = root.get('height')
        # Count elements
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        anim = len(root.findall('.//svg:animate', ns))
        trans = len(root.findall('.//svg:animateTransform', ns))
        motion = len(root.findall('.//svg:animateMotion', ns))
        groups = len(root.findall('.//svg:g', ns))
        print(f'✅ {svg}: {size:>7}B, {w}x{h}, {groups}g, {anim}a, {trans}t, {motion}m')
    except ET.ParseError as e:
        print(f'❌ PARSE ERROR: {svg}: {e}')
    except Exception as e:
        print(f'❌ ERROR: {svg}: {e}')

print('\n--- Validation Complete ---')
