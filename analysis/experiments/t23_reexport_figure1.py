"""Re-export Figure 1 without the superseded panels that survived cropping.

The delivered Figure 1 shows only the framework schematic, but the file still
carried the page description of two earlier panels below the visible box: the
form ``fzFrm1`` and its four raster objects. Cropping again would not remove
them. This script removes the invocation of that form from the page
description and garbage-collects every object that becomes unreachable, so the
superseded panels are no longer present in the file at all.

The visible rendering is verified to be unchanged at 600 dpi, so the figure
keeps exactly the appearance that was approved.
"""
import sys, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import re
import pymupdf
from lib import env

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'figure_src' / 'Figure_1_with_superseded_panels.pdf'
DST = ROOT / 'figure' / 'Figure_1.pdf'
DROP = 'fzFrm1'
DPI = 600


def render(path):
    d = pymupdf.open(path)
    pix = d[0].get_pixmap(dpi=DPI, alpha=False)
    h = hashlib.sha256(pix.samples).hexdigest()
    size = (pix.width, pix.height)
    d.close()
    return h, size


def outside_objects(path):
    d = pymupdf.open(path)
    p = d[0]
    crop = pymupdf.Rect(p.cropbox)
    n = sum(1 for b in p.get_image_info() if not crop.intersects(pymupdf.Rect(b['bbox'])))
    d.close()
    return n


if not SRC.exists():
    raise SystemExit(f'{SRC} not found\n'
                     '  this script documents and re-runs the Figure 1 re-export; the source '
                     'figure ships with the manuscript, not with the code')
before_hash, before_size = render(SRC)
before_out = outside_objects(SRC)

doc = pymupdf.open(SRC)
page = doc[0]
target = None
for xref, name, parent, bbox in page.get_xobjects():
    stream = doc.xref_stream(xref) if doc.xref_is_stream(xref) else b''
    if re.search(rb'/' + DROP.encode() + rb'\s+Do', stream):
        target = xref
        break
assert target is not None, f'{DROP} is not invoked by any form on the page'
stream = doc.xref_stream(target).decode('latin-1')
cleaned = re.sub(r'q[^q]*?/' + DROP + r'\s+Do\s*Q\s*', '', stream)
if cleaned == stream:
    cleaned = re.sub(r'/' + DROP + r'\s+Do\s*', '', stream)
assert cleaned != stream, 'the form invocation was not found'
doc.update_stream(target, cleaned.encode('latin-1'))

doc.save(DST, garbage=4, clean=True, deflate=True, deflate_images=True, deflate_fonts=True)
doc.close()

after_hash, after_size = render(DST)
after_out = outside_objects(DST)
report = dict(
    source=str(SRC.relative_to(ROOT)), output=str(DST.relative_to(ROOT)),
    removed_form=DROP, render_dpi=DPI,
    raster_objects_outside_visible_box_before=before_out,
    raster_objects_outside_visible_box_after=after_out,
    rendered_size_before=list(before_size), rendered_size_after=list(after_size),
    rendering_sha256_before=before_hash, rendering_sha256_after=after_hash,
    rendering_identical=bool(before_hash == after_hash),
    bytes_before=SRC.stat().st_size, bytes_after=DST.stat().st_size)
env.dump(report, 't23_reexport_figure1')
for k, v in report.items():
    print(f'{k}: {v}')
assert report['rendering_identical'], 'the visible figure changed; refusing to accept the export'
assert after_out == 0, 'objects remain outside the visible box'
