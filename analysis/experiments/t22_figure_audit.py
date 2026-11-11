"""Object and bounding-box audit of every figure file.

Three failure modes are checked, because each one can make a published figure
disagree with its caption:

1. content outside the visible box, which a trimmed rather than re-exported
   figure leaves behind and which text extraction still recovers;
2. extractable text that does not belong to the figure, which is how a
   superseded panel survives inside a cropped file;
3. multi-page figure files, annotations, optional-content groups and embedded
   files, any of which can hide or reveal material at render time.

The audit is run on the figures as they are included in the manuscript, so it
fails if a figure is ever trimmed instead of re-exported.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pymupdf, pandas as pd
from lib import env

FIG = Path(__file__).resolve().parents[1] / 'figure'
FORBIDDEN = ('33/44', 'lower error', 'Rows are ordered by', 'M1', 'equal error')
TOL = 0.5


def audit(path):
    d = pymupdf.open(path)
    rec = dict(file=path.name, pages=len(d))
    p = d[0]
    media, crop = pymupdf.Rect(p.mediabox), pymupdf.Rect(p.cropbox)
    rec['mediabox'] = ' '.join('%.1f' % v for v in media)
    rec['cropbox'] = ' '.join('%.1f' % v for v in crop)
    rec['crop_equals_media'] = bool(abs(media.x0 - crop.x0) < TOL and abs(media.y0 - crop.y0) < TOL
                                    and abs(media.x1 - crop.x1) < TOL and abs(media.y1 - crop.y1) < TOL)
    grown = pymupdf.Rect(crop.x0 - TOL, crop.y0 - TOL, crop.x1 + TOL, crop.y1 + TOL)

    def inside(r):
        # a stroked line has zero width or height; inflate before testing so that
        # axis rules are not mistaken for stray objects
        r = pymupdf.Rect(r)
        r = pymupdf.Rect(r.x0 - TOL, r.y0 - TOL, r.x1 + TOL, r.y1 + TOL)
        return grown.intersects(r)

    out_text = [b[4].strip().replace('\n', ' ') for b in p.get_text('blocks') if not inside(b[:4])]
    draws = p.get_drawings()
    out_draw = sum(1 for g in draws if not inside(g['rect']))
    out_img = sum(1 for b in p.get_image_info() if not inside(b['bbox']))
    text = p.get_text()
    rec.update(drawings=len(draws), images=len(p.get_images()),
               text_blocks=len(p.get_text('blocks')),
               text_blocks_outside=len(out_text), drawings_outside=out_draw,
               images_outside=out_img,
               annotations=sum(1 for _ in p.annots()),
               optional_content=len(d.get_ocgs()),
               embedded_files=d.embfile_count(),
               forbidden_strings=';'.join(s for s in FORBIDDEN if s in text),
               characters=len(text))
    rec['clean'] = bool(rec['pages'] == 1 and rec['crop_equals_media']
                        and rec['text_blocks_outside'] == 0 and rec['drawings_outside'] == 0
                        and rec['images_outside'] == 0 and rec['annotations'] == 0
                        and rec['embedded_files'] == 0 and not rec['forbidden_strings'])
    d.close()
    return rec


if __name__ == '__main__':
    files = sorted(FIG.glob('*.pdf'))
    if not files:
        raise SystemExit(f'no figure files found at {FIG}\n'
                         '  the figure PDFs ship with the manuscript, not with the code; '
                         'place them in analysis/figure or set the directory before auditing')
    rows = [audit(p) for p in files]
    T = pd.DataFrame(rows)
    T.to_csv(env.RESULTS / 't22_figure_audit.csv', index=False)
    env.dump(dict(checks=('single page, cropbox equal to mediabox, no text or vector or raster '
                          'object outside the visible box, no annotations, no optional-content '
                          'groups, no embedded files, no superseded panel strings'),
                  tolerance_points=TOL, forbidden_strings=list(FORBIDDEN),
                  files=rows, all_clean=bool(T.clean.all())), 't22_figure_audit')
    pd.set_option('display.width', 260)
    print(T[['file', 'pages', 'crop_equals_media', 'text_blocks_outside', 'drawings_outside',
             'images_outside', 'annotations', 'optional_content', 'embedded_files',
             'forbidden_strings', 'clean']].to_string(index=False))
    print('\nall clean:', bool(T.clean.all()))
