with open('Line_ext_4.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace('min_solid_ratio = self.cfg.MIN_LINE_SOLID_RATIO', '''min_solid_ratio = getattr(self.cfg, "MIN_LINE_SOLID_RATIO", 0.70)
        min_dashed_ratio = getattr(self.cfg, "MIN_LINE_DASHED_RATIO", 0.20)''')

code = code.replace('''if fill_ratio < min_solid_ratio:
                    dropped_dashed += 1
                    continue''', '''if fill_ratio >= min_solid_ratio:
                    style = "Solid"
                elif fill_ratio >= min_dashed_ratio:
                    style = "Dashed"
                else:
                    dropped_dashed += 1
                    continue''')

code = code.replace('segments.append([idx, orientation, x1, y1, x2, y2])', 'segments.append([idx, orientation, x1, y1, x2, y2, style])')

code = code.replace('def dedupe_segments(self, segments=None, axis_tol=None, touch_tol=None):', 'def dedupe_segments(self, segments=None, axis_tol=None, touch_tol=None):')

code = code.replace('''h_segs = [s for s in segments if s[1] == "Horizontal"]
        v_segs = [s for s in segments if s[1] == "Vertical"]''', '''h_segs_solid = [s for s in segments if s[1] == "Horizontal" and (s[6] == "Solid" if len(s) > 6 else True)]
        v_segs_solid = [s for s in segments if s[1] == "Vertical" and (s[6] == "Solid" if len(s) > 6 else True)]
        h_segs_dashed = [s for s in segments if s[1] == "Horizontal" and len(s) > 6 and s[6] == "Dashed"]
        v_segs_dashed = [s for s in segments if s[1] == "Vertical" and len(s) > 6 and s[6] == "Dashed"]''')

code = code.replace('def fuse_group(segs):', 'def fuse_group(segs, style="Solid"):')

code = code.replace('''if ori == "Horizontal":
                    fused.append([0, ori, s, ax, e, ax])
                else:
                    fused.append([0, ori, ax, s, ax, e])''', '''if ori == "Horizontal":
                    fused.append([0, ori, s, ax, e, ax, style])
                else:
                    fused.append([0, ori, ax, s, ax, e, style])''')

code = code.replace('''fused_h = fuse_group(h_segs)
        fused_v = fuse_group(v_segs)

        all_fused = fused_h + fused_v''', '''fused_h_solid = fuse_group(h_segs_solid, "Solid")
        fused_v_solid = fuse_group(v_segs_solid, "Solid")
        fused_h_dashed = fuse_group(h_segs_dashed, "Dashed")
        fused_v_dashed = fuse_group(v_segs_dashed, "Dashed")

        all_fused = fused_h_solid + fused_v_solid + fused_h_dashed + fused_v_dashed''')

# In filter_isolated_fragments
code = code.replace('''if si[1] == sj[1]:''', '''if si[1] == sj[1] and (si[6] if len(si)>6 else "Solid") == (sj[6] if len(sj)>6 else "Solid"):''')
code = code.replace('''else:
                    pts_i = self._endpoints(si)''', '''elif (si[6] if len(si)>6 else "Solid") == (sj[6] if len(sj)>6 else "Solid"):
                    pts_i = self._endpoints(si)''')

# Now output CSV headers
code = code.replace('"y_end"', '"y_end", "Line_Style"')
code = code.replace('row[2:6]', 'row[2:7]')
code = code.replace('row[3:7]', 'row[3:8]')

with open('Line_ext_4.py', 'w', encoding='utf-8') as f:
    f.write(code)
