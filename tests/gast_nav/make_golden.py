"""Independent stdlib-only scalar OBS1 golden oracle (does not import C2 code).

Only run intentionally when the owned OBS1 wire version changes; tests consume
committed JSON, never regenerate expected data. All writes stay next to this file.
"""
import base64
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib


def packed(values, bits):
    total = sum(int(v) << (i*bits) for i, v in enumerate(values))
    return total.to_bytes((len(values)*bits+7)//8, 'little')


def generate():
    destination = Path(__file__).with_name('golden')
    for label, cell_age in [('fresh', 0.), ('age050', .5), ('age051', .51)]:
        count = 49*43*3
        state = [0]*count; ages = [15]*count
        hit = 21*49+24
        state[hit] = 2 if cell_age <= .5 else 0
        ages[hit] = min(15, math.ceil(cell_age/.1))
        low = []
        for iy in range(25):
            for ix in range(37):
                x, y = -.9+.05*ix, -.6+.05*iy
                distance = min(.5, math.hypot(max(abs(x)-.025, 0), max(abs(y)-.025, 0)))
                low.append(round(distance/.5*255))
        support = bytes([50]*(49*43))
        fine = bytes(low+[255]*(37*25))
        unknown = packed([1]*(37*25), 1)
        near_ages = [15]*(37*25); near_ages[12*37+18] = 0
        near_age = packed(near_ages, 4)
        payload = packed(state, 2)+packed(ages, 4)+support+fine+unknown+near_age
        meta = struct.pack('<4H17f', 49, 43, 37, 25, .1, -2.4, -2.1, .05, -.9, -.6, .5, .1,
                           .03, .30, .30, .80, .80, 2.20, -1., .02, .5)
        header_len = struct.calcsize('<4sHHIQQQ4d')+len(meta)
        length = header_len+len(payload)+4
        header = struct.pack('<4sHHIQQQ4d', b'OBS1', 1, header_len, length, 1000, 0, 0, 0., 0., 0., 0.)+meta
        packet = header+payload
        packet += struct.pack('<I', zlib.crc32(packet))
        doc = {
            'oracle': 'stdlib scalar reference, no gd_lab/torch imports; nearest hit at cell centre',
            'input': {'point_cloud': [[0., 0., .15]], 'ray_origins': [[-.3, 0., .15]],
                      'ray_directions': [[1., 0., 0.]], 'first_hit_distances': [.3], 'self_hit': [False],
                      'sample_pose': [0., 0., 0., 0.], 'sample_time_ms': 1000, 'seq': 0, 'epoch': 0,
                      'support_height_default_m': 0., 'wide_hit_age_s': cell_age},
            'expected': {'packet_bytes_base64': base64.b64encode(packet).decode(),
                         'packet_sha256': hashlib.sha256(packet).hexdigest(), 'packet_length': length,
                         'wide_state_nonzero': [] if cell_age > .5 else [[0, 21, 24, 2]],
                         'wide_hit_age_code': ages[hit], 'support_default_code': 50,
                         'near_clearance_low': low, 'near_clearance_high_default': 255,
                         'near_unknown_default': True, 'near_age_default_code': 15, 'near_observed_age_spots': [[12, 18, 0]]}}
        (destination/f'obs1_{label}.json').write_text(json.dumps(doc, indent=2)+'\n')


if __name__ == '__main__':
    generate()
