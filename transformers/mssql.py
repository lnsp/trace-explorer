from trace_explorer import transformer
import xml.etree.ElementTree as ET
from collections import defaultdict

op_keywords = {
    'Top': 'Res',
    'Join': 'Hj',
    'Aggregate': 'Agg',
    'Scan': 'Scan',
    'Sort': 'Sort',
    'Streams': 'Other',
    'Spool': 'Buf',
    'Compute': 'Proj',
    'Bitmap Create': 'Other',
    'Clustered Index Seek': 'Other',
    'Concatenation': 'Res',
    'Assert': 'Other',
    'Segment': 'Agg',
    'Union': 'Hj'
}
schema_base = '{http://schemas.microsoft.com/sqlserver/2004/07/showplan}'
rel_op_tag = schema_base + 'RelOp'
runtime_tag = schema_base + 'RunTimeInformation'
runtime_info_tag = schema_base + 'RunTimeCountersPerThread'

columns = [
    'queryNumber', 'profAggRsoShare', 'profBufRsoShare', 'profFilterRsoShare',
    'profHjRsoShare', 'profOtherRsoShare', 'profProjRsoShare',
    'profResRsoShare', 'profScanRsoShare', 'profSortRsoShare']


class Transformer(transformer.Transformer):
    def columns(self):
        return columns

    def transform(self, content, path=None):
        tree = ET.fromstring(content)

        op_map = defaultdict(list)
        for tt in tree.findall('.//' + rel_op_tag):
            logical_op = tt.attrib['LogicalOp']
            for k in op_keywords:
                if k in logical_op:
                    logical_op = op_keywords[k]
            logical_op = 'prof%sRsoShare' % logical_op
            rt_inf = tt.find('./' + runtime_tag)
            if rt_inf is None:
                continue
            total_time = []
            for st in rt_inf.findall('./' + runtime_info_tag):
                total_time.append(float(st.attrib['ActualCPUms']))
            op_map[logical_op].append(max(total_time))
        op_map_sum = sum(sum(op_map[x]) for x in op_map)
        row = {x: sum(op_map[x])/op_map_sum for x in op_map}

        # try to detect queryNumber from path
        if path:
            # we expect a path with suffix _{queryNumber}_{iteration}.xml
            components = path.split('_')
            row['queryNumber'] = int(components[-2])

        return [row[c] if c in row else 0 for c in columns]
