import transformer
import json
import collections

op_map = {
    'TableScan': 'Scan',
    'ExternalScan': 'XtScan',
    'Filter': 'Filter',
    'Join': 'Hj',
    'Aggregate': 'Agg',
    'GroupingSets': 'Other',
    'WindowFunction': 'Other',
    'Sort': 'Sort',
    'SortWithLimit': 'Sort',
    'Flatten': 'Flat',
    'JoinFilter': 'Filter',
    'UnionAll': 'Other',
    'Result': 'Res',
}
op_cols = sorted([
    'profScanRsoShare',
    'profXtScanRsoShare',
    'profProjRsoShare',
    'profSortRsoShare',
    'profFilterRsoShare',
    'profResRsoShare',
    'profDmlRsoShare',
    'profHjRsoShare',
    'profBufRsoShare',
    'profFlatRsoShare',
    'profBloomRsoShare',
    'profAggRsoShare',
    'profBandRsoShare',
    'profPercentileRsoShare',
    'profUdtfRsoShare',
])
stat_map = {
    'execTime': 'xpExecTime',
    'scanFiles': 'scanFiles',
    'scanBytes': 'scanBytes',
    'compilationTime': 'compilationTime',
    'scheduleTime': 'scheduleTime',
}
stat_cols = sorted([
    'scanFiles',
    'scanBytes',
    'execTime',
    'compilationTime',
    'scheduleTime',
])

class Transformer(transformer.Transformer):
    def columns(self):
        return ['queryNumber', 'queryId'] + stat_cols + op_cols
    
    def transform(self, content):
        # expect merged metadata file
        obj = json.loads(content)
        queryNumber = obj['queryNumber']

        info_json = json.loads(obj['info'])
        profile_json = json.loads(obj['profile'])

        info_stats = info_json['data']['queries'][0]['stats']
        query_stats = [queryNumber, info_json['data']['queries'][0]['id']] + [info_stats[stat_map[c]] for c in stat_cols]

        # compute total execution time
        total_time = 0.0
        operator_time = collections.defaultdict(float)
        for step in profile_json['data']['steps']:
            total_time += step['graphData']['global']['totalStats']['value']
            for node in step['graphData']['nodes']:
                operator_time['prof%sRsoShare' % op_map[node['name']]] += node['totalStats']['value']

        # normalize operator time
        profile_shares = [operator_time[c] / total_time for c in op_cols]
        return query_stats + profile_shares
