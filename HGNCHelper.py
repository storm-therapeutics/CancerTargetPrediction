import httplib2 as http
import json

try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse

def getGeneOfficialSymbol(geneName="KRAS"):
    headers = {
        'Accept': 'application/json',
    }

    uri = 'http://rest.genenames.org'
    path = '/search/' + geneName

    target = urlparse(uri+path)
    method = 'GET'
    body = ''

    h = http.Http()

    response, content = h.request(target.geturl(), method, body, headers)

    if response['status'] == '200':
        data = json.loads(content)
        try:
            symbol = data['response']['docs'][0]['symbol']
        except:
            return None
        print('Input symbol: ' + geneName + ' - Official symbol: ' + data['response']['docs'][0]['symbol'])
        return symbol

    else:
        print('Error detected: ' + response['status'])
        print(response)
        print(geneName)
        return None