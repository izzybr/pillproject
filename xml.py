#tree = etree.parse(files[0])
#etree.tostring(tree)
import xml.etree.ElementTree as et

def main():
    tree = et.parse('cellosaurus.xml')
    root = tree.getroot()

    results = []
    for element in root.findall('.//cell-line'):
        key_values = {}
        for key in ['category', 'created', 'last_updated']:
            key_values[key] = element.attrib[key]
        for child in element.iter():
            if child.tag == 'accession':
                key_values['accession type'] = child.attrib['type']
            elif child.tag == 'name' and child.attrib['type'] == 'identifier':
                key_values['name type identifier'] = child.text
            elif child.tag == 'name' and child.attrib['type'] == 'synonym':
                key_values['name type synonym'] = child.text
        results.append([
                # Using the get method of the dict object in case any particular
                # entry does not have all the required attributes.
                 key_values.get('category'            , None)
                ,key_values.get('created'             , None)
                ,key_values.get('last_updated'        , None)
                ,key_values.get('accession type'      , None)
                ,key_values.get('name type identifier', None)
                ,key_values.get('name type synonym'   , None)
                ])

    print(results)