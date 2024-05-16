import re
import os
import sys
import pdb
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base)
from typing import List, Tuple

from tools.multiprocessingTool import MultiprocessingTool
from common.typing import OEAFileType
from config import Config


pref = {
    "rdf:": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs:": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd:": "http://www.w3.org/2001/XMLSchema#",
    "owl:": "http://www.w3.org/2002/07/owl#", 
    "skos:": "http://www.w3.org/2004/02/skos/core#",
    "dc:": "http://purl.org/dc/terms/",
    "foaf:": "http://xmlns.com/foaf/0.1/",
    "vcard:": "http://www.w3.org/2006/vcard/ns#",
    "dbp:": "http://dbpedia.org/",
    "y1:": "http://www.mpii.de/yago/resource/",
    "y2:": "http://yago-knowledge.org/resource/",
    "geo:": "http://www.geonames.org/ontology#",
    'wiki:': 'http://www.wikidata.org/',
    'schema:': 'http://schema.org/',
    'freebase:': 'http://rdf.freebase.com/',
    'dbp-zh:': 'http://zh.dbpedia.org/',
    'dbp-fr:': 'http://fr.dbpedia.org/',
    'dbp-ja:': 'http://ja.dbpedia.org/',
    'dbp-de:': 'http://de.dbpedia.org/',

}


def strip_square_brackets(s):
    # s = ""
    if s.startswith('"'):
        rindex = s.rfind('"')
        if rindex > 0:
            s = s[:rindex + 1]
    else:
        if s.startswith('<'):
            s = s[1:]
        if s.endswith('>'):
            s = s[:-1]
    return s


def retrieve_rdf_data(data):
    # Use regular expressions (regex) to find the parts of the data
    value = re.search(r'"(.+)"', data).group(1)
    try:
        # check if float and round off to 2 decimal points
        value = round(float(value), 2)
    except ValueError:
        pass

    type_ = re.search(r'<(.+?)#(.+)>', data)
    
    if type_:
        type_ = type_.group(2)
    else:
        type_search = re.search(r'"\^\^(\w+)', data)
        if type_search:
            type_ = type_search.group(1)
        else:
            type_ = None

    return f'{value}'

def compress_uri(uri):
    if not Config.simplify_url:
        return uri
    else:
        if uri.startswith("http://"):
            for key, val in pref.items():
                if uri.startswith(val):
                    surfix = uri.rsplit('/', 1)[-1]
                    uri = key + surfix
    return uri


def simplify_attr(fact):
    if len(fact) != 3:
        return None
    if not fact[2].startswith('"'):
        fact[2] = ''.join(('"', fact[2], '"'))
    if Config.remove_ID_attr and fact[1].endswith("ID"):
        return None
    else:
        return compress_uri(fact[0]), compress_uri(fact[1]), retrieve_rdf_data(fact[2])


def simplify_rel(fact) -> Tuple:
    return compress_uri(fact[0]), compress_uri(fact[1]), compress_uri(fact[2])


def simplify_truth(fact, is_rel_mapping=False, have_scores=False) -> Tuple:
    if not have_scores:
        return compress_uri(fact[0]), compress_uri(fact[1])
    else:
        if is_rel_mapping:
            return compress_uri(fact[0]), compress_uri(fact[1]), fact[2], fact[3]
        else:
            return compress_uri(fact[0]), compress_uri(fact[1]), fact[2]


class parser(object):
    name2oriname = {}
    oriname2name = {}
    kg1_name2oriname_dict_saved = False
    kg2_name2oriname_dict_saved = False

    def __init__(self, file_type: OEAFileType, have_scores=False, is_rel_mapping=False):
        self.file_type = file_type
        self.have_scores = have_scores
        self.is_rel_mapping = is_rel_mapping

    @staticmethod
    def _unique_name(name):
        while name in parser.name2oriname:
            name += "_"
        return name

    @staticmethod
    def _compress_uri(uri):
        if uri in parser.oriname2name:
            return parser.oriname2name[uri]
        else:
            uri = compress_uri(uri)
            parser.oriname2name[uri] = uri
            parser.name2oriname[uri] = uri
            return uri

    def simplify_rel(self, fact) -> Tuple:
        if fact[0] in parser.oriname2name:
            parsed_h = parser.oriname2name[fact[0]]
        else:
            parsed_h = self._unique_name(compress_uri(fact[0]))
            parser.name2oriname[parsed_h] = fact[0]
            parser.oriname2name[fact[0]] = parsed_h
        if fact[2] in parser.oriname2name:
            parsed_t = parser.oriname2name[fact[2]]
        else:
            parsed_t = self._unique_name(compress_uri(fact[2]))
            parser.name2oriname[parsed_t] = fact[2]
            parser.oriname2name[fact[2]] = parsed_t
        parsed_r = self._compress_uri(fact[1])

        return (parsed_h, parsed_r, parsed_t)


    def simplify_truth(self, fact, is_rel_mapping=False, have_scores=False) -> Tuple:
        if not have_scores:
            return parser.oriname2name[fact[0]], parser.oriname2name[fact[1]]
        else:
            if is_rel_mapping:
                return self._compress_uri(fact[0]), self._compress_uri(fact[1]), fact[2], fact[3]
            else:
                return parser.oriname2name[fact[0]], parser.oriname2name[fact[1]], fact[2]

    def simplify_attr(self, fact):
        if len(fact) != 3:
            return None
        if not fact[2].startswith('"'):
            fact[2] = ''.join(('"', fact[2], '"'))
        if Config.remove_ID_attr and fact[1].endswith("ID"):
            return None
        else:
            return parser.oriname2name[fact[0]], compress_uri(fact[1]), retrieve_rdf_data(fact[2])

    def parse(self, data):
        if self.file_type == OEAFileType.attr:
            return self.simplify_attr(data)
        elif self.file_type == OEAFileType.rel:
            return self.simplify_rel(data)
        elif self.file_type == OEAFileType.truth:
            return self.simplify_truth(data, self.is_rel_mapping, self.have_scores)
