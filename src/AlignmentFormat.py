import xml.etree.ElementTree as etree  # from lxml import etree #for parser
from io import BytesIO
import logging
import tempfile
from six.moves import urllib
from xml.sax.saxutils import quoteattr, escape

# Serialization


def write_alignments(file, alignments, source, target):

    serialize_mapping_to_file(file, alignments, [source], [target])

def __get_ontology_string(onto, name):
    onto_string = ""
    if onto is None:
        return onto_string
    if len(onto) > 0:
        onto_string += "  <" + name + ">\n"
        onto_string += '    <Ontology rdf:about="' + onto[0] + '">\n'
        if len(onto) > 1:
            onto_string += "      <location>" + onto[1] + "</location>\n"
            if len(onto) > 3:
                onto_string += "      <formalism>\n"
                onto_string += (
                    '        <Formalism align:name="'
                    + onto[2]
                    + '" align:uri="'
                    + onto[3]
                    + '"/>\n'
                )
                onto_string += "      </formalism>\n"
        onto_string += "    </Ontology>\n"
        onto_string += "  </" + name + ">\n"
    return onto_string


def __get_extension_string(extension):
    ext_string = ""
    if extension is None:
        return ext_string
    for key, value in extension:
        ext_string += "  <" + key + ">" + value + "</" + key + ">\n"
    return ext_string


def __get_xml_intro(onto_one=None, onto_two=None, extension=None):
    return (
        """<?xml version=\"1.0\" encoding=\"utf-8\"?>
    <rdf:RDF xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment"
      xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
      xmlns:xsd="http://www.w3.org/2001/XMLSchema#">
<Alignment>
  <xml>yes</xml>
  <level>0</level>
  <type>??</type>"""
        + __get_extension_string(extension)
        + __get_ontology_string(onto_one, "onto1")
        + __get_ontology_string(onto_two, "onto2")
    )

def __get_extension_label_and_base(full_uri):
    lastIndex = full_uri.rfind('#')
    if lastIndex > 0:
        return full_uri[lastIndex+1:], full_uri[:lastIndex+1]
    lastIndex = full_uri.rfind('/')
    if lastIndex > 0:
        return full_uri[lastIndex+1:], full_uri[:lastIndex+1]
    return full_uri, full_uri


def __get_mapping_string(source, target, relation, confidence, *args):
    cell_text = ["""
  <map>
    <Cell>
      <entity1 rdf:resource=%s/>
      <entity2 rdf:resource=%s/>
      <relation>%s</relation>
      <measure rdf:datatype="xsd:float">%s</measure>""" % (
        quoteattr(source),
        quoteattr(target),
        escape(str(relation)),
        escape(str(confidence))
    )]
    if len(args) > 0:
        
        for ext_key, ext_value in args[0].items():
            ext_label, ext_base = __get_extension_label_and_base(ext_key)
            ext_label = escape(str(ext_label))
            cell_text.append('      <alignapilocalns:%s xmlns:alignapilocalns=%s>%s</alignapilocalns:%s>' %
                (ext_label, quoteattr(ext_base), escape(str(ext_value)), ext_label))
    cell_text.append("""    </Cell>
  </map>""")
    return '\n'.join(cell_text)


def __get_xml_outro():
    return """
</Alignment>
</rdf:RDF>
"""


def serialize_mapping_to_file(
    file_path, alignment, onto_one=None, onto_two=None, extension=None
):
    """
    Serialize a alignment (iterable of (source, target, relation, confidence)) to a given file.
    :param file_path: represent the path of the file as a string
    :param alignment: iterable of (source, target, relation, confidence, extensions) - extensions is a dictionary and is optional
    :param onto_one: description of ontology one as (id, url, formalismName, formalismURI)
    :param onto_two: description of ontology two as (id, url, formalismName, formalismURI)
    :param extension: iterable of (key, value) describing the alignment
    """
    with open(file_path, "w", encoding="utf-8") as out_file:
        out_file.write(__get_xml_intro(onto_one, onto_two, extension))
        for correspondence in alignment:
            out_file.write(__get_mapping_string(*correspondence))
        out_file.write(__get_xml_outro())


def serialize_mapping_to_tmp_file(
    alignment, onto_one=None, onto_two=None, extension=None
):
    """
    Serialize a alignment (iterable of (source, target, relation, confidence)) to a file in the systems temp folder
    (which is not deleted) and return a file url of that file.
    :param alignment: iterable of (source, target, relation, confidence, extensions) - extensions is a dictionary and is optional
    :param onto_one: description of ontology one as (id, url, formalismName, formalismURI)
    :param onto_two: description of ontology two as (id, url, formalismName, formalismURI)
    :param extension: iterable of (key, value) describing the alignment
    :return: file url of the generated alignment file like file://tmp/alignment_123.rdf
    """
    with tempfile.NamedTemporaryFile(
        "w", prefix="alignment_", suffix=".rdf", delete=False
    ) as out_file:
        out_file.write(__get_xml_intro(onto_one, onto_two, extension))
        for correspondence in alignment:
            out_file.write(__get_mapping_string(*correspondence))
        out_file.write(__get_xml_outro())
    return "file:" + str(urllib.request.pathname2url(out_file.name))


# Parser


class AlignmentHandler(object):
    def __init__(self):
        self.base = "{http://knowledgeweb.semanticweb.org/heterogeneity/alignment}"
        self.rdf = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
        self.text = ""
        self.alignment = []
        self.one_cell = ["", "", "", "", {}]
        self.extension = {}
        self.onto1 = ""
        self.onto2 = ""
        self.onto_temp = ["", ""]
        self.in_cell = False
        self.used_tags = set(
            [
                self.base + name
                for name in [
                    "entity1",
                    "entity2",
                    "relation",
                    "measure",
                    "Cell",
                    "map",
                    "Alignment",
                    "xml",
                    "level",
                    "type",
                    "onto1",
                    "onto2",
                    "Ontology",
                    "location",
                    "formalism",
                    "Formalism",
                ]
            ]
        )
        self.used_tags.add(self.rdf + "RDF")

    def start(self, name, attrs):
        if name == self.base + "entity1":
            self.one_cell[0] = attrs[self.rdf + "resource"]  # .encode('utf-8')
        elif name == self.base + "entity2":
            self.one_cell[1] = attrs[self.rdf + "resource"]  # .encode('utf-8')
        elif name == self.base + "Ontology":
            self.onto_temp[0] = attrs[self.rdf + "about"]  # .encode('utf-8')
        elif name == self.base + "Cell":
            self.in_cell = True
        self.text = ""

    def end(self, name):
        if name == self.base + "relation":
            self.one_cell[2] = self.text.strip()
        elif name == self.base + "measure":
            self.one_cell[3] = self.text.strip()
        elif name == self.base + "Cell":
            self.alignment.append(self.one_cell)
            self.one_cell = ["", "", "", "", {}]
            self.in_cell = False
        elif name == self.base + "location":
            self.onto_temp[1] = self.text.strip()
        elif name == self.base + "onto1":
            if self.onto_temp[0] == "" and self.onto_temp[1] == "":
                self.onto_temp[0] = self.text.strip()
            self.onto1 = list(self.onto_temp)
        elif name == self.base + "onto2":
            if self.onto_temp[0] == "" and self.onto_temp[1] == "":
                self.onto_temp[0] = self.text.strip()
            self.onto2 = list(self.onto_temp)
        elif name not in self.used_tags:
            key = name.replace("{","",1).replace("}","",1) #name[name.index("}") + 1 :]
            if self.in_cell:
                self.one_cell[4][key] = self.text
            else:
                self.extension[key] = self.text

    def data(self, chars):
        self.text += chars

    def close(self):
        pass

def remove_cell_extensions(alignment):
    for c in alignment:
        c.pop()

def parse_mapping_from_string(s, parse_cell_extensions=False):
    """
    Parses a alignment from a given string.
    :param s: a string representing a alignment in alignment format
    :param parse_cell_extensions: if true, also parses the cell extensions 
    :return: (alignment: list of (source, target, relation, confidence, extensions - which are parsed only when parse_cell_extensions is True), 
    onto1 as (id, url, formalismName, formalismURI),
    onto2 similar to onto1, 
    extension (iterable of key, values)
    )
    """
    handler = AlignmentHandler()
    etree.parse(BytesIO(s.encode("utf-8")), etree.XMLParser(target=handler))
    if parse_cell_extensions == False:
        remove_cell_extensions(handler.alignment)
    return handler.alignment, handler.onto1, handler.onto2, handler.extension


def parse_mapping_from_file(source, parse_cell_extensions=False):
    """
    Parses a alignment from a filename or file object.
    :param source: is a filename or file object containing a alignment in alignment format
    :param parse_cell_extensions: if true, also parses the cell extensions 
    :return: (alignment: list of (source, target, relation, confidence, extensions - which are parsed only when parse_cell_extensions is True), 
    onto1 as (id, url, formalismName, formalismURI),
    onto2 similar to onto1, 
    extension (iterable of key, values)
    )
    """
    handler = AlignmentHandler()
    etree.parse(source, etree.XMLParser(target=handler))
    if parse_cell_extensions == False:
        remove_cell_extensions(handler.alignment)
    return handler.alignment, handler.onto1, handler.onto2, handler.extension


# if __name__ == "__main__":
#     logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
#     logging.info("Generate")
#     t = [('http://test.dwfwegwegwegtrh/12&34_' + str(i), 'http://test2.dwfwegwegwegtrh/' + str(i), '=', 1.0)
#       for i in range(200)]
#     logging.info("write")
#     serialize_mapping_to_file('test.txt', t)
#     # bla = serialize_mapping_to_tmp_file(t)
#     # logging.info(bla)
