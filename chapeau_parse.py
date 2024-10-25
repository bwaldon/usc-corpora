# TODO: when there are multiple sentences in the chapeau, put sentences that are not immediately before list elements in separate `header_constparse` and `header_depparse` attributes

from utils import replace_placeholder_subtree, replace_depparse_placeholder, coordinate_depparses, coordinate_constparses
from nltk.parse import DependencyGraph
from nltk.tree import ParentedTree
import re

class chapeau_parse():

  def __init__(self, chapeau, tokenizer, parser):
    # TODO: add support for multiple sentences in the chapeau
    self.header_constparses = None 
    self.header_depparses = None

    self.constparse, self.depparse, self.continuation_constparses, self.continuation_depparses, self.tok = self.get_chapeau_parse(chapeau, tokenizer, parser)

    self.constparse[0].append(ParentedTree('.', ['.']))

    period_head_address =  max([n['address'] for n in self.depparse.nodes.values()])
    period_address = period_head_address + 1
    self.depparse.nodes[period_address] = {
            'word': ".",
            'lemma': ".",
            'ctag': 'PUNCT',
            'tag': 'PUNCT',
            'address': period_address,
            'deps': {}
        }
    self.depparse.nodes[period_address]['head'] = period_head_address
    self.depparse.nodes[period_head_address]['deps'].setdefault('punct', [])
    self.depparse.nodes[period_head_address]['deps']['punct'].append(period_address)

    if self.tok[-1] != '.':
      self.tok += ' .'
    
  def get_chapeau_parse(self, chapeau, tokenizer, parser):

    assert chapeau.name == 'chapeau', 'Expected an element of type "chapeau", got "' + chapeau.name + '"'

    placeholder_sequence = '<PLCHLDR>'

    def _get_tokens(string):
      return " ".join([t.text for t in tokenizer(string).sentences[0].tokens])

    # strip final punctuation from chapeau if present
    chapeau_text = chapeau.get_text()
    if chapeau_text[-1] in [':', '—', '-']:
      chapeau_text = chapeau_text[:-1]

    tok = ""
    chapeau_tok = _get_tokens(chapeau_text)
    tok += chapeau_tok
    chapeau_frame_text = chapeau_tok + " " + placeholder_sequence
    
    parent = chapeau.find_parent()

    parent_identifier = parent.get('identifier')
    # use regex to find all elements with the same identifier plus /s, where s is a sequence not containing a slash
    list_elements = parent.find_all(attrs={'identifier': re.compile(parent_identifier + r'/[^/]+')})

    list_elements_constparses = []
    list_elements_depparses = []
    list_elements_finalccs = []
    list_elements_finalpunct = []

    coord_tok = ""
    for lelem in list_elements:
      finalccs = None
      finalpunct = None
      if lelem.find('chapeau'):
        constparse, depparse, _, _, _tok = self.get_chapeau_parse(lelem.find('chapeau'), tokenizer, parser)
        list_elements_constparses.append(constparse)
        list_elements_depparses.append(depparse)
        list_elements_finalccs.append(finalccs)
        list_elements_finalpunct.append(finalpunct)
        coord_tok += " " + _tok.strip()
      else:
        string = lelem.find('content').get_text()
        list_tok = _get_tokens(string)
        coord_tok += " " + list_tok.strip()
        tokens = list_tok.split(" ")
        if tokens[-1] in ['or', 'and']:
          finalccs = tokens[-1]
          tokens = tokens[:-1]
        if tokens[-1] in [';', ',', '.']:
          finalpunct = tokens[-1]
          tokens = tokens[:-1]
        list_elements_finalccs.append(finalccs)
        list_elements_finalpunct.append(finalpunct)
        lelem_parse = parser(" ".join(tokens))
        if len(lelem_parse.sentences) > 1:
          # if there are multiple sentences in the list element, treat as asyndetic coordination
          none_array = [None] * len(lelem_parse.sentences)
          lelem_constparse = coordinate_constparses([ParentedTree.fromstring(str(s.constituency)) for s in lelem_parse.sentences], none_array, none_array)
          lelem_depparse = coordinate_depparses([DependencyGraph("{:c}".format(s), top_relation_label='root') for s in lelem_parse.sentences], none_array, none_array)
        else:
          lelem_constparse = ParentedTree.fromstring(str(lelem_parse.sentences[0].constituency))
          lelem_depparse = DependencyGraph("{:c}".format(lelem_parse.sentences[0]), top_relation_label='root')
        list_elements_constparses.append(lelem_constparse)
        list_elements_depparses.append(lelem_depparse)

    tok += " " + coord_tok.strip()
    list_elements_constparses = [p[0] for p in list_elements_constparses]

    # get the indentation level of the chapeau element (or its parent) with regex that matches first numeral after string 'indent' in the class attribute
    indent_idx = None
    if chapeau.get('class') is not None:
      indent_idx = re.search(r'indent(\d+)', chapeau.get('class')).group(1)
    elif parent.get('class') is not None:
      indent_idx = re.search(r'indent(\d+)', parent.get('class')).group(1)
    # if none is found, the continuation is the first <continuation> element
    if indent_idx is None:
      continuation = parent.find('continuation')
    else: 
      # find the first <continuation> element with the same indentation. note: the attribute may include sequences after the number, so we use a regex to match the number
      continuation = parent.find('continuation', attrs={'class': re.compile(r'indent' + indent_idx + r'.*')})

    if continuation is None:
      chapeau_frame_text += ' '
    elif continuation is not None and list_elements_finalpunct[-1] != '.':
    # if continuation is present and the final punctuation of the final element is not a period, add the continuation (up to the first period, inclusive) to the chapeau
      cont_tok = _get_tokens(continuation.get_text().split('.')[0])
      cont_tok += " ."
      tok += " " + cont_tok
      chapeau_frame_text += " " + cont_tok

    chapeau_frame_parse = parser(chapeau_frame_text).sentences[0]

    # build the constituency tree

    conjunction = coordinate_constparses(list_elements_constparses, list_elements_finalccs, list_elements_finalpunct)
    chapeau_frame_constparse = ParentedTree.fromstring(str(chapeau_frame_parse.constituency))
    tree_out = replace_placeholder_subtree(chapeau_frame_constparse, conjunction, placeholder_sequence)

    # build the dependency tree

    conjunction = coordinate_depparses(list_elements_depparses, list_elements_finalccs, list_elements_finalpunct)
    chapeau_frame_depparse = DependencyGraph("{:c}".format(chapeau_frame_parse), top_relation_label='root')
    depparse_out = replace_depparse_placeholder(chapeau_frame_depparse, conjunction, placeholder_sequence)

    # parse subsequent continuation sentences if there are any

    continuation_constparses = []
    continuation_depparses = []

    if continuation is not None and list_elements_finalpunct[-1] == '.':
      cont_tok = _get_tokens(continuation.get_text().split('.')[0])
      cont_tok += " ."
      tok += " " + cont_tok
      continuation_parse = parser(cont_tok).sentences[0]
      continuation_constparse = ParentedTree.fromstring(str(continuation_parse.constituency))
      continuation_depparse = DependencyGraph("{:c}".format(continuation_parse), top_relation_label='root')
      continuation_constparses.append(continuation_constparse)
      continuation_depparses.append(continuation_depparse)
    
    if continuation is not None and len(continuation.get_text().split('.')) > 1:
      #[1:-1] to skip the last element, which is an empty string
      for continuation_sent in continuation.get_text().split('.')[1:-1]:
        cont_tok = _get_tokens(continuation_sent) + " ."
        tok += " " + cont_tok
        continuation_parse = parser(cont_tok).sentences[0]
        continuation_constparse = ParentedTree.fromstring(str(continuation_parse.constituency))
        continuation_depparse = DependencyGraph("{:c}".format(continuation_parse), top_relation_label='root')
        continuation_constparses.append(continuation_constparse)
        continuation_depparses.append(continuation_depparse)

    return tree_out, depparse_out, continuation_constparses, continuation_depparses, tok
  
  def continuation_constparses_to_pdf(self, filename):
    """
    Generate PDF images of the constituency parses of the continuation sentences.
    """
    for i, constparse in enumerate(self.continuation_constparses):
      from svgling import draw_tree
      import cairosvg

      # Render the tree to SVG
      layout = draw_tree(constparse)
      svg_data = layout.get_svg()

      # Convert SVG to PDF
      cairosvg.svg2pdf(bytestring=svg_data.tostring(), write_to=filename.split('.')[0] + '_' + str(i) + '.pdf')

  def continuation_depparses_to_pdf(self, filename):
    """
    Generate PDF images of the dependency parses of the continuation sentences.
    """
    for i, depparse in enumerate(self.continuation_depparses):
      from graphviz import Source
      dot = Source(depparse.to_dot())
      dot.render(filename.split('.')[0] + '_' + str(i), format='pdf', cleanup=True)

  def depparse_to_pdf(self, filename):
      """
      Generate a DF image of a dependency parse.

      Args:
          depparse (DependencyGraph): The dependency graph to visualize.
          filename (str): The name of the output PDF file.
      """
      from graphviz import Source
      dot = Source(self.depparse.to_dot())
      dot.render(filename.split('.')[0], format='pdf', cleanup=True) 

  def constparse_to_pdf(self, filename):
    """
    Generate a PDF image of a constituency parse.

    Args:
        constparse (ParentedTree): The constituency parse to visualize.
        filename (str): The name of the output PDF file.
    """
    from svgling import draw_tree
    import cairosvg

    # Render the tree to SVG
    layout = draw_tree(self.constparse)
    svg_data = layout.get_svg()

    # Convert SVG to PDF
    cairosvg.svg2pdf(bytestring=svg_data.tostring(), write_to=filename)