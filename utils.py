from nltk.parse import DependencyGraph
from nltk.tree import ParentedTree
import copy
from tqdm import tqdm

def remove_notes(element):
  # remove note elements and their contents
  note_sub_elements = element.find_all('note')
  for note in note_sub_elements:
    note.decompose()
  return element

def get_maximal_chapeaus(element):
  """ given an xml element, return all 'maximal' chapeau elements (chapeau elements that are not nested within another chapeau element)"""
  chapeaus = element.find_all('chapeau')
  maximal_chapeaus = {}
  for chapeau in chapeaus:
    if chapeau.find_parent('paragraph') is None:
      parent_with_id = chapeau.find_parent(attrs={'id': True})
      parent_id = parent_with_id.get('id')
      maximal_chapeaus[parent_id] = chapeau
  return maximal_chapeaus

def get_contentelems_outside_chapeaus(element, maximal_chapeaus):
  """ given an xml element, return all content elements that are not nested within a chapeau element's parent"""
  contentelems = element.find_all('content')
  contentelems_outside_chapeaus = []
  maximal_chapeaus_parentids = maximal_chapeaus.keys()
  for contentelem in tqdm(contentelems):
    if all([contentelem.find_parent(attrs={'id': parentid}) is None for parentid in maximal_chapeaus_parentids]):
      contentelems_outside_chapeaus.append(contentelem)
  return contentelems_outside_chapeaus

def coordinate_depparses(depgraphs, conjunctions, finalpuncts):

    conjunctions += [None]
    merged_graph = DependencyGraph()

    offset = 0

    for node in depgraphs[0].nodes.values():
      if node['head'] is None:
          merged_graph.nodes[node['address']] = node.copy()
          head_address = node['deps']['root'][0]

    for i in range(len(depgraphs)):

      if conjunctions[i-1] is not None:

        offset += 1
        conj_address = offset

        # Offset addresses for cc and depgraph to avoid clashes

        merged_graph.nodes[offset] = {
            'word': conjunctions[i-1],
            'lemma': conjunctions[i-1],
            'ctag': 'CONJ',
            'tag': 'CONJ',
            'address': conj_address,
            'deps': {}
        
        }

      for node in depgraphs[i].nodes.values():
          if node['head'] is not None:  # Skip the root node
              node_copy = node.copy()
              node_copy['address'] += offset
              node_copy['head'] += offset
              node_copy['deps'] = {key: [v + offset for v in val] for key, val in node_copy['deps'].items()}
              merged_graph.nodes[node_copy['address']] = node_copy
          else:
            coordinate_address = node['deps']['root'][0] + offset

      if i > 0:
        merged_graph.nodes[head_address]['deps'].setdefault('conj', [])
        merged_graph.nodes[head_address]['deps']['conj'].append(coordinate_address)

      if conjunctions[i-1] is not None:
        merged_graph.nodes[conj_address]['head'] = coordinate_address
        merged_graph.nodes[coordinate_address]['deps'].setdefault('cc', [])
        merged_graph.nodes[coordinate_address]['deps']['cc'].append(conj_address)

      # -1 to account for the root nodes of subsequent conjuncts
      offset += len(depgraphs[i].nodes) - 1

      if finalpuncts[i] == ',' or finalpuncts[i] == ';':

        offset += 1
        punct_address = offset

        # Offset addresses for cc and depgraph to avoid clashes

        merged_graph.nodes[offset] = {
            'word': finalpuncts[i],
            'lemma': finalpuncts[i],
            'ctag': 'PUNCT',
            'tag': 'PUNCT',
            'address': punct_address,
            'deps': {}
        }

        punct_head_address = max([n['address'] for n in merged_graph.nodes.values()]) - 1
        merged_graph.nodes[punct_address]['head'] = punct_head_address
        merged_graph.nodes[punct_head_address]['deps'].setdefault('punct', [])
        merged_graph.nodes[punct_head_address]['deps']['punct'].append(punct_address)

    nodes_sorted = sorted(merged_graph.nodes.values(), key=lambda x: x['address'])

    merged_graph.nodes = {i: node for i, node in enumerate(nodes_sorted)}

    return merged_graph

def coordinate_constparses(list_elements_constparses, list_elements_finalccs, list_elements_finalpunct):
       # conjoin the list element parses into a conjunction
    list_elements_labels = [p.label() for p in list_elements_constparses]
    majority_label = max(set(list_elements_labels), key = list_elements_labels.count)
    conjuncts = []
    for i, parse in enumerate(list_elements_constparses):
      conjuncts.append(ParentedTree.fromstring(str(parse)))
      if list_elements_finalpunct[i] is not None:
        if list_elements_finalpunct[i] == ';':
          conjuncts.append(ParentedTree(':', [';']))
        elif list_elements_finalpunct[i] == ',':
          conjuncts.append(ParentedTree(',', [',']))
      if list_elements_finalccs[i] is not None:
        conjuncts.append(ParentedTree('CC', [list_elements_finalccs[i]]))
    conjunction = ParentedTree(majority_label, conjuncts)
    return conjunction

def replace_depparse_placeholder(depparse, replacement, placeholder_sequence):
    """
    Replace the node of a dependency graph whose text matches 'placeholder_sequence'
    with a replacement subtree, whose head inherits the dependency relations of the placeholder node.
    """

    depparse = copy.deepcopy(depparse)
    replacement = copy.deepcopy(replacement)
    merged_graph = DependencyGraph()

    # Add nodes and dependencies from depparse
    for node in depparse.nodes.values():
        merged_graph.nodes[node['address']] = node.copy()
        if node['head'] is None:
            head_address = node['deps']['root'][0]
        if node['word'] == placeholder_sequence:
            placeholder_address = node['address']
            placeholder_head = node['head']

    replacement_length = len(replacement.nodes)

    # in depparse: offset head, address, and dep indices by the length of the replacement subtree
    for node in merged_graph.nodes.values():
        if node['head'] is not None:
          if node['head'] >= placeholder_address:  # Skip the root node
            node['head'] += replacement_length - 1
          if node['address'] >= placeholder_address:
            node['address'] += replacement_length - 1
          node['deps'] = {key: [v + replacement_length - 1 if v >= placeholder_address else v for v in val] for key, val in node['deps'].items()}

    # in replacement: offset head, address, and dep indices by placeholder address
    for node in replacement.nodes.values():
      if node['head'] is not None:
        node['head'] += placeholder_address - 1
        node['address'] += placeholder_address -1
        node['deps'] = {key: [v + placeholder_address - 1 for v in val] for key, val in node['deps'].items()}
      else:
        replacement_address = node['deps']['root'][0] + placeholder_address - 1

    nodes = list(merged_graph.nodes.values())
    for node in replacement.nodes.values():
        if node['head'] is not None:
          node_copy = node.copy()
          nodes.append(node_copy)

    placeholder_address += replacement_length - 1

    merged_graph.nodes = {node['address']: node for node in nodes}

    placeholder_head = merged_graph.nodes[placeholder_address]['head']
    placeholder_rel = merged_graph.nodes[placeholder_address]['rel']
    merged_graph.nodes[placeholder_head]['deps'].setdefault(placeholder_rel, [])
    merged_graph.nodes[placeholder_head]['deps'][placeholder_rel].append(replacement_address)

    del merged_graph.nodes[placeholder_address]

    for node in merged_graph.nodes.values():
      if node['head'] == placeholder_address:
        node['head'] = replacement_address
        rel = node['rel']
        merged_graph.nodes[replacement_address]['deps'].setdefault(rel, [])
        merged_graph.nodes[replacement_address]['deps'][rel].append(node['address'])

    for node in merged_graph.nodes.values():
      for key, val in node['deps'].items():
        if placeholder_address in val:
          val.remove(placeholder_address)

    nodes_sorted = sorted(merged_graph.nodes.values(), key=lambda x: x['address'])

    merged_graph.nodes = {i: node for i, node in enumerate(nodes_sorted)}

    for i, node in enumerate(merged_graph.nodes.values()):
      node['address'] = i
      if node['head'] is not None and node['head'] > placeholder_address:
        node['head'] -= 1
      node['deps'] = {key: [v - 1 if v > placeholder_address else v for v in val] for key, val in node['deps'].items()}

    return merged_graph

def replace_placeholder_subtree(tree, replacement, placeholder_sequence):
    """
    Replace the subtree in 'tree' whose leaves match '[placeholder_sequence]'
    with the given 'replacement', and remove any unary branching nodes above it.

    Args:
        tree (ParentedTree): The NLTK ParentedTree in which to find and replace the subtree.
        replacement (ParentedTree): The subtree to replace the placeholder subtree with.
        placeholder_sequence (list): The sequence of terminal leaves to search for.

    Returns:
        ParentedTree: The modified tree.
    """
    # Traverse the tree to find the subtree that matches placeholder_sequence
    for subtree in tree.subtrees():
        if subtree.leaves() == [placeholder_sequence]:
            parent = subtree.parent()  # Get the parent node

            # Replace the subtree with the replacement
            idx = parent.index(subtree)
            parent[idx] = replacement

            # Now remove unary branching nodes above the replacement
            while parent is not None and len(parent) == 1:
                grandparent = parent.parent()
                if grandparent:
                    # Replace parent with the replacement (unary node pruning)
                    idx = grandparent.index(parent)
                    grandparent[idx] = parent[0]
                    parent = grandparent
                else:
                    break

            break  # Exit after replacement is done
    return tree

   
    

