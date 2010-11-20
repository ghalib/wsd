from BeautifulSoup import BeautifulStoneSoup

def bad_instance(instance):
    return instance == '\n'

def next_lexelt(lexelt):
    return lexelt.nextSibling.nextSibling

def process_lexelt(lexelt, outfile):
    item = str(lexelt['item'])
    outfile.write('ITEM = %s\n' % item)
    for instance in lexelt:
        if not bad_instance(instance):
            process_instance(instance, outfile)

def extract_senses(instance):
    senses = [str(answer['senseid']) for answer in instance('answer')]
    return senses

def extract_contexts(instance):
    """Extract left and right contexts (relative to the word instance
    we are currently processing) from an instance node.  WINDOW is how
    many words of context on each side we will train on."""
    left = str(instance.context.contents[0]).strip()
    right = str(instance.context.contents[2]).strip()

    return left, right

def process_instance(instance, outfile, training=False):
    outfile.write('INSTANCE\n')
    instance_id = str(instance['id'])
    
    if training:
        senses = extract_senses(instance)
        outfile.write('SENSES = ')
        for sense in senses:
            outfile.write('%s ' % sense)

        outfile.write('\n')

    outfile.write('INSTANCE_ID = %s\n' % instance_id)
    left, right = extract_contexts(instance)
    instance_item = str(instance.context.contents[1].contents[0])
    outfile.write('LEFT_CONTEXT = %s\n' % left)
    outfile.write('INSTANCE_ITEM = %s\n' % instance_item)
    outfile.write('RIGHT_CONTEXT = %s\n' % right)
    outfile.write('END_INSTANCE\n')
    outfile.write('\n\n')

def main(orig_file, result_file):
    infile = open(orig_file)
    outfile = open(result_file, 'w')

    tagtree = BeautifulStoneSoup(infile)
    lexelt = tagtree.lexelt

    while(lexelt):
        process_lexelt(lexelt, outfile)
        outfile.write('END_ITEM\n\n')
        lexelt = next_lexelt(lexelt)

    outfile.write('END\n')

