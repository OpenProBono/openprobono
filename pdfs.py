from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Title, Text, NarrativeText, ListItem
import os

def uscode(return_objs: bool = False):
    subdir = '/data/pdf_uscAll@118-34not31/'
    path = os.getcwd() + subdir
    corpus = []
    filepaths = sorted(os.listdir(path))
    print(f'Loading {len(filepaths)} files')
    for i, fname in enumerate(filepaths, start=1):
        print(f' {i}: {fname}')
        if not fname.endswith('.pdf'):
            print('  skipping')
            continue
        filepath = path + fname
        chunks = partition_pdf(filename=filepath, chunking_strategy='auto')
        print(f'  {len(chunks)} chunks')
        text = [chunk.text for chunk in chunks]
        if return_objs:
            title = next((chunk for chunk in chunks if chunk.text.startswith('TITLE')), None)
            if title is None:
                title = fname[:len(fname) - 4]
            else:
                title = title.text
                title_split = title.split()
                j = 0
                while j < len(title_split) and not any(c for c in title_split[j] if c.islower()):
                    j += 1
                title = ' '.join(title_split[:j])
            print('  extracted title: ' + title)
            corpus.append({'source': fname, 'title': title, 'body': text})
        else:
            corpus.append(text)

        # early stop for debugging
        if i > 4:
            break
    return corpus