import numpy as np
from typing import List


def list_to_columnar_string(ListOfStrings:List[str], ncols:int=2, MinimumColumnWidth:int = 0, padding:str = "  ", ordering:str="columns"):
    """
    return a string representation of a list of strings. 
    the strings are lined up in a number of columns, newspaper-style. 
    Parameters are:
    - ListOfStrings : a list of strings to be formatted.
    - ncols : the number of columns to format them into.
    - MinimumColumnWidth : the minimum width for each column. Will be expanded to fit the largest string
    - padding : spacing string between columns, you can even do something fancy like " | "
    - ordering : 'columns' or 'rows'. If 'columns', then the content will appear in columns (like reading a newspaper), otherwise rows.
    """

    # only equal-width columns for now... set to the width of the largest single string    
    maxWidth = MinimumColumnWidth
    maxWidth = np.maximum(maxWidth, max([len(i) for i in ListOfStrings]))

    # find out where to cut the list into chunks
    cut = int(np.ceil(len(ListOfStrings) / ncols))
    # fill out the list so it is divisible by ncols, preventing indexing beyond list
    while len(ListOfStrings) % ncols != 0: # odd number of lines
        ListOfStrings += ['']
    
    # format each column
    ColumnarLines = []    
    for rowi in range(cut): 
        if ordering == 'columns':
            lineString = padding.join([ListOfStrings[rowi+coli*cut].ljust(maxWidth) for coli in range(ncols)])
        else: 
            lineString = padding.join([ListOfStrings[rowi*ncols+coli].ljust(maxWidth) for coli in range(ncols)])
        
        ColumnarLines.append(lineString + '\n')

    # add it all together
    return "".join(ColumnarLines)
