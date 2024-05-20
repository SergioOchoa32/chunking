import nltk
from nltk.chunk import RegexpParser
from nltk.tokenize import word_tokenize

# Descargar recursos necesarios
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Solicitar al usuario que ingrese el texto
text = input("Ingrese el texto para realizar el chunking: ")

# Tokenización de palabras
words = word_tokenize(text)

# Etiquetado POS (Part-of-Speech Tagging)
tagged = nltk.pos_tag(words)

# Definición de la gramática para diferentes tipos de frases
grammar = """
    NP: {<DT>?<JJ>*<NN>}
    VP: {<VB.*><NP|PP|CLAUSE>+$}
    PP: {<IN><NP>}
    CLAUSE: {<NP><VP>}
"""

# Creación del analizador de chunking
parser = RegexpParser(grammar)

# Aplicación del analizador al texto etiquetado
result = parser.parse(tagged)

# Imprimir el resultado
print(result)
result.draw()
