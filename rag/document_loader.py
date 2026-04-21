# Import library
from langchain_community.document_loaders import PyPDFLoader

# Create a document loader for rag_vs_fine_tuning.pdf
loader = PyPDFLoader('rag_vs_fine_tuning.pdf')

# Load the document
data = loader.load()
print(data[0])



# CSV file loader

# Import library
from langchain_community.document_loaders.csv_loader import CSVLoader

# Create a document loader for fifa_countries_audience.csv
loader = CSVLoader('fifa_countries_audience.csv')

# Load the document
data = loader.load()
print(data[0])



# HTML document loader

from langchain_community.document_loaders import UnstructuredHTMLLoader

# Create a document loader for unstructured HTML
loader = UnstructuredHTMLLoader("white_house_executive_order_nov_2023.html")

# Load the document
data = loader.load()

# Print the first document
print(data[0])

# Print the first document's metadata
print(data[0].metadata)