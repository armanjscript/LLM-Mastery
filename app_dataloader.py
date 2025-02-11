import os
from langchain_community.document_loaders import TextLoader
import pprint

script_directory = os.path.dirname(os.path.abspath(__file__))

data_folder = 'data'
file_name = 'Olympic_History_Part_1.txt'

file_path = os.path.join(script_directory, data_folder, file_name)

loader = TextLoader(file_path=file_path, encoding='utf-8')
data_docs = loader.load()

print(type(data_docs))
pprint.pprint(data_docs)
